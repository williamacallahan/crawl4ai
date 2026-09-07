"""Regression tests for LLM response-shape handling in
``EmbeddingStrategy.map_query_semantic_space``.

Bug: the prompt asked the model for "a JSON array of strings" while the parser
did ``variations['queries'].copy()``, assuming a JSON object with the exact key
``"queries"``. ``response_format={"type": "json_object"}`` only constrains the
top-level JSON to be an object; it pins no key names, and providers that drop
``response_format`` (``litellm.drop_params = True``) may emit the bare array the
prompt literally requests. Whenever the model did not happen to return
``{"queries": [...]}``, the method crashed with ``TypeError`` (bare array) or
``KeyError`` (any other wrapping key), before the first page was crawled.

The fix (a) makes the prompt explicitly request ``{"queries": [...]}`` so the
prompt and ``json_object`` mode agree, and (b) normalizes the parsed response
via ``EmbeddingStrategy._extract_query_variations``, which accepts a bare array,
an object with a ``"queries"`` key, or an object whose wrapping key is anything
else, and raises a clear ``ValueError`` for anything unrecognized.

Runs fully offline: mocks ``perform_completion_with_backoff`` (the LLM call) and
``_get_embeddings`` (the embedding call) so no network, API key, or
sentence-transformers model is required.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest

from crawl4ai import LLMConfig
from crawl4ai.adaptive_crawler import EmbeddingStrategy


# ``map_query_semantic_space`` imports ``perform_completion_with_backoff``
# lazily inside the method body (``from .utils import ...``), so patch the source
# module rather than the name bound in ``adaptive_crawler``.
ADAPTIVE_PATCH_TARGET = "crawl4ai.utils.perform_completion_with_backoff"


def _make_response(content, finish_reason="stop"):
    """Build a fake litellm-style completion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


def _adaptive_strategy():
    return EmbeddingStrategy(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        query_llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token="q-key"),
    )


def _expected_train_count(n_variations, n_synthetic=10):
    """Mirror the train/validation split math in map_query_semantic_space."""
    n_validation = max(2, int(n_variations * 0.2))
    return 1 + (n_variations - n_validation)


# ---------------------------------------------------------------------------
# map_query_semantic_space: end-to-end shape acceptance (mocked LLM/embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bare_array_response_works():
    """The bare JSON array the prompt body literally describes must work
    (previously: TypeError: list indices must be integers or slices, not str)."""
    strategy = _adaptive_strategy()
    n_synthetic = 10
    n_total = int(n_synthetic * 1.3)  # 13
    content = json.dumps([f"variation {i}" for i in range(n_total)])
    fake_resp = _make_response(content)
    train_count = _expected_train_count(n_total, n_synthetic)
    fake_embeddings = np.random.rand(train_count, 384).astype(np.float32)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(
            strategy, "_get_embeddings", new_callable=AsyncMock, return_value=fake_embeddings
        ):
            train_embeddings, train_queries = await strategy.map_query_semantic_space(
                "test query", n_synthetic=n_synthetic
            )

    assert np.array_equal(train_embeddings, fake_embeddings)
    assert train_queries[0] == "test query"
    assert len(train_queries) == train_count


@pytest.mark.asyncio
async def test_dict_with_wrong_wrapping_key_works():
    """A JSON object whose wrapping key isn't literally "queries" must work
    (previously: KeyError: 'queries')."""
    strategy = _adaptive_strategy()
    n_synthetic = 10
    n_total = int(n_synthetic * 1.3)  # 13
    content = json.dumps({"variations": [f"variation {i}" for i in range(n_total)]})
    fake_resp = _make_response(content)
    train_count = _expected_train_count(n_total, n_synthetic)
    fake_embeddings = np.random.rand(train_count, 384).astype(np.float32)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(
            strategy, "_get_embeddings", new_callable=AsyncMock, return_value=fake_embeddings
        ):
            train_embeddings, train_queries = await strategy.map_query_semantic_space(
                "test query", n_synthetic=n_synthetic
            )

    assert train_queries[0] == "test query"
    assert len(train_queries) == train_count


@pytest.mark.asyncio
async def test_dict_with_queries_key_works():
    """The happy path the original parser happened to accept must still work."""
    strategy = _adaptive_strategy()
    n_synthetic = 10
    n_total = int(n_synthetic * 1.3)  # 13
    content = json.dumps({"queries": [f"variation {i}" for i in range(n_total)]})
    fake_resp = _make_response(content)
    train_count = _expected_train_count(n_total, n_synthetic)
    fake_embeddings = np.random.rand(train_count, 384).astype(np.float32)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(
            strategy, "_get_embeddings", new_callable=AsyncMock, return_value=fake_embeddings
        ):
            train_embeddings, train_queries = await strategy.map_query_semantic_space(
                "test query", n_synthetic=n_synthetic
            )

    assert np.array_equal(train_embeddings, fake_embeddings)
    assert train_queries[0] == "test query"
    assert len(train_queries) == train_count


@pytest.mark.asyncio
async def test_unrecognized_object_shape_raises_clear_error():
    """A JSON object with no list value must raise a clear ValueError instead of
    KeyError, and skip the embedding step."""
    strategy = _adaptive_strategy()
    content = json.dumps({"count": 3, "label": "queries"})
    fake_resp = _make_response(content)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(strategy, "_get_embeddings", new_callable=AsyncMock) as emb_mock:
            with pytest.raises(ValueError) as exc_info:
                await strategy.map_query_semantic_space("test query", n_synthetic=10)

    assert "contained no list of strings" in str(exc_info.value)
    emb_mock.assert_not_called()


@pytest.mark.asyncio
async def test_empty_variations_raises_clear_error_and_skips_embeddings():
    """Empty list (or all-empty items) must raise a clear ValueError and never
    reach the embedding step."""
    strategy = _adaptive_strategy()
    content = json.dumps([])
    fake_resp = _make_response(content)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(strategy, "_get_embeddings", new_callable=AsyncMock) as emb_mock:
            with pytest.raises(ValueError) as exc_info:
                await strategy.map_query_semantic_space("test query", n_synthetic=10)

    assert "no query variations" in str(exc_info.value)
    emb_mock.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_requests_queries_key_object_shape():
    """Guard against the prompt and parser drifting apart again: the prompt
    passed to the LLM must explicitly request {"queries": [...]}, which is the
    shape the parser is built around and the shape ``json_object`` mode can
    actually produce (a top-level object, not a bare array)."""
    strategy = _adaptive_strategy()
    captured = {}

    def _capture(**kwargs):
        captured["prompt"] = kwargs.get("prompt_with_variables")
        return _make_response(json.dumps({"queries": ["v"] * 13}))

    with patch(ADAPTIVE_PATCH_TARGET, side_effect=_capture):
        with patch.object(
            strategy,
            "_get_embeddings",
            new_callable=AsyncMock,
            return_value=np.random.rand(12, 384).astype(np.float32),
        ):
            await strategy.map_query_semantic_space("test query", n_synthetic=10)

    prompt = captured["prompt"]
    assert prompt is not None
    assert '"queries"' in prompt
    assert "JSON array of strings" not in prompt.replace("JSON object", "")


# ---------------------------------------------------------------------------
# _extract_query_variations: direct guard for the non-obvious multi-list branch
# ---------------------------------------------------------------------------

def test_extract_prefers_single_string_list_when_multiple_lists():
    """When a dict without a "queries" key contains several list-valued fields,
    the normalizer must pick the single all-string list (not a non-string list
    or an arbitrary one). Guards the trickiest branch of the normalizer."""
    out = EmbeddingStrategy._extract_query_variations(
        {"variations": ["a", "b"], "meta": [1, 2]}
    )
    assert out == ["a", "b"]


# ---------------------------------------------------------------------------
# Script runner (matches the style of the other tests/regression/*.py files)
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("EmbeddingStrategy query-variation shape regression tests")
    print("=" * 60)

    test_fns = [
        test_bare_array_response_works,
        test_dict_with_wrong_wrapping_key_works,
        test_dict_with_queries_key_works,
        test_unrecognized_object_shape_raises_clear_error,
        test_empty_variations_raises_clear_error_and_skips_embeddings,
        test_prompt_requests_queries_key_object_shape,
        test_extract_prefers_single_string_list_when_multiple_lists,
    ]

    for fn in test_fns:
        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()
        print(f"PASS: {fn.__name__}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
