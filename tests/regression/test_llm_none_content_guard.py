"""Regression tests for None LLM content handling in helper call sites.

`RegexExtractionStrategy.generate_pattern` and
`EmbeddingStrategy.map_query_semantic_space` previously dereferenced
``response.choices[0].message.content`` without checking for ``None``, raising
unhelpful ``AttributeError`` / ``TypeError`` instead of a clear error. These
tests guard against the None-content path regressing (the same guard already
exists in ``LLMExtractionStrategy`` and ``JsonElementExtractionStrategy``).
"""

import json

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from crawl4ai import LLMConfig
from crawl4ai.adaptive_crawler import EmbeddingStrategy
from crawl4ai.extraction_strategy import RegexExtractionStrategy


# ``generate_pattern`` imports ``perform_completion_with_backoff`` at module
# level, so patch the name bound in the ``extraction_strategy`` namespace.
PATTERN_PATCH_TARGET = "crawl4ai.extraction_strategy.perform_completion_with_backoff"

# ``map_query_semantic_space`` imports it lazily inside the method body
# (``from .utils import ...``), so patch the source module.
ADAPTIVE_PATCH_TARGET = "crawl4ai.utils.perform_completion_with_backoff"


def _make_response(content, finish_reason="stop"):
    """Build a fake litellm-style completion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# RegexExtractionStrategy.generate_pattern
# ---------------------------------------------------------------------------

def test_generate_pattern_none_content_raises_clear_error():
    """None content must raise a clear ValueError citing finish_reason
    (previously: AttributeError: 'NoneType' object has no attribute 'replace')."""
    fake_resp = _make_response(None, finish_reason="length")
    llm_config = LLMConfig(provider="fake/model", api_token="tok")

    with patch(PATTERN_PATCH_TARGET, return_value=fake_resp):
        with pytest.raises(ValueError) as exc_info:
            RegexExtractionStrategy.generate_pattern(
                label="price", html="<html><span>9.99</span></html>", llm_config=llm_config
            )

    msg = str(exc_info.value)
    assert "LLM returned no content" in msg
    assert "finish_reason: length" in msg


def test_generate_pattern_valid_content_still_returns_pattern_dict():
    """Happy path is unaffected: valid JSON content still returns a {label: pattern} dict."""
    expected = {"price": r"\d+\.\d{2}"}
    fake_resp = _make_response(json.dumps(expected), finish_reason="stop")
    llm_config = LLMConfig(provider="fake/model", api_token="tok")

    with patch(PATTERN_PATCH_TARGET, return_value=fake_resp):
        result = RegexExtractionStrategy.generate_pattern(
            label="price", html="<html><span>$9.99</span></html>", llm_config=llm_config
        )

    assert result == expected
    import re
    re.compile(result["price"])


# ---------------------------------------------------------------------------
# EmbeddingStrategy.map_query_semantic_space
# ---------------------------------------------------------------------------

def _adaptive_strategy():
    return EmbeddingStrategy(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        query_llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token="q-key"),
    )


@pytest.mark.asyncio
async def test_map_query_semantic_space_none_content_raises_clear_error():
    """None content must raise a clear ValueError citing finish_reason
    (previously: TypeError: the JSON object must be str, bytes or bytearray,
    not NoneType), and must short-circuit before any embedding work."""
    strategy = _adaptive_strategy()
    fake_resp = _make_response(None, finish_reason="length")

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(strategy, "_get_embeddings", new_callable=AsyncMock) as emb_mock:
            with pytest.raises(ValueError) as exc_info:
                await strategy.map_query_semantic_space("test query", n_synthetic=10)

    assert "LLM returned no content" in str(exc_info.value)
    assert "finish_reason: length" in str(exc_info.value)
    emb_mock.assert_not_called()


@pytest.mark.asyncio
async def test_map_query_semantic_space_valid_content_still_returns_embeddings():
    """Happy path is unaffected: valid JSON content still yields
    (train_embeddings, train_queries) with the original query kept first."""
    strategy = _adaptive_strategy()
    n_synthetic = 10
    n_total = int(n_synthetic * 1.3)  # 13 variations requested by the method
    content = json.dumps({"queries": [f"variation {i}" for i in range(n_total)]})
    fake_resp = _make_response(content, finish_reason="stop")

    # n_total queries -> n_validation = max(2, int(n_total * 0.2)) = 2,
    # train_queries = [query] + (n_total - 2) = 1 + 11 = 12.
    expected_train_count = 1 + (n_total - max(2, int(n_total * 0.2)))
    fake_embeddings = np.random.rand(expected_train_count, 384).astype(np.float32)

    with patch(ADAPTIVE_PATCH_TARGET, return_value=fake_resp):
        with patch.object(
            strategy, "_get_embeddings", new_callable=AsyncMock, return_value=fake_embeddings
        ):
            train_embeddings, train_queries = await strategy.map_query_semantic_space(
                "test query", n_synthetic=n_synthetic
            )

    assert np.array_equal(train_embeddings, fake_embeddings)
    assert train_queries[0] == "test query"
    assert len(train_queries) == expected_train_count
