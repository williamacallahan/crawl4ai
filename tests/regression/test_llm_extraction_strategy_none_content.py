"""Regression tests for the LLM no-content error-block sentinel in
``LLMExtractionStrategy.extract`` / ``aextract``.

Background (commit ``b138c949``, issue #1606): when the LLM returns empty
``message.content`` the strategy builds an ``{"error": True, "tags": ["error"],
"content": "LLM returned no content (finish_reason: ...)"}`` sentinel block.
A pre-existing unconditional ``for block in blocks: block["error"] = False``
normalization loop immediately afterwards clobbered that ``error: True`` back
to ``False``, so downstream consumers that key off ``block["error"]`` (notably
``deploy/docker/api.py``'s ``LlmExtractionRejected`` filter) could not see the
no-content failure.

The fix normalizes only parsed provider content, preserving the locally
constructed no-content sentinel regardless of user-data tags.
These tests pin that contract for both the sync and async paths and guard
against the ``setdefault`` alternative, which would misclassify user-data
blocks whose schema legitimately contains ``"error": true`` as data.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from crawl4ai import LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# ``extract`` resolves ``perform_completion_with_backoff`` from the
# module-level import at the top of ``extraction_strategy.py``.
SYNC_PATCH_TARGET = "crawl4ai.extraction_strategy.perform_completion_with_backoff"

# ``aextract`` imports ``aperform_completion_with_backoff`` lazily inside the
# method body (``from .utils import ...``), so patch the source module.
ASYNC_PATCH_TARGET = "crawl4ai.utils.aperform_completion_with_backoff"


def _make_response(content, finish_reason="content_filter"):
    """Build a fake litellm-style completion response.

    ``LLMExtractionStrategy`` constructs a ``TokenUsage`` from
    ``response.usage`` (completion/prompt/total tokens and the
    ``*_tokens_details`` sub-objects) *before* testing ``content``, so the
    usage shape must be present for execution to reach the ``if not content:``
    branch; omitting it raises ``AttributeError`` in the outer ``except`` and
    masks the clobber.
    """
    usage = SimpleNamespace(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
        completion_tokens_details=None,  # falls through to {} by the strategy
        prompt_tokens_details=None,      # falls through to {} by the strategy
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _strategy(**kwargs):
    return LLMExtractionStrategy(
        llm_config=LLMConfig(provider="fake/model", api_token="tok"),
        **kwargs,
    )


def test_extract_none_content_preserves_error_flag():
    """The ``not content`` sentinel block must keep ``error: True`` after the
    normalization loop (the bug clobbered it to ``False``)."""
    fake_resp = _make_response(None, finish_reason="content_filter")
    strategy = _strategy()
    with patch(SYNC_PATCH_TARGET, return_value=fake_resp):
        blocks = strategy.extract("http://example.com", 0, "<html></html>")

    assert len(blocks) == 1
    assert blocks[0]["error"] is True, (
        "BUG: error flag was clobbered to False; the normalization loop "
        "must preserve the locally constructed no-content sentinel."
    )
    assert "LLM returned no content" in blocks[0]["content"]
    assert "finish_reason: content_filter" in blocks[0]["content"]
    assert blocks[0]["tags"] == ["error"]
    assert blocks[0]["index"] == 0


@pytest.mark.asyncio
async def test_aextract_none_content_preserves_error_flag():
    """The async twin ``aextract`` must also preserve ``error: True``."""
    fake_resp = _make_response(None, finish_reason="content_filter")
    strategy = _strategy()
    with patch(ASYNC_PATCH_TARGET, return_value=fake_resp):
        blocks = await strategy.aextract("http://example.com", 0, "<html></html>")

    assert len(blocks) == 1
    assert blocks[0]["error"] is True, (
        "BUG: aextract clobbered the error flag to False."
    )
    assert "LLM returned no content" in blocks[0]["content"]
    assert "finish_reason: content_filter" in blocks[0]["content"]
    assert blocks[0]["tags"] == ["error"]


def test_extract_user_data_error_field_is_clobbered_to_false():
    """A user-data block whose schema legitimately contains ``"error": true``
    as DATA must be clobbered to ``error=False``.

    This pins success normalization against the ``setdefault``
    alternative described in the bug report: ``setdefault`` would preserve the
    user's ``error: true`` data field, and ``deploy/docker/api.py``'s
    ``extraction_errors`` filter (``block.get("error")``) would then
    misclassify a successful extraction as a strategy-level
    ``LlmExtractionRejected`` failure.
    """
    content = json.dumps([{"error": True, "status": 503, "service": "downstream"}])
    fake_resp = _make_response(content, finish_reason="stop")
    strategy = _strategy(force_json_response=True)
    with patch(SYNC_PATCH_TARGET, return_value=fake_resp):
        blocks = strategy.extract("http://example.com", 0, "<html></html>")

    assert len(blocks) == 1
    assert blocks[0]["error"] is False, (
        "User-data 'error: true' field must be clobbered to False; otherwise "
        "api.py would misclassify a successful extraction as a strategy-level "
        "LlmExtractionRejected failure."
    )
    assert blocks[0]["status"] == 503
    assert blocks[0]["service"] == "downstream"


def _api_extraction_errors(content):
    """Mirror of ``deploy/docker/api.process_llm_extraction``'s error filter:
    ``[str(b.get('content') or 'LLM extraction failed') for b in content
    if isinstance(b, dict) and b.get('error')]``."""
    return [
        str(block.get("content") or "LLM extraction failed")
        for block in content
        if isinstance(block, dict) and block.get("error")
    ]


def test_no_content_block_is_detected_by_api_error_filter():
    """After the fix, the docker API error filter must surface the no-content
    block (it was invisible pre-fix because ``error`` was clobbered)."""
    fake_resp = _make_response(None, finish_reason="content_filter")
    strategy = _strategy()
    with patch(SYNC_PATCH_TARGET, return_value=fake_resp):
        blocks = strategy.extract("http://example.com", 0, "<html></html>")

    errors = _api_extraction_errors(blocks)
    assert len(errors) == 1
    assert "LLM returned no content" in errors[0]
    assert "content_filter" in errors[0]


def test_mixed_no_content_and_data_blocks_keep_per_block_error_flag():
    """In default ``apply_chunking=True`` library mode a no-content chunk's
    block is merged alongside data blocks from other chunks via
    ``extracted_content.extend(...)``. The error flag must be preserved
    per-block so a consumer can detect the partial failure."""
    # Chunk i: no content -> error sentinel
    no_content_resp = _make_response(None, finish_reason="content_filter")
    # Chunk j: valid data
    data_resp = _make_response(
        json.dumps([{"name": "Alice"}]), finish_reason="stop"
    )
    strategy_data = _strategy(force_json_response=True)
    strategy_none = _strategy(force_json_response=True)

    with patch(SYNC_PATCH_TARGET, return_value=no_content_resp):
        none_blocks = strategy_none.extract("http://example.com", 0, "<html></html>")
    with patch(SYNC_PATCH_TARGET, return_value=data_resp):
        data_blocks = strategy_data.extract("http://example.com", 1, "<html></html>")

    merged = []
    merged.extend(none_blocks)
    merged.extend(data_blocks)

    # The no-content sentinel survives with error=True ...
    error_blocks = [b for b in merged if b.get("error")]
    assert len(error_blocks) == 1
    assert "LLM returned no content" in error_blocks[0]["content"]
    # ... while the data block is normalized to error=False.
    data_blocks_merged = [b for b in merged if not b.get("error")]
    assert len(data_blocks_merged) == 1
    assert data_blocks_merged[0]["name"] == "Alice"
    # The docker API filter sees exactly the one real failure.
    errors = _api_extraction_errors(merged)
    assert len(errors) == 1
    assert "content_filter" in errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [["error"], None, ["ordinary"]])
async def test_successful_schema_tags_do_not_become_strategy_errors(tags):
    record = {"tags": tags, "error": True, "status": 503}
    response = _make_response(json.dumps([record]), finish_reason="stop")
    strategy = _strategy(force_json_response=True)
    with patch(SYNC_PATCH_TARGET, return_value=response):
        sync_blocks = strategy.extract("https://example.com", 0, "content")
    with patch(ASYNC_PATCH_TARGET, return_value=response):
        async_blocks = await strategy.aextract("https://example.com", 0, "content")
    expected = [{**record, "error": False}]
    assert sync_blocks == expected
    assert async_blocks == expected
