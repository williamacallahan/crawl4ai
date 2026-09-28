"""Regression guards for the browser-pool slot held idle across LLM roundtrips.

``handle_llm_qa`` (``/llm/{url}``) and the ``FilterType.LLM`` branch of
``handle_markdown_request`` (``/md`` with ``f=llm``) previously released
their pooled browser only in ``finally``, so a browser-pool slot (and its
``ADMISSION_SEM`` permit) sat idle for the whole LLM HTTP roundtrip wrapped
in ``async with llm_permit(...)`` (default 300s, auto-renewed). The fix
releases the crawler immediately after the crawl result is materialised,
before entering ``llm_permit``, and sets ``crawler = None`` so ``finally``
does not double-release.

These tests guard the load-bearing invariants of that fix:
- the crawler is released *before* the LLM permit is acquired (so the
  permit-acquire wait never holds a pool slot);
- the crawler is released *exactly once* (``release_crawler`` is not
idempotent for healthy pooled crawlers -- it never sets
``_docker_admission_released`` -- so a second release in ``finally`` would
leak an ``ADMISSION_SEM`` permit and over-report pool capacity);
- a failed crawl still releases the crawler exactly once via ``finally``;
- the non-LLM markdown branches keep the pre-fix single-release-in-finally
  pattern (scope guard: the early release must not be over-applied).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest
from utils import load_config


class PermitRedis:
    """Stub for the real ``llm_permit`` Redis dependency (offline)."""

    def __init__(self, acquired=True):
        self.acquired = acquired
        self.hset = AsyncMock()
        self.expire = AsyncMock()
        self.eval = AsyncMock(return_value=1)

    async def set(self, *_args, **_kwargs):
        return self.acquired


def _llm_config():
    return {**load_config(), "llm": {"provider": "openai/qwen3-32b", "api_key": "test-only"}}


def _resolve_llm():
    return lambda *_a, **_kw: {
        "provider": "openai/qwen3-32b",
        "api_token": "test-only",
        "temperature": 0.0,
        "base_url": None,
        "extra_args": {
            "timeout": 300,
            "num_retries": 0,
            "reasoning_effort": "low",
            "max_tokens": 4096,
        },
    }


def _completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )


def _md_completion():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<content># Filtered</content>")
            )
        ]
    )


def _patch_boundaries(monkeypatch, release_mock, *, crawl_result):
    """Stub crawl/egress boundaries; keep ``release_crawler`` observable."""
    async def fake_arun(_crawler, *args, **kwargs):
        return crawl_result

    async def fake_get_crawler(_bc):
        return SimpleNamespace()

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_crawler_arun", fake_arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_mock)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve_llm())


def _qa_crawl_result():
    return SimpleNamespace(
        success=True,
        markdown=SimpleNamespace(fit_markdown="context", raw_markdown="context"),
    )


def _md_crawl_result():
    return SimpleNamespace(
        success=True,
        cleaned_html="<h1>Example</h1>",
        markdown=SimpleNamespace(raw_markdown="# Example", fit_markdown=""),
    )


# ---------------------------------------------------------------------------
# handle_llm_qa: crawler released before the LLM permit is acquired, exactly once
# ---------------------------------------------------------------------------


def test_handle_llm_qa_releases_crawler_before_permit_acquire(monkeypatch):
    """The crawler must be released before ``llm_permit`` tries to acquire the
    Redis permit (``redis.set``), so the permit-acquire wait never holds a
    browser-pool slot. ``llm_permit`` calls ``redis.set`` first, so we observe
    ``release_crawler``'s call count at that point."""
    release = AsyncMock()
    _patch_boundaries(monkeypatch, release, crawl_result=_qa_crawl_result())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", AsyncMock(return_value=_completion()))

    redis = PermitRedis(acquired=True)
    permit_acquire_order = []

    async def observing_set(*args, **kwargs):
        permit_acquire_order.append(release.await_count)
        return True

    redis.set = observing_set

    asyncio.run(
        api.handle_llm_qa(
            "https://example.com", "question", _llm_config(), redis=redis,
        )
    )

    assert permit_acquire_order == [1], (
        "release_crawler must be called before llm_permit acquires the permit; "
        f"release count at permit acquire: {permit_acquire_order}"
    )


def test_handle_llm_qa_releases_crawler_exactly_once(monkeypatch):
    """The crawler must be released exactly once. A second release in
    ``finally`` would leak an ``ADMISSION_SEM`` permit: ``release_crawler``
    is not idempotent for healthy pooled crawlers (it never sets
    ``_docker_admission_released``), so the ``crawler = None`` guard is
    load-bearing."""
    release = AsyncMock()
    _patch_boundaries(monkeypatch, release, crawl_result=_qa_crawl_result())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", AsyncMock(return_value=_completion()))

    asyncio.run(
        api.handle_llm_qa(
            "https://example.com", "question", _llm_config(), redis=PermitRedis(),
        )
    )

    assert release.await_count == 1, (
        f"release_crawler must be called exactly once; got {release.await_count}"
    )


def test_handle_llm_qa_releases_crawler_on_crawl_failure(monkeypatch):
    """A failed crawl raises before the early-release point, so the only
    release happens in ``finally`` -- exactly once. No crawler is leaked on
    crawl failure."""
    release = AsyncMock()
    failed = SimpleNamespace(
        success=False,
        error_message="boom",
        markdown=SimpleNamespace(fit_markdown="", raw_markdown=""),
    )
    _patch_boundaries(monkeypatch, release, crawl_result=failed)
    monkeypatch.setattr(
        api, "aperform_completion_with_backoff",
        AsyncMock(side_effect=AssertionError("provider must not be called on crawl failure")),
    )

    with pytest.raises(api.HTTPException) as error:
        asyncio.run(
            api.handle_llm_qa(
                "https://example.com", "question", _llm_config(), redis=PermitRedis(),
            )
        )

    assert error.value.status_code == 502
    assert release.await_count == 1
    api.aperform_completion_with_backoff.assert_not_awaited()


# ---------------------------------------------------------------------------
# handle_markdown_request (LLM branch): same invariants
# ---------------------------------------------------------------------------


def test_handle_markdown_llm_releases_crawler_before_permit_acquire(monkeypatch):
    """``/md?f=llm``: crawler released before ``llm_permit`` acquires."""
    release = AsyncMock()
    _patch_boundaries(monkeypatch, release, crawl_result=_md_crawl_result())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", AsyncMock(return_value=_md_completion()))

    redis = PermitRedis(acquired=True)
    permit_acquire_order = []

    async def observing_set(*args, **kwargs):
        permit_acquire_order.append(release.await_count)
        return True

    redis.set = observing_set

    markdown = asyncio.run(
        api.handle_markdown_request(
            redis, "https://example.com", api.FilterType.LLM,
            query="Keep the title", config=_llm_config(),
        )
    )

    assert markdown == "# Filtered"
    assert permit_acquire_order == [1]


def test_handle_markdown_llm_releases_crawler_exactly_once(monkeypatch):
    """``/md?f=llm``: no double-release in ``finally``."""
    release = AsyncMock()
    _patch_boundaries(monkeypatch, release, crawl_result=_md_crawl_result())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", AsyncMock(return_value=_md_completion()))

    asyncio.run(
        api.handle_markdown_request(
            PermitRedis(), "https://example.com", api.FilterType.LLM,
            query="Keep the title", config=_llm_config(),
        )
    )

    assert release.await_count == 1


def test_handle_markdown_llm_releases_crawler_on_crawl_failure(monkeypatch):
    """``/md?f=llm``: a failed crawl releases exactly once in ``finally``."""
    release = AsyncMock()
    failed = SimpleNamespace(
        success=False,
        error_message="boom",
        cleaned_html="",
        markdown=SimpleNamespace(raw_markdown="", fit_markdown=""),
    )
    _patch_boundaries(monkeypatch, release, crawl_result=failed)
    monkeypatch.setattr(
        api, "aperform_completion_with_backoff",
        AsyncMock(side_effect=AssertionError("provider must not be called on crawl failure")),
    )

    with pytest.raises(api.HTTPException) as error:
        asyncio.run(
            api.handle_markdown_request(
                PermitRedis(), "https://example.com", api.FilterType.LLM,
                query="Keep the title", config=_llm_config(),
            )
        )

    assert error.value.status_code == 502
    assert release.await_count == 1
    api.aperform_completion_with_backoff.assert_not_awaited()


# ---------------------------------------------------------------------------
# Non-LLM markdown branches: scope guard (no early release, no LLM call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", [api.FilterType.RAW, api.FilterType.FIT, api.FilterType.BM25])
def test_markdown_non_llm_branches_release_in_finally_only(monkeypatch, filter_type):
    """RAW/FIT/BM25 branches must keep the pre-fix pattern: a single release
    in ``finally`` and never invoke the LLM provider. Guards against the
    early release being over-applied to non-LLM filters."""
    release = AsyncMock()
    _patch_boundaries(
        monkeypatch, release,
        crawl_result=SimpleNamespace(
            success=True,
            cleaned_html="<h1>Example</h1>",
            markdown=SimpleNamespace(raw_markdown="# raw", fit_markdown="# fit"),
        ),
    )
    monkeypatch.setattr(
        api, "aperform_completion_with_backoff",
        AsyncMock(side_effect=AssertionError("LLM provider must not be called for non-LLM filters")),
    )

    markdown = asyncio.run(
        api.handle_markdown_request(
            PermitRedis(), "https://example.com", filter_type,
            query="ignored for non-LLM", config=_llm_config(),
        )
    )

    expected = "# raw" if filter_type == api.FilterType.RAW else "# fit"
    assert markdown == expected
    assert release.await_count == 1
    api.aperform_completion_with_backoff.assert_not_awaited()
