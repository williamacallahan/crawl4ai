"""Anti-bot desync: a later retry's ``aprocess_html`` failure must not return
or cache an earlier attempt's challenge page.

``AsyncWebCrawler.arun`` runs an outer retry loop (``_max_attempts``) over an
inner proxy loop (``_proxy_list``).  Per attempt, commit 8401b204 moved the
``is_blocked(status_code, html)`` call to run BEFORE ``aprocess_html`` so the
loop-local ``_blocked`` flag is refreshed from the freshly fetched response.
``crawl_result`` is only replaced AFTER ``aprocess_html`` succeeds.  This
created a desync window: if a later attempt fetches a CLEAN page (setting
``_blocked = False``) and then ``aprocess_html`` raises, ``crawl_result`` is
left pointing at the EARLIER attempt's blocked/challenge page while the
post-loop block re-check trusts the stale ``_blocked = False`` and leaves
that challenge page marked ``success = True``.  The cache write gate (added in
cb01f495) only checks ``crawl_result.success``, so the challenge page is then
persisted with ``success = True`` -- poisoning the cache permanently because
the read-path guard that discards failed rows never fires on a ``True`` row.

The fix clears ``crawl_result`` right after ``is_blocked`` runs and before
``aprocess_html`` is awaited, so an ``aprocess_html`` exception leaves
``crawl_result = None`` (never a prior attempt's stale value).  The post-loop
``else`` branch then builds a clean failure result and the cache write gate
sees ``success = False`` and skips the write.

These tests lock in the fix and guard its scope:

* the desync (blocked-then-clean-but-raises) returns a failure, not the
  challenge page, and is never written to the cache (the reported bug);
* end-to-end against a real isolated SQLite cache DB, the challenge page is
  not persisted and a subsequent ENABLED crawl is NOT served a poisoned hit
  -- the site is re-fetched and the clean page is cached;
* no regression: a retry that recovers (aprocess_html raises on a clean fetch,
  the next attempt succeeds) still returns and caches the live page;
* the desync holds across BOTH loop axes (outer retry AND inner proxy loop);
* the existing ``RendererResourceLimitError`` clear-and-break path on a later
  attempt after a blocked first attempt still returns a failure, not the
  stale challenge page.

All tests mock the browser fetch (``crawler_strategy.crawl``) and
``aprocess_html``; the E2E test exercises the real
``aget_cached_url``/``acache_url`` against a throwaway SQLite DB + content
store under ``tmp_path`` (no DB mocks for the round-trip).

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_antibot_desync_cache_poisoning.py
"""
import os
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_configs import CrawlerRunConfig, ProxyConfig
from crawl4ai.async_crawler_strategy import RendererResourceLimitError
from crawl4ai.browser_manager import RendererKill
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult
from crawl4ai.utils import ensure_content_dirs

URL = "https://example.com"

# A Cloudflare-style anti-bot challenge page.  ``is_blocked(403, CHALLENGE_HTML)``
# returns ``(True, ...)`` -- the precondition for the desync (an earlier
# attempt that hits a block page and continues the retry loop).
CHALLENGE_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<h1>Just a moment...</h1>"
    "<p>Checking your browser before accessing the site. This may take "
    "a few seconds to complete.</p>"
    "<form id=\"challenge-form\" action=\"/cdn-cgi/challenge-platform/\">"
    "<input name=\"__cf_chl_f_tk__\" value=\"token\"/>"
    "</form></body></html>"
)

# A live page that passes ``is_blocked`` (HTTP 200, real visible text, real
# content elements, no challenge markers).  ``is_blocked(200, LIVE_HTML)``
# returns ``(False, "")``.
LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "after the retry attempt reached the real article body.</p>"
    "</article></body></html>"
)

PROXY_A = ProxyConfig(server="http://proxy-a.example:8080")
PROXY_B = ProxyConfig(server="http://proxy-b.example:8080")


class _RecordingLogger:
    def __init__(self):
        self.verbose = True
        self.events = []

    def info(self, message, tag="INFO", **kwargs):
        self.events.append(("info", tag, message))

    def warning(self, message, tag="WARNING", **kwargs):
        self.events.append(("warning", tag, message))

    def error_status(self, url, error, tag="ERROR", **kwargs):
        self.events.append(("error", tag, error))

    def url_status(self, url, success, timing, tag="FETCH", **kwargs):
        self.events.append(("success" if success else "error", tag, url))


class _SeqStrategy:
    """``AsyncCrawlerStrategy`` stub serving a scripted sequence of responses
    in crawl-call order, regardless of the active proxy.  Models the outer
    retry loop axis (the same proxy/direct hit more than once across
    retries)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.crawl_calls = 0

    async def crawl(self, url, config):
        idx = min(self.crawl_calls, len(self.responses) - 1)
        self.crawl_calls += 1
        return self.responses[idx]

    def update_user_agent(self, *args, **kwargs):
        pass


class _PerProxyStrategy:
    """``AsyncCrawlerStrategy`` stub keyed by identity on the ``ProxyConfig``
    instance (mirrors ``arun`` which passes the same instance to ``crawl``).
    Models the inner proxy loop axis."""

    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.crawl_calls = 0

    async def crawl(self, url, config):
        self.crawl_calls += 1
        behavior = self.behaviors.get(config.proxy_config)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    def update_user_agent(self, *args, **kwargs):
        pass


def _resp(html, status):
    return AsyncCrawlResponse(html=html, response_headers={}, status_code=status)


def _challenge_resp():
    return _resp(CHALLENGE_HTML, 403)


def _live_resp():
    return _resp(LIVE_HTML, 200)


def _make_crawler(tmp_path, strategy):
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=strategy,
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    return crawler, logger


def _echo(url, html, **kwargs):
    return CrawlResult(url=url, html=html, success=True)


def _patch_cache(monkeypatch, cached):
    """Mock the cache read/write entry points on the ``async_db_manager``
    singleton.  Returns the ``acache_url`` mock so each test asserts on the
    write path."""
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(return_value=cached),
    )
    acache = AsyncMock(return_value=None)
    monkeypatch.setattr(async_db_manager, "acache_url", acache)
    return acache


async def _isolate_db(monkeypatch, tmp_path):
    """Redirect the ``async_db_manager`` singleton to an isolated temp SQLite
    DB so the real ``aget_cached_url``/``acache_url`` round-trip is exercised
    without touching the shared ``~/.crawl4ai`` DB.  ``monkeypatch`` restores
    every attribute at teardown."""
    db_path = str(tmp_path / "test_desync.db")
    content_base = str(tmp_path / "content")
    monkeypatch.setattr(async_db_manager, "db_path", db_path)
    monkeypatch.setattr(
        async_db_manager, "content_paths", ensure_content_dirs(content_base)
    )
    monkeypatch.setattr(async_db_manager, "_initialized", True)
    monkeypatch.setattr(async_db_manager, "connection_pool", {})
    await async_db_manager.ainit_db()
    await async_db_manager.update_db_schema()


def _failing_aprocess(raise_on_call, exc):
    """Stateful ``aprocess_html`` side effect: succeeds (echo) until the Nth
    call, then raises ``exc``.  Mirrors a transient extraction/markdown/
    renderer failure after a successful fetch."""
    state = {"n": 0}

    async def _proc(url, html, **kwargs):
        state["n"] += 1
        if state["n"] == raise_on_call:
            raise exc
        return CrawlResult(url=url, html=html, success=True)

    return AsyncMock(side_effect=_proc)


# ---------------------------------------------------------------------------
# The reported bug: blocked attempt 1, clean fetch + aprocess_html raise on
# the retry.  Must NOT return or cache the challenge page.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_desync_clean_fetch_aprocess_raise_not_cached(tmp_path, monkeypatch):
    """``max_retries=1`` -> 2 attempts, single direct proxy.  Attempt 1:
    ``crawl()`` returns the 403 challenge; ``aprocess_html`` succeeds (echo)
    -> ``_blocked=True``, ``crawl_result=challenge``, loop continues.  Attempt
    2: ``crawl()`` returns the clean 200 page (``_blocked=False`` refreshed
    BEFORE ``aprocess_html``); ``aprocess_html`` raises a transient
    ``ValueError``.  Pre-fix: ``crawl_result`` stays at the attempt-1 challenge
    page, the post-loop re-check trusts the stale ``_blocked=False``, and the
    challenge page is returned with ``success=True`` and written to the cache.
    Post-fix: the result is a clear failure, the challenge page is NOT
    returned, and ``acache_url`` is never called."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(
        tmp_path, _SeqStrategy([_challenge_resp(), _live_resp()])
    )
    crawler.aprocess_html = _failing_aprocess(raise_on_call=2, exc=ValueError(
        "transient aprocess_html failure"))

    result = await crawler.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED, max_retries=1)
    )

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False, (
        "the challenge page must not be returned as a successful crawl"
    )
    assert result.html != CHALLENGE_HTML, (
        "the earlier attempt's challenge page must not be returned"
    )
    assert result.html == "", (
        "the desync failure must produce an empty-HTML failure result, not a page body"
    )
    assert result.error_message, "a failure result must carry an error message"
    assert "transient aprocess_html failure" in result.error_message
    # The cache write gate must NOT persist the challenge page.
    acache.assert_not_awaited()


# ---------------------------------------------------------------------------
# End-to-end with a real isolated SQLite cache DB: the challenge page is not
# persisted, so a subsequent ENABLED crawl is NOT served a poisoned hit.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_desync_does_not_poison_real_cache_e2e(tmp_path, monkeypatch):
    """Exercises the real ``aget_cached_url``/``acache_url`` round-trip against
    a throwaway SQLite DB.  Crawl 1 reproduces the desync (challenge then clean
    with ``aprocess_html`` raising on the clean fetch).  Crawl 2 (site
    recovered, ``aprocess_html`` works) must perform a REAL network fetch
    (cache miss -> not poisoned), return the live page, and write it.  Crawl 3
    is then served the live page from cache (hit, zero fetches) -- proving the
    cache now holds the clean page, not the challenge."""
    await _isolate_db(monkeypatch, tmp_path)

    # --- Crawl 1: desync (challenge then clean, aprocess_html raises on clean)
    crawler1, _ = _make_crawler(
        tmp_path, _SeqStrategy([_challenge_resp(), _live_resp()])
    )
    crawler1.aprocess_html = _failing_aprocess(raise_on_call=2, exc=ValueError(
        "transient aprocess_html failure"))
    result1 = await crawler1.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED, max_retries=1)
    )
    assert result1.success is False
    assert result1.html != CHALLENGE_HTML

    # The real DB must NOT contain the challenge page after the desync.
    cached_after_1 = await async_db_manager.aget_cached_url(URL)
    assert cached_after_1 is None, (
        "the challenge page must not have been persisted to the cache"
    )

    # --- Crawl 2: site recovered, aprocess_html works -- must FETCH (not a
    # poisoned cache hit), return the live page, and cache it.
    crawler2, _ = _make_crawler(tmp_path, _SeqStrategy([_live_resp()]))
    crawler2.aprocess_html = AsyncMock(side_effect=_echo)
    result2 = await crawler2.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
    )
    assert crawler2.crawler_strategy.crawl_calls == 1, (
        "crawl 2 must hit the network -- the cache was not poisoned"
    )
    assert result2.success is True
    assert result2.html == LIVE_HTML
    assert result2.cache_status == "miss"

    # The real DB now holds the live page.
    cached_after_2 = await async_db_manager.aget_cached_url(URL)
    assert cached_after_2 is not None
    assert cached_after_2.success is True
    assert cached_after_2.html == LIVE_HTML

    # --- Crawl 3: a clean page is served straight from cache (hit, no fetch).
    # A poisoned cache would have served the challenge page here; a clean
    # cache serves the live page with zero crawl() calls.
    crawler3, _ = _make_crawler(tmp_path, _SeqStrategy([_challenge_resp()]))
    crawler3.aprocess_html = AsyncMock(side_effect=_echo)
    result3 = await crawler3.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
    )
    assert crawler3.crawler_strategy.crawl_calls == 0, (
        "crawl 3 must be a pure cache hit -- zero network fetches"
    )
    assert result3.success is True
    assert result3.html == LIVE_HTML, (
        "the cached page must be the live page, not the challenge"
    )
    assert result3.cache_status == "hit"


# ---------------------------------------------------------------------------
# No regression: an aprocess_html failure on a clean fetch that a retry then
# recovers must still return and cache the live page.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_recovers_when_aprocess_html_raises_on_clean_fetch(
    tmp_path, monkeypatch
):
    """``max_retries=1`` -> 2 attempts.  Attempt 1: ``crawl()`` returns a clean
    200 page; ``aprocess_html`` raises (transient).  Attempt 2: ``crawl()``
    returns the clean page again; ``aprocess_html`` succeeds.  The fix must not
    turn this into a poisoning failure -- the retry recovers, returns the live
    page, and caches it (the normal retry-recovers path)."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(
        tmp_path, _SeqStrategy([_live_resp(), _live_resp()])
    )
    crawler.aprocess_html = _failing_aprocess(raise_on_call=1, exc=ValueError(
        "transient aprocess_html failure"))

    result = await crawler.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED, max_retries=1)
    )

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is True
    assert result.html == LIVE_HTML
    assert result.cache_status == "miss"
    acache.assert_awaited_once()
    written = acache.await_args.args[0]
    assert written.success is True
    assert written.html == LIVE_HTML


# ---------------------------------------------------------------------------
# No regression: a single attempt (default ``max_retries=0``) whose
# ``aprocess_html`` raises still returns a failure result and never caches.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_attempt_aprocess_html_failure_returns_failure_no_cache(
    tmp_path, monkeypatch
):
    """With the default ``max_retries=0`` there is only one attempt.  When
    ``aprocess_html`` raises, the in-loop re-raise fires (one proxy, one
    attempt) and the outer handler returns a failure result.  The fix must not
    change this path -- the caller still gets ``success=False`` and the cache
    write is skipped."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(tmp_path, _SeqStrategy([_live_resp()]))
    crawler.aprocess_html = AsyncMock(side_effect=ValueError(
        "transient aprocess_html failure"))

    result = await crawler.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
    )

    assert result.success is False
    assert result.html == ""
    assert result.error_message
    acache.assert_not_awaited()


# ---------------------------------------------------------------------------
# Cross-axis: the desync also holds across the INNER proxy loop (a blocked
# first proxy, then a clean second proxy whose aprocess_html raises).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_desync_across_inner_proxy_loop(tmp_path, monkeypatch):
    """``_max_attempts=1`` with two proxies ``[PROXY_A, PROXY_B]``.  PROXY_A:
    returns the 403 challenge; ``aprocess_html`` succeeds (echo) -> blocked,
    loop continues to the next proxy.  PROXY_B: returns the clean 200 page
    (``_blocked=False`` refreshed); ``aprocess_html`` raises.  Same desync as
    the outer retry axis, but across the inner proxy loop.  The fix must
    produce a failure, not the challenge page, and skip the cache write."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(
        tmp_path,
        _PerProxyStrategy(
            {
                PROXY_A: _challenge_resp(),
                PROXY_B: _live_resp(),
            }
        ),
    )
    crawler.aprocess_html = _failing_aprocess(raise_on_call=2, exc=ValueError(
        "transient aprocess_html failure"))
    config = CrawlerRunConfig(
        proxy_config=[PROXY_A, PROXY_B], cache_mode=CacheMode.ENABLED
    )

    result = await crawler.arun(URL, config=config)

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html != CHALLENGE_HTML
    assert result.html == ""
    assert result.error_message
    acache.assert_not_awaited()


# ---------------------------------------------------------------------------
# Guard: the existing ``RendererResourceLimitError`` clear-and-break path on
# a later attempt after a blocked first attempt must still yield a failure
# (not a stale challenge page).  This path already cleared ``crawl_result``
# pre-fix; it must continue to work alongside the new clear-before-aprocess.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_limit_on_later_attempt_after_block_no_stale_challenge(
    tmp_path, monkeypatch
):
    """``max_retries=1`` -> 2 attempts.  Attempt 1: ``crawl()`` returns the 403
    challenge; ``aprocess_html`` succeeds -> blocked, loop continues.  Attempt
    2: ``crawl()`` returns the clean 200 page; ``aprocess_html`` raises
    ``RendererResourceLimitError``.  The existing
    ``RendererResourceLimitError`` branch clears ``crawl_result`` and breaks;
    the post-loop ``else`` branch builds a failure result.  The fix must not
    regress this -- the caller gets ``success=False`` (no challenge page) and
    the cache is not written."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(
        tmp_path, _SeqStrategy([_challenge_resp(), _live_resp()])
    )
    crawler.aprocess_html = _failing_aprocess(
        raise_on_call=2,
        exc=RendererResourceLimitError(RendererKill(
            killed_at=0.0, reason="rss", renderer_pid=1, browser_pid=2,
            rss_bytes=100, rss_limit_bytes=50,
        )),
    )

    result = await crawler.arun(
        URL, CrawlerRunConfig(cache_mode=CacheMode.ENABLED, max_retries=1)
    )

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html != CHALLENGE_HTML
    assert result.html == ""
    assert result.error_message
    acache.assert_not_awaited()
