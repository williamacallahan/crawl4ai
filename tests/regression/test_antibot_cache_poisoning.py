"""Anti-bot challenge pages must not poison the default cache.

Under ``CacheMode.ENABLED`` (the default), a crawl that is blocked by
anti-bot protection returns a ``CrawlResult`` with ``success=False`` and a
non-empty ``html`` body (the challenge page).  Two cache gates used to
operate on this result without consulting ``success``:

* the **write gate** persisted the blocked result unconditionally, and
* the **read short-circuit** (``if not cached_result or not html``) treated
  a cached blocked row with non-empty HTML as a hit and served it forever.

Together they produced a permanent deadlock: a single anti-bot block
poisoned the cache for that URL and every subsequent crawl returned the
challenge page with ``success=False``, regardless of retries or whether
the site later became reachable.

These tests verify the fix: blocked results are never written to the
cache, and a poisoned row already in the cache is refetched (and
overwritten on success) under ``ENABLED``.  The ``READ_ONLY`` contract
(serve cached content verbatim, never hit the network) is preserved.

See: ``test_crawl_failure_severity.test_cached_target_refusal_remains_failed``
for the ``READ_ONLY`` contract test that must continue to pass unchanged.
"""
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult

URL = "https://example.com"

# A realistic anti-bot challenge page (Cloudflare "Just a moment...").
# Carries non-empty HTML -- the precondition for the cache-poisoning bug
# (an empty body would self-heal via the ``not html`` short-circuit).
CHALLENGE_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<h1>Just a moment...</h1>"
    "<p>Checking your browser before accessing the site. This may take "
    "a few seconds to complete.</p>"
    "<form id=\"challenge-form\" action=\"/cdn-cgi/challenge-platform/\">"
    "<input name=\"__cf_chl_f_tk__\" value=\"token\"/>"
    "</form></body></html>"
)

# A live page that passes the anti-bot ``is_blocked`` check: HTTP 200,
# >100 bytes of visible text, real content elements, and no challenge
# markers.  ``is_blocked(200, LIVE_HTML)`` returns ``(False, "")``.
LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "after the cached blocked entry was discarded so the cache could be "
    "repopulated with the real article body.</p></article></body></html>"
)


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


class _ResponseStrategy:
    """Minimal ``AsyncCrawlerStrategy`` stub that records ``crawl`` calls."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.crawl_calls = 0

    async def crawl(self, url, config):
        self.crawl_calls += 1
        if self.error:
            raise self.error
        return self.response

    def update_user_agent(self, *args, **kwargs):
        pass


def _make_crawler(tmp_path, response=None, error=None, process_html=None):
    """Build an ``AsyncWebCrawler`` wired for a mocked live fetch.

    ``aprocess_html`` is mocked so the live-fetch path returns a
    ``CrawlResult`` carrying the HTML that ``crawl`` just fetched (echo),
    mirroring what a real ``aprocess_html`` would attach from the
    ``AsyncCrawlResponse``.  Callers may override ``process_html`` to
    return a fixed result instead.
    """
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(response=response, error=error),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    if process_html is not None:
        crawler.aprocess_html = AsyncMock(return_value=process_html)
    else:

        def _echo(url, html, **kwargs):
            return CrawlResult(url=url, html=html, success=True)

        crawler.aprocess_html = AsyncMock(side_effect=_echo)
    return crawler, logger


def _patch_cache(monkeypatch, cached):
    """Mock the cache read/write entry points on the ``async_db_manager``
    singleton.  Returns the ``acache_url`` mock so each test can assert on
    the write path.
    """
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(return_value=cached),
    )
    acache = AsyncMock(return_value=None)
    monkeypatch.setattr(async_db_manager, "acache_url", acache)
    return acache


def _challenge_response():
    return AsyncCrawlResponse(
        html=CHALLENGE_HTML,
        response_headers={},
        status_code=403,
    )


def _live_response():
    return AsyncCrawlResponse(
        html=LIVE_HTML,
        response_headers={},
        status_code=200,
    )


def _poisoned_cached_result():
    """A cached blocked result, as it would have been written by the
    pre-fix write gate (``success=False`` with non-empty challenge HTML).
    """
    return CrawlResult(
        url=URL,
        html=CHALLENGE_HTML,
        success=False,
        status_code=403,
        error_message="Blocked by anti-bot protection: HTTP 403 with HTML content",
        cache_status="hit",
    )


# ---------------------------------------------------------------------------
# Write-side guard: blocked results are never persisted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_result_not_cached_under_enabled(tmp_path, monkeypatch):
    """Phase 1 of the deadlock (write side): a first crawl that is blocked
    by anti-bot protection with non-empty challenge HTML must NOT be written
    to the cache under the default ``CacheMode.ENABLED``."""
    acache = _patch_cache(monkeypatch, cached=None)  # cache miss
    crawler, logger = _make_crawler(tmp_path, response=_challenge_response())

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    # The blocked result is still surfaced to the caller with the
    # challenge page and a failure flag.
    assert result.success is False
    assert result.html == CHALLENGE_HTML
    assert result.error_message.startswith("Blocked by anti-bot protection")
    assert result.cache_status == "miss"

    # The write gate must NOT persist the blocked result -- this is the
    # core fix that prevents new cache poisoning.
    acache.assert_not_awaited()
    assert crawler.crawler_strategy.crawl_calls == 1


# ---------------------------------------------------------------------------
# Read-side guard: an already-poisoned cache row is refetched and, on
# success, overwritten.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poisoned_cache_refetched_and_overwritten_under_enabled(
    tmp_path, monkeypatch
):
    """Phase 2 of the deadlock (read side): a cached blocked row with
    non-empty HTML must trigger a fresh fetch under ``ENABLED``.  When the
    refetch succeeds, the good result overwrites the poison in the DB."""
    acache = _patch_cache(monkeypatch, cached=_poisoned_cached_result())
    crawler, logger = _make_crawler(tmp_path, response=_live_response())

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    # The read-side guard discarded the poison and hit the network.
    assert crawler.crawler_strategy.crawl_calls == 1
    # The caller sees the freshly fetched live page, not the challenge.
    assert result.success is True
    assert result.html == LIVE_HTML
    assert result.error_message in (None, "", "")
    assert result.cache_status == "miss"
    # The successful refetch overwrites the poison (cached_result was
    # nulled, so ``not bool(cached_result)`` is True, and success=True).
    acache.assert_awaited_once()
    written_result = acache.await_args.args[0]
    assert written_result.success is True
    assert written_result.html == LIVE_HTML


# ---------------------------------------------------------------------------
# No regression: successful crawls are still cached and served from cache.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_crawl_still_cached_under_enabled(tmp_path, monkeypatch):
    """Happy path: a successful crawl under ENABLED must still be written
    to the cache (the write guard only rejects ``success=False``)."""
    acache = _patch_cache(monkeypatch, cached=None)
    crawler, logger = _make_crawler(tmp_path, response=_live_response())

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    assert result.success is True
    assert result.html == LIVE_HTML
    assert result.cache_status == "miss"
    acache.assert_awaited_once()
    written_result = acache.await_args.args[0]
    assert written_result.success is True
    assert written_result.html == LIVE_HTML


@pytest.mark.asyncio
async def test_successful_crawl_served_from_cache_hit_under_enabled(
    tmp_path, monkeypatch
):
    """No regression on the cache-hit path: a cached successful result is
    served directly (no refetch, no rewrite)."""
    cached = CrawlResult(
        url=URL,
        html=LIVE_HTML,
        success=True,
        status_code=200,
        cache_status="hit",
    )
    acache = _patch_cache(monkeypatch, cached=cached)
    crawler, logger = _make_crawler(tmp_path, response=_live_response())

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    assert result.success is True
    assert result.html == LIVE_HTML
    assert result.cache_status == "hit"
    assert crawler.crawler_strategy.crawl_calls == 0
    acache.assert_not_awaited()



# ---------------------------------------------------------------------------
# End-to-end: the full two-phase deadlock is broken under ENABLED.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_antibot_cache_poisoning_deadlock(tmp_path, monkeypatch):
    """The canonical regression: a single anti-bot block used to poison
    the cache and then serve the challenge page on every subsequent crawl
    with no re-fetch.  After the fix, neither phase poisons nor deadlocks.

    Phase 1 -- cache miss, blocked response:
      * the write gate refuses to persist the blocked result.
    Phase 2 -- a poisoned row is in the cache, the site has recovered:
      * the read guard discards the poison, refetches, and overwrites it
        with the live page.
    """
    crawler, logger = _make_crawler(tmp_path, response=_challenge_response())

    # --- Phase 1: first crawl (cache miss, blocked) ---
    acache_phase1 = AsyncMock(return_value=None)
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(async_db_manager, "acache_url", acache_phase1)

    result1 = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    assert result1.success is False
    assert result1.html == CHALLENGE_HTML
    assert result1.error_message.startswith("Blocked by anti-bot protection")
    # Pre-fix: acache_url was called with the blocked result (the poison).
    # Post-fix: the write gate skips it.
    acache_phase1.assert_not_awaited()
    crawl_calls_after_phase1 = crawler.crawler_strategy.crawl_calls
    assert crawl_calls_after_phase1 == 1  # Phase 1 fetched the challenge page

    # --- Phase 2: second crawl (poison already in the cache, site back) ---
    # Simulate the poison that *would* have been written by Phase 1 before
    # the fix (or by an older, pre-fix version of crawl4ai).  The site has
    # now recovered, so crawl() returns a live 200 page.
    crawler.crawler_strategy.response = _live_response()
    acache_phase2 = AsyncMock(return_value=None)
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(return_value=_poisoned_cached_result()),
    )
    monkeypatch.setattr(async_db_manager, "acache_url", acache_phase2)

    result2 = await crawler.arun(
        URL,
        CrawlerRunConfig(cache_mode=CacheMode.ENABLED),
    )

    # Pre-fix: crawl() was never called and the cached challenge was
    # returned with cache_status "hit".  Post-fix: the read guard forces a
    # refetch and the live page overwrites the poison.
    assert crawler.crawler_strategy.crawl_calls == crawl_calls_after_phase1 + 1
    assert result2.success is True
    assert result2.html == LIVE_HTML
    assert result2.cache_status == "miss"
    acache_phase2.assert_awaited_once()
    assert acache_phase2.await_args.args[0].success is True
    assert acache_phase2.await_args.args[0].html == LIVE_HTML
