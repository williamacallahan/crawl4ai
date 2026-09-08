"""Cached pages missing requested artifacts must fall through to a fresh fetch.

Database cache reads preserve an absent screenshot as an empty string. PDF
content is not persisted, so cached results reconstruct it as None.
"""
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult

URL = "https://example.com"
# The live HTML is deliberately larger than ``_EMPTY_CONTENT_THRESHOLD`` (100
# bytes) with >50 chars of visible text and real content elements, so the
# anti-bot ``is_blocked`` check on the fresh-fetch path does not flag it as a
# silent block (which would mark the result failed and obscure the regression
# under test).  The cached HTML is never run through ``is_blocked`` (the
# cache-hit branch returns it directly), so it can stay small.
CACHED_HTML = (
    "<html><body><h1>Cached page</h1>"
    "<p>Cached content here for the regression test.</p></body></html>"
)
LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "strategy after the cached entry was invalidated because it lacked the "
    "requested screenshot artifact.</p></article></body></html>"
)
LIVE_SCREENSHOT_B64 = "LIVE_SCREENSHOT_BASE64_PNG"
LIVE_PDF_BYTES = b"%PDF-1.4 LIVE_PDF_BYTES %%EOF"


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


def _make_crawler(tmp_path, response):
    """Build an ``AsyncWebCrawler`` wired for a mocked live fetch.

    ``aprocess_html`` is mocked so the live-fetch path returns a ``CrawlResult``
    carrying the captured screenshot/pdf, mirroring what a real
    ``aprocess_html`` would attach from the ``AsyncCrawlResponse`` artifacts.
    """
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(response=response),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    crawler.aprocess_html = AsyncMock(
        return_value=CrawlResult(
            url=URL,
            html=LIVE_HTML,
            success=True,
            screenshot=LIVE_SCREENSHOT_B64,
            pdf=LIVE_PDF_BYTES,
        )
    )
    return crawler, logger


def _patch_cache(monkeypatch, cached):
    """Mock the cache read/write entry points on the ``async_db_manager`` singleton.

    Returns the ``acache_url`` mock so each test can assert on the write path.
    """
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(return_value=cached),
    )
    acache = AsyncMock(return_value=None)
    monkeypatch.setattr(async_db_manager, "acache_url", acache)
    return acache


def _no_none_deref_error(logger):
    """True if no error_status event carries the original AttributeError text."""
    return not any(
        event[0] == "error" and "NoneType" in str(event[2])
        for event in logger.events
    )


@pytest.mark.asyncio
async def test_screenshot_missing_from_cache_falls_through_to_fresh_fetch(
    tmp_path, monkeypatch
):
    """``screenshot=True`` against a cached entry whose ``screenshot`` is ``""``
    (the exact shape the DB write path stores when the original crawl lacked
    one) must invalidate the cache entry and re-crawl to capture the screenshot,
    not crash on ``cached_result.success``."""
    acache = _patch_cache(
        monkeypatch,
        CrawlResult(
            url=URL,
            html=CACHED_HTML,
            success=True,
            screenshot="",  # empty string is what the DB persists/restores
            pdf=None,
        ),
    )
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
            screenshot=LIVE_SCREENSHOT_B64,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(screenshot=True, cache_mode=CacheMode.READ_ONLY),
    )

    assert result.success is True
    assert result.error_message is None
    assert result.screenshot == LIVE_SCREENSHOT_B64
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    crawler.aprocess_html.assert_awaited_once()
    assert crawler.aprocess_html.await_args.kwargs["screenshot_data"] == LIVE_SCREENSHOT_B64
    acache.assert_not_awaited()  # READ_ONLY never writes
    assert _no_none_deref_error(logger)
    # The cache-hit FETCH log line now executes (pre-fix it raised before the
    # call) and reports success for the cached HTML.
    assert any(event[:2] == ("success", "FETCH") for event in logger.events)


@pytest.mark.asyncio
async def test_pdf_missing_from_cache_falls_through_to_fresh_fetch(
    tmp_path, monkeypatch
):
    """``pdf=True`` against any cached entry must invalidate and re-crawl: pdf is
    never persisted to cache (absent from both the content_map write side and
    the content_fields read side), so ``cached_result.pdf`` is always ``None``
    on a cache read, always tripping the guard."""
    acache = _patch_cache(
        monkeypatch,
        CrawlResult(
            url=URL,
            html=CACHED_HTML,
            success=True,
            screenshot=None,
            pdf=None,  # always None on cache read — pdf is never persisted
        ),
    )
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
            pdf_data=LIVE_PDF_BYTES,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(pdf=True, cache_mode=CacheMode.READ_ONLY),
    )

    assert result.success is True
    assert result.error_message is None
    assert result.pdf == LIVE_PDF_BYTES
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    crawler.aprocess_html.assert_awaited_once()
    assert crawler.aprocess_html.await_args.kwargs["pdf_data"] == LIVE_PDF_BYTES
    acache.assert_not_awaited()
    assert _no_none_deref_error(logger)
    assert any(event[:2] == ("success", "FETCH") for event in logger.events)


@pytest.mark.asyncio
async def test_screenshot_present_in_cache_is_returned_without_refetch(
    tmp_path, monkeypatch
):
    """No regression: when the cache already holds the requested screenshot,
    the screenshot guard does not null ``cached_result`` and ``arun`` returns
    the cached entry directly (no fresh fetch)."""
    acache = _patch_cache(
        monkeypatch,
        CrawlResult(
            url=URL,
            html=CACHED_HTML,
            success=True,
            screenshot="CACHED_SCREENSHOT_B64",
        ),
    )
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
            screenshot=LIVE_SCREENSHOT_B64,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(screenshot=True, cache_mode=CacheMode.READ_ONLY),
    )

    assert result.success is True
    assert result.screenshot == "CACHED_SCREENSHOT_B64"
    assert result.html == CACHED_HTML
    assert result.cache_status == "hit"
    assert crawler.crawler_strategy.crawl_calls == 0
    acache.assert_not_awaited()
    assert _no_none_deref_error(logger)
    assert any(event[:2] == ("success", "FETCH") for event in logger.events)
