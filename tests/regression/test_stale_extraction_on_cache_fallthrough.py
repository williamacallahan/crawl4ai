"""Cache fallthrough must re-run extraction instead of returning stale content.

When a cache hit is missing a requested artifact (screenshot/pdf), ``arun``
nulls ``cached_result`` and falls through to a fresh fetch.  Before the fix,
the cached ``extracted_content`` was *not* cleared, so the fresh-fetch call
``aprocess_html`` received the stale value and its ``not bool(extracted_content)``
guard *skipped* the configured extraction strategy, returning the stale cached
extraction alongside freshly fetched HTML/screenshot/pdf.  Under the default
``CacheMode.ENABLED`` the stale value was then re-persisted to the cache.

These tests exercise the real ``aprocess_html`` (not mocked) with a tracking
extraction strategy so the extraction-skip guard inside ``aprocess_html`` is
actually exercised.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_stale_extraction_on_cache_fallthrough.py
"""
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.extraction_strategy import ExtractionStrategy
from crawl4ai.models import AsyncCrawlResponse, CrawlResult

URL = "https://example.com"
# Live HTML is large enough (>100 bytes, real visible text) to pass the
# anti-bot ``is_blocked`` check on the fresh-fetch path.  The cached HTML is
# never run through ``is_blocked`` (returned directly on a cache hit).
CACHED_HTML = (
    "<html><body><h1>Cached page</h1>"
    "<p>Old cached content for the regression test.</p></body></html>"
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
# Genuinely non-empty cached extraction (NOT the literal "[]" that ``arun``
# normalizes to None).  Carrying this through the fallthrough is the bug.
STALE_CACHED_EXTRACTION = '[{"stale_from_cache": true}]'
FRESH_EXTRACTION = [{"fresh": True}]


class TrackingStrategy(ExtractionStrategy):
    """Extraction strategy that records whether it was invoked.

    Returns a deterministic fresh value so tests can distinguish a re-run
    from the stale cached value being returned unchanged.
    """

    def __init__(self):
        super().__init__(input_format="markdown")
        self.extract_called = False
        self.extract_calls = 0

    def extract(self, url, html, *q, **kwargs):
        self.extract_called = True
        self.extract_calls += 1
        return FRESH_EXTRACTION

    async def arun(self, url, sections, *q, **kwargs):
        self.extract_called = True
        self.extract_calls += 1
        return FRESH_EXTRACTION


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
    """Minimal ``AsyncCrawlerStrategy`` stub that returns a captured response."""

    def __init__(self, response=None):
        self.response = response
        self.crawl_calls = 0

    async def crawl(self, url, config):
        self.crawl_calls += 1
        return self.response

    def update_user_agent(self, *args, **kwargs):
        pass


def _make_crawler(tmp_path, response):
    """Build an ``AsyncWebCrawler`` wired for a *real* ``aprocess_html`` run.

    ``aprocess_html`` is intentionally left un-mocked so the extraction-skip
    guard inside it is actually exercised.
    """
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(response=response),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    return crawler, logger


def _patch_cache(monkeypatch, cached):
    """Mock the cache read/write entry points on the ``async_db_manager``.

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


def _cached_result(**overrides):
    """Cached ``CrawlResult`` with non-empty ``extracted_content`` by default."""
    base = dict(
        url=URL,
        html=CACHED_HTML,
        success=True,
        extracted_content=STALE_CACHED_EXTRACTION,
        screenshot="",  # empty string is what the DB persists/restores
        pdf=None,  # always None on cache read — pdf is never persisted
    )
    base.update(overrides)
    return CrawlResult(**base)


def _assert_fresh_extraction(result, strategy):
    """Verify the configured strategy actually re-ran on the fresh fetch."""
    assert strategy.extract_called is True, (
        "extraction strategy was not invoked on the fresh fetch — stale cached "
        "extracted_content must have short-circuited extraction"
    )
    assert "stale_from_cache" not in (result.extracted_content or ""), (
        "stale cached extracted_content was returned unchanged by the fresh fetch"
    )
    assert "fresh" in (result.extracted_content or ""), (
        "fresh extraction result was not returned after the fallthrough"
    )


@pytest.mark.asyncio
async def test_screenshot_missing_read_only_reruns_extraction(tmp_path, monkeypatch):
    """``READ_ONLY`` + missing screenshot must re-run extraction on fallthrough
    and return fresh (not stale) extracted_content; READ_ONLY never writes."""
    acache = _patch_cache(monkeypatch, _cached_result(screenshot=""))
    strategy = TrackingStrategy()
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
        CrawlerRunConfig(
            screenshot=True,
            extraction_strategy=strategy,
            cache_mode=CacheMode.READ_ONLY,
        ),
    )

    assert result.success is True
    assert not result.error_message
    assert result.screenshot == LIVE_SCREENSHOT_B64
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)
    acache.assert_not_awaited()  # READ_ONLY never writes


@pytest.mark.asyncio
async def test_screenshot_missing_enabled_reruns_and_persists_fresh(tmp_path, monkeypatch):
    """``ENABLED`` (default) + missing screenshot must re-run extraction and
    persist the *fresh* (not stale) extracted_content back to the cache."""
    acache = _patch_cache(monkeypatch, _cached_result(screenshot=""))
    strategy = TrackingStrategy()
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
        CrawlerRunConfig(
            screenshot=True,
            extraction_strategy=strategy,
            cache_mode=CacheMode.ENABLED,
        ),
    )

    assert result.success is True
    assert result.screenshot == LIVE_SCREENSHOT_B64
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)

    # Under ENABLED the fresh-fetch result is re-persisted; it must carry the
    # fresh extraction, NOT the stale cached value.
    acache.assert_awaited_once()
    persisted = acache.await_args.args[0]
    assert "stale_from_cache" not in (persisted.extracted_content or "")
    assert "fresh" in (persisted.extracted_content or "")


@pytest.mark.asyncio
async def test_pdf_missing_enabled_reruns_and_persists_fresh(tmp_path, monkeypatch):
    """``pdf=True`` always misses on a cache read (pdf is never persisted to
    the cache DB), so every re-crawl of a cached URL with ``pdf=True`` + an
    extraction strategy triggers the fallthrough and must re-run extraction.

    This is the highest-impact variant of the bug."""
    acache = _patch_cache(monkeypatch, _cached_result(pdf=None))
    strategy = TrackingStrategy()
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
        CrawlerRunConfig(
            pdf=True,
            extraction_strategy=strategy,
            cache_mode=CacheMode.ENABLED,
        ),
    )

    assert result.success is True
    assert result.pdf == LIVE_PDF_BYTES
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)

    acache.assert_awaited_once()
    persisted = acache.await_args.args[0]
    assert "stale_from_cache" not in (persisted.extracted_content or "")
    assert "fresh" in (persisted.extracted_content or "")


@pytest.mark.asyncio
async def test_artifact_present_returns_cached_extraction_without_refetch(
    tmp_path, monkeypatch
):
    """Non-regression: when the cache already holds the requested artifact AND
    a non-empty extracted_content, the fallthrough is NOT triggered — ``arun``
    returns the cached entry directly without re-running extraction."""
    acache = _patch_cache(
        monkeypatch,
        _cached_result(screenshot="CACHED_SCREENSHOT_B64", pdf=None),
    )
    strategy = TrackingStrategy()
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
        CrawlerRunConfig(
            screenshot=True,
            extraction_strategy=strategy,
            cache_mode=CacheMode.READ_ONLY,
        ),
    )

    assert result.success is True
    assert result.screenshot == "CACHED_SCREENSHOT_B64"
    assert result.html == CACHED_HTML
    assert result.cache_status == "hit"
    assert crawler.crawler_strategy.crawl_calls == 0  # no fresh fetch
    assert strategy.extract_called is False  # cache hit — no re-extraction
    # Cached extraction is returned verbatim (this is correct cache-hit behavior).
    assert result.extracted_content == STALE_CACHED_EXTRACTION
    acache.assert_not_awaited()
