"""Empty-HTML cache hit must re-run extraction instead of returning stale content.

When a cache hit holds an EMPTY (``""``/``None``) HTML body together with a
non-empty ``extracted_content`` (a row only a *content-independent* extraction
strategy can produce — e.g. a strategy that tags output from URL/metadata
rather than the HTML body — written by an empty-HTML producer such as a binary
download or a near-empty HTTP 200 response), ``arun`` falls through to a fresh
fetch via the ``not html`` branch of the gate ``if not cached_result or not
html:``.  Before the fix, the stale cached ``extracted_content`` was *not*
cleared on this path (the ``f82b301`` clear only fired when ``cached_result is
None``, i.e. the screenshot/pdf artifact-missing fallthrough), so the
fresh-fetch call ``aprocess_html`` received the stale truthy value and its
``not bool(extracted_content)`` guard *skipped* the configured extraction
strategy, returning the stale cached extraction alongside freshly fetched HTML.

The fix broadens the clear to ``if cached_result is None or not html:``, so the
stale ``extracted_content`` is dropped whenever a fresh fetch is about to run
for any reason, and the configured extraction strategy re-runs on the freshly
fetched page exactly as the artifact-missing fallthrough already does.

These tests exercise the real ``aprocess_html`` (not mocked) with a tracking
extraction strategy so the extraction-skip guard inside ``aprocess_html`` is
actually exercised, mirroring ``test_stale_extraction_on_cache_fallthrough.py``
which covers the artifact axis.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_empty_html_cache_fallthrough.py
"""
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.extraction_strategy import ExtractionStrategy
from crawl4ai.models import AsyncCrawlResponse, CrawlResult

URL = "https://example.com"
# The cached HTML is the EMPTY string — the empty-HTML fallthrough trigger.
# It is never run through ``is_blocked`` (the cache-hit branch returns it
# directly), so it can be empty here without tripping the anti-bot detector.
CACHED_HTML = ""
# Live HTML is large enough (>100 bytes, real visible text) to pass the
# anti-bot ``is_blocked`` check on the fresh-fetch path and to distinguish the
# freshly fetched HTML from the empty cached HTML.
LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "strategy after the cached entry was found to have an empty HTML body "
    "and a fresh fetch was triggered by the not-html fallthrough branch.</p>"
    "</article></body></html>"
)
LIVE_SCREENSHOT_B64 = "LIVE_SCREENSHOT_BASE64_PNG"
# Genuinely non-empty cached extraction (NOT the literal "[]" that ``arun``
# normalizes to None).  Carrying this through the fallthrough unchanged is
# the bug — it is the value a content-independent strategy would have
# persisted alongside the empty HTML on the first crawl.
STALE_CACHED_EXTRACTION = '[{"stale_from_cache": true}]'
FRESH_EXTRACTION = [{"fresh": True}]


class TrackingStrategy(ExtractionStrategy):
    """Extraction strategy that records whether it was invoked.

    Returns a deterministic fresh value so tests can distinguish a re-run
    from the stale cached value being returned unchanged.  Content-
    independent (ignores the HTML body) on purpose: the empty-HTML trigger
    row can only be seeded by a strategy that emits non-empty output for
    empty content.
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
    """Cached ``CrawlResult``: empty HTML + non-empty extraction by default."""
    base = dict(
        url=URL,
        html=CACHED_HTML,  # ""  -> the empty-HTML fallthrough trigger
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
        "extraction strategy was not invoked on the empty-HTML fresh fetch — "
        "the stale cached extracted_content must have short-circuited extraction"
    )
    assert "stale_from_cache" not in (result.extracted_content or ""), (
        "stale cached extracted_content was returned unchanged by the fresh fetch"
    )
    assert "fresh" in (result.extracted_content or ""), (
        "fresh extraction result was not returned after the empty-HTML fallthrough"
    )


@pytest.mark.asyncio
async def test_empty_html_read_only_reruns_extraction(tmp_path, monkeypatch):
    """``READ_ONLY`` + empty cached HTML (no artifact requested) must trigger the
    ``not html`` fallthrough, re-run extraction, and return fresh (not stale)
    extracted_content alongside the freshly fetched HTML."""
    _patch_cache(monkeypatch, _cached_result())
    strategy = TrackingStrategy()
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.READ_ONLY,
        ),
    )

    assert result.success is True
    assert not result.error_message
    assert result.html == LIVE_HTML  # freshly fetched HTML, not the empty cache
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1  # a fresh fetch happened
    _assert_fresh_extraction(result, strategy)


@pytest.mark.asyncio
async def test_empty_html_enabled_reruns_extraction_and_returns_fresh(
    tmp_path, monkeypatch
):
    """``ENABLED`` (default) + empty cached HTML must re-run extraction and return
    the fresh (not stale) extracted_content.

    The fresh result is NOT re-persisted on this path: the empty-HTML fallthrough
    keeps ``cached_result`` truthy, so the cache-write guard
    ``if cache_context.should_write() and not bool(cached_result)`` stays False
    (the artifact-missing fallthrough differs — it nulls ``cached_result`` and
    therefore re-persists).  This stickiness is a known consequence of the
    minimal clear-only fix and is asserted here to pin the actual behavior.
    """
    acache = _patch_cache(monkeypatch, _cached_result())
    strategy = TrackingStrategy()
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.ENABLED,
        ),
    )

    assert result.success is True
    assert result.html == LIVE_HTML
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)

    # The empty-HTML fallthrough leaves cached_result truthy, so the write
    # guard's ``not bool(cached_result)`` is False and the fresh result is not
    # re-persisted (the stale empty-HTML cache row survives — stickiness).
    acache.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_html_with_screenshot_missing_reruns_and_persists_fresh(
    tmp_path, monkeypatch
):
    """Empty cached HTML + a requested-but-missing screenshot fires BOTH triggers
    (the artifact check nulls ``cached_result`` AND ``not html`` is true).  The
    artifact null makes the cache-write guard True, so the fresh result IS
    re-persisted — contrasting with the pure empty-HTML fallthrough above and
    matching the artifact-axis behavior."""
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
    assert result.html == LIVE_HTML
    assert result.screenshot == LIVE_SCREENSHOT_B64
    assert result.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)

    # The artifact null set cached_result=None, so the write guard fires and
    # the fresh extraction is re-persisted (NOT the stale cached value).
    acache.assert_awaited_once()
    persisted = acache.await_args.args[0]
    assert "stale_from_cache" not in (persisted.extracted_content or "")
    assert "fresh" in (persisted.extracted_content or "")


@pytest.mark.asyncio
async def test_empty_html_with_empty_cached_extraction_runs_extraction(
    tmp_path, monkeypatch
):
    """Non-regression / safe-case guard: when the cached extracted_content is
    empty or the literal ``"[]"`` (which ``arun`` normalizes to None), there is
    no stale value to carry, so extraction runs on the fresh fetch regardless of
    the fix.  This is the case built-in content-dependent strategies produce on
    empty HTML (they return ``[]``), and it must keep working."""
    _patch_cache(
        monkeypatch,
        _cached_result(extracted_content="[]"),  # normalizes to None on read
    )
    strategy = TrackingStrategy()
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.READ_ONLY,
        ),
    )

    assert result.success is True
    assert result.html == LIVE_HTML
    assert crawler.crawler_strategy.crawl_calls == 1
    _assert_fresh_extraction(result, strategy)


@pytest.mark.asyncio
async def test_non_empty_html_cache_hit_returns_cached_without_refetch(
    tmp_path, monkeypatch
):
    """Non-regression: a cache hit with NON-empty HTML and a non-empty
    extracted_content, no artifact requested, must NOT trigger a fresh fetch and
    must return the cached extraction verbatim.  This confirms the broadened
    clear ``if cached_result is None or not html:`` does not accidentally clear
    extraction (or re-fetch) on a normal, valid cache hit."""
    acache = _patch_cache(
        monkeypatch,
        _cached_result(
            html=(
                "<html><body><h1>Cached page</h1>"
                "<p>Non-empty cached content for the regression test.</p></body></html>"
            ),
        ),
    )
    strategy = TrackingStrategy()
    crawler, logger = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML,
            response_headers={},
            status_code=200,
        ),
    )

    result = await crawler.arun(
        URL,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.READ_ONLY,
        ),
    )

    assert result.success is True
    # No fresh fetch — the cached HTML is returned directly.
    assert crawler.crawler_strategy.crawl_calls == 0
    assert strategy.extract_called is False  # cache hit — no re-extraction
    # Cached extraction is returned verbatim (correct cache-hit behavior).
    assert result.extracted_content == STALE_CACHED_EXTRACTION
    assert "stale_from_cache" in (result.extracted_content or "")
    acache.assert_not_awaited()
