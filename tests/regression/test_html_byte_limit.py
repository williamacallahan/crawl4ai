"""Bound captured HTML before the scraper, including cache and fallback paths."""

from unittest.mock import AsyncMock

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import TargetNavigationError
from crawl4ai.async_dispatcher import SemaphoreDispatcher
from crawl4ai.async_webcrawler import async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult


HEALED_PAGE = "<html><body><h1>Healed page</h1>" + "<p>Real article text after a refetch.</p>" * 40 + "</body></html>"


def _crawler(tmp_path):
    crawler = AsyncWebCrawler(
        config=BrowserConfig(verbose=False), base_directory=str(tmp_path)
    )
    # raw: uses the real strategy's browser-free path; do not launch Chromium.
    crawler.ready = True
    return crawler


@pytest.mark.asyncio
async def test_raw_html_byte_limit_precedes_scraping(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    config = CrawlerRunConfig(max_html_bytes=134, word_count_threshold=0)
    scraper = AsyncMock(wraps=config.scraping_strategy.ascrap)
    monkeypatch.setattr(config.scraping_strategy, "ascrap", scraper)

    oversized = await crawler.arun("raw:<p>" + "é" * 64 + "</p>", config=config)

    assert oversized.success is False
    assert oversized.error_message == "html_too_large: 135 exceeds 134"
    assert oversized.html == ""
    scraper.assert_not_awaited()

    under_limit = await crawler.arun("raw:<p>" + "é" * 63 + "</p>", config=config)

    assert under_limit.success is True, under_limit.error_message
    assert "é" in under_limit.markdown.raw_markdown
    scraper.assert_awaited_once()


@pytest.mark.asyncio
async def test_oversized_page_does_not_abort_raw_batch(tmp_path):
    results = await _crawler(tmp_path).arun_many(
        ["raw:<p>" + "x" * 128 + "</p>", "raw:<p>small page</p>"],
        config=CrawlerRunConfig(max_html_bytes=64, word_count_threshold=0),
        dispatcher=SemaphoreDispatcher(semaphore_count=2),
    )

    assert len(results) == 2
    assert results[0].success is False
    assert results[0].error_message == "html_too_large: 135 exceeds 64"
    assert results[1].success is True


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cache", "fallback"])
async def test_alternate_html_sources_cannot_bypass_limit(tmp_path, monkeypatch, source):
    crawler = _crawler(tmp_path)
    html = "<p>" + "x" * 128 + "</p>"
    config = CrawlerRunConfig(max_html_bytes=64, cache_mode=CacheMode.BYPASS)
    scraper = AsyncMock(side_effect=AssertionError("oversized input reached scraper"))
    monkeypatch.setattr(config.scraping_strategy, "ascrap", scraper)
    if source == "cache":
        config.cache_mode = CacheMode.READ_ONLY
        monkeypatch.setattr(
            async_db_manager, "aget_cached_url",
            AsyncMock(return_value=CrawlResult(url="https://example.test", html=html, success=True)),
        )
        crawl = AsyncMock(side_effect=AssertionError("cache hit refetched"))
    else:
        crawl = AsyncMock(side_effect=TargetNavigationError("net::ERR_CONNECTION_REFUSED"))
        config.fallback_fetch_function = AsyncMock(return_value=html)
    monkeypatch.setattr(crawler.crawler_strategy, "crawl", crawl)

    result = await crawler.arun("https://example.test", config=config)

    assert result.success is False
    assert result.error_message == "html_too_large: 135 exceeds 64"
    scraper.assert_not_awaited()
    if source == "cache":
        crawl.assert_not_awaited()
    else:
        config.fallback_fetch_function.assert_awaited_once()


def test_html_limit_survives_config_round_trips():
    config = CrawlerRunConfig(max_html_bytes=1234)
    assert config.clone().max_html_bytes == 1234
    assert CrawlerRunConfig.load(config.dump()).max_html_bytes == 1234
    assert CrawlerRunConfig.from_kwargs(config.to_dict()).max_html_bytes == 1234


@pytest.mark.asyncio
async def test_unusable_oversized_cache_still_heals_by_refetching(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    url = "https://example.test/"
    monkeypatch.setattr(
        async_db_manager, "aget_cached_url",
        AsyncMock(return_value=CrawlResult(url=url, html="x" * 8192, success=False)),
    )
    persist = AsyncMock()
    monkeypatch.setattr(async_db_manager, "acache_url", persist)
    monkeypatch.setattr(
        crawler.crawler_strategy, "crawl",
        AsyncMock(return_value=AsyncCrawlResponse(
            html=HEALED_PAGE, status_code=200, response_headers={}
        )),
    )

    result = await crawler.arun(
        url, CrawlerRunConfig(max_html_bytes=4096, cache_mode=CacheMode.ENABLED, word_count_threshold=0)
    )

    assert result.success is True, result.error_message
    assert "Healed page" in result.html
    persist.assert_awaited_once()


@pytest.mark.parametrize("limit", [True, 0, -1, None, 1.5, "1024"])
def test_html_limit_requires_positive_integer(limit):
    with pytest.raises(ValueError, match="max_html_bytes must be a positive integer"):
        CrawlerRunConfig(max_html_bytes=limit)
