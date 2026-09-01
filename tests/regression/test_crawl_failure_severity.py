from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, ProxyConfig
from crawl4ai.async_crawler_strategy import (
    AsyncPlaywrightCrawlerStrategy,
    TargetNavigationError,
)
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_webcrawler import async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult


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
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def crawl(self, url, config):
        if self.error:
            raise self.error
        return self.response


def _navigation_strategy(error_detail):
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True),
        logger=AsyncLogger(verbose=False),
    )
    page = MagicMock()
    page.goto = AsyncMock(side_effect=Error(error_detail))
    page.context.browser.contexts = []
    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = AsyncMock(
        return_value=(page, MagicMock())
    )
    strategy.browser_manager.release_page_with_context = AsyncMock()
    return strategy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    (
        "net::ERR_NAME_NOT_RESOLVED",
        "net::ERR_CONNECTION_REFUSED",
        "net::ERR_CERT_AUTHORITY_INVALID",
        "net::ERR_SSL_PROTOCOL_ERROR",
    ),
)
async def test_direct_target_navigation_refusals_preserve_provenance(error_code):
    strategy = _navigation_strategy(f"page.goto: {error_code}")

    with pytest.raises(TargetNavigationError, match="Target navigation refused"):
        await strategy.crawl("https://example.invalid", CrawlerRunConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_detail",
    (
        "page.goto: net::ERR_TIMED_OUT",
        "Target page, context or browser has been closed",
    ),
)
async def test_unproven_navigation_failures_remain_platform_errors(error_detail):
    strategy = _navigation_strategy(error_detail)

    with pytest.raises(RuntimeError, match="Failed on navigating ACS-GOTO"):
        await strategy.crawl("https://example.invalid", CrawlerRunConfig())


@pytest.mark.asyncio
async def test_proxy_navigation_failures_remain_platform_errors():
    strategy = _navigation_strategy("page.goto: net::ERR_CONNECTION_REFUSED")
    config = CrawlerRunConfig(
        proxy_config=ProxyConfig(server="http://proxy.invalid:8080")
    )

    with pytest.raises(RuntimeError, match="Failed on navigating ACS-GOTO"):
        await strategy.crawl("https://example.invalid", config)


@pytest.mark.asyncio
async def test_antibot_outcome_is_info_at_crawl_owner(tmp_path):
    html = "<html><body><h1>Access Denied</h1></body></html>"
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(
            response=AsyncCrawlResponse(
                html=html,
                response_headers={},
                status_code=403,
            )
        ),
        config=BrowserConfig(verbose=True),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    crawler.aprocess_html = AsyncMock(
        return_value=CrawlResult(
            url="https://example.com",
            html=html,
            success=True,
        )
    )

    result = await crawler.arun(
        "https://example.com",
        CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
    )

    assert not result.success
    assert any(event[:2] == ("info", "TARGET") for event in logger.events)
    assert any(event[:2] == ("info", "COMPLETE") for event in logger.events)
    assert not any(event[0] == "error" for event in logger.events)


@pytest.mark.asyncio
async def test_cached_target_refusal_remains_failed(tmp_path, monkeypatch):
    html = "<html><body><h1>Access Denied</h1></body></html>"
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(error=AssertionError("cache miss")),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    monkeypatch.setattr(
        async_db_manager,
        "aget_cached_url",
        AsyncMock(
            return_value=CrawlResult(
                url="https://example.com",
                html=html,
                success=False,
                status_code=403,
                error_message="Blocked by anti-bot protection",
            )
        ),
    )

    result = await crawler.arun(
        "https://example.com",
        CrawlerRunConfig(cache_mode=CacheMode.READ_ONLY),
    )

    assert not result.success
    assert result.error_message == "Blocked by anti-bot protection"


@pytest.mark.asyncio
async def test_fallback_failure_remains_error(tmp_path):
    html = "<html><body><h1>Access Denied</h1></body></html>"
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(
            response=AsyncCrawlResponse(
                html=html,
                response_headers={},
                status_code=403,
            )
        ),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    crawler.aprocess_html = AsyncMock(
        return_value=CrawlResult(
            url="https://example.com",
            html=html,
            success=True,
        )
    )

    async def failed_fallback(url):
        raise RuntimeError("fallback unavailable")

    result = await crawler.arun(
        "https://example.com",
        CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            fallback_fetch_function=failed_fallback,
        ),
    )

    assert not result.success
    assert any(
        event[0] == "error" and "Fallback fetch failed" in event[2]
        for event in logger.events
    )


@pytest.mark.asyncio
async def test_target_navigation_is_info_but_browser_failure_is_error(tmp_path):
    target_logger = _RecordingLogger()
    target_crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(
            error=TargetNavigationError("Target navigation refused: DNS")
        ),
        base_directory=str(tmp_path / "target"),
        logger=target_logger,
    )
    target_crawler.ready = True

    target_result = await target_crawler.arun(
        "https://example.invalid",
        CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
    )

    platform_logger = _RecordingLogger()
    platform_crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(error=RuntimeError("browser crashed")),
        base_directory=str(tmp_path / "platform"),
        logger=platform_logger,
    )
    platform_crawler.ready = True

    platform_result = await platform_crawler.arun(
        "https://example.invalid",
        CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
    )

    assert not target_result.success
    assert any(event[:2] == ("info", "TARGET") for event in target_logger.events)
    assert not any(event[0] == "error" for event in target_logger.events)
    assert not platform_result.success
    assert any(event[0] == "error" for event in platform_logger.events)
    assert not any(
        event[:2] == ("info", "TARGET") for event in platform_logger.events
    )
