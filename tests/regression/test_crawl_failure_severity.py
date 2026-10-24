import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, Mock

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
from crawl4ai.browser_manager import BrowserManager
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


def _navigation_strategy(error_detail, *, browser_manager=None):
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True),
        logger=AsyncLogger(verbose=False),
    )
    page = MagicMock()
    page.goto = AsyncMock(side_effect=Error(error_detail))
    page.context.browser.contexts = []
    if browser_manager is None:
        browser_manager = MagicMock()
        browser_manager.tearing_down = False
        browser_manager.page_crashes = deque()
        browser_manager.attribute_renderer_kill = Mock(return_value=None)
    strategy.browser_manager = browser_manager
    context = MagicMock()
    context.new_cdp_session = AsyncMock(return_value=AsyncMock())
    strategy.browser_manager.get_page = AsyncMock(
        return_value=(page, context)
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
async def test_page_crash_without_renderer_pid_stays_a_fault_despite_recent_kill():
    # A nearby kill can belong to another page. Playwright's crash event
    # supplies no renderer PID, so timing alone cannot explain this crash.
    manager = BrowserManager(BrowserConfig(headless=True))
    strategy = _navigation_strategy("ignored", browser_manager=manager)
    page, _ = strategy.browser_manager.get_page.return_value
    crash_handlers = {}
    page.on = lambda event, handler: crash_handlers.__setitem__(event, handler)
    strategy.browser_manager._close_page_quietly = AsyncMock()
    manager.renderer_kills.append((time.monotonic(), "rss", 12345))

    async def crash_then_fail(*args, **kwargs):
        crash_handlers["crash"](page)
        raise Error("Page crashed")

    page.goto = AsyncMock(side_effect=crash_then_fail)

    with pytest.raises(Error) as raised:
        await strategy.crawl("https://example.invalid", CrawlerRunConfig())

    assert str(raised.value) == "Target page crashed"
    assert manager.consecutive_failures == 1


@pytest.mark.asyncio
async def test_page_crash_stays_a_fault_when_crashes_outnumber_kills():
    # Multiple unknown renderers crashing near a kill remain browser faults.
    manager = BrowserManager(BrowserConfig(headless=True))
    strategy = _navigation_strategy("ignored", browser_manager=manager)
    page, _ = strategy.browser_manager.get_page.return_value
    crash_handlers = {}
    page.on = lambda event, handler: crash_handlers.__setitem__(event, handler)
    strategy.browser_manager._close_page_quietly = AsyncMock()
    manager.page_crashes.append(time.monotonic())
    manager.renderer_kills.append((time.monotonic(), "rss", 12345))

    async def crash_then_fail(*args, **kwargs):
        crash_handlers["crash"](page)
        raise Error("Page crashed")

    page.goto = AsyncMock(side_effect=crash_then_fail)

    with pytest.raises(Error) as raised:
        await strategy.crawl("https://example.invalid", CrawlerRunConfig())

    assert str(raised.value) == "Target page crashed"


@pytest.mark.asyncio
async def test_started_browser_clears_previous_renderer_failure_ledgers(monkeypatch):
    manager = BrowserManager(BrowserConfig(headless=True))
    manager.renderer_kills.append((time.monotonic(), "rss", 12345))
    manager.page_crashes.append(time.monotonic())
    monkeypatch.setattr(manager, "_attribute_browser_pid", AsyncMock())

    await manager._started_browser()

    assert not manager.renderer_kills
    assert not manager.page_crashes


@pytest.mark.asyncio
@pytest.mark.parametrize("recycle_trigger", ("page_count", "request_recycle"))
async def test_closed_call_during_session_deferred_recycle_stays_unexpected(
    recycle_trigger,
):
    manager = BrowserManager(
        BrowserConfig(
            headless=True,
            max_pages_before_recycle=1 if recycle_trigger == "page_count" else 0,
        )
    )
    strategy = _navigation_strategy(
        "Target page, context or browser has been closed",
        browser_manager=manager,
    )
    page, context = manager.get_page.return_value
    retained_session = (context, page, time.time())
    manager.sessions["retained"] = retained_session
    await manager._note_page_served(session_id="retained")
    if recycle_trigger == "request_recycle":
        assert not await manager.request_recycle()

    assert manager._recycle_due
    assert manager._recycle_done.is_set()
    assert not manager.tearing_down
    with pytest.raises(RuntimeError, match="Failed on navigating ACS-GOTO"):
        await strategy.crawl(
            "https://example.invalid", CrawlerRunConfig(session_id="retained")
        )

    assert manager.sessions["retained"] is retained_session
    assert manager._recycle_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("recycling", (False, True))
async def test_closed_call_during_browser_drain_is_a_watchdog_rejection(recycling):
    # A page/context/protocol call closed while the pool deliberately drains
    # this browser is a designed rejection, not an unexpected fault.
    manager = BrowserManager(BrowserConfig(headless=True))
    strategy = _navigation_strategy(
        "Target page, context or browser has been closed",
        browser_manager=manager,
    )
    manager._recycle_due = True
    manager._recycle_done.clear()
    manager._recycling = recycling

    with pytest.raises(Error, match="browser drained by the resource watchdog"):
        await strategy.crawl("https://example.invalid", CrawlerRunConfig())


@pytest.mark.asyncio
async def test_closed_call_without_drain_stays_unexpected():
    strategy = _navigation_strategy(
        "Target page, context or browser has been closed"
    )

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refusal", "fallback_runs"),
    [
        ("Target navigation refused: net::ERR_CERT_AUTHORITY_INVALID", False),
        ("Target navigation refused: net::ERR_CONNECTION_REFUSED", True),
    ],
)
async def test_tls_refusal_never_reaches_the_fallback(tmp_path, refusal, fallback_runs):
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(error=TargetNavigationError(refusal)),
        base_directory=str(tmp_path),
        logger=_RecordingLogger(),
    )
    crawler.ready = True
    crawler.aprocess_html = AsyncMock(
        return_value=CrawlResult(url="https://example.com", html="<p>unlocked</p>", success=True)
    )
    fetched = []

    async def fallback(url):
        fetched.append(url)
        return "<p>unlocked</p>"

    result = await crawler.arun(
        "https://example.com",
        CrawlerRunConfig(cache_mode=CacheMode.BYPASS, fallback_fetch_function=fallback),
    )

    assert result.success is fallback_runs
    assert fetched == (["https://example.com"] if fallback_runs else [])
