import time
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from playwright.async_api import Error

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, ProxyConfig
from crawl4ai.async_crawler_strategy import (
    AsyncPlaywrightCrawlerStrategy,
    RENDERER_KILL_ATTRIBUTION_WINDOW_SECONDS,
    TargetNavigationError,
)
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_webcrawler import async_db_manager
from crawl4ai.browser_manager import BrowserManager, RendererKill
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

    def error(self, message, tag="ERROR", **kwargs):
        self.events.append(("error", tag, message))

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
@pytest.mark.parametrize(
    "record,expected_code,phase",
    [
        ("rss", "renderer_memory_limit", "goto"),
        ("cpu", "renderer_cpu_limit", "goto"),
        ("unrecorded", None, "goto"),
        ("unknown_browser", None, "goto"),
        ("foreign_browser", None, "goto"),
        ("orphaned", None, "goto"),
        ("fallback", "renderer_memory_limit", "goto"),
        ("fallback_failed", "renderer_memory_limit", "goto"),
        ("ambiguous", None, "goto"),
        ("consumed", None, "goto"),
        ("rss", "renderer_memory_limit", "mhtml"),
        ("unrecorded", None, "mhtml"),
        ("rss", "renderer_memory_limit", "screenshot"),
        ("unrecorded", None, "screenshot"),
    ],
)
async def test_renderer_kill_provenance_controls_crawl_failure_severity(
    tmp_path, record, expected_code, phase
):
    # Playwright supplies no renderer PID. Only one recent, unconsumed kill
    # from this crawl's known browser can explain the terminal crash.
    manager = BrowserManager(BrowserConfig(headless=True))
    manager.browser_pid = None if record == "unknown_browser" else 1234
    now = time.monotonic()
    rss_bytes = 1053999104
    rss_limit_bytes = 805306368
    kill_age = RENDERER_KILL_ATTRIBUTION_WINDOW_SECONDS + 0.1 if record == "orphaned" else 0.1
    if record != "unrecorded":
        manager.renderer_kills.append(
            RendererKill(
                killed_at=now - kill_age,
                reason="cpu" if record == "cpu" else "rss",
                renderer_pid=12345,
                browser_pid=4321 if record == "foreign_browser" else 1234,
                rss_bytes=rss_bytes,
                rss_limit_bytes=rss_limit_bytes,
                cpu_cores=1.0 if record == "cpu" else None,
                cpu_limit_cores=0.9 if record == "cpu" else None,
            )
        )
    if record == "ambiguous":
        manager.renderer_kills.append(
            RendererKill(now - 0.1, "rss", 12346, 1234, rss_bytes, rss_limit_bytes)
        )
    if record == "consumed":
        manager.attribute_renderer_kill(now, 10, browser_pid=1234)

    strategy = _navigation_strategy("ignored", browser_manager=manager)
    page, context = strategy.browser_manager.get_page.return_value
    crash_handlers = {}
    page.on = lambda event, handler: crash_handlers.__setitem__(event, handler)
    strategy.browser_manager._close_page_quietly = AsyncMock()

    async def crash_then_fail(*args, **kwargs):
        crash_handlers["crash"](page)
        raise Error("Page crashed")

    page.goto = AsyncMock(side_effect=crash_then_fail)
    logger = _RecordingLogger()
    strategy.logger = logger
    if phase != "goto":
        # Exercise helpers through the supported page hook. These helpers
        # normally swallow export errors; a terminal crash must escape them.
        async def export_from_hook(page, **kwargs):
            if phase == "mhtml":
                await strategy.capture_mhtml(page)
            else:
                await strategy.take_screenshot_naive(page)

        strategy.set_hook("before_goto", export_from_hook)
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.evaluate = AsyncMock()
        page.context = context
        context.new_cdp_session.return_value.send = AsyncMock(side_effect=crash_then_fail)
        page.screenshot = AsyncMock(side_effect=crash_then_fail)
    crawler = AsyncWebCrawler(
        crawler_strategy=strategy,
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    fallback = AsyncMock(return_value="<p>fallback</p>")
    if record == "fallback_failed":
        fallback.side_effect = RuntimeError("fallback unavailable")
    crawler.aprocess_html = AsyncMock(return_value=CrawlResult(
        url="https://example.invalid", html="<p>fallback</p>", success=True,
    ))

    result = await crawler.arun(
        "https://example.invalid",
        CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            max_retries=2 if expected_code else 0,
            fallback_fetch_function=fallback if record.startswith("fallback") else None,
            session_id="renderer-session",
        ),
    )

    assert result.success is (record == "fallback")
    assert page.goto.await_count == (1 if phase == "goto" else 0)
    if record.startswith("fallback"):
        fallback.assert_awaited_once_with("https://example.invalid")
        assert result.crawl_stats["fallback_fetch_used"]
    else:
        fallback.assert_not_awaited()
    if expected_code:
        assert result.crawl_stats["attempts"] == 1
        assert result.session_id == "renderer-session"
        if record == "fallback":
            assert result.crawl_stats["resolved_by"] == "fallback_fetch"
            assert result.html == "<p>fallback</p>"
        else:
            detail = (
                "cpu_cores=1.0 cpu_limit_cores=0.9" if record == "cpu" else
                f"rss_bytes={rss_bytes} rss_limit_bytes={rss_limit_bytes}"
            )
            assert result.error_message == f"{expected_code}: {detail}"
            assert any(event[:2] == ("warning", "WARNING") for event in logger.events)
            assert any(event[:2] == ("info", "COMPLETE") for event in logger.events)
        if record != "fallback_failed":
            assert not any(event[0] == "error" for event in logger.events)
        assert manager.consecutive_failures == 0
    else:
        assert "Unexpected error in" in result.error_message
        assert "Target page crashed" in result.error_message
        assert any(event[0] == "error" for event in logger.events)
        assert not any(event[0] == "warning" for event in logger.events)
        assert manager.consecutive_failures == 1
    assert not strategy._page_crash_errors


@pytest.mark.asyncio
async def test_started_browser_clears_previous_renderer_failure_ledgers(monkeypatch):
    manager = BrowserManager(BrowserConfig(headless=True))
    manager.renderer_kills.append(
        RendererKill(time.monotonic(), "rss", 12345, 1234, 1053999104, 805306368)
    )
    monkeypatch.setattr(manager, "_attribute_browser_pid", AsyncMock())

    await manager._started_browser()

    assert not manager.renderer_kills


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
