"""Health-driven restarts drain first and bound their rollback when startup or the driver hangs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai import browser_manager as browser_manager_module
from tests.regression.test_browser_lifecycle_residuals import (  # noqa: F401
    _Context,
    _Page,
    _manager,
)


@pytest.mark.asyncio
async def test_failed_restart_cleans_up_startup_without_joining_its_parent(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_START_SECONDS", 5.0)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_WAIT_SECONDS", 5.0)
    monkeypatch.setattr(browser_manager_module, "CANCELLATION_GRACE_SECONDS", 0.05)
    manager = _manager(BrowserConfig(headless=True, sleep_on_close=False))
    context = _Context()

    class _Browser:
        closed = False

        async def close(self):
            self.closed = True

    class _Driver:
        stopped = False

        async def stop(self):
            self.stopped = True

    browser, driver = _Browser(), _Driver()

    async def fail_start():
        manager.playwright = driver
        manager.browser = browser
        manager.default_context = context
        manager._default_context_owned = True
        raise RuntimeError("browser setup failed")

    manager._start_impl = fail_start
    manager.request_pool_recycle("browser health fault")
    await asyncio.wait_for(manager._pool_recycle_task, timeout=1)
    recycle = manager._recycle_task
    assert recycle is not None
    try:
        # This must finish before either internal five-second wait expires.
        await asyncio.wait_for(asyncio.shield(recycle), timeout=1)
        assert browser.closed
        assert context.closed
        assert driver.stopped
        assert manager.playwright is None
        assert manager._closing
        assert not manager._recycling
        assert manager._recycle_done.is_set()
        with pytest.raises(RuntimeError, match="unavailable"):
            await manager._admit_page_acquisition()
    finally:
        if not recycle.done():
            recycle.cancel()
        await asyncio.gather(recycle, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_until_deadline", [False, True])
async def test_external_cdp_health_recycle_drains_then_reconnects(
    monkeypatch, drain_until_deadline
):
    """A cdp_url manager's health recycle drains in-flight pages first, then
    reconnects this manager's own CDP client via start() instead of latching
    pool_recycle_requested -- the bug fixed here permanently bricked a
    standalone-library (non-pool) manager after one transient close fault.
    """
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MIN_SECONDS", 0.05)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MAX_SECONDS", 0.05)
    manager = _manager(
        BrowserConfig(cdp_url="http://external.example:9222", cdp_cleanup_on_close=False)
    )
    reconnected = asyncio.Event()

    async def start():
        # start() reconnects this manager's own CDP client to the external
        # endpoint (the external process is left alone); _started_browser
        # publishes the fresh connection and clears the recycle latch.
        manager.browser = _Context()
        manager.default_context = _Context()
        await manager._started_browser()
        reconnected.set()

    manager.start = start
    connection = _Context()
    manager.browser = connection
    token = await manager._admit_page_acquisition(page_timeout_ms=10)
    if not drain_until_deadline:
        manager._drain_seconds = 10

    manager.request_pool_recycle("external browser health fault")
    assert not manager.pool_recycle_requested
    assert not manager._recycle_done.is_set()
    await asyncio.wait_for(manager._pool_recycle_task, timeout=1)
    deadline = manager._drain_deadline
    assert deadline is not None
    assert token in manager._active_acquisitions
    assert not manager.pool_recycle_requested
    waiter = asyncio.create_task(manager._admit_page_acquisition(page_timeout_ms=10))
    try:
        await asyncio.sleep(0)
        assert not waiter.done(), "admission opened before in-flight pages drained"
        if drain_until_deadline:
            await asyncio.wait_for(asyncio.shield(deadline), timeout=1)
        else:
            await manager._finish_admission(token)

        # The drain released the in-flight token and triggered a reconnect
        # rather than latching pool_recycle_requested forever.
        assert not manager.pool_recycle_requested
        await asyncio.wait_for(asyncio.shield(manager._recycle_task), timeout=1)
        assert reconnected.is_set()
        assert manager._recycle_done.is_set()
        # The in-flight token was drained by the recycle (a parked waiter may
        # already have re-added its own -- that admission reopens is asserted
        # below).
        assert token not in manager._active_acquisitions
        # This client's old connection was torn down; start() built a fresh one.
        assert manager.browser is not connection
        assert connection.closed
        # A parked acquisition reopens once the reconnect lands -- no
        # "Browser manager is unavailable" (the permanent-latch symptom).
        token_after = await asyncio.wait_for(waiter, timeout=1)
        assert token_after in manager._active_acquisitions
        assert token_after is not token
        assert not manager.pool_recycle_requested
    finally:
        waiter.cancel()
        if deadline is not None and not deadline.done():
            deadline.cancel()
        await asyncio.gather(waiter, deadline, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
async def test_cdp_url_manager_keeps_serving_crawls_after_transient_close_fault():
    """End-to-end: a standalone-library (non-pool) cdp_url manager keeps
    serving crawls after one transient page.close() failure. Pre-fix the
    first failure latched pool_recycle_requested and every later get_page
    raised "Browser manager is unavailable" with no in-library remedy short
    of rebuilding the whole AsyncWebCrawler.
    """
    manager = _manager(
        BrowserConfig(cdp_url="http://external.example:9222", cdp_cleanup_on_close=False)
    )
    reconnected = asyncio.Event()

    async def start():
        manager.browser = object()
        manager.default_context = _Context()
        await manager._started_browser()
        reconnected.set()

    manager.start = start
    manager.browser = _Context()

    first_page, _ = await manager.get_page(CrawlerRunConfig())

    async def transient_close():
        raise RuntimeError("page close timed out")

    first_page.close = transient_close
    await manager.release_page_with_context(first_page)

    for _ in range(200):
        await asyncio.sleep(0.01)
        if manager._recycle_task is not None and manager._recycle_task.done():
            break
    assert reconnected.is_set()
    assert not manager.pool_recycle_requested

    # A later crawl succeeds instead of failing with "unavailable".
    second_page, _ = await manager.get_page(CrawlerRunConfig())
    assert second_page is not first_page
    await manager.release_page_with_context(second_page)
    await manager.close()


@pytest.mark.asyncio
async def test_cdp_url_recycle_failure_closes_admission_not_loops_forever():
    """If the external CDP endpoint is genuinely down, the cdp_url reconnect
    fails and the manager closes admission (so the pool discards the crawler
    via _is_live/_closing and a standalone user gets a clear error), rather
    than looping reconnects or latching pool_recycle_requested. This is the
    shared _recycle_browser failure path, exercised here on the new cdp_url
    close(_for_recycle=True) disconnect branch.
    """
    manager = _manager(
        BrowserConfig(cdp_url="http://external.example:9222", cdp_cleanup_on_close=False)
    )

    async def start():
        raise RuntimeError("CDP endpoint is not ready after startup")

    manager.start = start
    manager.browser = _Context()

    # No in-flight pages, so the drain completes immediately and the
    # reconnect attempt runs at once.
    manager.request_pool_recycle("external browser health fault")
    await asyncio.wait_for(manager._pool_recycle_task, timeout=2)
    await asyncio.wait_for(asyncio.shield(manager._recycle_task), timeout=2)

    assert manager._closing
    assert not manager._recycling
    assert manager._recycle_done.is_set()
    # A failed reconnect must not latch pool_recycle_requested; _closing is the
    # pool's discard signal instead (crawler_pool._is_live checks _closing).
    assert not manager.pool_recycle_requested
    with pytest.raises(RuntimeError, match="unavailable"):
        await manager._admit_page_acquisition()
    await manager.close()


@pytest.mark.asyncio
async def test_failed_restart_bounds_rollback_when_driver_stop_hangs(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_CLOSE_SECONDS", 0.02)
    monkeypatch.setattr(browser_manager_module, "DRIVER_STOP_SECONDS", 0.02)
    manager = _manager()

    async def never_stops():
        await asyncio.Event().wait()

    driver = MagicMock(stop=AsyncMock(side_effect=never_stops))

    async def fail_start():
        manager.playwright = driver
        raise RuntimeError("browser setup failed")

    manager._start_impl = fail_start
    manager.request_pool_recycle("browser health fault")
    await asyncio.wait_for(manager._pool_recycle_task, 1)
    await asyncio.wait_for(manager._recycle_task, 1)

    # Both the ordinary rollback close and its driver-stop fallback are bounded.
    assert driver.stop.await_count == 2
    assert manager.playwright is None
    assert manager._closing and not manager._recycling
    assert manager._recycle_done.is_set()


@pytest.mark.asyncio
async def test_a_hung_page_close_still_returns_its_admission_token(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "PAGE_CLOSE_SECONDS", 0.05)
    manager = _manager()
    # Resource-close failure now schedules a restart. Keep that lifecycle on
    # the fake manager instead of launching Playwright during pytest teardown.
    manager.start = AsyncMock()
    manager.config.sleep_on_close = False
    token = await manager._admit_page_acquisition()
    assert manager._active_acquisitions

    hung = asyncio.Event()

    async def never_closes():
        await hung.wait()

    page = _Page()
    page.close = never_closes
    await manager._rollback_page_acquisition(
        token, page=page, close_page=True
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if not manager._active_acquisitions:
            break
    assert not manager._active_acquisitions, "a wedged close stranded the token"
    hung.set()
    await manager._pool_recycle_task
    await manager._recycle_task
    manager.start.assert_awaited_once()
