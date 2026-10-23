"""Health-driven restarts drain first and bound their rollback when startup or the driver hangs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai import BrowserConfig
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
async def test_external_cdp_health_quarantine_waits_for_drain(
    monkeypatch, drain_until_deadline
):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MIN_SECONDS", 0.05)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MAX_SECONDS", 0.05)
    manager = _manager(
        BrowserConfig(cdp_url="http://external.example:9222", cdp_cleanup_on_close=False)
    )
    manager.start = AsyncMock()
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
        assert not waiter.done()
        if drain_until_deadline:
            await asyncio.wait_for(asyncio.shield(deadline), timeout=1)
        else:
            await manager._finish_admission(token)

        assert manager.pool_recycle_requested
        assert manager._recycle_done.is_set()
        manager.request_pool_recycle("late page close failure")
        assert manager._recycle_done.is_set()
        assert not manager._active_acquisitions
        with pytest.raises(RuntimeError, match="unavailable"):
            await asyncio.wait_for(waiter, timeout=1)
        manager.start.assert_not_awaited()
        assert manager._recycle_task is None
        assert manager.browser is connection
        assert not connection.closed
    finally:
        waiter.cancel()
        deadline.cancel()
        await asyncio.gather(waiter, deadline, return_exceptions=True)
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
