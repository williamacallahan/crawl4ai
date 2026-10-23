"""A browser at its fault limit drains and restarts; only a wedged one is disposed of at once."""

import asyncio
from dataclasses import replace

import pytest
from playwright.async_api import Error as PlaywrightError

from crawl4ai import BrowserConfig, CrawlerRunConfig
from test_is_live_recycle import (  # noqa: F401
    _Browser as _PlainBrowser,
    _Context,
    _configure_pool,
    _crawler_with_manager,
    _drain_close_tasks,
    _live_factory,
    _manager,
)
import crawler_pool  # noqa: E402  (test_is_live_recycle puts deploy/docker on sys.path)


class _CDPSession:
    def __init__(self):
        self.commands = []
        self.error = None
        self.blocked = None

    async def send(self, command):
        self.commands.append(command)
        if self.blocked is not None:
            await self.blocked.wait()
        if self.error is not None:
            raise self.error
        return {"product": "Chrome"}


class _Browser(_PlainBrowser):
    """A browser double that also answers the janitor's CDP health probe."""

    def __init__(self, connected=True):
        super().__init__(connected)
        self.session = _CDPSession()

    async def new_browser_cdp_session(self):
        return self.session


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["admission", "watchdog"])
async def test_browser_at_its_fault_limit_drains_instead_of_closing_under_crawls(
    monkeypatch, trigger,
):
    """A responsive browser drains real page leases before restarting in place."""
    _configure_pool(monkeypatch, _live_factory([]))
    manager = _manager(BrowserConfig(headless=True))
    browser = manager.browser = _Browser()
    manager.default_context = _Context()
    page, _ = await manager.get_page(CrawlerRunConfig())
    crawler = _crawler_with_manager(manager)
    crawler.active_requests = 1
    crawler_pool.COLD_POOL["draining"] = crawler

    async def start():
        manager.browser = _Browser()
        manager.default_context = _Context()

    manager.start = start
    manager.consecutive_failures = crawler_pool.WATCHDOG.consecutive_failure_limit
    try:
        if trigger == "admission":
            assert crawler_pool._is_live(crawler) is True
        await crawler_pool._watchdog_pass()

        assert crawler_pool.COLD_POOL["draining"] is crawler
        assert not crawler.closed
        assert not page.is_closed()
        assert browser.is_connected()
        assert manager._recycle_due
        assert not manager._recycling
        assert not manager._recycle_done.is_set()
        assert manager.pool_recycle_reason == "consecutive browser faults"
        assert manager.consecutive_failures == 0
        assert browser.session.commands == ["Browser.getVersion"]

        await manager.release_page_with_context(page)
        await manager._recycle_task
        assert not browser.is_connected()
        assert manager.browser is not browser
        assert manager.browser.is_connected()
        assert crawler_pool.COLD_POOL["draining"] is crawler
        assert crawler_pool._is_live(crawler)
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["target_closed", "timeout"])
async def test_fault_limit_still_probes_and_disposes_an_unresponsive_browser(
    monkeypatch, failure,
):
    """A fault-triggered drain must not bypass an exhausted CDP failure budget."""
    _configure_pool(monkeypatch, _live_factory([]))
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(0))
    manager = _manager()
    browser = manager.browser = _Browser()
    manager.default_context = _Context()
    await manager.get_page(CrawlerRunConfig())
    crawler = _crawler_with_manager(manager)
    crawler.active_requests = 1
    crawler_pool.COLD_POOL["unresponsive"] = crawler
    manager.consecutive_failures = crawler_pool.WATCHDOG.consecutive_failure_limit
    manager._docker_health_browser = browser
    manager._docker_health_failures = crawler_pool.WATCHDOG.consecutive_failure_limit - 1
    manager._docker_health_session = browser.session
    if failure == "target_closed":
        browser.session.error = PlaywrightError("Target closed")
    else:
        browser.session.blocked = asyncio.Event()
        monkeypatch.setattr(
            crawler_pool, "WATCHDOG",
            replace(crawler_pool.WATCHDOG, browser_probe_timeout_seconds=0.01),
        )
    try:
        await crawler_pool._watchdog_pass()
        await _drain_close_tasks()

        assert browser.session.commands == ["Browser.getVersion"]
        assert "unresponsive" not in crawler_pool.COLD_POOL
        assert crawler.closed
        assert crawler.active_requests == 0
        assert crawler_pool.ADMISSION_SEM._value == 1
        assert manager.consecutive_failures == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["closing", "disconnected"])
async def test_fault_limit_does_not_make_an_unavailable_manager_live(state):
    manager = _manager()
    manager.browser = _Browser(connected=state != "disconnected")
    manager._closing = state == "closing"
    manager.consecutive_failures = crawler_pool.WATCHDOG.consecutive_failure_limit

    try:
        assert not crawler_pool._is_live(_crawler_with_manager(manager))
        assert manager._pool_recycle_task is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_probe_retires_a_closed_manager_before_scheduling_fault_recycle():
    manager = _manager()
    manager.browser = _Browser()
    manager._closing = True
    manager.consecutive_failures = crawler_pool.WATCHDOG.consecutive_failure_limit

    try:
        assert await crawler_pool._probe_browser(_crawler_with_manager(manager))
        assert manager._pool_recycle_task is None
        assert manager.browser.session.commands == []
    finally:
        await manager.close()
