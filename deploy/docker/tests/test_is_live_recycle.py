"""A browser remains live during its own close/start recycle window.

These tests drive BrowserManager.close with start parked on an asyncio.Event,
then exercise pool liveness, discard and acquisition. A recycling manager
must stay registered while its browser reference is temporarily None.
The separate test_real_browser_recycle_window.py covers real Chromium.
"""

import asyncio
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from playwright.async_api import Error as PlaywrightError

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool  # noqa: E402


# Playwright-shaped doubles mirroring the stubs the library's own recycle
# tests use (tests/regression/test_browser_lifecycle_residuals.py), so the real
# ``close(_for_recycle=True)`` path can run without launching a browser.


class _Page:
    def __init__(self, context=None):
        self.context = context
        self.closed = False

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True

    async def evaluate(self, script):
        pass


class _Context:
    def __init__(self):
        self.pages = []
        self.closed = False
        self.handlers = {}
        self.headers = {}

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    async def set_extra_http_headers(self, headers):
        self.headers = dict(headers)

    def is_closed(self):
        return self.closed

    def remove_listener(self, event, handler):
        callbacks = self.handlers.get(event, [])
        if handler in callbacks:
            callbacks.remove(handler)

    async def new_page(self):
        page = _Page(self)
        self.pages.append(page)
        for handler in self.handlers.get("page", []):
            handler(page)
        return page

    async def close(self):
        self.closed = True
        for page in self.pages:
            await page.close()


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


class _Browser:
    def __init__(self, connected=True):
        self._connected = connected
        self.session = _CDPSession()

    def is_connected(self):
        return self._connected

    async def new_context(self, **_settings):
        return _Context()

    async def new_browser_cdp_session(self):
        return self.session

    async def close(self):
        self._connected = False


@pytest.fixture(autouse=True)
def _reset_global_pages():
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    yield
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None


def _manager(config=None):
    manager = BrowserManager(
        config or BrowserConfig(headless=True), logger=None
    )
    manager.managed_browser = None
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    return manager


class _RealManagerCrawler:
    """A crawler wrapping a real BrowserManager with the pool's required surface."""

    def __init__(self, manager):
        self.crawler_strategy = SimpleNamespace(browser_manager=manager)
        self.active_requests = 0
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True


def _crawler_with_manager(manager):
    return _RealManagerCrawler(manager)


async def _drive_to_mid_recycle(manager, permit_restart):
    """Drive the real recycle to the mid-recycle window and park there.

    Uses the real ``close(_for_recycle=True)`` (which nulls ``browser``/
    ``default_context``) and a stub ``start()`` that blocks on
    ``permit_restart`` so the manager stays in the healthy-but-null state.
    """
    manager.default_context = _Context()
    manager.browser = _Browser(connected=True)
    closed_event = asyncio.Event()
    real_close = manager.close

    async def tracking_close(_for_recycle=False):
        await real_close(_for_recycle=_for_recycle)
        if _for_recycle:
            closed_event.set()

    manager.close = tracking_close

    async def start():
        await permit_restart.wait()
        manager.default_context = _Context()
        manager.browser = _Browser(connected=True)

    manager.start = start

    page, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(page)
    await closed_event.wait()
    return page


def _configure_pool(monkeypatch, factory):
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(10))
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 10)
    monkeypatch.setattr(crawler_pool, "MEM_LIMIT", 100)
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "DEFAULT_CONFIG_SIG", None)
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "get_container_memory_percent", lambda: 0)
    monkeypatch.setattr(crawler_pool, "AsyncWebCrawler", factory)


def _live_factory(created):
    """Factory that produces a crawler owning a live, connected browser."""

    def factory(**_kwargs):
        manager = _manager(BrowserConfig(headless=True))
        manager.browser = _Browser(connected=True)
        manager.default_context = _Context()
        crawler = _RealManagerCrawler(manager)
        created.append(crawler)
        return crawler

    return factory


async def _drain_close_tasks():
    tasks = list(crawler_pool._CLOSE_TASKS)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_is_live_returns_true_for_recycling_manager_with_browser_none():
    """The fix: a healthy mid-recycle manager (browser=None) is live."""
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = None
    manager.default_context = None
    manager._recycling = True
    manager._closing = False

    assert crawler_pool._is_live(_crawler_with_manager(manager)) is True


@pytest.mark.asyncio
async def test_is_live_returns_true_for_recycling_manager_even_if_browser_disconnected():
    """Even if a stale browser object lingers, recycling takes precedence.

    Guards against reordering the checks to put ``browser.is_connected()``
    before the ``_recycling`` short-circuit.
    """
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=False)
    manager._recycling = True
    manager._closing = False

    assert crawler_pool._is_live(_crawler_with_manager(manager)) is True


def test_is_live_uses_getattr_default_for_managers_without_recycling_attr():
    """Mock managers from older tests have no _recycling attr; must fall through.

    Guards the ``getattr(manager, "_recycling", False)`` backward-compat fallback
    so the fix does not break existing pool mocks (e.g. ``_PoolCrawler``).
    """
    browser = MagicMock()
    browser.is_connected.return_value = True
    manager = SimpleNamespace(browser=browser, default_context=None)
    crawler = SimpleNamespace(crawler_strategy=SimpleNamespace(browser_manager=manager))

    assert crawler_pool._is_live(crawler) is True


@pytest.mark.asyncio
async def test_real_recycle_mid_window_state_is_live():
    """Drive the real close path to the mid-recycle state; _is_live is True."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()

    await _drive_to_mid_recycle(manager, permit_restart)

    assert manager._recycling is True
    assert manager._closing is False
    assert manager.browser is None
    assert manager.default_context is None
    assert crawler_pool._is_live(_crawler_with_manager(manager)) is True

    permit_restart.set()
    await manager._recycle_task
    assert manager.browser is not None
    assert manager.browser.is_connected() is True


@pytest.mark.asyncio
async def test_discard_if_unavailable_keeps_a_mid_recycle_pool_crawler():
    """The pool-hit path must not discard a healthy recycling browser."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)

    crawler = _crawler_with_manager(manager)
    pool = {"deadbeef": crawler}

    try:
        result = crawler_pool._discard_if_unavailable(pool, "deadbeef", "Cold")
        assert result is None
        assert "deadbeef" in pool
        assert pool["deadbeef"] is crawler
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_get_crawler_serves_recycling_permanent_without_force_replace(monkeypatch):
    """get_crawler returns the recycling PERMANENT; no replacement is created."""
    created = []
    _configure_pool(monkeypatch, _live_factory(created))

    config = BrowserConfig(headless=True)
    sig = crawler_pool._sig(config)

    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)

    recycling = _crawler_with_manager(manager)
    crawler_pool.PERMANENT = recycling
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    force_true = 0
    real_init = crawler_pool._init_permanent_locked

    async def spy_init(cfg, *, force=False, target=None):
        nonlocal force_true
        if force:
            force_true += 1
        return await real_init(cfg, force=force, target=target)

    monkeypatch.setattr(crawler_pool, "_init_permanent_locked", spy_init)

    try:
        crawler = await asyncio.wait_for(
            crawler_pool.get_crawler(config), timeout=1
        )
        assert crawler is recycling
        assert crawler_pool.PERMANENT is recycling
        assert force_true == 0
        assert created == []

        await crawler_pool.release_crawler(crawler)
    finally:
        permit_restart.set()
        await manager._recycle_task
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_blast_radius_zero_force_replaces_with_fix(monkeypatch):
    """5 concurrent default-config requests during one recycle -> 0 force-replaces.

    Guards the lock-contention path: under concurrency the first request must
    still serve the recycling PERMANENT (not force-replace it), and every other
    request must share it - no force-replace storm.
    """
    created = []
    _configure_pool(monkeypatch, _live_factory(created))

    config = BrowserConfig(headless=True)
    sig = crawler_pool._sig(config)

    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)

    recycling = _crawler_with_manager(manager)
    crawler_pool.PERMANENT = recycling
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    force_true = 0
    real_init = crawler_pool._init_permanent_locked

    async def spy_init(cfg, *, force=False, target=None):
        nonlocal force_true
        if force:
            force_true += 1
        return await real_init(cfg, force=force, target=target)

    monkeypatch.setattr(crawler_pool, "_init_permanent_locked", spy_init)

    requests = [
        asyncio.create_task(crawler_pool.get_crawler(config)) for _ in range(5)
    ]
    try:
        results = await asyncio.wait_for(asyncio.gather(*requests), timeout=2)
        assert all(result is recycling for result in results)
        assert crawler_pool.PERMANENT is recycling
        assert force_true == 0
        assert created == []
        for crawler in results:
            await crawler_pool.release_crawler(crawler)
    finally:
        permit_restart.set()
        await manager._recycle_task
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_failed_recycle_is_treated_as_dead_and_replaced(monkeypatch):
    """A recycle whose start() fails is genuinely dead and must be replaced.

    On a failed restart ``_recycle_browser`` clears ``_recycling`` and sets
    ``_closing=True`` in its finally, so ``_is_live`` returns False - the fix
    must NOT mask a failed recycle as live. Guards the fix being over-broad.
    """
    created = []
    _configure_pool(monkeypatch, _live_factory(created))

    config = BrowserConfig(headless=True)
    sig = crawler_pool._sig(config)

    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    manager.default_context = _Context()
    manager.browser = _Browser(connected=True)

    async def failing_start():
        raise RuntimeError("restart failed")

    async def close(_for_recycle=False):
        manager.browser = None
        manager.default_context = None

    manager.start = failing_start
    manager.close = close

    page, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(page)
    await manager._recycle_task

    assert manager._recycling is False
    assert manager._closing is True
    assert manager.browser is None
    assert crawler_pool._is_live(_crawler_with_manager(manager)) is False

    dead = _crawler_with_manager(manager)
    crawler_pool.PERMANENT = dead
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    crawler = await asyncio.wait_for(crawler_pool.get_crawler(config), timeout=1)
    assert crawler is not dead
    assert created == [crawler]
    assert crawler_pool.PERMANENT is crawler

    await crawler_pool.release_crawler(crawler)
    await _drain_close_tasks()


@pytest.mark.asyncio
async def test_failed_recycle_with_a_still_connected_browser_is_replaced(monkeypatch):
    """A restart that fails after its browser connected still bricks the manager.

    ``start()`` assigns ``browser`` before ``_started_browser()`` clears
    ``_closing``; a failure in between leaves a connected browser on a manager
    that refuses every page. Liveness must follow ``_closing``, not the socket.
    """
    created = []
    _configure_pool(monkeypatch, _live_factory(created))

    config = BrowserConfig(headless=True)
    sig = crawler_pool._sig(config)

    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    manager.default_context = _Context()
    manager.browser = _Browser(connected=True)

    async def start_fails_after_connecting():
        manager.browser = _Browser(connected=True)
        manager.default_context = _Context()
        raise RuntimeError("context setup failed")

    async def close(_for_recycle=False):
        manager.browser = None
        manager.default_context = None

    manager.start = start_fails_after_connecting
    manager.close = close

    page, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(page)
    await manager._recycle_task

    assert manager._closing is True
    assert manager.browser.is_connected()
    with pytest.raises(RuntimeError, match="unavailable"):
        await manager.get_page(CrawlerRunConfig())

    dead = _crawler_with_manager(manager)
    crawler_pool.PERMANENT = dead
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    crawler = await asyncio.wait_for(crawler_pool.get_crawler(config), timeout=1)
    assert crawler is not dead
    assert created == [crawler]
    assert crawler_pool.PERMANENT is crawler

    await crawler_pool.release_crawler(crawler)
    await _drain_close_tasks()
    assert dead.closed


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
