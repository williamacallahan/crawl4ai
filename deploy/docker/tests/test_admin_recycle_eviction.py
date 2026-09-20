"""Regression tests for the admin browser restart/cleanup paths evicting a
browser mid-recycle.

``BrowserManager`` recycles an owned Chromium process in place after
``max_pages_before_recycle`` pages, running ``close(_for_recycle=True)`` ->
``start()`` in a shielded, fire-and-forget background task
(``_recycle_browser``). While that runs, the manager lies fallow at
``active_requests == 0`` and ``_recycling is True`` with its ``browser``/
``default_context`` temporarily ``None``.

Commits ``d15fdd6`` and ``881307d`` added an ``_is_recycling`` guard to the
*automatic* eviction paths (``_janitor_pass`` and ``_make_browser_capacity``)
so their bounded ``_close_in_background`` 60s close cap cannot cancel the
shielded recycle and orphan a fresh driver + Chromium on a detached,
``_closing=True`` manager. Two production-reachable *admin* detach paths -
``_init_permanent_locked(force=True)`` (reached by ``POST /actions/restart_browser``
for the permanent browser) and ``retire_pool_crawlers`` (reached by
``POST /actions/restart_browser`` / ``/actions/kill_browser`` /
``/actions/cleanup`` for hot/cold browsers) - were never given that guard, so
an admin action landing on a mid-recycle browser routed it back into the
exact orphan mechanism the commits closed.

This fix mirrors ``d15fdd6``/``881307d`` literally: both admin paths now skip a
mid-recycle entry (``_is_recycling`` is True) instead of detaching it.

These tests drive the real ``BrowserManager.close(_for_recycle=True)`` recycle
machinery with only the Playwright/Chromium transport stubbed (mirroring the
doubles ``tests/regression/test_browser_lifecycle_residuals.py``,
``test_is_live_recycle.py`` and the in-tree ``test_janitor_recycle_eviction.py``
use), then exercise the real ``crawler_pool._init_permanent_locked`` /
``retire_pool_crawlers``. They assert:

* the new ``_is_recycling`` gate skips a mid-recycle permanent browser on a
  force-restart (the fix) and the browser stays registered, healthy and
  closeable after the recycle completes (no orphan),
* the gate skips a mid-recycle hot/cold entry on retire/cleanup (the fix),
* plain (non-recycling) idle browsers are still detached and closed (no
  regression), and
* the orphan mechanism the admin guard now avoids reproduces directly when the
  close budget fires during a straddling ``start()`` (the counterfactual the
  guarded paths already document).

Browser-free: ``-m "not browser"`` picks these up in the CI unit gate.
"""

import asyncio
import os
import sys
import time
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool  # noqa: E402


# --- Playwright-shaped doubles (browser-free) -------------------------------


class _Driver:
    """Stands in for the Playwright driver returned by ``async_playwright().start()``."""

    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


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

    def is_closed(self):
        return self.closed

    async def new_page(self):
        page = _Page(self)
        self.pages.append(page)
        return page

    async def close(self):
        self.closed = True
        for page in self.pages:
            await page.close()


class _Browser:
    def __init__(self, connected=True):
        self._connected = connected
        self.closed = False

    def is_connected(self):
        return self._connected

    async def close(self):
        self._connected = False
        self.closed = True


# --- shared fixtures / helpers ----------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_pages():
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    yield
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None


@pytest.fixture(autouse=True)
def _reset_pool(monkeypatch):
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(10))
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 10)
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "DEFAULT_CONFIG_SIG", None)
    monkeypatch.setattr(crawler_pool, "MEM_LIMIT", 100)
    monkeypatch.setattr(crawler_pool, "BASE_IDLE_TTL", 300)
    monkeypatch.setattr(crawler_pool, "get_container_memory_percent", lambda: 0)
    crawler_pool._CLOSE_TASKS.clear()
    yield
    crawler_pool._CLOSE_TASKS.clear()


async def _drain_close_tasks():
    tasks = list(crawler_pool._CLOSE_TASKS)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _manager(config=None):
    manager = BrowserManager(
        config or BrowserConfig(headless=True), logger=None
    )
    manager.managed_browser = None
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    return manager


class _RealManagerCrawler:
    """A crawler wrapping a real BrowserManager with the pool's required surface.

    ``close()`` is the pool-stub close (just marks closed); used where the test
    only needs the admin path's detach/close bookkeeping, not the real manager
    teardown.
    """

    def __init__(self, manager):
        self.crawler_strategy = SimpleNamespace(browser_manager=manager)
        self.active_requests = 0
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True


class _RealCloseCrawler:
    """A crawler whose ``close()`` delegates to the real ``manager.close()``.

    Used to reproduce the orphan mechanism ``_close_in_background``-style:
    close drives ``manager.close(_for_recycle=False)`` which awaits the
    shielded recycle task and can be cancelled by the close budget.
    """

    def __init__(self, manager):
        self.crawler_strategy = SimpleNamespace(browser_manager=manager)
        self.active_requests = 0
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        await self.crawler_strategy.browser_manager.close()
        self.closed = True


def _crawler_with_manager(manager):
    return _RealManagerCrawler(manager)


def _stale_last_used(seconds_ago):
    """A LAST_USED timestamp that is ``seconds_ago`` in the past."""
    return time.time() - seconds_ago


async def _drive_to_mid_recycle(manager, permit_restart):
    """Drive the real recycle to the mid-recycle window and park there.

    Uses the real ``close(_for_recycle=True)`` (which nulls ``browser``/
    ``default_context`` and stops the driver) and a stub ``start()`` that
    blocks on ``permit_restart`` so the manager stays in the
    healthy-but-null state (``_recycling is True``, ``_closing is False``).
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


async def _drive_to_straddle(manager, permit_start, driver2):
    """Drive the real recycle so ``start()`` is parked with the close phase done.

    Like ``_drive_to_mid_recycle`` but the stubbed ``start()`` installs a fresh
    fake driver (``driver2``) + browser when released, exposing the orphan
    window: while ``start()`` is parked, ``manager.playwright is None`` (the
    real close phase stopped it), and once released ``manager.playwright`` is
    the fresh ``driver2`` nobody else references.
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
        await permit_start.wait()
        manager.playwright = driver2
        manager.default_context = _Context()
        manager.browser = _Browser(connected=True)

    manager.start = start

    page, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(page)
    await closed_event.wait()
    return page


async def _close_with_budget(crawler, budget):
    """Mirror ``_close_in_background._close`` with a configurable close budget.

    The production close cap is 60s; a tiny ``budget`` makes the straddle race
    deterministic and fast for the orphan-mechanism counterfactual.
    """
    try:
        await asyncio.wait_for(crawler.close(), timeout=budget)
    except asyncio.TimeoutError:
        with suppress(Exception):
            await asyncio.wait_for(
                crawler.crawler_strategy.browser_manager.playwright.stop(), timeout=5
            )
    except Exception:
        pass


def _recycle_config():
    return BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )


# --- _init_permanent_locked(force=True): the fix (skip mid-recycle) ----------


@pytest.mark.asyncio
async def test_init_permanent_force_skips_recycling_permanent():
    """A mid-recycle PERMANENT is NOT force-detached (the fix).

    The admin ``/actions/restart_browser`` path calls
    ``_init_permanent_locked(cfg, force=True)``. Mirroring d15fdd6/881307d, a
    mid-recycle permanent must be skipped so the 60s close cap cannot cancel
    the shielded ``start()`` and orphan a fresh driver + Chromium on the
    detached, ``_closing=True`` manager.
    """
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "permdefault"

    try:
        assert manager._recycling is True
        assert crawler_pool._is_recycling(crawler) is True

        close_task = await crawler_pool._init_permanent_locked(
            BrowserConfig(headless=True), force=True
        )

        # The guard skips the detach: no background close, PERMANENT retained.
        assert close_task is None
        assert crawler_pool.PERMANENT is crawler
        assert crawler_pool.DEFAULT_CONFIG_SIG == "permdefault"
        assert not crawler_pool._CLOSE_TASKS
        assert getattr(crawler, "_docker_close_task", None) is None
        # The manager is untouched: not closed, still recycling.
        assert manager._closing is False
        assert manager._recycling is True
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_init_permanent_force_skips_recycling_permanent_loop_returns():
    """``init_permanent(force=True)`` exits its loop without awaiting when the
    permanent browser is mid-recycle, so ``restart_browser`` reports success
    while the in-place recycle continues (mirroring the guarded paths' silent
    skip). No close task is scheduled, no replacement is created concurrently.
    """
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = crawler_pool._sig(BrowserConfig(headless=True))

    try:
        # init_permanent's loop must return promptly (no shield on a close_task).
        await asyncio.wait_for(
            crawler_pool.init_permanent(BrowserConfig(headless=True), force=True),
            timeout=1,
        )

        assert crawler_pool.PERMANENT is crawler
        assert not crawler_pool._CLOSE_TASKS
        assert manager._closing is False
        assert manager._recycling is True
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_init_permanent_force_still_detaches_non_recycling_permanent():
    """No regression: a non-recycling PERMANENT is still force-detached.

    The guard only skips mid-recycle managers; a plain (recycle-complete or
    idle) permanent must still be detached and background-closed so the
    operator's restart intent is honored.
    """
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=True)
    manager.default_context = _Context()
    crawler = _crawler_with_manager(manager)

    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "permdefault"

    try:
        assert crawler_pool._is_recycling(crawler) is False

        close_task = await crawler_pool._init_permanent_locked(
            BrowserConfig(headless=True), force=True
        )

        assert isinstance(close_task, asyncio.Task)
        # Detached: PERMANENT is None, a close task is scheduled.
        assert crawler_pool.PERMANENT is None
        # DEFAULT_CONFIG_SIG deliberately survives (admission coordination).
        assert crawler_pool.DEFAULT_CONFIG_SIG == "permdefault"
        assert getattr(crawler, "_docker_close_task", None) is close_task
    finally:
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_init_permanent_no_force_live_recycling_permanent_returns_none():
    """No regression on the non-force path: ``_is_live`` already returns True
    for a recycling manager, so ``force=False`` returns None without the new
    guard firing. Confirms the new guard is scoped to ``force=True``.
    """
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "permdefault"

    try:
        close_task = await crawler_pool._init_permanent_locked(
            BrowserConfig(headless=True), force=False
        )

        assert close_task is None
        assert crawler_pool.PERMANENT is crawler
        assert not crawler_pool._CLOSE_TASKS
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_init_permanent_force_recycling_permanent_stays_closeable_no_orphan():
    """End-to-end: after the skip + recycle completion the manager is healthy
    and fully closeable through the pool - the admin skip did not strand a
    driver on a detached, ``_closing=True`` manager (no orphan)."""
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "permdefault"
    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    recycle_done = False
    try:
        close_task = await crawler_pool._init_permanent_locked(
            BrowserConfig(headless=True), force=True
        )
        assert close_task is None  # guard skipped the detach

        # The shielded recycle completes normally; the fresh browser is still
        # reachable through PERMANENT and closeable by a normal close().
        permit_restart.set()
        await manager._recycle_task
        recycle_done = True

        assert manager._recycling is False
        assert manager._closing is False
        assert manager.browser is not None
        assert manager.browser.is_connected() is True
        assert crawler_pool.PERMANENT is crawler  # still the pool's reference

        await manager.close()
        assert manager.browser is None
        assert manager._closing is True
    finally:
        if not recycle_done:
            permit_restart.set()
            with suppress(Exception):
                await manager._recycle_task
        with suppress(Exception):
            await manager.close()


# --- retire_pool_crawlers: the fix (skip mid-recycle) -----------------------


@pytest.mark.asyncio
async def test_retire_pool_crawlers_skips_recycling_cold_entry():
    """A mid-recycle cold entry is NOT retired by an admin action (the fix).

    ``POST /actions/kill_browser`` / ``/actions/restart_browser`` / ``/actions/cleanup``
    reach ``retire_pool_crawlers``; mirroring d15fdd6/881307d, a mid-recycle
    entry must be skipped so the 60s close cap cannot cancel the shielded
    ``start()`` and orphan the fresh driver + Chromium.
    """
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "deadbeef"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    try:
        assert crawler_pool._is_recycling(crawler) is True

        retired = await crawler_pool.retire_pool_crawlers(sig)

        assert retired == []
        assert sig in crawler_pool.COLD_POOL
        assert crawler_pool.COLD_POOL[sig] is crawler
        assert not crawler_pool._CLOSE_TASKS
        assert getattr(crawler, "_docker_close_task", None) is None
        assert manager._closing is False
        assert manager._recycling is True
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_retire_pool_crawlers_skips_recycling_hot_entry():
    """A mid-recycle hot entry is NOT retired either (the fix, hot branch)."""
    manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "feedface"
    crawler_pool.HOT_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(1000)

    try:
        retired = await crawler_pool.retire_pool_crawlers(sig)

        assert retired == []
        assert sig in crawler_pool.HOT_POOL
        assert crawler_pool.HOT_POOL[sig] is crawler
        assert not crawler_pool._CLOSE_TASKS
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_retire_pool_crawlers_retires_non_recycling_entry():
    """No regression: a plain (non-recycling) idle entry is still retired and
    closed by an admin action."""
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=True)
    manager.default_context = _Context()
    crawler = _crawler_with_manager(manager)

    sig = "c0ffee"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    try:
        assert crawler_pool._is_recycling(crawler) is False

        retired = await crawler_pool.retire_pool_crawlers(sig)

        assert retired == [(sig, "cold")]
        assert sig not in crawler_pool.COLD_POOL
        await _drain_close_tasks()
        assert crawler.closed is True
    finally:
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_retire_pool_crawlers_cold_only_skips_recycling_retires_rest():
    """``cold_only=True`` (``/actions/cleanup``) skips the mid-recycle cold entry
    and still retires the non-recycling idle ones (the fix + no regression)."""
    recycling_manager = _manager(_recycle_config())
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(recycling_manager, permit_restart)
    recycling_crawler = _crawler_with_manager(recycling_manager)

    idle_manager = _manager(BrowserConfig(headless=True))
    idle_manager.browser = _Browser(connected=True)
    idle_manager.default_context = _Context()
    idle_crawler = _crawler_with_manager(idle_manager)

    recycle_sig = "recycle01"
    idle_sig = "idle00002"
    crawler_pool.COLD_POOL[recycle_sig] = recycling_crawler
    crawler_pool.COLD_POOL[idle_sig] = idle_crawler
    crawler_pool.LAST_USED[recycle_sig] = _stale_last_used(100)
    crawler_pool.LAST_USED[idle_sig] = _stale_last_used(100)

    try:
        assert crawler_pool._is_recycling(recycling_crawler) is True
        assert crawler_pool._is_recycling(idle_crawler) is False

        retired = await crawler_pool.retire_pool_crawlers(cold_only=True)

        # Only the idle, non-recycling entry is retired.
        assert retired == [(idle_sig, "cold")]
        assert recycle_sig in crawler_pool.COLD_POOL
        assert crawler_pool.COLD_POOL[recycle_sig] is recycling_crawler
        assert idle_sig not in crawler_pool.COLD_POOL
        await _drain_close_tasks()
        assert idle_crawler.closed is True
        assert recycling_crawler.closed is False
        assert recycling_manager._closing is False
    finally:
        permit_restart.set()
        await recycling_manager._recycle_task
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_retire_pool_crawlers_mixed_hot_cold_only_recycles_skipped():
    """Without ``cold_only``: hot + cold are scanned; recycling entries in both
    tiers are skipped, non-recycling entries are retired (fix + no regression)."""
    hot_recycling = _manager(_recycle_config())
    hot_permit = asyncio.Event()
    await _drive_to_mid_recycle(hot_recycling, hot_permit)
    hot_recycling_crawler = _crawler_with_manager(hot_recycling)

    cold_idle_manager = _manager(BrowserConfig(headless=True))
    cold_idle_manager.browser = _Browser(connected=True)
    cold_idle_manager.default_context = _Context()
    cold_idle_crawler = _crawler_with_manager(cold_idle_manager)

    hot_sig = "hotrecyc"
    cold_sig = "coldidle"
    crawler_pool.HOT_POOL[hot_sig] = hot_recycling_crawler
    crawler_pool.COLD_POOL[cold_sig] = cold_idle_crawler
    crawler_pool.LAST_USED[hot_sig] = _stale_last_used(1000)
    crawler_pool.LAST_USED[cold_sig] = _stale_last_used(100)

    try:
        retired = await crawler_pool.retire_pool_crawlers()  # all pools, no prefix

        # The recycling hot entry is skipped; the idle cold entry is retired.
        assert retired == [(cold_sig, "cold")]
        assert hot_sig in crawler_pool.HOT_POOL
        assert cold_sig not in crawler_pool.COLD_POOL
        await _drain_close_tasks()
        assert cold_idle_crawler.closed is True
        assert hot_recycling_crawler.closed is False
        assert hot_recycling._closing is False
    finally:
        hot_permit.set()
        await hot_recycling._recycle_task
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_retire_pool_crawlers_no_prefix_skips_all_recycling():
    """With every idle entry mid-recycle (no prefix), nothing is retired - the
    admin action is a no-op rather than orphaning. Mirrors the capacity path's
    fail-closed behavior."""
    cold_recycling = _manager(_recycle_config())
    cold_permit = asyncio.Event()
    await _drive_to_mid_recycle(cold_recycling, cold_permit)
    cold_crawler = _crawler_with_manager(cold_recycling)

    hot_recycling = _manager(_recycle_config())
    hot_permit = asyncio.Event()
    await _drive_to_mid_recycle(hot_recycling, hot_permit)
    hot_crawler = _crawler_with_manager(hot_recycling)

    crawler_pool.COLD_POOL["coldsig"] = cold_crawler
    crawler_pool.HOT_POOL["hotsig"] = hot_crawler
    crawler_pool.LAST_USED["coldsig"] = _stale_last_used(100)
    crawler_pool.LAST_USED["hotsig"] = _stale_last_used(1000)

    try:
        retired = await crawler_pool.retire_pool_crawlers()

        assert retired == []
        assert "coldsig" in crawler_pool.COLD_POOL
        assert "hotsig" in crawler_pool.HOT_POOL
        assert not crawler_pool._CLOSE_TASKS
    finally:
        cold_permit.set()
        hot_permit.set()
        await cold_recycling._recycle_task
        await hot_recycling._recycle_task


# --- orphan mechanism counterfactual (admin path) ---------------------------


@pytest.mark.asyncio
async def test_admin_force_replace_close_budget_orphans_driver():
    """The orphan mechanism the admin guard now avoids: closing a recycling
    browser whose ``start()`` outlasts the close budget leaves a fresh driver
    running on a ``_closing=True`` manager that nothing will ever close again.

    Mirrors ``test_close_during_straddling_start_orphans_driver`` from the
    janitor suite but via the admin path's shared ``_close_in_background._close``
    close-budget sequence, proving the shared close the admin detach routes
    into is the same orphan class. ``_init_permanent_locked(force=True)`` /
    ``retire_pool_crawlers`` now skip a mid-recycle entry so this path never
    fires on one.
    """
    manager = _manager(_recycle_config())
    permit_start = asyncio.Event()
    driver2 = _Driver()
    await _drive_to_straddle(manager, permit_start, driver2)

    crawler = _RealCloseCrawler(manager)

    try:
        # Mid-recycle, start() parked: playwright is None (close phase stopped
        # the old driver) and the fresh driver is not yet installed.
        assert manager._recycling is True
        assert manager._closing is False
        assert manager.playwright is None

        # The admin path used to detach-and-close the mid-recycle browser via
        # this exact budget sequence; the 60s cap (here 0.2s) fires while
        # start() is still parked.
        await _close_with_budget(crawler, budget=0.2)

        # Fallback tried None.stop() -> AttributeError swallowed; the cancelled
        # close() already set _closing=True before the shielded recycle await.
        assert manager._closing is True
        # The shielded recycle keeps running; release start() now.
        permit_start.set()
        await manager._recycle_task

        # _restart_browser had already passed the _closing TOCTOU before the
        # close set _closing=True, so start() ran and installed a fresh driver
        # on the now-_closing manager. Nothing references it to close it again.
        assert manager._recycling is False
        assert manager.playwright is driver2
        assert driver2.stopped is False            # <-- orphaned driver
        assert manager.browser is not None
        assert manager.browser.closed is False       # <-- orphaned browser

        # The leak is that NOBODY calls close() on this detached manager; only
        # the test does, for hygiene, and it cleans both up - proving the only
        # reason they survive is the missing cleanup path, not a real pin.
        await manager.close()
        assert driver2.stopped is True
        assert manager.browser is None
    finally:
        permit_start.set()
        with suppress(Exception):
            await manager._recycle_task
        with suppress(Exception):
            await manager.close()


@pytest.mark.asyncio
async def test_admin_force_replace_during_close_sub_phase_does_not_orphan():
    """Sub-phase scoping: an admin close that lands during the recycle's *close*
    sub-phase is orphan-safe even without the guard, because ``_restart_browser``
    rechecks ``_closing`` (set by the admin close before the shielded recycle
    await) and skips ``start()``. With the guard, the entry is skipped outright
    (consistent), and the recycle completes normally - no orphan either way.
    This is the negative control isolating the start sub-phase as the sole
    orphaning straddle."""
    manager = _manager(_recycle_config())
    manager.default_context = _Context()
    manager.browser = _Browser(connected=True)
    real_close = manager.close
    close_in_progress = asyncio.Event()
    permit_close_release = asyncio.Event()
    started = asyncio.Event()

    async def tracking_close(_for_recycle=False):
        await real_close(_for_recycle=_for_recycle)
        if _for_recycle:
            # Park the close sub-phase: _restart_browser is still awaiting this
            # close() to return, so its _closing recheck has not happened yet.
            close_in_progress.set()
            await permit_close_release.wait()

    manager.close = tracking_close

    async def start():
        started.set()
        raise AssertionError("start() must not run when close sub-phase set _closing")

    manager.start = start

    page, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(page)
    await close_in_progress.wait()

    crawler = _RealCloseCrawler(manager)

    try:
        assert manager._recycling is True
        assert manager._closing is False
        # The close sub-phase already ran its real teardown (browser/playwright
        # nulled) but _restart_browser has not returned to recheck _closing.
        assert manager.browser is None

        # Admin close-budget sequence: sets _closing=True (budget cancels the
        # shielded recycle await), while the close sub-phase is still parked.
        await _close_with_budget(crawler, budget=0.2)

        # _closing is now True (admin close set it before the shield await).
        assert manager._closing is True

        # Release the close sub-phase: _restart_browser rechecks _closing and
        # skips start() - no fresh driver/Chromium installed (no orphan).
        permit_close_release.set()
        await manager._recycle_task

        assert started.is_set() is False
        assert manager._recycling is False
        assert manager.browser is None
    finally:
        permit_close_release.set()
        with suppress(Exception):
            await manager._recycle_task
        with suppress(Exception):
            await manager.close()
