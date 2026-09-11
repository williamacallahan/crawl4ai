"""Regression tests for the crawler pool janitor evicting a browser mid-recycle.

``BrowserManager`` recycles an owned Chromium process in place after
``max_pages_before_recycle`` pages, running ``close(_for_recycle=True)`` ->
``start()`` in a shielded, fire-and-forget background task
(``_recycle_browser``). While that runs, the manager lies fallow at
``active_requests == 0`` and ``_recycling is True`` with its ``browser``/
``default_context`` temporarily ``None``.

Commit ``93b3d77`` taught the *admission* path (``_is_live`` /
``_discard_if_unavailable``) to treat a recycling manager as live so the pool
blocks on the library's own restart instead of force-replacing a healthy
browser. The *janitor's* eviction loop (``_janitor_pass``) was never updated:
a recycling browser at ``active_requests == 0`` past its TTL is
indistinguishable from a long-dead idle browser, so the janitor detached and
``_close_in_background``-ed it. The janitor's 60s close cap then cancelled
``close()`` at the shielded recycle-task await, the keeps-running shielded
``start()`` finished a fresh driver + Chromium on the now-detached manager
(left ``_closing=True``), and nothing ever closed it again - a monotonic
Chromium process leak (the exact class ``bdccf62`` / ``d0f57c0`` closed).

These tests drive the real ``BrowserManager.close(_for_recycle=True)`` recycle
machinery with only the Playwright/Chromium transport stubbed (mirroring the
doubles ``tests/regression/test_browser_lifecycle_residuals.py`` and
``test_is_live_recycle.py`` use), then exercise the real
``crawler_pool._janitor_pass``. They assert:

* the new ``_is_recycling`` gate skips a mid-recycle browser past TTL (the fix),
* the browser stays registered, healthy and closeable after the pass (no
  orphan), and is evicted normally once the recycle clears (no regression),
* plain idle browsers past TTL are still evicted (no regression), and
* the orphan mechanism the janitor now avoids reproduces directly when the
  close budget fires during a straddling ``start()``.

Browser-free: ``-m "not browser"`` picks these up in the CI unit gate.
"""

import asyncio
import os
import sys
import time
from contextlib import contextmanager, suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool  # noqa: E402


# ─── Playwright-shaped doubles (browser-free) ──────────────────────────────


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


# ─── shared fixtures / helpers ─────────────────────────────────────────────


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
    only needs the janitor's detach/close bookkeeping, not the real manager
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


@contextmanager
def _janitor_clock(monkeypatch, mem_pct):
    """Run ``_janitor_pass`` with a deterministic clock and no leading sleep.

    ``_janitor_pass`` does ``await asyncio.sleep(interval)`` first; under
    ``mem_pct > 80`` that is 10s. Patch ``asyncio.sleep`` to a no-op only for
    the duration of the pass (the recycle is parked on an ``asyncio.Event``,
    not ``asyncio.sleep``, so the patch does not disturb it).
    """
    monkeypatch.setattr(crawler_pool, "get_container_memory_percent", lambda: mem_pct)
    real_sleep = asyncio.sleep

    async def _noop_sleep(_delay=0):
        return None

    asyncio.sleep = _noop_sleep
    try:
        yield
    finally:
        asyncio.sleep = real_sleep


async def _run_janitor(monkeypatch, mem_pct=85):
    with _janitor_clock(monkeypatch, mem_pct):
        await crawler_pool._janitor_pass()


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


# ─── _is_recycling helper ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_recycling_true_for_mid_recycle_manager():
    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)

    try:
        assert manager._recycling is True
        assert manager._closing is False
        assert crawler_pool._is_recycling(_crawler_with_manager(manager)) is True
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_is_recycling_false_for_idle_healthy_manager():
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=True)
    manager.default_context = _Context()

    assert manager._recycling is False
    assert crawler_pool._is_recycling(_crawler_with_manager(manager)) is False


def test_is_recycling_false_when_manager_lacks_recycling_attr():
    """Older pool mocks have no ``_recycling`` attribute; must fall through.

    Mirrors ``test_is_live_uses_getattr_default_for_managers_without_recycling_attr``
    so the backward-compat fallback (``getattr(manager, "_recycling", False)``)
    is preserved.
    """
    browser = MagicMock()
    browser.is_connected.return_value = True
    manager = SimpleNamespace(browser=browser, default_context=None)
    crawler = SimpleNamespace(crawler_strategy=SimpleNamespace(browser_manager=manager))

    assert crawler_pool._is_recycling(crawler) is False


def test_is_recycling_false_when_manager_access_raises():
    """A crawler whose strategy/manager access blows up must not be treated as
    recycling (which would pin it in the pool forever); defaults to evictable."""
    class _Boom:
        @property
        def browser_manager(self):
            raise RuntimeError("broken")

    crawler = SimpleNamespace(crawler_strategy=_Boom())
    assert crawler_pool._is_recycling(crawler) is False


# ─── janitor: the fix (skip recycling browsers past TTL) ────────────────────


@pytest.mark.asyncio
async def test_janitor_skips_recycling_cold_browser_past_ttl(monkeypatch):
    """A mid-recycle cold browser idle past its TTL is NOT evicted (the fix)."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "deadbeef"
    crawler_pool.COLD_POOL[sig] = crawler
    # idle >> 30s (cold_ttl at mem_pct>80); active==0 just like a recycling browser
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    try:
        assert manager._recycling is True
        assert crawler_pool._is_recycling(crawler) is True

        await _run_janitor(monkeypatch, mem_pct=85)

        # Retained in the pool, no close task created, manager untouched.
        assert sig in crawler_pool.COLD_POOL
        assert crawler_pool.COLD_POOL[sig] is crawler
        assert not crawler_pool._CLOSE_TASKS
        assert getattr(crawler, "_docker_close_task", None) is None
        assert manager._closing is False
        assert manager._recycling is True
        # The skip refreshes LAST_USED so the fresh post-recycle browser gets a
        # full idle window before any future eviction.
        assert crawler_pool.LAST_USED[sig] > _stale_last_used(1)
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_janitor_skips_recycling_hot_browser_past_ttl(monkeypatch):
    """A mid-recycle hot browser idle past its TTL is NOT evicted either."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "feedface"
    crawler_pool.HOT_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(1000)  # idle >> 120s (hot_ttl at >80)

    try:
        await _run_janitor(monkeypatch, mem_pct=85)

        assert sig in crawler_pool.HOT_POOL
        assert crawler_pool.HOT_POOL[sig] is crawler
        assert not crawler_pool._CLOSE_TASKS
        assert manager._closing is False
    finally:
        permit_restart.set()
        await manager._recycle_task


@pytest.mark.asyncio
async def test_janitor_leaves_recycling_browser_closeable_no_orphan(monkeypatch):
    """End-to-end: after the skip + recycle completion the manager is healthy
    and fully closeable - the janitor did not strand a driver on a detached,
    ``_closing=True`` manager (no orphan)."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "cafebabe"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    recycle_done = False
    try:
        await _run_janitor(monkeypatch, mem_pct=85)

        # The janitor left it alone; the shielded recycle completes normally.
        permit_restart.set()
        await manager._recycle_task
        recycle_done = True

        assert manager._recycling is False
        # restarted=True in _recycle_browser's finally does NOT set _closing.
        assert manager._closing is False
        assert manager.browser is not None
        assert manager.browser.is_connected() is True
        # The freshly-recycled browser is still reachable through the pool, so
        # it can - and is - closed cleanly by a normal close(). No orphan.
        assert sig in crawler_pool.COLD_POOL
        await manager.close()
        assert manager.browser is None
        assert manager._closing is True
    finally:
        if not recycle_done:
            permit_restart.set()
            with suppress(Exception):
                await manager._recycle_task


# ─── janitor: no regression (still evicts once not recycling) ───────────────


@pytest.mark.asyncio
async def test_janitor_evicts_idle_browser_after_recycle_completes(monkeypatch):
    """Once the recycle clears, the same browser past TTL is evicted normally."""
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)
    crawler = _crawler_with_manager(manager)

    sig = "baadf00d"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    try:
        # Pass 1: recycling -> skipped (refreshed LAST_USED, retained).
        await _run_janitor(monkeypatch, mem_pct=85)
        assert sig in crawler_pool.COLD_POOL

        # Recycle completes; the browser is now healthy but no longer recycling.
        permit_restart.set()
        await manager._recycle_task
        assert manager._recycling is False

        # Re-stale LAST_USED so the now-healthy idle browser is past TTL again.
        crawler_pool.LAST_USED[sig] = _stale_last_used(100)

        # Pass 2: not recycling -> evicted normally.
        await _run_janitor(monkeypatch, mem_pct=85)
        assert sig not in crawler_pool.COLD_POOL
        assert crawler_pool._CLOSE_TASKS or getattr(crawler, "_docker_close_task", None) is not None

        await _drain_close_tasks()
        assert crawler.closed is True
    finally:
        permit_restart.set()
        with suppress(Exception):
            await manager._recycle_task
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_janitor_evicts_normal_idle_browser_past_ttl(monkeypatch):
    """A plain (non-recycling) idle browser past TTL is still evicted."""
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=True)
    manager.default_context = _Context()
    crawler = _crawler_with_manager(manager)

    sig = "c0ffee"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    try:
        assert crawler_pool._is_recycling(crawler) is False

        await _run_janitor(monkeypatch, mem_pct=85)

        assert sig not in crawler_pool.COLD_POOL
        await _drain_close_tasks()
        assert crawler.closed is True
    finally:
        await _drain_close_tasks()


@pytest.mark.asyncio
async def test_janitor_skip_is_gated_on_is_recycling(monkeypatch):
    """The janitor's skip decision is exactly ``_is_recycling`` (wiring guard).

    Patching ``_is_recycling`` to True keeps a normal idle browser in the pool;
    patching it to False evicts a recycling browser. Proves the gate is wired
    through ``_is_recycling`` and not e.g. ``_is_live`` (which would also pin
    healthy idle browsers forever).
    """
    manager = _manager(BrowserConfig(headless=True))
    manager.browser = _Browser(connected=True)
    manager.default_context = _Context()
    crawler = _crawler_with_manager(manager)

    sig = "decade"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    # Force the gate True -> retained even though not actually recycling.
    monkeypatch.setattr(crawler_pool, "_is_recycling", lambda c: True)
    await _run_janitor(monkeypatch, mem_pct=85)
    assert sig in crawler_pool.COLD_POOL
    assert not crawler_pool._CLOSE_TASKS

    # The True-gate pass refreshed LAST_USED; re-stale so pass 2 actually
    # reaches the eviction decision (otherwise the TTL check skips it).
    crawler_pool.LAST_USED[sig] = _stale_last_used(100)

    # Force the gate False -> evicted (the stub close just marks closed; no
    # real manager teardown, so no orphan here - this proves the decision only).
    monkeypatch.setattr(crawler_pool, "_is_recycling", lambda c: False)
    await _run_janitor(monkeypatch, mem_pct=85)
    assert sig not in crawler_pool.COLD_POOL
    await _drain_close_tasks()
    assert crawler.closed is True


# ─── orphan mechanism counterfactual ───────────────────────────────────────


@pytest.mark.asyncio
async def test_close_during_straddling_start_orphans_driver():
    """The orphan mechanism the janitor now avoids: closing a recycling browser
    whose ``start()`` outlasts the close budget leaves a fresh driver running on
    a ``_closing=True`` manager that nothing will ever close again.

    Mirrors ``_close_in_background._close`` (a ``wait_for(crawler.close(),
    budget)`` then a ``suppress(Exception): playwright.stop()`` fallback) with
    a tiny budget so the straddle is fast and deterministic. The janitor's
    ``_is_recycling`` skip is precisely what prevents this path from ever
    firing on a recycling pool entry.
    """
    config = BrowserConfig(
        use_managed_browser=True, headless=True, max_pages_before_recycle=1
    )
    manager = _manager(config)
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

        # The janitor used to call this on the detached recycling browser; the
        # 60s budget (here 0.2s) fires while start() is still parked.
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


# ─── capacity eviction (_make_browser_capacity) ──────────────────────────────


@pytest.mark.asyncio
async def test_make_browser_capacity_skips_mid_recycle_victim(monkeypatch):
    """Capacity eviction must not pick a mid-recycle entry as its victim.

    A mid-recycle browser is idle (active_requests == 0) and - because its
    recycle fires at the end of an aged crawl - typically the oldest
    LAST_USED candidate, so pre-fix it is exactly the entry min() picks.
    Detaching it cancels its shielded close() at the 60s cap and orphans
    the fresh driver + Chromium the recycle installs (the janitor case's
    mechanism, reached from the automatic admission path). The eviction
    must fall through to the younger non-recycling idle entry instead.
    """
    old_manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(old_manager, permit_restart)

    try:
        monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 2)
        old_sig, fresh_sig = "sig:old", "sig:fresh"
        crawler_pool.COLD_POOL[old_sig] = _crawler_with_manager(old_manager)
        crawler_pool.LAST_USED[old_sig] = _stale_last_used(600)
        crawler_pool.COLD_POOL[fresh_sig] = _crawler_with_manager(_manager())
        crawler_pool.LAST_USED[fresh_sig] = _stale_last_used(10)

        close_task = crawler_pool._make_browser_capacity()

        assert close_task is not None
        assert old_sig in crawler_pool.COLD_POOL  # recycling victim skipped
        assert fresh_sig not in crawler_pool.COLD_POOL  # next-oldest evicted
        await _drain_close_tasks()
    finally:
        permit_restart.set()
        await old_manager._recycle_task


@pytest.mark.asyncio
async def test_make_browser_capacity_fails_closed_when_all_idle_recycling(monkeypatch):
    """Only-idle-recycling capacity: reject the request, don't orphan.

    With every idle candidate mid-recycle, eviction must fail closed with
    the existing capacity RuntimeError rather than detach a recycling
    browser. _recycle_browser's finally clears _recycling within <= 180s
    and the janitor then evicts it, so the rejection window is bounded.
    """
    manager = _manager(
        BrowserConfig(
            use_managed_browser=True, headless=True, max_pages_before_recycle=1
        )
    )
    permit_restart = asyncio.Event()
    await _drive_to_mid_recycle(manager, permit_restart)

    try:
        monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 1)
        sig = "sig:old"
        crawler_pool.COLD_POOL[sig] = _crawler_with_manager(manager)
        crawler_pool.LAST_USED[sig] = _stale_last_used(600)

        with pytest.raises(RuntimeError, match="at capacity"):
            crawler_pool._make_browser_capacity()

        assert sig in crawler_pool.COLD_POOL  # untouched, not orphaned
    finally:
        permit_restart.set()
        await manager._recycle_task
