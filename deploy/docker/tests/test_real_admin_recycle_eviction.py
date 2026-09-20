"""E2E regression test for the admin recycle-eviction guard against a real
headless Chromium.

The control-flow suite (``test_admin_recycle_eviction.py``) drives the real
``BrowserManager.close(_for_recycle=True)`` path with a stubbed ``start()`` so
it can park the manager deterministically in the mid-recycle window. This file
complements it by exercising the *whole* in-place recycle - a real Playwright
launch, the real ``close`` teardown, and the real ``start()`` relaunch - and
calling the real ``crawler_pool._init_permanent_locked(force=True)`` /
``retire_pool_crawlers`` while the manager is genuinely mid-recycle on a live
Chromium.

Preconditions:
  * A real headless Chromium installed for Playwright (``playwright install
    chromium``). The environment used by CI has ``chromium-1179``/``chromium-1194``
    under ``~/.cache/ms-playwright/``.
  * The Chromium binary must be launchable as the test user. When running as
    root, ``extra_args=["--no-sandbox"]`` is supplied.
"""

import asyncio
import os
import subprocess
import sys
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool  # noqa: E402

pytestmark = pytest.mark.browser


def _crawler(manager):
    return SimpleNamespace(
        crawler_strategy=SimpleNamespace(browser_manager=manager),
        active_requests=0,
        closed=False,
    )


def _count_chromium():
    try:
        r = subprocess.run(
            ["pgrep", "-f", "chromium"], capture_output=True, text=True, timeout=5
        )
        return len([l for l in r.stdout.strip().splitlines() if l]) if r.stdout.strip() else 0
    except Exception:
        return -1


async def _await_mid_recycle(manager, *, max_iters=2000, step=0.002):
    """Poll until the manager is in the real mid-recycle window
    (_recycling True, _closing False, browser None) or the recycle completes."""
    for _ in range(max_iters):
        if manager._recycling and not manager._closing and manager.browser is None:
            return True
        if manager._recycle_task is not None and manager._recycle_task.done() and not manager._recycling:
            return False
        await asyncio.sleep(step)
    return False


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


def _install_delayed_start(manager, permit_start):
    """Wrap ``manager.start`` so the *recycle's* ``start()`` waits on the event.

    The initial browser must already be launched (via the original ``start``)
    before this is installed, so the first launch is not blocked. The recycle
    re-enters ``start()`` and parks until ``permit_start`` is set.
    """
    original_start = manager.start

    async def delayed_start():
        await permit_start.wait()
        return await original_start()

    manager.start = delayed_start


@pytest.mark.asyncio
async def test_real_recycle_admin_force_restart_skips_mid_recycle_no_orphan():
    """A real in-place recycle on a real Chromium: ``_init_permanent_locked(
    force=True)`` issued while ``_recycling`` is True must skip the detach
    (the fix) so the 60s close cap cannot cancel the shielded ``start()``.

    The real ``start()`` takes ~0.4s on a healthy manager, so we extend the
    recycle window deterministically by wrapping ``start()`` to wait on an
    ``asyncio.Event`` we release after asserting the guard. The close phase
    and the manager state are entirely real; only the start phase is paused
    so the mid-recycle window is reliably observable. After release, the real
    ``start()`` runs and installs a fresh real Chromium driver.
    """
    config = BrowserConfig(
        headless=True,
        max_pages_before_recycle=1,
        extra_args=["--no-sandbox"],
    )
    manager = BrowserManager(config, logger=None)
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    permit_start = asyncio.Event()

    crawler = _crawler(manager)
    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "permdefault"

    chromium_before = _count_chromium()
    recycle_observed = False
    try:
        # Initial real launch with the original start (before the wrapper).
        await manager.start()
        assert manager.browser is not None
        assert manager.browser.is_connected() is True

        # Now install the delayed start so the recycle's start() parks.
        _install_delayed_start(manager, permit_start)

        page, _ = await manager.get_page(CrawlerRunConfig())
        await manager.release_page_with_context(page)

        recycle_observed = await _await_mid_recycle(manager)
        assert recycle_observed, "never observed the real mid-recycle window"

        # The guard under test: admin force-restart of a mid-recycle PERMANENT
        # must skip the detach without scheduling a close.
        assert crawler_pool._is_recycling(crawler) is True
        close_task, _done = await crawler_pool._init_permanent_locked(
            BrowserConfig(headless=True, extra_args=["--no-sandbox"]),
            force=True, target=crawler,
        )
        assert close_task is None, "guard did not skip mid-recycle PERMANENT"
        assert crawler_pool.PERMANENT is crawler, "PERMANENT was detached despite the guard"
        assert manager._closing is False, "manager was closed despite the guard"
        assert not crawler_pool._CLOSE_TASKS, "a close was scheduled despite the guard"

        # Release the real start(); the recycle completes and installs a fresh
        # real Chromium driver on the still-retained PERMANENT.
        permit_start.set()
        if manager._recycle_task is not None:
            await asyncio.wait_for(manager._recycle_task, timeout=30)

        assert manager._recycling is False
        assert manager._closing is False
        assert manager.browser is not None
        assert manager.browser.is_connected() is True

        # The recycled browser is still reachable through PERMANENT and is
        # fully closeable by a normal close() - no driver stranded.
        second_page, _ = await asyncio.wait_for(
            manager.get_page(CrawlerRunConfig()), timeout=10
        )
        await manager.release_page_with_context(second_page)

        await manager.close()
        assert manager.browser is None

        # Allow the driver exit handler to reap the Chromium process group.
        await asyncio.sleep(1.5)
        chromium_after = _count_chromium()
        assert chromium_after <= chromium_before, (
            f"Chromium orphan: {chromium_before} -> {chromium_after}"
        )
    finally:
        permit_start.set()
        if manager._recycle_task is not None and not manager._recycle_task.done():
            with suppress(Exception):
                await manager._recycle_task
        with suppress(Exception):
            await manager.close()
        await asyncio.sleep(1.0)


@pytest.mark.asyncio
async def test_real_recycle_retire_pool_crawlers_skips_mid_recycle_no_orphan():
    """A real in-place recycle on a real Chromium in the cold pool:
    ``retire_pool_crawlers`` issued while ``_recycling`` is True must skip the
    entry (the fix) so the 60s close cap cannot cancel the shielded start().
    Mirrors the permanent-restart E2E on the hot/cold retire path.
    """
    config = BrowserConfig(
        headless=True,
        max_pages_before_recycle=1,
        extra_args=["--no-sandbox"],
    )
    manager = BrowserManager(config, logger=None)
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    permit_start = asyncio.Event()

    crawler = _crawler(manager)
    sig = "realcold1"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = time.time()

    chromium_before = _count_chromium()
    recycle_observed = False
    try:
        await manager.start()
        assert manager.browser is not None

        _install_delayed_start(manager, permit_start)

        page, _ = await manager.get_page(CrawlerRunConfig())
        await manager.release_page_with_context(page)

        recycle_observed = await _await_mid_recycle(manager)
        assert recycle_observed, "never observed the real mid-recycle window"

        assert crawler_pool._is_recycling(crawler) is True
        retired = await crawler_pool.retire_pool_crawlers(sig)
        assert retired == [], "guard did not skip mid-recycle cold entry"
        assert sig in crawler_pool.COLD_POOL
        assert crawler_pool.COLD_POOL[sig] is crawler
        assert manager._closing is False
        assert not crawler_pool._CLOSE_TASKS

        permit_start.set()
        if manager._recycle_task is not None:
            await asyncio.wait_for(manager._recycle_task, timeout=30)

        assert manager._recycling is False
        assert manager._closing is False
        assert manager.browser is not None
        assert manager.browser.is_connected() is True

        await manager.close()
        assert manager.browser is None

        await asyncio.sleep(1.5)
        chromium_after = _count_chromium()
        assert chromium_after <= chromium_before, (
            f"Chromium orphan: {chromium_before} -> {chromium_after}"
        )
    finally:
        permit_start.set()
        if manager._recycle_task is not None and not manager._recycle_task.done():
            with suppress(Exception):
                await manager._recycle_task
        with suppress(Exception):
            await manager.close()
        await asyncio.sleep(1.0)


@pytest.mark.asyncio
async def test_real_admin_force_restart_non_recycling_detaches_and_replaces():
    """No-regression E2E: a force-restart of a healthy (non-recycling) real
    PERMANENT detaches it and a fresh real Chromium is created on the next
    acquire. Confirms the guard does not over-broadly skip healthy browsers."""
    from crawl4ai import AsyncWebCrawler

    config = BrowserConfig(
        headless=True,
        max_pages_before_recycle=500,
        extra_args=["--no-sandbox"],
    )
    crawler = AsyncWebCrawler(config=config, thread_safe=False)
    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "permdefault2"

    chromium_before = _count_chromium()
    try:
        await crawler_pool.init_permanent(config, force=True)
        # Force-restart of a non-recycling PERMANENT detaches it; the wrapper
        # creates a fresh PERMANENT.
        await asyncio.sleep(1.0)
        assert crawler_pool.PERMANENT is not None
        assert crawler_pool.PERMANENT is not crawler
        new_manager = crawler_pool.PERMANENT.crawler_strategy.browser_manager
        assert new_manager.browser is not None
        assert new_manager.browser.is_connected() is True

        # The detached old crawler was background-closed via close_all() below;
        # close_all drains the bounded closes, so after it the old Chromium is
        # reaped and only the new PERMANENT's Chromium remains (baseline).
    finally:
        await crawler_pool.close_all()
        # Allow the driver exit handlers to reap both Chromium process groups.
        await asyncio.sleep(2.5)
        chromium_after = _count_chromium()
        assert chromium_after <= chromium_before, (
            f"Chromium orphan after close_all: {chromium_before} -> {chromium_after}"
        )


@pytest.mark.asyncio
async def test_real_managed_browser_recycle_admin_force_restart_no_orphan():
    """E2E on the opt-in managed-browser path (browser_mode='builtin',
    use_managed_browser=True), where Chromium runs in its own process group
    (``os.setpgrp``) and only ``managed_browser.cleanup()`` (``os.killpg``)
    can reclaim it.

    This is the severe-consequence cell of the bug's 2x2 matrix: without the
    guard, an admin force-restart during the recycle's start sub-phase would
    detach the manager, the 60s close cap cancels the shielded await, and the
    keeps-running ``start()`` spawns a fresh managed Chromium in its own
    process group on the detached, ``_closing=True`` manager with no cleanup
    path - a structurally-unreclaimable orphan. The fix skips the detach so
    ``cleanup()`` remains reachable through the retained PERMANENT.

    Uses the real ``ManagedBrowser`` subprocess launch against the bundled
    Playwright Chromium; only the recycle's ``start()`` is paused on an
    ``asyncio.Event`` so the mid-recycle window is reliably observable before
    the fresh managed Chromium is spawned.
    """
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_logger import AsyncLogger

    config = BrowserConfig(
        headless=True,
        use_managed_browser=True,
        browser_mode="builtin",
        max_pages_before_recycle=1,
        extra_args=["--no-sandbox"],
    )
    crawler = AsyncWebCrawler(
        config=config, logger=AsyncLogger(verbose=False), thread_safe=False
    )
    await crawler.start()
    manager = crawler.crawler_strategy.browser_manager
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    permit_start = asyncio.Event()
    _install_delayed_start(manager, permit_start)

    crawler_pool.PERMANENT = crawler
    crawler_pool.DEFAULT_CONFIG_SIG = "managedperm"

    chromium_before = _count_chromium()
    recycle_observed = False
    try:
        assert manager.managed_browser is not None, "managed browser did not start"
        first_managed = manager.managed_browser
        assert manager.browser is not None
        assert manager.browser.is_connected() is True

        page, _ = await manager.get_page(CrawlerRunConfig())
        await manager.release_page_with_context(page)

        recycle_observed = await _await_mid_recycle(manager)
        assert recycle_observed, "never observed the real mid-recycle window"

        # The guard under test: skip the detach on a mid-recycle managed PERMANENT.
        assert crawler_pool._is_recycling(crawler) is True
        close_task, _done = await crawler_pool._init_permanent_locked(config, force=True, target=crawler)
        assert close_task is None, "guard did not skip mid-recycle managed PERMANENT"
        assert crawler_pool.PERMANENT is crawler
        assert manager._closing is False
        assert not crawler_pool._CLOSE_TASKS

        # Release the real start(); the recycle spawns a fresh managed Chromium
        # in its own process group on the still-retained PERMANENT.
        permit_start.set()
        if manager._recycle_task is not None:
            await asyncio.wait_for(manager._recycle_task, timeout=45)

        assert manager._recycling is False
        assert manager._closing is False
        assert manager.browser is not None
        assert manager.browser.is_connected() is True
        assert manager.managed_browser is not None
        assert manager.managed_browser is not first_managed

        # close() calls managed_browser.cleanup() (os.killpg) on the fresh
        # managed Chromium - no orphan, the process group is reaped.
        await manager.close()
        assert manager.browser is None
        assert manager.managed_browser is None

        await asyncio.sleep(2.0)
        chromium_after = _count_chromium()
        assert chromium_after <= chromium_before, (
            f"managed Chromium orphan: {chromium_before} -> {chromium_after}"
        )
    finally:
        permit_start.set()
        if manager._recycle_task is not None and not manager._recycle_task.done():
            with suppress(Exception):
                await manager._recycle_task
        with suppress(Exception):
            await manager.close()
        # Belt-and-suspenders: kill any stray managed Chromium process group.
        with suppress(Exception):
            await crawler.close()
        await asyncio.sleep(1.5)
