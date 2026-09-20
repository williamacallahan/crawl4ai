"""Regression tests for the admin force-restart race in ``crawler_pool``.

``init_permanent(cfg, force=True)`` (the ``POST /monitor/actions/restart_browser?
sig=permanent`` admin route) used to release the pool ``LOCK`` while awaiting
the old permanent's background close, then re-loop into ``_init_permanent_locked``
and unconditionally detach whatever ``PERMANENT`` happened to be on that later
iteration - including a *fresh* permanent a concurrent default-config request
had just rebuilt and admitted an in-flight crawl on. The restart would then
rebuild a *third* permanent, killing the in-flight crawl's browser under it and
returning ``{"success": True, "restarted": "permanent"}`` regardless.

The fix scopes a force restart to the identity of the permanent it captured at
the start (``original``). If a concurrent request already replaced that original,
the force restart returns ``False`` (*superseded*) and leaves the fresh permanent
- and any in-flight crawl admitted on it - untouched. These tests assert that
fixed behaviour, the new ``init_permanent`` return contract, the admin route's
new reporting, and that the non-force admission path still replaces a stale
permanent (the ``target=PERMANENT`` change must not regress it).

The harness mirrors ``test_resource_policy.py``'s ``_PoolCrawler`` /
``_BlockingCloseCrawler`` / ``_configure_pool`` doubles so the real
``crawler_pool`` code runs against stubbed browsers with no Chromium.
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from crawl4ai import BrowserConfig

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool  # noqa: E402
import monitor_routes  # noqa: E402


# ───────────────────────── stub crawlers ──────────────────────────


class _PoolCrawler:
    """Minimal AsyncWebCrawler double for the pool's required surface."""

    def __init__(self, connected=True):
        browser = MagicMock()
        browser.is_connected.return_value = connected
        self.crawler_strategy = SimpleNamespace(
            browser_manager=SimpleNamespace(browser=browser, default_context=None)
        )
        self.active_requests = 0
        self.closed = False
        self._docker_admission_released = False
        self._arun_gate = asyncio.Event()

    async def start(self):
        pass

    async def close(self):
        self.closed = True
        self.crawler_strategy.browser_manager.browser = None

    async def arun(self, url="https://example.com", *_a, **_kw):
        # Simulate a crawl: if the browser was closed under us, surface the
        # Playwright "Browser has been closed" error the user would see. The
        # gate keeps the crawl in-flight across the force restart's second
        # iteration so the race is observed deterministically.
        if self.closed:
            raise RuntimeError("Browser has been closed")
        await self._arun_gate.wait()
        if self.closed:
            raise RuntimeError("Browser has been closed")
        return {"url": url, "success": True}


class _BlockingCloseCrawler(_PoolCrawler):
    def __init__(self, connected=True):
        super().__init__(connected=connected)
        self.close_started = asyncio.Event()
        self.continue_close = asyncio.Event()

    async def close(self):
        self.close_started.set()
        await self.continue_close.wait()
        await super().close()


class _SlowLaunchCrawler(_PoolCrawler):
    """A crawler whose ``start()`` parks on a gate, modelling a slow launch.

    ``start()`` runs under the pool LOCK, so a slow launch holds LOCK for its
    duration. This is what serializes a force restart's second iteration behind
    a concurrent rebuild even when the *old* permanent's close is instant.
    """

    def __init__(self):
        super().__init__(connected=True)

    async def start(self):
        self._launch_started.set()
        await self._launch_gate.wait()


# ───────────────────────── pool harness ────────────────────────────


def _configure_pool(monkeypatch, factory):
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(2))
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 2)
    monkeypatch.setattr(crawler_pool, "MEM_LIMIT", 100)
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "DEFAULT_CONFIG_SIG", None)
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "get_container_memory_percent", lambda: 0)
    monkeypatch.setattr(crawler_pool, "AsyncWebCrawler", factory)


async def _drain_close_tasks():
    tasks = list(crawler_pool._CLOSE_TASKS)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _assert_pool_lock_available():
    await asyncio.wait_for(crawler_pool.LOCK.acquire(), timeout=1)
    crawler_pool.LOCK.release()


# ───────────────────────── the race, fixed ─────────────────────────


@pytest.mark.asyncio
async def test_force_restart_does_not_kill_inflight_crawl_on_fresh_permanent(monkeypatch):
    """The canonical race: a default-config request rebuilds PERMANENT during
    the force restart's lock-free close await and is admitted on that fresh
    browser. The restart must leave the fresh permanent - and the in-flight
    crawl - alone and report ``False`` (superseded)."""
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)

    stale = _BlockingCloseCrawler(connected=True)
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    force_task = asyncio.create_task(crawler_pool.init_permanent(config, force=True))
    await stale.close_started.wait()
    assert crawler_pool.PERMANENT is None  # old detached, closing in background

    # A default-config request arrives while LOCK is released; it rebuilds
    # PERMANENT and is admitted on the fresh browser.
    admitted = await crawler_pool.get_crawler(config)
    fresh = crawler_pool.PERMANENT
    assert admitted is fresh
    assert fresh.active_requests == 1

    # An in-flight crawl is now running on `fresh`.
    arun_task = asyncio.create_task(admitted.arun("https://example.com"))
    await asyncio.sleep(0)  # let arun enter and park on its gate

    stale.continue_close.set()  # old close completes; force-caller re-loops
    replaced = await asyncio.wait_for(force_task, timeout=2)

    # FIX: the fresh permanent serving the in-flight crawl is untouched.
    assert replaced is False
    assert fresh.closed is False
    assert fresh._docker_admission_released is False
    assert crawler_pool.PERMANENT is fresh
    assert created == [fresh]  # only the concurrent rebuild; no third permanent

    # The in-flight crawl completes successfully.
    admitted._arun_gate.set()
    result = await asyncio.wait_for(arun_task, timeout=2)
    assert result == {"url": "https://example.com", "success": True}

    await crawler_pool.release_crawler(admitted)
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_slow_launch_under_lock_still_preserves_fresh_permanent(monkeypatch):
    """The race does not depend on a slow close. Even with an *instant* close of
    the old permanent, a concurrent rebuild whose launch (``crawler.start()``
    under LOCK) serializes the force restart's second iteration behind it must
    still leave the freshly-launched permanent alone."""
    created = []
    launch_started = asyncio.Event()
    allow_launch = asyncio.Event()

    def factory(**_kwargs):
        crawler = _SlowLaunchCrawler()
        crawler._launch_started = launch_started
        crawler._launch_gate = allow_launch
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)

    stale = _PoolCrawler(connected=True)  # instant close
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    force_task = asyncio.create_task(crawler_pool.init_permanent(config, force=True))
    # The old permanent's close is instant, so the force-caller parks on its
    # shield only briefly. Start the concurrent request; its slow launch holds
    # LOCK once it acquires it, serializing the force-caller's next iteration.
    admitted = asyncio.create_task(crawler_pool.get_crawler(config))
    await asyncio.wait_for(launch_started.wait(), timeout=2)
    assert crawler_pool.PERMANENT is None  # not yet set: launch in progress under LOCK

    # Let the (instant) old close drain and the force-caller queue its second
    # iteration behind the held LOCK. Then complete the launch: PERMANENT
    # becomes the fresh crawler, LOCK is freed, the force-caller re-enters and
    # must NOT detach it.
    allow_launch.set()
    replaced = await asyncio.wait_for(force_task, timeout=2)
    admitted_crawler = await asyncio.wait_for(admitted, timeout=2)

    fresh = crawler_pool.PERMANENT
    assert admitted_crawler is fresh
    assert replaced is False
    assert fresh.closed is False
    assert fresh._docker_admission_released is False
    assert created == [fresh]  # only the concurrent rebuild; no third permanent

    await crawler_pool.release_crawler(admitted_crawler)
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_force_restart_preserves_initialization_already_in_progress(monkeypatch):
    created = []
    launch_started = asyncio.Event()
    allow_launch = asyncio.Event()

    def factory(**_kwargs):
        crawler = _SlowLaunchCrawler()
        crawler._launch_started = launch_started
        crawler._launch_gate = allow_launch
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    initializing = asyncio.create_task(crawler_pool.init_permanent(config))
    await asyncio.wait_for(launch_started.wait(), timeout=2)
    restarting = asyncio.create_task(crawler_pool.init_permanent(config, force=True))
    await asyncio.sleep(0)  # Capture the absent original, then wait on the pool lock.
    allow_launch.set()

    assert await asyncio.wait_for(initializing, timeout=2) is True
    assert await asyncio.wait_for(restarting, timeout=2) is False
    assert len(created) == 1
    assert crawler_pool.PERMANENT is created[0]
    assert created[0].closed is False
    await crawler_pool.close_all()


# ───────────────────── _init_permanent_locked contract ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("recycling", [False, True])
async def test_init_permanent_locked_force_leaves_a_concurrent_rebuild_alone(monkeypatch, recycling):
    """The fix: a force restart whose original target was already replaced by a
    concurrent rebuild returns ``(None, False)`` and does not detach the fresh
    permanent."""
    _configure_pool(monkeypatch, lambda **_kw: _PoolCrawler())
    stale = _PoolCrawler(connected=True)
    fresh = _PoolCrawler(connected=True)
    fresh.crawler_strategy.browser_manager._recycling = recycling
    crawler_pool.PERMANENT = fresh  # a concurrent caller already rebuilt it
    crawler_pool.DEFAULT_CONFIG_SIG = "sig"

    close_task, done = await crawler_pool._init_permanent_locked(
        BrowserConfig(), force=True, target=stale,
    )
    assert close_task is None
    assert done is False
    assert crawler_pool.PERMANENT is fresh  # untouched
    assert fresh.closed is False
    assert fresh._docker_admission_released is False
    await _assert_pool_lock_available()


@pytest.mark.asyncio
async def test_init_permanent_locked_force_creates_replacement_when_none(monkeypatch):
    """After the force-caller finishes closing its original and nothing rebuilt
    it (``PERMANENT is None``), it must fall through to create the replacement,
    not misread the empty slot as "superseded"."""
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()

    close_task, done = await crawler_pool._init_permanent_locked(
        config, force=True, target=None,
    )
    assert close_task is None
    assert done is True
    assert crawler_pool.PERMANENT is created[0]
    assert crawler_pool.DEFAULT_CONFIG_SIG == crawler_pool._sig(config)


@pytest.mark.asyncio
async def test_init_permanent_force_returns_true_when_it_replaces(monkeypatch):
    """Happy path: with no concurrent rebuild, the force restart replaces the
    permanent and returns ``True``."""
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    stale = _PoolCrawler(connected=True)
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    replaced = await asyncio.wait_for(
        crawler_pool.init_permanent(config, force=True), timeout=2,
    )
    assert replaced is True
    assert crawler_pool.PERMANENT is created[0]
    assert crawler_pool.PERMANENT is not stale
    await _drain_close_tasks()
    assert stale.closed


# ─────────────────────── admin route reporting ──────────────────────


@pytest.mark.asyncio
async def test_restart_browser_route_reports_superseded_when_rebuild_won(monkeypatch):
    """When ``init_permanent(force=True)`` returns ``False`` (a concurrent
    rebuild superseded the restart), the admin route reports that instead of
    claiming success."""
    _configure_pool(monkeypatch, lambda **_kw: _PoolCrawler())

    async def superseded_init(cfg, *, force=False):
        return False

    monkeypatch.setattr(crawler_pool, "init_permanent", superseded_init)
    monkeypatch.setattr("server.get_default_browser_config", lambda: BrowserConfig())
    crawler_pool.DEFAULT_CONFIG_SIG = "permanent-sig"

    response = await asyncio.wait_for(
        monitor_routes.restart_browser(monitor_routes.KillBrowserRequest(sig="permanent")),
        timeout=1,
    )
    assert response == {
        "success": True,
        "restarted": False,
        "reason": "superseded by a concurrent rebuild",
    }


# ──────────────────── admission-path regression ─────────────────────


@pytest.mark.asyncio
async def test_admission_path_still_replaces_a_stale_permanent(monkeypatch):
    """The ``target=PERMANENT`` change in the non-force admission path must not
    regress stale-permanent replacement: a default-config request whose
    permanent is unavailable still detaches it, rebuilds, and is admitted on
    the replacement."""
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    stale = _PoolCrawler(connected=False)  # not live
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    crawler = await crawler_pool.get_crawler(config)
    assert crawler is created[0]
    assert crawler_pool.PERMANENT is crawler
    await _drain_close_tasks()
    assert stale.closed  # the stale permanent was replaced

    await crawler_pool.release_crawler(crawler)
    await crawler_pool.close_all()
