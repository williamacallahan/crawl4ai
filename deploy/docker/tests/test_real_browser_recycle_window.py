"""End-to-end regression test for ``_is_live`` against a real headless Chromium.

The control-flow suite (``test_is_live_recycle.py``) drives the real
``BrowserManager.close(_for_recycle=True)`` path with a stubbed ``start()`` so it
can park the manager deterministically in the mid-recycle window. This file
complements it by exercising the *whole* in-place recycle - a real Playwright
launch, the real ``close`` teardown, and the real ``start()`` relaunch - and
sampling ``crawler_pool._is_live`` throughout the cycle on a live manager.

Preconditions:
  * A real headless Chromium installed for Playwright
    (``playwright install chromium``). The environment used by CI has
    ``chromium-1179``/``chromium-1194`` under ``~/.cache/ms-playwright/``.
  * The Chromium binary must be launchable as the test user. When running as
    root, ``extra_args=["--no-sandbox"]`` is supplied (the same operator policy
    ``deploy/docker/config.yml`` applies via ``api._apply_server_browser_policy``).
"""

import asyncio
import os
import sys
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
    return SimpleNamespace(crawler_strategy=SimpleNamespace(browser_manager=manager))


async def _sample_until_recycle_done(manager, crawler, *, max_iters=1000, step=0.003):
    """Sample (_recycling, browser_connected, _is_live) until the recycle ends."""
    observations = []
    for _ in range(max_iters):
        observations.append((
            manager._recycling,
            manager.browser is not None and manager.browser.is_connected(),
            crawler_pool._is_live(crawler),
        ))
        if (
            manager._recycle_task is not None
            and manager._recycle_task.done()
            and not manager._recycling
            and len(observations) > 1
        ):
            break
        await asyncio.sleep(step)
    return observations


@pytest.fixture(autouse=True)
def _reset_global_pages():
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    yield
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None


@pytest.mark.asyncio
async def test_real_browser_recycle_keeps_is_live_true_throughout():
    """A real in-place recycle must not be misclassified as dead by _is_live.

    Drives the full ``get_page`` -> ``release_page_with_context`` ->
    fire-and-forget ``_recycle_browser`` -> real ``close(_for_recycle=True)``
    -> real ``start()`` cycle on a real headless Chromium and asserts that
    ``_is_live`` never returns False while the manager is mid-recycle, then
    returns True again once the browser has restarted.
    """
    config = BrowserConfig(
        headless=True,
        max_pages_before_recycle=1,
        extra_args=["--no-sandbox"],
    )
    manager = BrowserManager(config, logger=None)
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    try:
        await manager.start()
        first_browser = manager.browser
        assert manager.browser is not None
        assert manager.browser.is_connected() is True

        crawler = _crawler(manager)
        assert crawler_pool._is_live(crawler) is True

        page, _ = await manager.get_page(CrawlerRunConfig())
        assert manager._pages_served >= 1

        await manager.release_page_with_context(page)
        observations = await _sample_until_recycle_done(manager, crawler)
        if manager._recycle_task is not None:
            await manager._recycle_task

        during_recycle = [obs for obs in observations if obs[0] is True]
        assert during_recycle, (
            "never observed the mid-recycle window; the recycle likely finished "
            "before sampling began - increase sample rate or reduce churn"
        )

        false_during_recycle = [obs for obs in during_recycle if obs[2] is False]
        assert not false_during_recycle, (
            f"_is_live returned False {len(false_during_recycle)} of "
            f"{len(during_recycle)} times while the manager was mid-recycle "
            "(browser=None/default_context=None, _recycling=True) - a healthy "
            "in-place recycle was misclassified as dead"
        )

        null_during_recycle = [obs for obs in during_recycle if obs[1] is False]
        assert null_during_recycle, (
            "never observed the browser=None window inside the recycle - the "
            "test did not exercise the vulnerable state"
        )

        assert manager.browser is not None
        assert manager.browser.is_connected() is True
        assert manager._recycling is False
        assert crawler_pool._is_live(crawler) is True
        second_page, _ = await asyncio.wait_for(
            manager.get_page(CrawlerRunConfig()), timeout=10
        )
        assert second_page is not page
        assert manager.browser is not first_browser
        await manager.release_page_with_context(second_page)
    finally:
        await manager.close()


def _real_manager(**config_overrides):
    config = BrowserConfig(headless=True, extra_args=["--no-sandbox"], **config_overrides)
    manager = BrowserManager(config, logger=None)
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    return manager


@pytest.mark.asyncio
async def test_concurrent_get_page_under_real_recycling_never_deadlocks():
    """Concurrent acquisitions across several real close/start cycles all finish.

    The mock suite stubs ``close``/``start``; only a real recycle shows that
    the drain barrier, the restart, and the waiters it wakes cannot wedge.
    """
    manager = _real_manager(max_pages_before_recycle=3)
    acquisitions, concurrency = 24, 6
    browsers = []  # holding each Browser keeps CPython from reusing its id
    gate = asyncio.Semaphore(concurrency)

    async def acquire_and_release():
        async with gate:
            page, _ = await manager.get_page(CrawlerRunConfig())
            browsers.append(manager.browser)
            await asyncio.sleep(0.01)
            await manager.release_page_with_context(page)

    try:
        await manager.start()
        await asyncio.wait_for(
            asyncio.gather(*(acquire_and_release() for _ in range(acquisitions))),
            timeout=120,
        )
        if manager._recycle_task is not None:
            await asyncio.wait_for(manager._recycle_task, timeout=60)

        assert len({id(browser) for browser in browsers}) > 1, "no real recycle ran under concurrent load"
        assert not manager._active_acquisitions
        assert manager._closing is False
        assert crawler_pool._is_live(_crawler(manager)) is True
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_context_refcount_reaches_zero_after_a_real_browser_crash():
    manager = _real_manager()
    try:
        await manager.start()
        page, _ = await manager.get_page(CrawlerRunConfig())
        assert sum(manager._context_refcounts.values()) == 1

        # The browser dies under an in-flight crawl; its release must still
        # return the context reference and the drain token.
        await manager.browser.close()
        await asyncio.wait_for(manager.release_page_with_context(page), timeout=30)

        assert all(count == 0 for count in manager._context_refcounts.values())
        assert not manager._active_acquisitions
        assert crawler_pool._is_live(_crawler(manager)) is False
    finally:
        await manager.close()
