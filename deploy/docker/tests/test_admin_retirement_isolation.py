"""Admin pool retirement must not tear down in-flight dedicated hook crawlers.

`crawler_pool.retire_pool_crawlers` is the shared function behind three admin
routes (``kill_browser`` / ``restart_browser`` / ``force_cleanup``). Dedicated
hook crawlers are request-owned and live in ``COLD_POOL`` under
``dedicated:<uuid>`` signatures; their only intended disposal is
``release_dedicated_crawler``. These tests pin the invariant that admin
retirement skips the ``dedicated:`` namespace, while still retiring ordinary
sha1-keyed pooled browsers (including intentionally active ones, the documented
admin escape hatch).
"""
import asyncio
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from crawl4ai import BrowserConfig
from fastapi import HTTPException

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawler_pool
import monitor_routes


class _PoolCrawler:
    """Stand-in AsyncWebCrawler for pool retirement tests (see test_resource_policy)."""

    def __init__(self, connected=True):
        browser = MagicMock()
        browser.is_connected.return_value = connected
        self.crawler_strategy = SimpleNamespace(
            browser_manager=SimpleNamespace(browser=browser, default_context=None)
        )
        self.active_requests = 0
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True
        self.crawler_strategy.browser_manager.browser = None


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


class _Monitor:
    def get_active_requests(self):
        return []

    async def track_janitor_event(self, *_args, **_kwargs):
        return None


async def _make_dedicated(monkeypatch):
    """Create a request-owned dedicated crawler exactly as production does."""
    dedicated = _PoolCrawler()
    _configure_pool(monkeypatch, lambda **_kw: dedicated)
    crawler = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    assert crawler is dedicated
    sig = getattr(dedicated, "_docker_pool_sig")
    assert sig.startswith("dedicated:")
    return dedicated, sig


async def _drain_close_tasks():
    tasks = list(crawler_pool._CLOSE_TASKS)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_kill_browser_skips_inflight_dedicated_crawler(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())
    original_value = crawler_pool.ADMISSION_SEM._value

    with pytest.raises(HTTPException) as exc:
        await monitor_routes.kill_browser(
            monitor_routes.KillBrowserRequest(sig="dedicate")
        )

    assert exc.value.status_code == 404
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated
    assert getattr(dedicated, "_docker_admission_released", False) is False
    assert dedicated.active_requests == 1
    assert crawler_pool.ADMISSION_SEM._value == original_value


@pytest.mark.asyncio
async def test_kill_browser_with_full_dedicated_sig_returns_404(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    with pytest.raises(HTTPException) as exc:
        await monitor_routes.kill_browser(
            monitor_routes.KillBrowserRequest(sig=sig)
        )

    assert exc.value.status_code == 404
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated


@pytest.mark.asyncio
async def test_restart_browser_skips_inflight_dedicated_crawler(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    with pytest.raises(HTTPException) as exc:
        await monitor_routes.restart_browser(
            monitor_routes.KillBrowserRequest(sig="dedicate")
        )

    assert exc.value.status_code == 404
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated
    assert getattr(dedicated, "_docker_admission_released", False) is False


@pytest.mark.asyncio
async def test_force_cleanup_skips_inflight_dedicated_crawler(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())
    original_value = crawler_pool.ADMISSION_SEM._value

    response = await monitor_routes.force_cleanup()

    assert response == {"success": True, "killed_browsers": 0}
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated
    assert getattr(dedicated, "_docker_admission_released", False) is False
    assert dedicated.active_requests == 1
    assert crawler_pool.ADMISSION_SEM._value == original_value


@pytest.mark.asyncio
async def test_retire_pool_crawlers_skips_dedicated_namespace_directly(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)

    # Blanket cold-only close (force_cleanup's path).
    assert await crawler_pool.retire_pool_crawlers(cold_only=True) == []
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated

    # The 8-char dashboard prefix ("dedicate") and the full sig both miss.
    assert await crawler_pool.retire_pool_crawlers("dedicate") == []
    assert await crawler_pool.retire_pool_crawlers(sig) == []
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[sig] is dedicated
    assert getattr(dedicated, "_docker_admission_released", False) is False


@pytest.mark.asyncio
async def test_force_cleanup_skips_dedicated_but_closes_idle_sha1_cold_browsers(
    monkeypatch,
):
    dedicated, dedicated_sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    idle = _PoolCrawler()
    idle_sig = crawler_pool._sig(BrowserConfig())
    crawler_pool.COLD_POOL[idle_sig] = idle
    crawler_pool.LAST_USED[idle_sig] = 0
    crawler_pool.USAGE_COUNT[idle_sig] = 1

    response = await monitor_routes.force_cleanup()

    assert response == {"success": True, "killed_browsers": 1}
    assert idle.closed is True
    assert idle_sig not in crawler_pool.COLD_POOL
    # The dedicated crawler is untouched.
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[dedicated_sig] is dedicated
    assert dedicated.active_requests == 1


@pytest.mark.asyncio
async def test_kill_browser_still_kills_idle_sha1_pooled_browser_with_dedicated_present(
    monkeypatch,
):
    dedicated, dedicated_sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    idle = _PoolCrawler()
    idle_sig = crawler_pool._sig(BrowserConfig())
    crawler_pool.COLD_POOL[idle_sig] = idle
    crawler_pool.LAST_USED[idle_sig] = 0
    crawler_pool.USAGE_COUNT[idle_sig] = 1

    response = await monitor_routes.kill_browser(
        monitor_routes.KillBrowserRequest(sig=idle_sig[:8])
    )

    assert response["success"] is True
    assert response["killed_sig"] == idle_sig[:8]
    assert idle.closed is True
    assert idle_sig not in crawler_pool.COLD_POOL
    # Dedicated crawler remains in-flight.
    assert dedicated.closed is False
    assert crawler_pool.COLD_POOL[dedicated_sig] is dedicated


@pytest.mark.asyncio
async def test_kill_browser_still_kills_active_sha1_pooled_browser(monkeypatch):
    """The fix targets the dedicated namespace, not active pooled browsers:
    an admin may still intentionally kill an active sha1-keyed pooled browser
    (the documented escape hatch that logs a warning)."""
    _configure_pool(monkeypatch, lambda **_kw: _PoolCrawler())
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    active = _PoolCrawler()
    active.active_requests = 1
    sig = crawler_pool._sig(BrowserConfig())
    crawler_pool.COLD_POOL[sig] = active
    crawler_pool.LAST_USED[sig] = 0
    crawler_pool.USAGE_COUNT[sig] = 1

    response = await monitor_routes.kill_browser(
        monitor_routes.KillBrowserRequest(sig=sig[:8])
    )

    assert response["success"] is True
    assert active.closed is True
    assert sig not in crawler_pool.COLD_POOL


@pytest.mark.asyncio
async def test_dedicated_browser_remains_live_for_inflight_owner_after_admin_kill(
    monkeypatch,
):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    assert dedicated.crawler_strategy.browser_manager.browser is not None

    with pytest.raises(HTTPException):
        await monitor_routes.kill_browser(
            monitor_routes.KillBrowserRequest(sig="dedicate")
        )
    # The in-flight browser is still alive for its owning request.
    assert dedicated.crawler_strategy.browser_manager.browser is not None
    assert dedicated.closed is False

    with pytest.raises(HTTPException):
        await monitor_routes.restart_browser(
            monitor_routes.KillBrowserRequest(sig="dedicate")
        )
    assert dedicated.crawler_strategy.browser_manager.browser is not None
    assert dedicated.closed is False

    response = await monitor_routes.force_cleanup()
    assert response["killed_browsers"] == 0
    assert dedicated.crawler_strategy.browser_manager.browser is not None
    assert dedicated.closed is False


@pytest.mark.asyncio
async def test_release_dedicated_crawler_closes_browser_after_admin_noop(monkeypatch):
    dedicated, sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())
    capacity = crawler_pool.ADMISSION_SEM._value

    with pytest.raises(HTTPException):
        await monitor_routes.kill_browser(
            monitor_routes.KillBrowserRequest(sig="dedicate")
        )

    # The owner's intended disposal path still fully closes the crawler and
    # releases its admission lease (no early no-op from a prior retire).
    await crawler_pool.release_dedicated_crawler(dedicated)

    assert dedicated.closed is True
    assert sig not in crawler_pool.COLD_POOL
    assert getattr(dedicated, "_docker_admission_released", False) is True
    assert getattr(dedicated, "_docker_request_owned", False) is False
    assert getattr(dedicated, "_docker_pool_sig") is None
    assert crawler_pool.ADMISSION_SEM._value == capacity + 1
    await _drain_close_tasks()


@pytest.mark.asyncio
async def test_multiple_dedicated_crawlers_survive_force_cleanup(monkeypatch):
    dedicated, dedicated_sig = await _make_dedicated(monkeypatch)
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _Monitor())

    # Second dedicated crawler created through the production path. Reconfigure
    # the factory only (the first dedicated entry already in COLD_POOL survives,
    # and MAX_BROWSER_INSTANCES=2 still has one slot free).
    other = _PoolCrawler()
    monkeypatch.setattr(crawler_pool, "AsyncWebCrawler", lambda **_kw: other)
    second = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    assert second is other
    second_sig = getattr(other, "_docker_pool_sig")
    assert len(crawler_pool.COLD_POOL) == 2

    response = await monitor_routes.force_cleanup()

    assert response == {"success": True, "killed_browsers": 0}
    assert dedicated.closed is False
    assert other.closed is False
    assert crawler_pool.COLD_POOL[dedicated_sig] is dedicated
    assert crawler_pool.COLD_POOL[second_sig] is other

    await crawler_pool.release_dedicated_crawler(dedicated)
    await crawler_pool.release_dedicated_crawler(other)
    assert crawler_pool.COLD_POOL == {}
    await _drain_close_tasks()
