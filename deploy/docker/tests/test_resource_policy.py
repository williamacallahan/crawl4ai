import os
import sys
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from crawl4ai import BrowserConfig, MemoryAdaptiveDispatcher

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api
import crawler_pool
import monitor_routes


def _config():
    return {
        "crawler": {
            "browser": {
                "kwargs": {
                    "memory_saving_mode": True,
                    "max_pages_before_recycle": 500,
                },
                "extra_args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
        }
    }


def test_resource_policy_enforces_server_browser_limits(monkeypatch):
    monkeypatch.delenv("CRAWL4AI_CHROMIUM_SANDBOX", raising=False)
    browser_config = BrowserConfig(
        memory_saving_mode=False,
        max_pages_before_recycle=0,
    )

    result = api._apply_server_browser_policy(browser_config, _config())

    assert result is browser_config
    assert result.memory_saving_mode is True
    assert result.max_pages_before_recycle == 500
    assert result.extra_args == ["--no-sandbox", "--disable-dev-shm-usage"]


def test_resource_policy_honors_chromium_sandbox_opt_in(monkeypatch):
    monkeypatch.setenv("CRAWL4AI_CHROMIUM_SANDBOX", "true")

    result = api._apply_server_browser_policy(BrowserConfig(), _config())

    assert result.extra_args == ["--disable-dev-shm-usage"]


def test_resource_policy_preserves_stricter_recycle_limit():
    browser_config = BrowserConfig(
        memory_saving_mode=True,
        max_pages_before_recycle=100,
    )

    result = api._apply_server_browser_policy(browser_config, _config())

    assert result.max_pages_before_recycle == 100


def test_resource_policy_replaces_malformed_recycle_limit():
    browser_config = BrowserConfig()
    browser_config.max_pages_before_recycle = None

    result = api._apply_server_browser_policy(browser_config, _config())

    assert result.max_pages_before_recycle == 500


def test_dispatcher_accepts_server_capacity_and_hysteresis():
    dispatcher = MemoryAdaptiveDispatcher(
        max_session_permit=20,
        memory_threshold_percent=80,
        recovery_threshold_percent=65,
    )

    assert dispatcher.max_session_permit == 20
    assert dispatcher.memory_threshold_percent == 80
    assert dispatcher.recovery_threshold_percent == 65


def test_crawl_result_projection_omits_unrequested_heavy_fields():
    result = {
        "url": "https://example.com",
        "success": True,
        "markdown": {"fit_markdown": "# Example"},
        "html": "<html>large body</html>",
        "cleaned_html": "<main>large body</main>",
    }

    projected = api._project_crawl_result(result, ["url", "success", "markdown"])

    assert projected == {
        "url": "https://example.com",
        "success": True,
        "markdown": {"fit_markdown": "# Example"},
    }


@pytest.mark.asyncio
async def test_browser_pool_evicts_least_recent_idle_browser(monkeypatch):
    oldest = MagicMock(active_requests=0)
    oldest.close = AsyncMock()
    active = MagicMock(active_requests=1)
    active.close = AsyncMock()

    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 3)
    monkeypatch.setattr(crawler_pool, "PERMANENT", MagicMock())
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {"oldest": oldest, "active": active})
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {"oldest": 1, "active": 2})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {"oldest": 1, "active": 1})

    close_task = crawler_pool._make_browser_capacity()
    assert close_task is not None
    await close_task

    oldest.close.assert_awaited_once()
    active.close.assert_not_awaited()
    assert set(crawler_pool.COLD_POOL) == {"active"}


def test_browser_pool_refuses_growth_when_all_browsers_are_active(monkeypatch):
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 2)
    monkeypatch.setattr(crawler_pool, "PERMANENT", MagicMock())
    monkeypatch.setattr(
        crawler_pool,
        "COLD_POOL",
        {"active": MagicMock(active_requests=1)},
    )
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})

    try:
        crawler_pool._make_browser_capacity()
    except RuntimeError as error:
        assert str(error) == "Crawler browser pool is at capacity"
    else:
        raise AssertionError("Expected browser pool capacity rejection")


class _PoolCrawler:
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


class _BlockingCloseCrawler(_PoolCrawler):
    def __init__(self, connected=True):
        super().__init__(connected=connected)
        self.close_started = asyncio.Event()
        self.continue_close = asyncio.Event()

    async def close(self):
        self.close_started.set()
        await self.continue_close.wait()
        await super().close()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_name", ["HOT_POOL", "COLD_POOL"])
@pytest.mark.parametrize("stale_state", ["closed", "disconnected"])
async def test_browser_pool_replaces_unavailable_browser(
    monkeypatch, pool_name, stale_state
):
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    stale = _PoolCrawler(connected=stale_state != "disconnected")
    if stale_state == "closed":
        await stale.close()
    getattr(crawler_pool, pool_name)[sig] = stale
    crawler_pool.LAST_USED[sig] = 1
    crawler_pool.USAGE_COUNT[sig] = 1

    crawler = await crawler_pool.get_crawler(config)

    assert crawler is created[0]
    assert stale.closed
    assert crawler_pool.COLD_POOL[sig] is crawler
    await crawler_pool.release_crawler(crawler)


@pytest.mark.asyncio
async def test_permanent_browser_start_failure_closes_partial_browser(monkeypatch):
    crawler = _PoolCrawler()
    crawler.start = AsyncMock(side_effect=RuntimeError("start failed"))
    _configure_pool(monkeypatch, lambda **_kwargs: crawler)

    with pytest.raises(RuntimeError, match="start failed"):
        await crawler_pool.init_permanent(BrowserConfig())

    await _drain_close_tasks()
    assert crawler.closed
    assert crawler_pool.PERMANENT is None


@pytest.mark.asyncio
async def test_pooled_browser_start_failure_closes_partial_browser(monkeypatch):
    crawler = _PoolCrawler()
    crawler.start = AsyncMock(side_effect=asyncio.CancelledError)
    _configure_pool(monkeypatch, lambda **_kwargs: crawler)

    with pytest.raises(asyncio.CancelledError):
        await crawler_pool.get_crawler(BrowserConfig())

    await _drain_close_tasks()
    assert crawler.closed
    assert crawler_pool.COLD_POOL == {}
    assert crawler_pool.ADMISSION_SEM._value == 2


@pytest.mark.asyncio
async def test_permanent_browser_replacement_and_monitor_restart_do_not_deadlock(
    monkeypatch,
):
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    stale = _PoolCrawler(connected=False)
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = sig

    crawler = await crawler_pool.get_crawler(config)
    await crawler_pool.release_crawler(crawler)
    assert crawler is created[0]
    assert stale.closed

    monkeypatch.setattr("server.get_default_browser_config", BrowserConfig)
    response = await asyncio.wait_for(
        monitor_routes.restart_browser(
            monitor_routes.KillBrowserRequest(sig="permanent")
        ),
        timeout=1,
    )

    assert response == {"success": True, "restarted": "permanent"}
    await _drain_close_tasks()
    assert crawler.closed
    assert crawler_pool.PERMANENT is created[1]


@pytest.mark.asyncio
async def test_capacity_eviction_waits_for_a_wedged_close_without_exceeding_cap(
    monkeypatch,
):
    state = {"open": 0, "max_open": 0}
    created = []

    class CapacityCrawler(_PoolCrawler):
        async def start(self):
            state["open"] += 1
            state["max_open"] = max(state["max_open"], state["open"])

        async def close(self):
            if not self.closed:
                state["open"] -= 1
            await super().close()

    class BlockingCapacityCrawler(CapacityCrawler):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.continue_close = asyncio.Event()

        async def close(self):
            self.close_started.set()
            await self.continue_close.wait()
            await super().close()

    def factory(**_kwargs):
        crawler = CapacityCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 1)
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(2))
    old = BlockingCapacityCrawler()
    await old.start()
    crawler_pool.COLD_POOL["idle"] = old
    crawler_pool.LAST_USED["idle"] = 0
    crawler_pool.USAGE_COUNT["idle"] = 1

    acquisition = asyncio.create_task(crawler_pool.get_crawler(BrowserConfig()))
    await old.close_started.wait()
    await _assert_pool_lock_available()
    assert not created
    assert state == {"open": 1, "max_open": 1}

    old.continue_close.set()
    crawler = await asyncio.wait_for(acquisition, timeout=1)
    assert crawler is created[0]
    assert state == {"open": 1, "max_open": 1}

    await crawler_pool.release_crawler(crawler)
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_unavailable_replacement_leaves_the_lock_free_while_close_is_wedged(
    monkeypatch,
):
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 1)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    stale = _BlockingCloseCrawler(connected=False)
    crawler_pool.COLD_POOL[sig] = stale
    crawler_pool.LAST_USED[sig] = 0
    crawler_pool.USAGE_COUNT[sig] = 1

    acquisition = asyncio.create_task(crawler_pool.get_crawler(config))
    await stale.close_started.wait()
    await _assert_pool_lock_available()
    assert not created

    stale.continue_close.set()
    crawler = await asyncio.wait_for(acquisition, timeout=1)
    assert crawler is created[0]
    await crawler_pool.release_crawler(crawler)
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_permanent_replacement_leaves_the_lock_free_while_close_is_wedged(
    monkeypatch,
):
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 1)
    config = BrowserConfig()
    stale = _BlockingCloseCrawler(connected=False)
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = crawler_pool._sig(config)

    replacement = asyncio.create_task(crawler_pool.init_permanent(config, force=True))
    await stale.close_started.wait()
    await _assert_pool_lock_available()
    assert not created

    stale.continue_close.set()
    await asyncio.wait_for(replacement, timeout=1)
    assert crawler_pool.PERMANENT is created[0]
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_stale_permanent_replacement_preserves_a_concurrent_replacement(
    monkeypatch,
):
    created = []

    def factory(**_kwargs):
        crawler = _PoolCrawler()
        created.append(crawler)
        return crawler

    _configure_pool(monkeypatch, factory)
    config = BrowserConfig()
    stale = _BlockingCloseCrawler(connected=False)
    crawler_pool.PERMANENT = stale
    crawler_pool.DEFAULT_CONFIG_SIG = crawler_pool._sig(config)

    acquisition = asyncio.create_task(crawler_pool.get_crawler(config))
    await stale.close_started.wait()
    await crawler_pool.init_permanent(config)
    replacement = crawler_pool.PERMANENT

    stale.continue_close.set()
    assert await asyncio.wait_for(acquisition, timeout=1) is replacement
    assert created == [replacement]
    await crawler_pool.release_crawler(replacement)
    await crawler_pool.close_all()


@pytest.mark.asyncio
async def test_cancelled_promotion_rolls_back_the_active_request(monkeypatch):
    started = asyncio.Event()

    class Monitor:
        async def track_janitor_event(self, *_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

    monitor = Monitor()
    _configure_pool(monkeypatch, lambda **_kwargs: _PoolCrawler())
    monkeypatch.setattr("monitor.get_monitor", lambda: monitor)
    config = BrowserConfig()
    sig = crawler_pool._sig(config)
    crawler = _PoolCrawler()
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = 0
    crawler_pool.USAGE_COUNT[sig] = 2

    acquisition = asyncio.create_task(crawler_pool.get_crawler(config))
    await started.wait()
    acquisition.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert crawler_pool.HOT_POOL[sig] is crawler
    assert crawler.active_requests == 0
    assert crawler_pool.ADMISSION_SEM._value == 2


@pytest.mark.asyncio
async def test_start_failure_schedules_its_wedged_close_outside_the_lock(monkeypatch):
    class FailingCrawler(_BlockingCloseCrawler):
        async def start(self):
            raise RuntimeError("start failed")

    crawler = FailingCrawler()
    _configure_pool(monkeypatch, lambda **_kwargs: crawler)

    with pytest.raises(RuntimeError, match="start failed"):
        await crawler_pool.get_crawler(BrowserConfig())

    await crawler.close_started.wait()
    await _assert_pool_lock_available()
    crawler.continue_close.set()
    await _drain_close_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cleanup", "kill", "restart"])
async def test_monitor_retirement_actions_leave_the_lock_free_while_close_is_wedged(
    monkeypatch, action
):
    class Monitor:
        def get_active_requests(self):
            return []

        async def track_janitor_event(self, *_args, **_kwargs):
            return None

    _configure_pool(monkeypatch, lambda **_kwargs: _PoolCrawler())
    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: Monitor())
    crawler = _BlockingCloseCrawler()
    sig = "cold-browser"
    crawler_pool.COLD_POOL[sig] = crawler
    crawler_pool.LAST_USED[sig] = 0
    crawler_pool.USAGE_COUNT[sig] = 1

    if action == "cleanup":
        task = asyncio.create_task(monitor_routes.force_cleanup())
    elif action == "kill":
        task = asyncio.create_task(
            monitor_routes.kill_browser(monitor_routes.KillBrowserRequest(sig="cold"))
        )
    else:
        task = asyncio.create_task(
            monitor_routes.restart_browser(monitor_routes.KillBrowserRequest(sig="cold"))
        )

    await crawler.close_started.wait()
    await _assert_pool_lock_available()
    assert not task.done()
    crawler.continue_close.set()
    response = await asyncio.wait_for(task, timeout=1)

    assert response["success"] is True
    assert sig not in crawler_pool.COLD_POOL
