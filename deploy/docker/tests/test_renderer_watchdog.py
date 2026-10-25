"""Resource thresholds, CDP failure budgets, and disposal through the real pool."""

import asyncio
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from playwright.async_api import Error as PlaywrightError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crawler_pool
from crawl4ai.browser_manager import RendererKill
from utils import BrowserWatchdogSettings, CrawlResourceSettings


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "DEFAULT_CONFIG_SIG", None)
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(0))
    monkeypatch.setattr(crawler_pool, "_CLOSE_TASKS", set())
    monkeypatch.setattr(crawler_pool, "_RENDERER_SAMPLES", {})
    monkeypatch.setattr(crawler_pool, "RENDERER_KILLS", {"cpu": 0, "rss": 0})
    return crawler_pool


def crawler(*, send=None, active=1):
    session = SimpleNamespace(send=send or AsyncMock(return_value={"product": "Chrome"}))
    browser = SimpleNamespace(
        new_browser_cdp_session=AsyncMock(return_value=session),
        is_connected=lambda: True,
    )
    manager = SimpleNamespace(
        browser=browser,
        browser_pid=None,
        _recycling=False,
        _closing=False,
        pool_recycle_requested=False,
        consecutive_failures=0,
        renderer_kills=deque(maxlen=12),
    )

    def request_pool_recycle(reason):
        manager._recycle_due = True
        manager.pool_recycle_reason = reason

    manager.request_pool_recycle = Mock(side_effect=request_pool_recycle)
    return SimpleNamespace(
        crawler_strategy=SimpleNamespace(browser_manager=manager),
        active_requests=active,
        close=AsyncMock(),
    )


async def drain_closes(pool):
    await asyncio.gather(*list(pool._CLOSE_TASKS))


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [0, 2])
async def test_three_target_closed_probes_retire_idle_or_busy_browser(pool, active):
    failed = crawler(send=AsyncMock(side_effect=PlaywrightError("Target closed")), active=active)
    pool.COLD_POOL["bad"] = failed
    pool.LAST_USED["bad"] = 1
    pool.USAGE_COUNT["bad"] = 1

    await pool._watchdog_pass()
    await pool._watchdog_pass()
    assert pool.COLD_POOL["bad"] is failed
    await pool._watchdog_pass()
    await drain_closes(pool)

    assert "bad" not in pool.COLD_POOL
    assert "bad" not in pool.LAST_USED
    assert failed.active_requests == 0
    assert pool.ADMISSION_SEM._value == active
    failed.close.assert_awaited_once()
    await pool.release_crawler(failed)
    assert pool.ADMISSION_SEM._value == active


@pytest.mark.asyncio
async def test_success_resets_probe_failures_and_reuses_browser_session(pool):
    send = AsyncMock(side_effect=[asyncio.TimeoutError, {}, asyncio.TimeoutError, {}])
    healthy = crawler(send=send)
    pool.HOT_POOL["healthy"] = healthy
    for _ in range(4):
        await pool._watchdog_pass()
    manager = healthy.crawler_strategy.browser_manager
    assert manager._docker_health_failures == 0
    manager.browser.new_browser_cdp_session.assert_awaited_once()
    healthy.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_wedged_probe_is_bounded_and_does_not_hold_pool_lock(pool, monkeypatch):
    started = asyncio.Event()

    async def hung_send(_command):
        started.set()
        await asyncio.Event().wait()

    failed = crawler(send=hung_send)
    pool.COLD_POOL["hung"] = failed
    monkeypatch.setattr(pool, "WATCHDOG", replace(pool.WATCHDOG, browser_probe_timeout_seconds=0.02))
    task = asyncio.create_task(pool._watchdog_pass())
    await started.wait()
    async with pool.LOCK:
        assert not task.done()
    await asyncio.wait_for(task, 1)
    assert failed.crawler_strategy.browser_manager._docker_health_failures == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["requested", "consecutive"])
async def test_manager_failure_signal_drains_permanent_and_still_probes(pool, failure):
    failed = crawler(active=2)
    manager = failed.crawler_strategy.browser_manager
    if failure == "requested":
        manager.request_pool_recycle("cleanup failed")
    else:
        manager.consecutive_failures = pool.WATCHDOG.consecutive_failure_limit
    pool.PERMANENT = failed
    pool.DEFAULT_CONFIG_SIG = "default"
    assert pool._is_live(failed)
    await pool._watchdog_pass()
    await drain_closes(pool)
    assert pool.PERMANENT is failed
    assert failed.active_requests == 2
    assert pool.ADMISSION_SEM._value == 0
    assert manager.consecutive_failures == 0
    assert manager._recycle_due
    assert not manager.pool_recycle_requested
    failed.close.assert_not_awaited()
    manager.request_pool_recycle.assert_called_once_with(
        "cleanup failed" if failure == "requested" else "consecutive browser faults"
    )
    manager.browser.new_browser_cdp_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_budget_recycle_transition_is_not_unhealthy(pool):
    healthy = crawler(active=0)
    manager = healthy.crawler_strategy.browser_manager
    manager._recycling = True
    manager.browser = None
    pool.HOT_POOL["recycling"] = healthy
    assert pool._is_live(healthy)
    await pool._watchdog_pass()
    assert pool.HOT_POOL["recycling"] is healthy
    healthy.close.assert_not_awaited()


class Renderer:
    """The psutil fields consumed by the resource threshold calculation."""

    pid = 42

    def __init__(self):
        self.cpu = 0.0
        self.runnable_ns_by_tid = {}
        self.rss = 10 * 1024 * 1024
        self.killed = 0
        self.arguments = ["chromium", "--type=renderer"]

    def create_time(self):
        return 123.0

    def cmdline(self):
        return self.arguments

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0)

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def kill(self):
        self.killed += 1


@pytest.fixture
def renderer_tree(pool, monkeypatch):
    target = crawler()
    target.crawler_strategy.browser_manager.browser_pid = 1234
    renderer = Renderer()

    def process(pid):
        assert pid == 1234
        return SimpleNamespace(children=lambda recursive: [renderer],
                               memory_info=lambda: SimpleNamespace(rss=0))

    monkeypatch.setattr(pool.psutil, "Process", process)
    monkeypatch.setattr(
        pool, "_runnable_ns_by_tid", lambda pid: dict(renderer.runnable_ns_by_tid)
    )
    return target, renderer


def test_renderer_cpu_requires_sustained_window(pool, renderer_tree):
    target, renderer = renderer_tree
    window = pool.WATCHDOG.renderer_cpu_seconds
    pool._watch_renderers(target, 0)
    for second in (window / 4, window / 2, window * 3 / 4):
        renderer.cpu = float(second)
        pool._watch_renderers(target, second)
        assert renderer.killed == 0
    renderer.cpu = float(window)
    pool._watch_renderers(target, window)
    assert renderer.killed == 1
    assert pool.RENDERER_KILLS == {"cpu": 1, "rss": 0}
    assert list(target.crawler_strategy.browser_manager.renderer_kills) == [
        RendererKill(window, "cpu", renderer.pid, 1234, renderer.rss, 805306368,
                     cpu_cores=1.0, cpu_limit_cores=pool.WATCHDOG.renderer_cpu_cores)
    ]


SECOND_NS = 1_000_000_000


def test_renderer_starved_by_host_load_still_counts_its_cpu_demand(pool, renderer_tree):
    # On a loaded host a spinning thread runs a quarter of the time and waits
    # in the run queue for the rest; process usage alone never crosses 0.9.
    target, renderer = renderer_tree
    window = pool.WATCHDOG.renderer_cpu_seconds
    pool._watch_renderers(target, 0)
    for second in (window / 2, window):
        renderer.cpu = second / 4
        renderer.runnable_ns_by_tid = {7: int(second * SECOND_NS)}
        pool._watch_renderers(target, second)
    assert renderer.killed == 1
    assert pool.RENDERER_KILLS == {"cpu": 1, "rss": 0}


def test_exited_thread_does_not_reset_a_spinning_threads_window(pool, renderer_tree):
    # The process's runnable total falls when a long-lived helper thread
    # exits; the spinning thread's own growth must still count.
    target, renderer = renderer_tree
    window = pool.WATCHDOG.renderer_cpu_seconds
    renderer.runnable_ns_by_tid = {7: 0, 8: int(window * 10 * SECOND_NS)}
    pool._watch_renderers(target, 0)
    for second in (window / 2, window):
        renderer.cpu = second / 4
        renderer.runnable_ns_by_tid = {7: int(second * SECOND_NS)}
        pool._watch_renderers(target, second)
    assert renderer.killed == 1


def test_many_briefly_runnable_threads_are_not_a_spin(pool, renderer_tree):
    # Twenty threads each runnable a tenth of the time on a saturated host
    # sum to two cores of runnable time, but no thread is spinning.
    target, renderer = renderer_tree
    window = pool.WATCHDOG.renderer_cpu_seconds
    pool._watch_renderers(target, 0)
    for tick in range(1, 9):
        second = tick * window / 4
        renderer.cpu = second * 0.1
        renderer.runnable_ns_by_tid = {
            tid: int(second * 0.1 * SECOND_NS) for tid in range(100, 120)
        }
        pool._watch_renderers(target, second)
    assert renderer.killed == 0


def test_renderer_cpu_lull_resets_sustained_window(pool, renderer_tree):
    target, renderer = renderer_tree
    step = pool.WATCHDOG.renderer_cpu_seconds / 4
    # Busy for three steps, idle for one (the lull), then busy for four more.
    for tick, cpu_steps in ((0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 5)):
        renderer.cpu = cpu_steps * step
        pool._watch_renderers(target, tick * step)
    assert renderer.killed == 0
    renderer.cpu = 6 * step
    pool._watch_renderers(target, 7 * step)
    assert renderer.killed == 1


def test_renderer_rss_limit_is_immediate_and_ignores_nonrenderers(pool, renderer_tree, caplog):
    target, renderer = renderer_tree
    renderer.rss = (pool.WATCHDOG.renderer_rss_mb + 1) * 1024 * 1024
    renderer.arguments = ["chromium", "--type=utility"]
    pool._watch_renderers(target, 0)
    assert renderer.killed == 0
    renderer.arguments = ["chromium", "--type=renderer"]
    pool._watch_renderers(target, 1)
    assert renderer.killed == 1
    assert pool.get_pool_snapshot()["renderer_kills"] == {"cpu": 0, "rss": 1}
    assert list(target.crawler_strategy.browser_manager.renderer_kills) == [
        RendererKill(1, "rss", renderer.pid, 1234, renderer.rss, 805306368)
    ]
    assert (
        "pid=42 reason=rss browser_pid=1234 created_at=123.0 "
        f"rss_bytes={renderer.rss} rss_limit_bytes=805306368"
    ) in caplog.text


@pytest.mark.parametrize("reason", ["cpu", "rss"])
def test_renderer_kill_attribution_requires_matching_owner_and_consumes_once(
    pool, renderer_tree, reason
):
    from crawl4ai import BrowserConfig
    from crawl4ai.browser_manager import BrowserManager

    target, renderer = renderer_tree
    manager = BrowserManager(BrowserConfig(headless=True, sleep_on_close=False))
    manager.browser_pid = 1234
    target.crawler_strategy.browser_manager = manager
    killed_at = pool.WATCHDOG.renderer_cpu_seconds
    if reason == "cpu":
        pool._watch_renderers(target, 0)
        renderer.cpu = killed_at
    else:
        renderer.rss = (pool.WATCHDOG.renderer_rss_mb + 1) * 1024 * 1024
    pool._watch_renderers(target, killed_at)

    assert renderer.killed == 1
    record = RendererKill(
        killed_at, reason, renderer.pid, 1234, renderer.rss, 805306368,
        cpu_cores=1.0 if reason == "cpu" else None,
        cpu_limit_cores=pool.WATCHDOG.renderer_cpu_cores if reason == "cpu" else None,
    )
    assert list(manager.renderer_kills) == [record]
    crashed_at = killed_at + 0.1
    assert manager.attribute_renderer_kill(crashed_at, 1, browser_pid=5678) is None
    assert manager.attribute_renderer_kill(crashed_at, 1) is None
    assert manager.attribute_renderer_kill(killed_at - 0.1, 1, browser_pid=1234) is None
    assert manager.attribute_renderer_kill(killed_at + 1.1, 1, browser_pid=1234) is None
    assert manager.attribute_renderer_kill(crashed_at, 1, browser_pid=1234) == record
    assert manager.attribute_renderer_kill(crashed_at, 1, browser_pid=1234) is None
    assert not manager.renderer_kills


def test_renderer_kill_attribution_ignores_newer_kill_for_another_browser():
    from crawl4ai import BrowserConfig
    from crawl4ai.browser_manager import BrowserManager

    manager = BrowserManager(BrowserConfig(headless=True, sleep_on_close=False))
    cpu_kill = RendererKill(10, "cpu", 42, 1234, 1024, 805306368)
    rss_kill = RendererKill(10.1, "rss", 43, 5678, 805306369, 805306368)
    manager.renderer_kills.extend([cpu_kill, rss_kill])

    assert manager.attribute_renderer_kill(10.3, 1, browser_pid=9999) is None
    assert manager.attribute_renderer_kill(10.3, 1, browser_pid=1234) == cpu_kill
    assert manager.attribute_renderer_kill(10.3, 1, browser_pid=1234) is None
    assert manager.attribute_renderer_kill(10.3, 1, browser_pid=5678) == rss_kill


def test_excess_renderer_kills_do_not_attribute_an_unrelated_crash():
    from crawl4ai import BrowserConfig
    from crawl4ai.browser_manager import BrowserManager

    manager = BrowserManager(BrowserConfig(headless=True, sleep_on_close=False))
    manager.renderer_kills.extend(
        RendererKill(10, "cpu", pid, 1234, 1024, 805306368) for pid in range(42, 54)
    )

    assert manager.attribute_renderer_kill(10.1, 1) is None
    assert manager.attribute_renderer_kill(10.1, 1, browser_pid=1234) is None
    assert len(manager.renderer_kills) == 12


@pytest.mark.parametrize("process_type,recycle_due,drain_pending,recycled", [
    ("--type=gpu-process", False, False, True),   # non-renderer growth
    ("--type=renderer", False, False, False),     # one heavy page, under renderer_rss_mb
    ("--type=gpu-process", True, False, True),    # session-deferred page-count recycle
    ("--type=gpu-process", True, True, False),    # its drain already runs
])
def test_non_renderer_rss_over_cap_drains_and_restarts(
    pool, renderer_tree, process_type, recycle_due, drain_pending, recycled
):
    from prometheus_client import REGISTRY

    target, child = renderer_tree
    manager = target.crawler_strategy.browser_manager
    manager._pages_served, manager._recycle_due = 1, recycle_due
    manager._pool_recycle_pending = drain_pending
    child.arguments = ["chromium", process_type]
    child.rss = (pool.WATCHDOG.browser_rss_mb + 100) * 1024 * 1024
    assert child.rss < pool.WATCHDOG.renderer_rss_mb * 1024 * 1024

    def recycles():
        return REGISTRY.get_sample_value(
            "crawl4ai_browser_recycles_total", {"reason": "browser_tree_rss"}) or 0

    before = recycles()
    pool._watch_renderers(target, 0)
    assert child.killed == 0
    assert (getattr(manager, "pool_recycle_reason", None) == "browser process tree rss") is recycled
    assert recycles() - before == (1 if recycled else 0)


@pytest.mark.parametrize("field,value", [
    ("renderer_cpu_cores", True), ("renderer_cpu_seconds", 0),
    ("renderer_rss_mb", 1.5), ("consecutive_failure_limit", -1),
    ("interval_seconds", float("inf")), ("browser_probe_timeout_seconds", "5"),
])
def test_watchdog_settings_reject_invalid_values(field, value):
    config = {"crawler": {"pool": {"watchdog": {field: value}}}}
    with pytest.raises(ValueError, match=field):
        BrowserWatchdogSettings.from_config(config)


def test_server_resource_settings_and_defaults():
    assert BrowserWatchdogSettings.from_config({}).renderer_rss_mb == 768
    assert CrawlResourceSettings.from_config({}).max_iframes == 20
    with pytest.raises(ValueError, match="max_iframes"):
        CrawlResourceSettings.from_config({"crawler": {"base_config": {"max_iframes": 0}}})


@pytest.mark.parametrize("requested,expected", [(50, 20), (7, 7)])
def test_api_clamps_iframe_budget_and_preserves_stricter_request(requested, expected):
    from api import apply_server_crawler_defaults
    from crawl4ai import CrawlerRunConfig

    run_config = CrawlerRunConfig(max_iframes=requested, page_termination_timeout=10)
    configured = {"crawler": {"base_config": {
        "max_iframes": 20, "page_termination_timeout": 5.0,
    }}}
    apply_server_crawler_defaults(
        run_config,
        {"type": "CrawlerRunConfig", "params": {"max_iframes": requested}},
        configured,
    )
    assert run_config.max_iframes == expected
    assert run_config.page_termination_timeout == 5.0


def test_api_applies_lower_server_iframe_ceiling_to_omitted_field():
    from api import apply_server_crawler_defaults
    from crawl4ai import CrawlerRunConfig

    run_config = CrawlerRunConfig()
    apply_server_crawler_defaults(
        run_config, {}, {"crawler": {"base_config": {"max_iframes": 4}}},
    )
    assert run_config.max_iframes == 4


@pytest.mark.asyncio
async def test_probe_failure_during_browser_restart_does_not_retire_replacement(pool):
    target = crawler()
    manager = target.crawler_strategy.browser_manager
    old_browser = manager.browser

    async def restart_during_probe(_command):
        manager.browser = crawler().crawler_strategy.browser_manager.browser
        raise PlaywrightError("Target closed")

    old_browser.new_browser_cdp_session.return_value.send = restart_during_probe
    pool.HOT_POOL["restarted"] = target
    manager._docker_health_browser = old_browser
    manager._docker_health_failures = 2
    manager._docker_health_session = None
    await pool._watchdog_pass()
    assert manager.browser is not old_browser
    assert pool.HOT_POOL["restarted"] is target
    target.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_disposal_rechecks_browser_identity_after_waiting_for_lock(pool):
    target = crawler()
    manager = target.crawler_strategy.browser_manager
    old_browser = manager.browser
    pool.HOT_POOL["restarted"] = target
    async with pool.LOCK:
        disposal = asyncio.create_task(pool._dispose_unhealthy(target, old_browser))
        await asyncio.sleep(0)
        manager.browser = crawler().crawler_strategy.browser_manager.browser
    await disposal
    assert pool.HOT_POOL["restarted"] is target
    target.close.assert_not_awaited()
