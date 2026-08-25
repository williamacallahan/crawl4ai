import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock

from crawl4ai import BrowserConfig, MemoryAdaptiveDispatcher

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api
import crawler_pool


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


def test_browser_pool_evicts_least_recent_idle_browser(monkeypatch):
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

    asyncio.run(crawler_pool._make_browser_capacity())

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
        asyncio.run(crawler_pool._make_browser_capacity())
    except RuntimeError as error:
        assert str(error) == "Crawler browser pool is at capacity"
    else:
        raise AssertionError("Expected browser pool capacity rejection")
