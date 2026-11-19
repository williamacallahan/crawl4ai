"""A server browser's V8 heap cap sits above the renderer memory watchdog.

A 512 MiB cap below the 768 MiB watchdog crashed JS-heavy renderers as an
unclassified "Target page crashed" ERROR (infra#732) instead of letting the
watchdog end them as a typed renderer_memory_limit WARNING.
"""

from crawl4ai import BrowserConfig
from crawl4ai.browser_manager import BrowserManager

import api
from utils import BrowserWatchdogSettings, load_config


def test_server_launch_args_cap_the_heap_above_the_renderer_watchdog(monkeypatch):
    monkeypatch.delenv("CRAWL4AI_CHROMIUM_SANDBOX", raising=False)
    shipped = load_config()

    launch_args = BrowserManager(
        api._apply_server_browser_policy(BrowserConfig(headless=True), shipped)
    )._build_browser_args()["args"]

    (js_flags,) = [arg for arg in launch_args if arg.startswith("--js-flags=")]
    heap_cap_mb = int(js_flags.rsplit("=", 1)[1])
    assert heap_cap_mb > BrowserWatchdogSettings.from_config(shipped).renderer_rss_mb
    assert "--aggressive-cache-discard" in launch_args
