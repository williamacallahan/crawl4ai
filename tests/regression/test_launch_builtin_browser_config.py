"""Regression test for launch_builtin_browser constructing ManagedBrowser.

Guards the fix for `crwl browser start` crashing: launch_builtin_browser
previously passed legacy kwargs (browser_type=..., debugging_port=...) instead
of a BrowserConfig, and ManagedBrowser.__init__ dereferences
browser_config.browser_type, raising AttributeError before any browser
started. Second occurrence of this bug class (launch_standalone_browser was
fixed in 260e2dc). Runs no real browser.
"""

import pytest

from crawl4ai.async_configs import BrowserConfig
from crawl4ai.browser_profiler import BrowserProfiler


@pytest.mark.asyncio
async def test_launch_builtin_browser_passes_browser_config(monkeypatch, tmp_path):
    captured = {}

    class FakeManagedBrowser:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.browser_process = None

        async def start(self):
            return None

    monkeypatch.setattr(
        "crawl4ai.browser_profiler.ManagedBrowser", FakeManagedBrowser
    )

    profiler = BrowserProfiler()
    # Point builtin-browser state at the tmp dir so no prior browser is found
    # and nothing is written to the real home folder.
    profiler.builtin_browser_dir = str(tmp_path)
    profiler.builtin_config_file = str(tmp_path / "browser_config.json")

    result = await profiler.launch_builtin_browser(
        browser_type="chromium", debugging_port=9455, headless=True
    )

    # The fake exposes no browser process, so launch reports failure — but
    # only after construction, which is the surface under test.
    assert result is None
    config = captured.get("browser_config")
    assert isinstance(config, BrowserConfig)
    assert config.browser_type == "chromium"
    assert config.debugging_port == 9455
    assert config.headless is True
    assert config.user_data_dir == str(tmp_path / "user_data")
