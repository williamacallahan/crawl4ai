"""Every context crawl4ai creates blocks service-worker registration.

A registered service worker keeps its renderer process alive after every page
of the context closes, so long-lived contexts accumulated one renderer per
service-worker site until the browser died.
"""

from unittest.mock import AsyncMock, MagicMock

import playwright.async_api
import pytest

from crawl4ai import BrowserConfig
from crawl4ai.browser_manager import BrowserManager


@pytest.mark.asyncio
async def test_created_browser_context_blocks_service_workers():
    manager = BrowserManager(BrowserConfig(headless=True, text_mode=True))
    manager.browser = MagicMock(new_context=AsyncMock(return_value=AsyncMock()))

    await manager.create_browser_context()

    _, context_options = manager.browser.new_context.call_args
    assert context_options["service_workers"] == "block"


@pytest.mark.asyncio
async def test_persistent_context_blocks_service_workers(monkeypatch, tmp_path):
    launch_persistent_context = AsyncMock(return_value=AsyncMock())
    driver = MagicMock()
    driver.chromium.launch_persistent_context = launch_persistent_context
    starter = MagicMock(start=AsyncMock(return_value=driver))
    monkeypatch.setattr(playwright.async_api, "async_playwright", lambda: starter)
    manager = BrowserManager(
        BrowserConfig(
            headless=True,
            use_persistent_context=True,
            user_data_dir=str(tmp_path),
        )
    )
    monkeypatch.setattr(manager, "setup_context", AsyncMock())

    await manager.start()

    _, launch_options = launch_persistent_context.call_args
    assert launch_options["service_workers"] == "block"
