"""The synthetic capability cookie is origin-scoped, not derived from a page path."""

from unittest.mock import AsyncMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager


@pytest.mark.asyncio
@pytest.mark.parametrize("url,origin", [
    ("https://example.com/reports/a;b/index.html?x=1#part", "https://example.com/"),
    ("https://example.com:8443/deep/path", "https://example.com:8443/"),
])
async def test_default_cookie_uses_origin_instead_of_document_path(url, origin):
    manager = BrowserManager(BrowserConfig())
    context = AsyncMock()

    await manager.setup_context(context, CrawlerRunConfig(url=url))

    context.add_cookies.assert_awaited_once_with([
        {"name": "cookiesEnabled", "value": "true", "url": origin},
    ])
