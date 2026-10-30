from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error

from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

NAVIGATING = Error(
    "Page.content: Unable to retrieve content because the page is navigating "
    "and changing the content."
)


@pytest.mark.asyncio
async def test_client_redirect_chain_reads_the_document_it_settles_on():
    # Production refused twice in one crawl: each hop of a client redirect
    # chain replaces the document while the load-state wait has already resolved.
    page = MagicMock()
    page.content = AsyncMock(side_effect=[NAVIGATING, NAVIGATING, "<html>final</html>"])
    page.wait_for_load_state = AsyncMock()

    assert await AsyncPlaywrightCrawlerStrategy._settled_content(page) == "<html>final</html>"


@pytest.mark.asyncio
async def test_other_content_failures_are_not_retried():
    page = MagicMock()
    page.content = AsyncMock(side_effect=Error("Target page crashed"))
    page.wait_for_load_state = AsyncMock()

    with pytest.raises(Error, match="Target page crashed"):
        await AsyncPlaywrightCrawlerStrategy._settled_content(page)
