"""Screenshots stay within the layout viewport and never picture their own failure.

A full-page capture sized its viewport from the page's scroll extent, so a
sideways carousel made a 20000 px wide capture and an endless page a canvas
held whole in memory. A failed capture returned an 800x600 JPEG of the error
text, which callers stored as the page's screenshot.
"""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from crawl4ai import BrowserConfig
from crawl4ai.async_configs import _MAX_RENDER_PIXELS
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


def _strategy(page_width, page_height):
    strategy = object.__new__(AsyncPlaywrightCrawlerStrategy)
    strategy.logger = MagicMock()
    strategy._page_crash_errors = {}
    strategy.get_page_dimensions = AsyncMock(
        return_value={"width": page_width, "height": page_height}
    )
    return strategy


def _page(width, height):
    """A page whose screenshots are the size of its current viewport."""
    page = MagicMock()
    page.viewport_size = {"width": width, "height": height}

    async def resize(size):
        page.viewport_size = dict(size)

    async def screenshot(**_kwargs):
        buffer = BytesIO()
        Image.new("RGB", (page.viewport_size["width"], page.viewport_size["height"])).save(
            buffer, format="JPEG"
        )
        return buffer.getvalue()

    page.set_viewport_size = AsyncMock(side_effect=resize)
    page.screenshot = AsyncMock(side_effect=screenshot)
    page.evaluate = AsyncMock(return_value=None)
    return page


@pytest.mark.asyncio
async def test_full_page_capture_keeps_the_layout_width_and_the_pixel_budget():
    page = _page(1280, 800)
    strategy = _strategy(page_width=20_000, page_height=50_000)

    with patch("crawl4ai.async_crawler_strategy.asyncio.sleep", new_callable=AsyncMock):
        encoded = await strategy.take_screenshot_scroller(page, screenshot_height_threshold=4000)

    image = Image.open(BytesIO(base64.b64decode(encoded)))
    assert image.width == 1280
    assert image.height == _MAX_RENDER_PIXELS // 1280
    assert max(call.args[0]["width"] for call in page.set_viewport_size.call_args_list) == 1280
    assert page.viewport_size == {"width": 1280, "height": 800}


@pytest.mark.asyncio
async def test_a_short_narrow_page_is_captured_whole():
    page = _page(1280, 800)
    strategy = _strategy(page_width=1280, page_height=2000)

    with patch("crawl4ai.async_crawler_strategy.asyncio.sleep", new_callable=AsyncMock):
        encoded = await strategy.take_screenshot_scroller(page)

    image = Image.open(BytesIO(base64.b64decode(encoded)))
    assert (image.width, image.height) == (1280, 2000)


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", ["take_screenshot_naive", "take_screenshot_scroller"])
async def test_a_failed_capture_returns_no_screenshot(capture):
    page = _page(1280, 800)
    page.screenshot = AsyncMock(side_effect=RuntimeError("Target closed"))
    strategy = _strategy(page_width=1280, page_height=4000)

    with patch("crawl4ai.async_crawler_strategy.asyncio.sleep", new_callable=AsyncMock):
        assert await getattr(strategy, capture)(page) is None


def test_a_partial_viewport_keeps_the_named_other_dimension():
    config = BrowserConfig(viewport={"width": 1920}, viewport_height=1080)

    assert (config.viewport_width, config.viewport_height) == (1920, 1080)
