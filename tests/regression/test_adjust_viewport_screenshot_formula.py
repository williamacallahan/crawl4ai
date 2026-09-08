"""
Regression tests for the `adjust_viewport_to_content` screenshot-truncation bug.

Bug (introduced in c51e901f, 2024-12-08): the formula
    target_height = int(target_width * page_width / page_height * 0.95)
swapped `page_width` and `page_height`, producing a *tiny* viewport for any
page taller than it is wide. `page.screenshot(full_page=False)` then honoured
Playwright's cached viewport size (set by `page.set_viewport_size`) and
captured only the top strip of the page (e.g. a 1080x8000 page rendered as
1080x138, hiding ~98% of the content). See
docs/md_v2/api/parameters.md:172 ("Resizes viewport to match page content
height") for the documented intent the buggy formula violated.

Fix: preserve the page aspect ratio and add a wide-page guard so the bare
operand swap does not regress pages wider than the configured viewport:
    target_height = int(target_width * page_height / page_width)
    if target_height < page_height:
        target_height = page_height

The browser tests decode the resulting PNG and assert it covers the full
content height. They run only in the browser lane because the regression is
visible screenshot output, not the source expression used to produce it.

Run:
    .venv/bin/python -m pytest tests/regression/test_adjust_viewport_screenshot_formula.py -v
"""

import base64
import io
import os
import tempfile

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

_STRIP_COLORS = [
    "#f0a0a0", "#a0f0a0", "#a0a0f0", "#f0f0a0",
    "#f0a0f0", "#a0f0f0", "#e0c080", "#c080e0",
]


def _tall_stripped_html(page_height: int = 8000, page_width: int = 1080) -> str:
    """Build a tall page split into 8 vertically-stacked sections, each with a
    distinct solid background color. The number of distinguished colors in the
    decoded screenshot tells us how much of the page was captured, and the
    exact `scrollHeight` (== `page_height`) lets us assert 100% coverage."""
    strip_count = 8
    strip_height = page_height // strip_count
    strips = []
    for i in range(strip_count):
        strips.append(
            f'<section style="width:{page_width}px;height:{strip_height}px;'
            f'background:{_STRIP_COLORS[i]};margin:0;padding:0;border:0;'
            f'overflow:hidden">'
            f'<h2>Section {i + 1}</h2>'
            f'<p>This is paragraph {i + 1} of the tall viewport-adjustment '
            f'fixture, written with enough visible text content to clear the '
            f'antibot detector structural-integrity threshold and produce a '
            f'successful crawl result against the synthetic page.</p>'
            f'</section>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Tall Page Fixture</title></head>
<body style="margin:0;padding:0;overflow:hidden">
{"".join(strips)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_html(html: str):
    """Write `html` to a temp file and return (url, path). The caller removes
    `path` in a finally block."""
    fd, path = tempfile.mkstemp(suffix=".html", prefix="c4ai_viewport_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(html)
    return f"file://{path}", path


def _decode_png_size(b64: str):
    """Decode a base64 PNG and return (width, height)."""
    from PIL import Image
    raw = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw)) as img:
        return img.size  # (width, height)


# ---------------------------------------------------------------------------
# End-to-end browser tests (Playwright, decode PNG, assert 100% coverage)
# ---------------------------------------------------------------------------

@pytest.mark.browser
@pytest.mark.asyncio
async def test_adjust_viewport_to_content_screenshot_covers_page():
    """scan_full_page=False (the default dispatch path): the screenshot PNG
    must cover the full content height of a tall static page.

    Buggy code yields 1080x138 for an 8000px-tall fixture (~1.7% coverage);
    the fix must produce a viewport tall enough to capture the full page.
    """
    html = _tall_stripped_html(page_height=8000, page_width=1080)
    url, path = _write_temp_html(html)
    try:
        browser_config = BrowserConfig(
            headless=True, verbose=False,
            viewport_width=1080, viewport_height=600,
        )
        run_config = CrawlerRunConfig(
            screenshot=True,
            adjust_viewport_to_content=True,
            scan_full_page=False,
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=0,
            exclude_external_links=True,
            verbose=False,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        assert result.success, f"Crawl failed: {result.error_message}"
        assert result.screenshot, "Screenshot should be captured"
        width, height = _decode_png_size(result.screenshot)
        assert height >= 8000, (
            f"truncated screenshot (scan_full_page=False): {width}x{height} "
            f"covers only {height / 8000:.1%} of the 8000px page; the fix must "
            f"make the viewport tall enough to capture the full content."
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.browser
@pytest.mark.asyncio
async def test_adjust_viewport_to_content_screenshot_scan_full_page_true():
    """scan_full_page=True: the assertion encodes the user-visible coverage
    contract (the screenshot must cover the full static page), stable across
    Playwright versions. The mechanism that produces the capture may differ
    across driver versions (naive top-crop under the corrected formula on
    1.53.0, where the CDP override persists across `cdp.detach()`; rescue
    scroller on a future Playwright that clears the override on detach), but
    the contract this test pins is the contract the docs describe: 'Resizes
    viewport to match page content height'.
    """
    html = _tall_stripped_html(page_height=8000, page_width=1080)
    url, path = _write_temp_html(html)
    try:
        browser_config = BrowserConfig(
            headless=True, verbose=False,
            viewport_width=1080, viewport_height=600,
        )
        run_config = CrawlerRunConfig(
            screenshot=True,
            adjust_viewport_to_content=True,
            scan_full_page=True,
            scroll_delay=0.2,
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=0,
            exclude_external_links=True,
            verbose=False,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        assert result.success, f"Crawl failed: {result.error_message}"
        assert result.screenshot, "Screenshot should be captured"
        width, height = _decode_png_size(result.screenshot)
        assert height >= 8000, (
            f"truncated screenshot (scan_full_page=True): {width}x{height} "
            f"covers only {height / 8000:.1%} of the 8000px page; the fix "
            f"must cover the full content height on the scan_full_page=True "
            f"dispatch path too."
        )
    finally:
        if os.path.exists(path):
            os.remove(path)
