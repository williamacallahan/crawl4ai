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

The suite is split into:
  - browser-free AST tests (no Playwright, no browser; fail in <0.1s on buggy
    code; fully version-independent of Playwright's screenshot-handling)
    validating the formula's arithmetic and the presence of the wide-page
    guard. These pin the arithmetic defect directly and survive any future
    change to Playwright's screenshot pipeline.
  - end-to-end browser tests that decode the resulting PNG and assert the
    screenshot covers the full content height of the page (100% vertical
    coverage - the bug's thesis is content loss, so the regression must pin
    content completeness, not a loose threshold).

Run:
    .venv/bin/python -m pytest tests/regression/test_adjust_viewport_screenshot_formula.py -v
"""

import ast
import base64
import inspect
import io
import os
import tempfile

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
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
# Browser-free AST tests (no Playwright, no browser; <0.1s, version-independent)
# ---------------------------------------------------------------------------

def _extract_target_height_assignment() -> ast.Assign:
    """Locate the first `target_height = <expr>` assignment inside
    `_crawl_web` and return the AST Assign node so the test can validate the
    arithmetic structure directly (operand order, presence of `0.95`)."""
    src = inspect.cleandoc(inspect.getsource(
        AsyncPlaywrightCrawlerStrategy._crawl_web))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "target_height":
                    return node
    raise AssertionError(
        "No `target_height = ...` assignment found in _crawl_web source; "
        "the adjust_viewport_to_content branch must compute target_height."
    )


def _eval_target_height(node: ast.Assign, target_width: int,
                        page_width: int, page_height: int) -> int:
    """Evaluate the RHS of the located assignment with the given inputs.

    The expression is evaluated directly from the AST (not via exec) so the
    test stays sensitive to operand order and the `0.95` constant."""
    expr = ast.Expression(body=node.value)
    return eval(compile(expr, "<target_height>", mode="eval"),
                {"__builtins__": {"int": int}},
                {"target_width": target_width,
                 "page_width": page_width,
                 "page_height": page_height})


def test_formula_arithmetic_preserves_page_aspect_ratio():
    """The target_height formula must preserve the page aspect ratio when
    reprojected onto `target_width`.

    Bug: `target_width * page_width / page_height * 0.95` swaps operands; for
         tall pages this yields a tiny viewport height (1080x8000 -> 138).
    Fix: `target_width * page_height / page_width` produces a viewport whose
         height is proportional to the page content height (1080x8000 -> 8000).
    """
    node = _extract_target_height_assignment()
    # Document case: page_width <= target_width.
    case_matrix = [
        # (target_width, page_width, page_height, expected_fixed)
        (1080, 1080, 8000,  8000),
        (1080, 1080, 25049, 25049),
        (1080, 1080, 2848,  2848),
        (1080, 960,  5000,  5625),   # narrower than viewport -> taller viewport
    ]
    for tw, pw, ph, expected in case_matrix:
        got = _eval_target_height(node, tw, pw, ph)
        assert got == expected, (
            f"target_height for (target_width={tw}, page_width={pw}, "
            f"page_height={ph}) = {got}, expected {expected} (the "
            f"aspect-ratio-preserving value). The formula must compute "
            f"target_width * page_height / page_width (no `0.95`)."
        )


def test_recommended_fix_includes_wide_page_guard():
    """The wide-page guard is required so the bare operand-swap does not
    regress pages wider than the configured viewport (for `page_width >
    target_width` the unguarded formula yields `target_height < page_height`,
    truncating the page vertically as well as horizontally).

    This test pins the guard at the AST level so a maintainer applying the
    bare one-liner `int(target_width * page_height / page_width)` cannot
    silently ship the wide-page regression.
    """
    src = inspect.getsource(AsyncPlaywrightCrawlerStrategy._crawl_web)
    has_if_guard = (
        "if target_height < page_height" in src
        and "target_height = page_height" in src
    )
    has_max_guard = "max(" in src and "page_height" in src
    assert has_if_guard or has_max_guard, (
        "Neither `if target_height < page_height: target_height = page_height` "
        "nor a `max(target_height, page_height)` wide-page guard is present "
        "in _crawl_web source; the bare operand-swap without the guard would "
        "vertically truncate pages wider than the configured viewport."
    )
    # Wide-page matrix: page_width > target_width. The bare swap underflows;
    # the guard clamps to page_height (no loss).
    for tw, pw, ph in [(1080, 2000, 800), (1080, 3000, 600)]:
        bare = int(tw * ph / pw)
        guarded = max(bare, ph)
        assert bare < ph, (
            f"Precondition failed for ({tw=}, {pw=}, {ph=}): the bare "
            f"swap must shrink the viewport on wide pages (got bare={bare}, "
            f"which is not less than page_height={ph}); the wide-page test "
            f"matrix is mis-configured."
        )
        assert guarded == ph, (
            f"Guarded target_height for ({tw=}, {pw=}, {ph=}) = {guarded}, "
            f"expected {ph} (guard must prevent vertical truncation on wide "
            f"pages)."
        )


def test_formula_arithmetic_tall_page_yields_tall_viewport():
    """Doubling page_height must (at least) *enlarge* target_height — the buggy
    formula inverts this (a doubly-tall page *halves* target_height)."""
    node = _extract_target_height_assignment()
    tw = pw = 1080
    h1 = _eval_target_height(node, tw, pw, 4000)
    h2 = _eval_target_height(node, tw, pw, 8000)
    # Fixed: 4000 -> 8000 (doubles).  Buggy: 275 -> 138 (inverts; 138 < 275).
    assert h2 > h1, (
        f"Doubling page_height must increase target_height (got {h1}->{h2}); "
        f"the buggy formula inverts this and made tall pages yield tiny "
        f"viewports, which is the truncation root cause."
    )
    assert h2 >= 2 * h1 * 0.95, (
        f"Doubling page_height should ~double target_height: got {h1}->{h2}."
    )


# ---------------------------------------------------------------------------
# End-to-end browser tests (Playwright, decode PNG, assert 100% coverage)
# ---------------------------------------------------------------------------

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

