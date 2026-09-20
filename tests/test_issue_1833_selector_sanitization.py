"""Regression tests for the `cleaned_html` sanitization contract under
``css_selector`` / ``target_elements``.

Guards against re-introducing the defect where
``LXMLWebScrapingStrategy._scrap`` built ``content_element`` via
``copy.deepcopy`` *before* the script/style/empty-element/attribute cleaning
steps (which run on ``body``), then serialized the stale, un-cleaned
``content_element`` into ``cleaned_html``. With the fix, ``content_element``
is constructed from the already-cleaned ``body``.
"""

import pytest
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


SAMPLE_HTML = """<html><head><title>T</title>
  <style>body { font-family: sans-serif; }</style>
  <noscript>enable js</noscript>
</head><body>
  <script>console.log("top-level");</script>
  <div class="sidebar" data-tracking-id="xyz" onclick="evil()">
    <p>Sidebar content with enough words to pass threshold easily here.</p>
    <script>console.log("inside sidebar");</script>
  </div>
  <div class="main-content" data-tracking-id="abc123" onclick="evil()">
    <p>Main paragraph one with enough words to pass threshold easily here.</p>
    <script>console.log("inside main content");</script>
    <style>.inner { font-weight: bold; }</style>
    <noscript>main ns</noscript>
    <div class="empty-div"></div>
    <a href="https://example.com/page" data-track="hit">link</a>
  </div>
  <div class="footer">Footer content here</div>
</body></html>"""


@pytest.fixture
def scraper():
    return LXMLWebScrapingStrategy()


def _assert_sanitized(ch):
    assert "<script" not in ch
    assert "</script>" not in ch
    assert "console.log" not in ch
    assert "<style" not in ch
    assert "</style>" not in ch
    assert "<noscript" not in ch
    assert "onclick" not in ch
    assert "data-tracking-id" not in ch
    assert "data-track" not in ch
    assert 'class="empty-div"' not in ch


@pytest.mark.parametrize(
    "kwargs",
    [
        {"css_selector": ".main-content"},
        {"target_elements": [".main-content"]},
        {"css_selector": ".main-content", "target_elements": ["p"]},
    ],
)
def test_selector_paths_produce_sanitized_cleaned_html(scraper, kwargs):
    """css_selector / target_elements paths must strip <script>/<style>/<noscript>,
    inline event-handler attributes, tracking data-* attributes, and empty
    elements - same contract as the no-selector path."""
    res = scraper._scrap("https://example.com", SAMPLE_HTML, **kwargs)
    _assert_sanitized(res["cleaned_html"])


def test_no_selector_path_still_sanitized(scraper):
    """Baseline regression guard: the no-selector path must remain sanitized."""
    res = scraper._scrap("https://example.com", SAMPLE_HTML)
    _assert_sanitized(res["cleaned_html"])


def test_selector_paths_preserve_content_restriction(scraper):
    """The fix is sanitization-only; filtering must still restrict output to
    the matched region (content outside the match must not appear)."""
    res = scraper._scrap(
        "https://example.com", SAMPLE_HTML, css_selector=".main-content"
    )
    ch = res["cleaned_html"]
    assert "Main paragraph one" in ch
    assert "Sidebar content" not in ch
    assert "Footer content" not in ch


def test_keep_data_attributes_honored_under_selector(scraper):
    """remove_unwanted_attributes_fast runs on body before the selector
    extraction, so keep_data_attributes=True must still preserve data-*
    attributes in the selector-path output (and on* handlers stay stripped)."""
    res = scraper._scrap(
        "https://example.com",
        SAMPLE_HTML,
        css_selector=".main-content",
        keep_data_attributes=True,
    )
    ch = res["cleaned_html"]
    assert "data-tracking-id" in ch
    assert "<script" not in ch
    assert "<style" not in ch
    assert "onclick" not in ch
