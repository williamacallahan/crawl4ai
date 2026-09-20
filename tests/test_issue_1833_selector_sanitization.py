"""Regression tests for the `cleaned_html` sanitization contract under
``css_selector`` / ``target_elements``.

Selected content must be scoped using its original attributes, then have
scripts, styles, empty nodes and unwanted attributes removed.
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
    """Data retention applies to selected output; event handlers stay stripped."""
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


@pytest.mark.parametrize("selector", ['[data-region="article"]', '[role="main"]'])
@pytest.mark.parametrize("mode", ["css", "targets", "combined"])
def test_attribute_selectors_scope_cleaned_output(scraper, selector, mode):
    html = """<body><aside><a href="/outside">Outside prose</a></aside>
      <article data-region="article" role="main" onclick="bad()">
        <p>Selected article text</p><script>bad()</script></article></body>"""
    options = {"css_selector": selector} if mode == "css" else {"target_elements": [selector]}
    if mode == "combined":
        options["css_selector"] = "body"
    result = scraper._scrap("https://example.com", html, **options)
    cleaned = result["cleaned_html"]
    assert "Selected article text" in cleaned
    assert "Outside prose" not in cleaned
    assert "data-region" not in cleaned
    assert "role=" not in cleaned
    assert "onclick" not in cleaned
    assert "<script" not in cleaned
    assert result["links"]["internal"][0]["href"] == "https://example.com/outside"


@pytest.mark.parametrize("options", [
    {"css_selector": "strong[data-region]"},
    {"target_elements": ["strong[data-region]"]},
])
def test_only_text_runs_after_selector_matching(scraper, options):
    html = """<body><p>Outside prose</p><strong data-region="article">
      Selected <em>nested text</em></strong></body>"""
    result = scraper._scrap("https://example.com", html, only_text=True, **options)
    cleaned = result["cleaned_html"]
    assert "Selected" in cleaned
    assert "nested text" in cleaned
    assert "Outside prose" not in cleaned
    assert "<strong" not in cleaned
    assert "<em" not in cleaned
    assert "data-region" not in cleaned
