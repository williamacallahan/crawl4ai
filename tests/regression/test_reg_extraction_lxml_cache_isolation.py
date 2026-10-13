"""JsonLxmlExtractionStrategy result-cache must not leak across pages.

Before the fix, ``JsonLxmlExtractionStrategy`` memoized per-element selector
results in ``_result_cache`` under a key derived from
``element.get('id', '') or str(hash(element))`` plus the selector string.
HTML ``id`` attributes are unique only *within* a single page, so when one
strategy instance was reused across pages (the default ``arun``/``arun_many``
behavior via ``CrawlerRunConfig.extraction_strategy``) and two pages' matched
base/child elements shared an ``id``, the second page silently read cached
results produced from the first page's lxml parse tree -- including live lxml
``Element`` object references -- corrupting ``CrawlResult.extracted_content``
with field values from the wrong page.

The fix:
  * ``JsonLxmlExtractionStrategy.extract`` calls ``_clear_caches()`` before
    every page so the element-result cache cannot outlive a single parse.
  * The cache key is now scoped to the element's parse tree
    (``id(element.getroottree().getroot())``) so page A and page B cannot
    share an entry even when their matched elements carry the same ``id``.

These tests isolate the cache mechanism from the crawler (no-browser unit
tests driving ``extract`` directly) and then verify the public ``arun_many``
path end-to-end against a real local server.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_reg_extraction_lxml_cache_isolation.py

    # No-browser subset (runs in the CI unit gate):
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_reg_extraction_lxml_cache_isolation.py -m "not browser"
"""
import socket

import pytest
from aiohttp import web

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.extraction_strategy import JsonLxmlExtractionStrategy


# Schema whose base element is selected by class but the matched element
# *carries* an id (the real bug surface -- the id need not be selected by #id).
_SCHEMA_CLASS_BASE_WITH_ID = {
    "baseSelector": "div.product-list",
    "fields": [
        {"name": "title", "selector": "h2.title", "type": "text"},
        {"name": "price", "selector": "span.price", "type": "text"},
    ],
}


def _page(title: str, price: str, *, base_id: str = "product-list") -> str:
    """A page whose base element carries ``base_id`` and the given values."""
    return (
        "<html><head><title>Test</title></head><body>"
        "<p>Some genuinely visible body text so the anti-bot minimal-text "
        "check is satisfied on the end-to-end crawl path.</p>"
        f'<div class="product-list" id="{base_id}">'
        f'<h2 class="title">{title}</h2>'
        f'<span class="price">{price}</span>'
        "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Unit-level isolation (no browser) -- runs in the CI unit gate.
# ---------------------------------------------------------------------------


def test_lxml_cache_isolates_pages_with_shared_base_id():
    """Reusing one JsonLxmlExtractionStrategy across two pages whose matched
    base elements share an ``id`` must return each page's own field values.

    Before the fix, page B silently returned page A's values because the
    cache key ``f"{element_id}::{selector_str}"`` collided across pages.
    """
    strategy = JsonLxmlExtractionStrategy(schema=_SCHEMA_CLASS_BASE_WITH_ID)
    result_a = strategy.extract(url="", html_content=_page("Alpha Title", "$10.00"))
    result_b = strategy.extract(url="", html_content=_page("Beta Title", "$30.00"))

    assert result_a == [{"title": "Alpha Title", "price": "$10.00"}], result_a
    assert result_b == [{"title": "Beta Title", "price": "$30.00"}], result_b


def test_lxml_cache_does_not_leak_element_object_identity():
    """The tree-scoped cache key prevents a cached element from a previous
    page's parse from being handed back for a different page even when the
    strategy's ``extract``-level cache clear is bypassed (i.e. direct
    ``_get_elements`` calls). Two pages whose base elements share an ``id``
    must each return their own element, not a stale element from the other
    page's tree.

    This isolates the per-tree key (defense-in-depth) from the per-extract
    cache clear: we never call ``extract`` here, so ``_clear_caches`` is never
    triggered -- only the tree-scoped key can prevent the collision."""
    strategy = JsonLxmlExtractionStrategy(schema=_SCHEMA_CLASS_BASE_WITH_ID)

    # Page A: select and remember the actual element object returned.
    parsed_a = strategy._parse_html(_page("Alpha Title", "$10.00"))
    base_a = strategy._get_base_elements(parsed_a, "div.product-list")
    titles_a = strategy._get_elements(base_a[0], "h2.title")
    assert titles_a and titles_a[0].text == "Alpha Title"
    alpha_element = titles_a[0]

    # Page B: same base id, different values. Must NOT reuse page A's element.
    parsed_b = strategy._parse_html(_page("Beta Title", "$30.00"))
    base_b = strategy._get_base_elements(parsed_b, "div.product-list")
    titles_b = strategy._get_elements(base_b[0], "h2.title")
    assert titles_b and titles_b[0].text == "Beta Title", (
        f"page B returned stale page A element text: {titles_b[0].text!r}"
    )
    # The element object handed back for page B must be a distinct object
    # from page A's cached element (lxml proxy identity is stable within a
    # parse, so distinct proxies here means distinct underlying nodes).
    assert titles_b[0] is not alpha_element, (
        "page B selection returned the exact lxml element cached from page A"
    )


def test_lxml_cache_isolation_class_selector_matching_id_bearing_element():
    """The trigger surface is 'matched element has an id', not 'selected by
    #id'. A class baseSelector that matches an element which merely *carries*
    an id must still isolate across pages."""
    schema = {
        "baseSelector": "div.product-list",
        "fields": [
            {"name": "title", "selector": "h2.title", "type": "text"},
        ],
    }
    strategy = JsonLxmlExtractionStrategy(schema=schema)
    html_a = (
        '<html><body><div class="product-list" id="main">'
        '<h2 class="title">Alpha</h2></div></body></html>'
    )
    html_b = (
        '<html><body><div class="product-list" id="main">'
        '<h2 class="title">Beta</h2></div></body></html>'
    )
    result_a = strategy.extract(url="", html_content=html_a)
    result_b = strategy.extract(url="", html_content=html_b)
    assert result_a[0]["title"] == "Alpha", result_a
    assert result_b[0]["title"] == "Beta", result_b


def test_lxml_cache_cleared_between_extracts():
    """Every ``extract`` call must clear the element-result cache before doing
    any selection, so cached entries cannot accumulate across pages (memory
    hygiene) and stale element references cannot outlive a single parse.
    Correctness across pages is already provided by the object-identity key,
    but the per-extract clear bounds growth and is the documented contract."""
    strategy = JsonLxmlExtractionStrategy(schema=_SCHEMA_CLASS_BASE_WITH_ID)

    clear_calls = 0
    original_clear_caches = strategy._clear_caches

    def counting_clear():
        nonlocal clear_calls
        clear_calls += 1
        original_clear_caches()

    strategy._clear_caches = counting_clear

    # Three extracts across three different pages -- _clear_caches must fire
    # exactly once per extract (before any selection runs).
    strategy.extract(url="", html_content=_page("Alpha Title", "$10.00"))
    strategy.extract(url="", html_content=_page("Beta Title", "$30.00"))
    strategy.extract(url="", html_content=_page("Gamma Title", "$70.00"))

    assert clear_calls == 3, (
        f"_clear_caches should be called once per extract (3 expected), "
        f"got {clear_calls}"
    )

    # And the cache must not accumulate across pages: after the last extract it
    # holds only the last page's entries (one base element, two field
    # selectors = at most a couple of entries), not three pages' worth.
    assert len(strategy._result_cache) <= 4, (
        f"cache should be bounded to one page's entries, got "
        f"{len(strategy._result_cache)}"
    )


def test_lxml_within_page_caching_still_works():
    """Defense against over-fixing: within a single page, repeated (element,
    selector) queries must still hit the cache so perf is not regressed. With
    an object-identity key, the same element queried twice with the same
    selector produces one cache entry and the second call is a hit."""
    schema = {
        "baseSelector": "div.product-list",
        "fields": [
            {"name": "title", "selector": "h2.title", "type": "text"},
        ],
    }
    strategy = JsonLxmlExtractionStrategy(schema=schema)
    parsed = strategy._parse_html(
        _page("Alpha Title", "$10.00", base_id="shared-id")
    )
    base = strategy._get_base_elements(parsed, "div.product-list")[0]

    strategy._result_cache.clear()  # isolate the within-page cache behavior
    first = strategy._get_elements(base, "h2.title")
    second = strategy._get_elements(base, "h2.title")

    assert first and first == second, "repeated selection must return equal results"
    # Exactly one entry for (base, "h2.title"); the second call hit the cache.
    assert len(strategy._result_cache) == 1, (
        f"repeated (element, selector) query should produce 1 cache entry, "
        f"got {len(strategy._result_cache)}"
    )


def test_lxml_cache_isolates_multiple_base_elements_within_one_page():
    """A page with several base elements that share the same class but carry
    different ids must extract each element's own fields (no within-page
    cross-element collision). Also covers the case where two base elements
    share the same id (invalid HTML, but common) -- they must still each
    return their own values, not a single cached value."""
    schema = {
        "baseSelector": "div.product-list",
        "fields": [
            {"name": "title", "selector": "h2.title", "type": "text"},
        ],
    }
    strategy = JsonLxmlExtractionStrategy(schema=schema)
    html = (
        "<html><body>"
        '<div class="product-list" id="dup"><h2 class="title">First</h2></div>'
        '<div class="product-list" id="dup"><h2 class="title">Second</h2></div>'
        "</body></html>"
    )
    result = strategy.extract(url="", html_content=html)
    titles = [item["title"] for item in result]
    assert titles == ["First", "Second"], result


# ---------------------------------------------------------------------------
# End-to-end through AsyncWebCrawler.arun_many (real browser, real server)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def _start_two_page_server(port: int):
    """Serve /a and /b -- both with <div class="product-list" id="product-list">
    but with DIFFERENT field values."""
    app = web.Application()

    async def page_a(_req):
        return web.Response(
            text=_page("Alpha Title", "$10.00"),
            content_type="text/html",
        )

    async def page_b(_req):
        return web.Response(
            text=_page("Beta Title", "$30.00"),
            content_type="text/html",
        )

    app.router.add_get("/a", page_a)
    app.router.add_get("/b", page_b)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


@pytest.mark.browser
@pytest.mark.asyncio
async def test_lxml_cache_isolation_arun_many_shared_strategy():
    """End-to-end: one JsonLxmlExtractionStrategy shared via
    CrawlerRunConfig.extraction_strategy across two URLs via arun_many. Both
    pages expose <div class="product-list" id="product-list"> with different
    field values. Each CrawlResult.extracted_content must carry its own page's
    values, not the other page's. Before the fix, page B silently returned
    page A's values (silent per-page data corruption through the dispatcher)."""
    port = _free_port()
    runner = await _start_two_page_server(port)
    try:
        base = f"http://127.0.0.1:{port}"
        urls = [f"{base}/a", f"{base}/b"]
        strategy = JsonLxmlExtractionStrategy(schema=_SCHEMA_CLASS_BASE_WITH_ID)
        config = CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        )
        async with AsyncWebCrawler(
            config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])
        ) as crawler:
            results = await crawler.arun_many(urls, config=config)

        by_url = {r.url: r for r in results}
        a = by_url[f"{base}/a"]
        b = by_url[f"{base}/b"]
        assert a.success, f"crawl /a failed: {a.error_message}"
        assert b.success, f"crawl /b failed: {b.error_message}"

        import json

        data_a = json.loads(a.extracted_content)
        data_b = json.loads(b.extracted_content)
        assert data_a == [{"title": "Alpha Title", "price": "$10.00"}], data_a
        assert data_b == [{"title": "Beta Title", "price": "$30.00"}], data_b
    finally:
        await runner.cleanup()
