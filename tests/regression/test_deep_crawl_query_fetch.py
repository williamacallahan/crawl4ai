"""Deduplication must not rewrite the query sent to the crawl transport."""

from types import SimpleNamespace

import pytest

from crawl4ai import CrawlResult, CrawlerRunConfig
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
)
from crawl4ai.utils import quick_extract_links


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_type", [
    BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy,
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
async def test_query_dedup_preserves_discovered_request(strategy_type, stream, prefetch):
    start = "https://example.com/?z=1&a=2"
    target = "https://example.com/target?z=2&a=1&sig=A%2fb+X&blank="
    hrefs = [
        "/?a=2&z=1",  # The start page is already visited under its canonical key.
        "/target?z=2&a=1&sig=A%2fb+X&blank=",
        "/target?a=1&z=2&sig=A%2fb+X&blank=",
    ]
    links = {"internal": [{"href": href} for href in hrefs], "external": []}
    if prefetch:
        html = "".join(f'<a href="{href}">link</a>' for href in hrefs)
        links = quick_extract_links(html, start)
    fetched = []

    async def arun_many(urls, config):
        results = []
        for url in urls:
            fetched.append(url)
            results.append(CrawlResult(
                url=url, html="", success=True,
                links=links if url == start else {"internal": [], "external": []},
            ))
        if config.stream:
            async def results_stream():
                for result in results:
                    yield result
            return results_stream()
        return results

    strategy = strategy_type(max_depth=2, max_pages=10)
    results = await strategy.arun(
        start, SimpleNamespace(arun_many=arun_many), CrawlerRunConfig(stream=stream),
    )
    if stream:
        results = [result async for result in results]

    assert fetched == [start, target]
    assert [result.url for result in results] == [start, target]
    assert results[1].metadata["depth"] == 1
    assert results[1].metadata["parent_url"] == start


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_type", [
    BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy,
])
async def test_public_shutdown_signals_cancellation(strategy_type):
    strategy = strategy_type(max_depth=1)
    await strategy.shutdown()
    assert strategy.cancelled
    assert strategy.stats.end_time is not None


# ---------------------------------------------------------------------------
# Blank-value query parameter dedup (regression for the dedup-key / fetch-URL
# divergence bug)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_type", [
    BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy,
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("order", [
    ("/page?q=", "/page"),
    ("/page", "/page?q="),
], ids=["blank-first", "bare-first"])
async def test_blank_param_distinct_from_absent_both_fetched(
    strategy_type, stream, prefetch, order,
):
    """Both /page and /page?q= must be fetched when linked from the same page.

    Regression for the bug where ``normalize_url_for_deep_crawl`` parsed the
    query with ``parse_qs`` (``keep_blank_values=False``), dropping the blank
    param ``?q=`` and collapsing ``/page?q=`` and ``/page`` into one dedup key.
    The fetched URL preserved ``?q=`` via ``urljoin``, so the two hrefs
    collapsed to one key but remained two distinct fetched URIs — one was
    silently starved. This test drives the real strategy ``.arun`` through the
    same ``arun_many`` mock pattern used by the existing query-fetch test,
    across both discovery orders, to prove neither variant starves the other.
    """
    start = "https://example.com/"
    href_blank, href_bare = order
    target_blank = "https://example.com/page?q="
    target_bare = "https://example.com/page"
    links = {
        "internal": [{"href": href_blank}, {"href": href_bare}],
        "external": [],
    }
    if prefetch:
        html = "".join(f'<a href="{h}">link</a>' for h in (href_blank, href_bare))
        links = quick_extract_links(html, start)
    fetched = []

    async def arun_many(urls, config):
        results = []
        for url in urls:
            fetched.append(url)
            results.append(CrawlResult(
                url=url, html="", success=True,
                links=links if url == start else {"internal": [], "external": []},
            ))
        if config.stream:
            async def results_stream():
                for result in results:
                    yield result
            return results_stream()
        return results

    strategy = strategy_type(max_depth=2, max_pages=10)
    results = await strategy.arun(
        start, SimpleNamespace(arun_many=arun_many), CrawlerRunConfig(stream=stream),
    )
    if stream:
        results = [result async for result in results]

    target_urls = [u for u in fetched if "/page" in u and u != start]
    assert len(target_urls) == 2, (
        f"Both /page and /page?q= must be fetched, got {fetched}"
    )
    qs_set = set()
    for u in target_urls:
        qs = u.split("?", 1)[1] if "?" in u else ""
        qs_set.add(qs)
    assert "" in qs_set, (
        f"Bare /page not fetched (only /page?q= found): {fetched}"
    )
    assert "q=" in qs_set, (
        f"/page?q= not fetched (only bare /page found): {fetched}"
    )
    assert [r.url for r in results] == fetched, (
        f"Result URLs must match fetched URLs: {[r.url for r in results]} vs {fetched}"
    )


@pytest.mark.asyncio
async def test_resume_blank_param_does_not_block_bare():
    """BFS resumed crawl must not let a persisted /page?q= key block /page.

    Regression for the resume half of the dedup-key divergence bug. When a
    crawl fetched ``/page?q=`` and stored it under the (fixed) dedup key
    ``http://example.com/page?q=``, a resumed crawl that later discovers the
    bare ``/page`` must still fetch it — the two keys are distinct. Before the
    fix, the persisted key was ``http://example.com/page`` (blank dropped on
    resume load via ``normalize_url_for_deep_crawl(u, u)``), so the resumed
    crawl found ``/page`` already in ``visited`` and skipped it before it
    ever reached ``arun_many``.

    Tests BFS specifically (the primary strategy); DFS and BFF carry equivalent
    ``visited``-rebuild paths all flowing through the same
    ``normalize_url_for_deep_crawl``, so the function-level regression tests
    in ``test_reg_utils.py`` cover those paths.
    """
    start = "https://example.com/"
    # Resume state: a prior crawl fetched /page?q= (stored under its dedup key)
    # and was interrupted. The start URL is in pending so the strategy re-fetches
    # it and re-discovers its links.
    resume_state = {
        "visited": ["https://example.com/page?q="],
        "pending": [{"url": start, "parent_url": None}],
        "depths": {start: 0},
        "pages_crawled": 1,
    }
    links = {
        "internal": [{"href": "/page?q="}, {"href": "/page"}],
        "external": [],
    }
    fetched = []

    async def arun_many(urls, config):
        results = []
        for url in urls:
            fetched.append(url)
            results.append(CrawlResult(
                url=url, html="", success=True,
                links=links if url == start else {"internal": [], "external": []},
            ))
        return results

    strategy = BFSDeepCrawlStrategy(max_depth=2, max_pages=10, resume_state=resume_state)
    await strategy.arun(
        start, SimpleNamespace(arun_many=arun_many), CrawlerRunConfig(stream=False),
    )

    # The start page (re-fetched from pending) plus the bare /page must be
    # fetched. /page?q= is already in visited (resume) and must NOT be
    # re-fetched. Without the fix, /page would also be blocked because the
    # resume-loaded key drops the blank param, collapsing to /page.
    assert "https://example.com/" in fetched, f"Start page not fetched: {fetched}"
    assert "https://example.com/page" in fetched, (
        f"Resumed crawl must fetch bare /page (not blocked by /page?q= key): "
        f"{fetched}"
    )
    # /page?q= was already visited (from resume state) and must not be re-fetched.
    assert "https://example.com/page?q=" not in fetched, (
        f"Already-visited /page?q= should not be re-fetched: {fetched}"
    )

