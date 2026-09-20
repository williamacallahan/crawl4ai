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
