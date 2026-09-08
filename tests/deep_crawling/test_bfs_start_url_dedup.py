"""
Regression guard: BFS deep-crawl start-URL deduplication (batch/stream parity).

Commit f891133 moved marking of *discovered* URLs into ``link_discovery``
(``visited.add(base_url)``) and removed the per-level ``visited.update(urls)``
line from ``_arun_batch`` only — leaving the identical line standing in
``_arun_stream``. The seed/start URL is seeded directly into ``current_level``
and never passes through ``link_discovery``, so the removed line was the *only*
code path that marked the canonical (trailing-slash) start URL before its own
depth-0 self-link could be discovered. A canonical start URL whose page emits a
self-link (e.g. ``href="/"``) normalizing back to that exact canonical string was
re-enqueued at depth 0 and re-fetched at depth 1 — a duplicate fetch and a
duplicate ``CrawlResult`` in the returned list.

The fix restores ``visited.update(urls)`` in ``_arun_batch`` to mirror
``_arun_stream``. These tests pin that parity so a future change cannot silently
re-introduce the batch/stream asymmetry.
"""

import pytest
from typing import List
from unittest.mock import MagicMock

from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


def _make_config(stream=False):
    config = MagicMock()
    config.stream = stream

    def clone_config(**kwargs):
        new_config = MagicMock()
        new_config.stream = kwargs.get("stream", stream)
        new_config.clone = MagicMock(side_effect=clone_config)
        return new_config

    config.clone = MagicMock(side_effect=clone_config)
    return config


def _make_self_link_crawler(crawled: List[str], self_link: str = "/"):
    """Every page emits a single self-link that normalizes back to its own
    canonical URL. Tracks every URL passed to ``arun_many``."""

    async def mock_arun_many(urls, config):
        results = []
        for url in urls:
            crawled.append(url)
            result = MagicMock()
            result.url = url
            result.success = True
            result.metadata = {}
            result.links = {"internal": [{"href": self_link}], "external": []}
            results.append(result)
        if config.stream:
            async def gen():
                for r in results:
                    yield r
            return gen()
        return results

    crawler = MagicMock()
    crawler.arun_many = mock_arun_many
    return crawler


class TestStartUrlDedup:
    """The canonical (trailing-slash) start URL must be crawled exactly once
    when its page emits a self-link normalizing back to the same canonical
    string. Batch and stream must agree."""

    @pytest.mark.asyncio
    async def test_batch_crawls_canonical_start_url_exactly_once(self):
        START_URL = "https://example.com/"
        crawled: List[str] = []
        crawler = _make_self_link_crawler(crawled, self_link="/")

        strategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=10)
        results = await strategy._arun_batch(START_URL, crawler, _make_config(False))

        assert crawled.count(START_URL) == 1, (
            f"start URL crawled {crawled.count(START_URL)} times, expected 1; "
            f"fetched={crawled}"
        )
        assert [r.url for r in results] == [START_URL]

    @pytest.mark.asyncio
    async def test_stream_crawls_canonical_start_url_exactly_once(self):
        START_URL = "https://example.com/"
        crawled: List[str] = []
        crawler = _make_self_link_crawler(crawled, self_link="/")

        strategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=10)
        result_urls = [
            r.url async for r in strategy._arun_stream(START_URL, crawler, _make_config(True))
        ]

        assert crawled.count(START_URL) == 1
        assert result_urls == [START_URL]

    @pytest.mark.asyncio
    async def test_batch_and_stream_agree_on_canonical_self_link(self):
        """The root cause was a batch/stream asymmetry. This guard fails if batch
        and stream diverge again for the canonical self-link idiom."""
        START_URL = "https://example.com/"
        batch_crawled: List[str] = []
        stream_crawled: List[str] = []

        strategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=10)
        await strategy._arun_batch(
            START_URL, _make_self_link_crawler(batch_crawled, "/"), _make_config(False)
        )
        async for _ in strategy._arun_stream(
            START_URL, _make_self_link_crawler(stream_crawled, "/"), _make_config(True)
        ):
            pass

        assert batch_crawled == stream_crawled == [START_URL]
