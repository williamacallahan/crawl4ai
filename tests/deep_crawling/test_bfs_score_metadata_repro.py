"""
Regression: per-page ``metadata["score"]`` for BFS/DFS deep crawl with url_scorer.

Bug (introduced in commit c308a79): ``BFSDeepCrawlStrategy`` never wrote the
crawled page's *own* score onto its own ``CrawlResult.metadata`` (the
result-annotation loops in ``_arun_batch``/``_arun_stream`` set only ``depth``
and ``parent_url``), while ``link_discovery`` wrote each *child* URL's score
onto the *parent* page's ``CrawlResult.metadata["score"]`` -- so a hub page
ended up carrying its last non-zero child's score, and leaf pages had no
``score`` key at all.

The sibling strategies establish the contract: ``DFSDeepCrawlStrategy`` sets
``result.metadata["score"] = self.url_scorer.score(url)`` for every crawled
page, and ``BestFirstCrawlingStrategy`` sets ``result.metadata["score"]`` on
every yielded result. The DFS ``link_discovery`` had the same parent-
contamination defect; this fix removes the metadata mutation from both BFS and
DFS ``link_discovery`` and adds the own-score assignment to both BFS modes.

These tests pin the contract so the bug cannot silently return:
  - every crawled BFS/DFS result carries ``metadata["score"]`` equal to
    ``url_scorer.score(result.url)`` when a scorer is configured;
  - hub pages are no longer contaminated by a child's score;
  - leaf pages carry their own (possibly 0.0) score instead of no key;
  - the no-scorer path is unchanged (no ``score`` key for BFS);
  - score-threshold filtering and capacity-trimming-by-score in
    ``link_discovery`` still work (they rely on the local ``score`` only).
"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


# ---------------------------------------------------------------------------
# Shared mock helpers (recursive-clone config + tree crawler).
# ---------------------------------------------------------------------------


def _make_config(stream=False):
    """A mock config whose ``clone`` propagates the requested stream flag.

    The recursive form is required so BestFirst (which always clones with
    ``stream=True``) and BFS/DFS (which clone per mode) all work.
    """
    config = MagicMock()
    config.stream = stream

    def clone_config(**kwargs):
        new_config = MagicMock()
        new_config.stream = kwargs.get("stream", stream)
        new_config.clone = MagicMock(side_effect=clone_config)
        return new_config

    config.clone = MagicMock(side_effect=clone_config)
    return config


def _make_tree_crawler(tree: Dict[str, List[str]]):
    """Mock crawler where each URL's children come from ``tree``.

    Each requested URL returns one successful result with ``links`` populated
    from ``tree``. Honours both batch and streaming configs.
    """

    async def mock_arun_many(urls, config):
        results = []
        for url in urls:
            result = MagicMock()
            result.url = url
            result.success = True
            result.metadata = {}
            result.links = {
                "internal": [{"href": child} for child in tree.get(url, [])],
                "external": [],
            }
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


# The canonical scenario from the bug report: a hub at example.com/ links to
# /alpha (scores 1.0) and /beta (scores 0.0); the hub itself scores 0.0.
_HUB_TREE: Dict[str, List[str]] = {
    "https://example.com/": [
        "https://example.com/alpha",
        "https://example.com/beta",
    ],
    "https://example.com/alpha": [],
    "https://example.com/beta": [],
}


def _expected_scores(scorer: KeywordRelevanceScorer) -> Dict[str, float]:
    return {url: scorer.score(url) for url in _HUB_TREE}


async def _run(strategy, start_url: str, crawler, stream: bool) -> List:
    if stream:
        return [r async for r in strategy._arun_stream(start_url, crawler, _make_config(True))]
    return await strategy._arun_batch(start_url, crawler, _make_config(False))


# ---------------------------------------------------------------------------
# BFS: per-page own score (the primary bug).
# ---------------------------------------------------------------------------


class TestBFSScoreMetadata:
    """BFS must write each crawled page's own ``metadata["score"]`` and must
    not contaminate hub pages with a child's score.

    Regression for the missing/incorrect per-page score metadata introduced
    in commit c308a79.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_bfs_writes_each_page_own_score(self, stream):
        scorer = KeywordRelevanceScorer(keywords=["alpha"])
        strategy = BFSDeepCrawlStrategy(
            max_depth=1, max_pages=10, url_scorer=scorer
        )
        results = await _run(
            strategy,
            "https://example.com/",
            _make_tree_crawler(_HUB_TREE),
            stream,
        )

        by_url = {r.url: r.metadata.get("score") for r in results}
        expected = _expected_scores(scorer)
        assert by_url == expected, (
            f"BFS {'stream' if stream else 'batch'} metadata['score'] mismatch.\n"
            f"  got:      {by_url}\n"
            f"  expected: {expected}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_bfs_no_scorer_leaves_no_score_key(self, stream):
        """Without a url_scorer the BFS path must not synthesize a 'score' key
        (parity with the original behaviour and with DFS)."""
        strategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=10)
        results = await _run(
            strategy,
            "https://example.com/",
            _make_tree_crawler(_HUB_TREE),
            stream,
        )

        for r in results:
            assert "score" not in r.metadata, (
                f"unexpected 'score' key without url_scorer on {r.url}: "
                f"{r.metadata!r}"
            )
            assert r.metadata.get("depth") is not None
            assert "parent_url" in r.metadata


# ---------------------------------------------------------------------------
# BFS: link_discovery's legitimate use of score (filtering / sorting) still
# works after removing the parent-mutation.
# ---------------------------------------------------------------------------


class TestBFSScoreDiscoveryStillWorks:
    """The local ``score`` in ``link_discovery`` must still drive threshold
    filtering and capacity-trimming-by-score once the metadata mutation is
    gone."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_score_threshold_filters_low_score_children(self, stream):
        scorer = KeywordRelevanceScorer(keywords=["alpha"])
        strategy = BFSDeepCrawlStrategy(
            max_depth=1,
            max_pages=10,
            url_scorer=scorer,
            score_threshold=0.5,  # /alpha (1.0) kept, /beta (0.0) dropped
        )
        results = await _run(
            strategy,
            "https://example.com/",
            _make_tree_crawler(_HUB_TREE),
            stream,
        )

        urls = {r.url for r in results}
        assert "https://example.com/" in urls
        assert "https://example.com/alpha" in urls
        assert "https://example.com/beta" not in urls, (
            "score_threshold filtering broken: /beta should have been dropped"
        )
        # Scores still correct on the surviving pages.
        by_url = {r.url: r.metadata["score"] for r in results}
        assert by_url == {
            "https://example.com/": 0.0,
            "https://example.com/alpha": 1.0,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_capacity_trimming_keeps_highest_scored_children(self, stream):
        """With only one remaining slot, the higher-scored child (/alpha)
        must win the sort and be the one crawled -- proving the local score
        still drives the ordering."""
        scorer = KeywordRelevanceScorer(keywords=["alpha"])
        strategy = BFSDeepCrawlStrategy(
            max_depth=1, max_pages=2, url_scorer=scorer  # hub + exactly 1 child
        )
        results = await _run(
            strategy,
            "https://example.com/",
            _make_tree_crawler(_HUB_TREE),
            stream,
        )

        urls = [r.url for r in results]
        assert urls[0] == "https://example.com/"
        assert "https://example.com/alpha" in urls, (
            "capacity trimming should have kept the highest-scoring child /alpha"
        )
        assert "https://example.com/beta" not in urls
        assert strategy._pages_crawled == 2


# ---------------------------------------------------------------------------
# DFS: the secondary parent-contamination defect (shared with BFS).
# ---------------------------------------------------------------------------


class TestDFSScoreMetadata:
    """DFS already set each page's own score, but its ``link_discovery`` also
    overwrote the parent's score with a child's. The hub must keep its own
    score."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_dfs_hub_score_not_contaminated_by_children(self, stream):
        """The DFS hub must keep its own 0.0 score; before the fix
        ``link_discovery`` overwrote it with /alpha's 1.0."""
        scorer = KeywordRelevanceScorer(keywords=["alpha"])
        strategy = DFSDeepCrawlStrategy(
            max_depth=1, max_pages=10, url_scorer=scorer
        )
        results = await _run(
            strategy,
            "https://example.com/",
            _make_tree_crawler(_HUB_TREE),
            stream,
        )

        hub = next(r for r in results if r.url == "https://example.com/")
        assert hub.metadata["score"] == 0.0, (
            f"DFS hub contaminated by a child score: got {hub.metadata['score']!r}"
        )
        by_url = {r.url: r.metadata["score"] for r in results}
        assert by_url == _expected_scores(scorer)
