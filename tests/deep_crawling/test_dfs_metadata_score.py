"""
Regression tests for DFS per-page ``metadata["score"]`` annotation.

``DFSDeepCrawlStrategy`` sets each crawled page's own score in
``_arun_batch`` / ``_arun_stream`` (``dfs_strategy.py:89-90`` / ``:204-205``).
The DFS-overridden ``link_discovery`` previously overwrote that value in-place
with each discovered child link's score, so the value the caller received was
the last non-zero child's score rather than the parent page's own. The fix
removes the child-score write from ``link_discovery``.

These tests guard against re-introducing the clobber in both the common path
and the capacity-trim path, in both batch and stream modes. They are mock-based
(no browser, no network) and follow the helper conventions of
``test_deep_crawl_resume.py``.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


def create_mock_config(stream=False):
    """A mock ``CrawlerRunConfig`` whose ``clone`` honours the requested stream flag."""
    config = MagicMock()
    config.stream = stream

    def clone_config(**kwargs):
        new_config = MagicMock()
        new_config.stream = kwargs.get("stream", stream)
        new_config.clone = MagicMock(side_effect=clone_config)
        return new_config

    config.clone = MagicMock(side_effect=clone_config)
    return config


def _make_tree_crawler(
    tree: Dict[str, List[str]],
    fetch_log: Optional[List[str]] = None,
):
    """Mock crawler where each URL's children come from ``tree``.

    Each call records the requested URLs into ``fetch_log`` (when provided) and
    returns one successful ``CrawlResult``-like object per URL whose ``links``
    are the children listed in ``tree`` for that URL. Honours both batch
    (``stream=False``) and streaming (``stream=True``) configs.
    """

    async def mock_arun_many(urls, config):
        if fetch_log is not None:
            fetch_log.extend(urls)
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


class TestDFSParentScoreNotClobbered:
    """The parent page's ``metadata["score"]`` must equal the scorer's score for
    the parent's own URL, not a child's score.

    Reproduction (from the bug report): a parent whose own score is 0.5 links to
    two children scoring 0.5 and 1.0. Before the fix the parent's
    ``metadata["score"]`` ended up as 1.0 (the last non-zero child's score) in
    both batch and stream modes.
    """

    PARENT = "http://example.com/docs/python"          # score 0.5 (matches "python")
    CHILD_A = "http://example.com/guide/article"       # score 0.5 (matches "guide")
    CHILD_B = "http://example.com/python/guide/intro"  # score 1.0 (matches both)

    TREE = {
        PARENT: [CHILD_A, CHILD_B],
        CHILD_A: [],
        CHILD_B: [],
    }

    def _scorer(self):
        return KeywordRelevanceScorer(keywords=["python", "guide"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_parent_keeps_own_score(self, stream):
        scorer = self._scorer()
        # Sanity-check the scorer's expected values so the test is self-documenting.
        assert scorer.score(self.PARENT) == 0.5
        assert scorer.score(self.CHILD_A) == 0.5
        assert scorer.score(self.CHILD_B) == 1.0

        strategy = DFSDeepCrawlStrategy(
            max_depth=1,          # children at depth 1 -> link_discovery returns early
            max_pages=10,         # no capacity trimming
            url_scorer=scorer,
        )
        crawler = _make_tree_crawler(self.TREE)
        config = create_mock_config(stream=stream)

        results = []
        if stream:
            async for r in strategy._arun_stream(self.PARENT, crawler, config):
                results.append(r)
        else:
            results = await strategy._arun_batch(self.PARENT, crawler, config)

        by_url = {r.url: r for r in results}
        assert set(by_url) == {self.PARENT, self.CHILD_A, self.CHILD_B}

        parent_result = by_url[self.PARENT]
        # Core guarantee: the parent's score is its own, not a child's.
        assert parent_result.metadata["score"] == scorer.score(self.PARENT), (
            f"Parent score clobbered: got {parent_result.metadata['score']}, "
            f"expected {scorer.score(self.PARENT)} (parent's own score). Before the "
            f"fix this was {scorer.score(self.CHILD_B)} (last non-zero child)."
        )
        # Children keep their own scores (link_discovery returns early at
        # max_depth, so they were never clobbered even before the fix).
        assert by_url[self.CHILD_A].metadata["score"] == scorer.score(self.CHILD_A)
        assert by_url[self.CHILD_B].metadata["score"] == scorer.score(self.CHILD_B)


class TestDFSParentScoreNotClobberedCapacityTrim:
    """When ``valid_links`` exceeds ``remaining_capacity``, DFS sorts by score
    descending and slices ``[:remaining_capacity]``. Before the fix, the loop
    then wrote each surviving child's score onto the parent's metadata, leaving
    the parent with the *lowest-scoring surviving* child's score instead of its
    own. This guards the capacity-trim branch, which is a separate code path
    from the common case above.
    """

    PARENT = "http://example.com/python/guide/landing"  # score 1.0 (both)
    CHILD_A = "http://example.com/guide/python/c"       # score 1.0 (both)
    CHILD_B = "http://example.com/guide/c"              # score 0.5 (guide)
    CHILD_C = "http://example.com/python/x"             # score 0.5 (python)

    TREE = {
        PARENT: [CHILD_A, CHILD_B, CHILD_C],
        CHILD_A: [],
        CHILD_B: [],
        CHILD_C: [],
    }

    def _scorer(self):
        return KeywordRelevanceScorer(keywords=["python", "guide"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_parent_keeps_own_score_under_capacity_trim(self, stream):
        scorer = self._scorer()
        assert scorer.score(self.PARENT) == 1.0
        assert scorer.score(self.CHILD_A) == 1.0
        assert scorer.score(self.CHILD_B) == 0.5
        assert scorer.score(self.CHILD_C) == 0.5

        strategy = DFSDeepCrawlStrategy(
            max_depth=2,
            max_pages=3,          # parent + 2 surviving of 3 children
            url_scorer=scorer,
        )
        fetch_log: List[str] = []
        crawler = _make_tree_crawler(self.TREE, fetch_log=fetch_log)
        config = create_mock_config(stream=stream)

        results = []
        if stream:
            async for r in strategy._arun_stream(self.PARENT, crawler, config):
                results.append(r)
        else:
            results = await strategy._arun_batch(self.PARENT, crawler, config)

        by_url = {r.url: r for r in results}
        assert len(results) == 3, (
            f"Expected exactly 3 results (max_pages=3), got {len(results)}"
        )
        assert self.PARENT in by_url
        # The lowest-priority child must not be fetched (trim removed it).
        assert self.CHILD_C not in fetch_log

        parent_result = by_url[self.PARENT]
        # Core guarantee under the capacity-trim branch: the parent's score is
        # its own (1.0), not the lowest surviving child's (0.5).
        assert parent_result.metadata["score"] == scorer.score(self.PARENT), (
            f"Parent score clobbered under capacity trim: got "
            f"{parent_result.metadata['score']}, expected "
            f"{scorer.score(self.PARENT)} (parent's own score). Before the fix "
            f"this was 0.5 (the lowest surviving child's score)."
        )
