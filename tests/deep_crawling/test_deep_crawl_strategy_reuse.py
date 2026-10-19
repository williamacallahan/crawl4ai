"""
Regression: reusing a deep-crawl strategy instance must NOT carry the
``max_pages`` page budget (or stats) across runs.

Bug (introduced in commit d0247496, made severe by 7b9aabc6):
``_pages_crawled`` was initialized only in ``BFSDeepCrawlStrategy.__init__``
(BFF/DFS got the same pattern in the same commit; DFS inherits BFS's counter)
and never reset at the start of a non-resume run. The strategy instance is
stored on ``CrawlerRunConfig`` and reused across ``arun()`` calls (including
inside ``arun_many``, which loops seeds through the same ``primary_cfg``), so
the budget silently accumulated across runs:

* If the previous run exhausted ``max_pages``, the next run's top-of-loop
  guard ``if self._pages_crawled >= self.max_pages: break`` fired immediately
  and the crawl returned zero results.
* If the previous run ended below ``max_pages``, the next run proceeded with
  a silently shrunken budget.

The cancellation suite's ``test_strategy_reuse_after_cancellation`` only
passed because its first run was cancelled before any page was crawled.

These tests pin the per-run reset for batch and stream modes across BFS,
DFS, and Best-First strategies, covering: reuse after an exhausted budget
(zero-result symptom); reuse after a partial run below ``max_pages``
(silently shrunken-budget symptom, anchored against a fresh-instance
control); the ``arun_many`` shared-config path (production-reachable
silent-sharing bug); stats isolation (``urls_skipped`` / ``start_time``);
and the resume path being unaffected (the ``resume_state`` branch still
overrides ``_pages_crawled`` from the checkpoint).

Mock-based, following the conventions of ``test_deep_crawl_cancellation.py``
and ``test_deep_crawl_resume.py``: no browser, no network.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock

from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    FilterChain,
    URLFilter,
)


# ============================================================================
# Helper functions
# ============================================================================

def create_mock_config(stream=False):
    """Mock CrawlerRunConfig whose clone honours the requested stream flag."""
    config = MagicMock()
    config.stream = stream

    def clone_config(**kwargs):
        new_config = MagicMock()
        new_config.stream = kwargs.get("stream", stream)
        new_config.clone = MagicMock(side_effect=clone_config)
        return new_config

    config.clone = MagicMock(side_effect=clone_config)
    return config


def make_mock_crawler(num_links: int = 2):
    """Mock crawler whose ``arun_many`` returns one successful result per URL,
    each with ``num_links`` unique child links. Returns an async generator in
    stream mode and a list in batch mode -- matching the contract the deep
    crawl strategies rely on."""
    call_count = 0

    async def mock_arun_many(urls, config):
        nonlocal call_count
        results = []
        for url in urls:
            call_count += 1
            result = MagicMock()
            result.url = url
            result.success = True
            result.metadata = {}
            links = [
                {"href": f"{url}/child{call_count}_{i}"}
                for i in range(num_links)
            ]
            result.links = {"internal": links, "external": []}
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


async def run_strategy(strategy, start_url, crawler, config, stream):
    """Drive a strategy in batch or stream mode and collect all results."""
    if stream:
        return [r async for r in strategy._arun_stream(start_url, crawler, config)]
    return await strategy._arun_batch(start_url, crawler, config)


class _RejectAllFilter(URLFilter):
    """URLFilter that rejects every URL; drives ``stats.urls_skipped`` up
    deterministically so its per-run reset can be asserted. Used inside a
    ``FilterChain`` so ``FilterChain.apply`` (the async method the strategies
    call) computes rejections the same way production filters do."""

    def apply(self, url: str) -> bool:
        return False


# ============================================================================
# TEST SUITE: per-run budget reset on instance reuse
# ============================================================================

class TestBudgetResetOnReuse:
    """A reused strategy instance must start each non-resume run with a
    fresh ``max_pages`` budget and a clean stats object."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy_class", [
        BFSDeepCrawlStrategy,
        DFSDeepCrawlStrategy,
        BestFirstCrawlingStrategy,
    ])
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_reuse_after_exhausted_budget(
        self, strategy_class, stream,
    ):
        """Run 1 exhausts ``max_pages``; run 2 must crawl a fresh budget,
        not return zero results from a stale counter (the primary symptom
        described in the bug report)."""
        max_pages = 3
        strategy = strategy_class(max_depth=2, max_pages=max_pages)
        crawler = make_mock_crawler(num_links=2)
        config = create_mock_config(stream=stream)

        results1 = await run_strategy(
            strategy, "https://example.com", crawler, config, stream,
        )
        assert len(results1) == max_pages, (
            f"{strategy_class.__name__} run1 returned {len(results1)}; "
            f"precondition: run1 must fully exhaust the budget"
        )
        assert strategy._pages_crawled == max_pages

        # Without the fix: run 2 returns 0 results because the stale
        # counter trips the top-of-loop guard immediately.
        results2 = await run_strategy(
            strategy, "https://example.com", crawler, config, stream,
        )
        assert len(results2) == max_pages, (
            f"{strategy_class.__name__} (stream={stream}) reused instance "
            f"returned {len(results2)} after run1 exhausted the budget; "
            f"expected a fresh budget of {max_pages}"
        )
        assert strategy._pages_crawled == max_pages  # reset to 0, grown back

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy_class", [
        BFSDeepCrawlStrategy,
        DFSDeepCrawlStrategy,
        BestFirstCrawlingStrategy,
    ])
    @pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
    async def test_reuse_after_partial_run_below_budget(
        self, strategy_class, stream,
    ):
        """Run 1 ends below ``max_pages`` (here: capped by ``max_depth``).
        Run 2 must use a full fresh budget, mirroring a fresh-instance
        control; without the fix it crawls silently fewer pages because the
        stale counter eats part of the budget."""
        max_pages = 5
        strategy = strategy_class(max_depth=1, max_pages=max_pages)
        crawler = make_mock_crawler(num_links=2)
        config = create_mock_config(stream=stream)

        results1 = await run_strategy(
            strategy, "https://example.com", crawler, config, stream,
        )
        # max_depth=1 caps the run at start + 2 children = 3 pages, well
        # below max_pages=5; the budget is NOT exhausted by run 1.
        assert len(results1) == 3, (
            f"{strategy_class.__name__} run1 returned {len(results1)}; "
            f"precondition: run1 must end with _pages_crawled < max_pages"
        )
        assert strategy._pages_crawled == 3

        control = strategy_class(max_depth=1, max_pages=max_pages)
        ctrl_crawler = make_mock_crawler(num_links=2)
        control_results = await run_strategy(
            control, "https://example.com", ctrl_crawler, config, stream,
        )
        assert len(control_results) == 3, (
            f"fresh {strategy_class.__name__} control returned "
            f"{len(control_results)}; test expectation unmet"
        )

        results2 = await run_strategy(
            strategy, "https://example.com", crawler, config, stream,
        )
        assert len(results2) == len(control_results), (
            f"{strategy_class.__name__} (stream={stream}) reused instance "
            f"returned {len(results2)} pages vs fresh control "
            f"{len(control_results)}; the stale counter silently shrank "
            f"this run's budget (expected fresh-budget parity)"
        )
        assert strategy._pages_crawled == 3  # reset to 0, then back to 3

    @pytest.mark.asyncio
    async def test_reuse_matches_fresh_instance_budget_bfs_batch(self):
        """Stronger than a count assertion: the reused instance's second run
        must end with the same ``_pages_crawled`` as a fresh instance's one
        run, not a left-shifted (shrunken) value."""
        max_pages = 4
        reused = BFSDeepCrawlStrategy(max_depth=2, max_pages=max_pages)
        fresh = BFSDeepCrawlStrategy(max_depth=2, max_pages=max_pages)
        crawler = make_mock_crawler(num_links=2)
        config = create_mock_config(stream=False)

        await reused._arun_batch("https://example.com", crawler, config)
        assert reused._pages_crawled == max_pages

        # The reuse condition: same instance is run a second time.
        await reused._arun_batch("https://example.com", crawler, config)
        reused_run2_pages = reused._pages_crawled

        await fresh._arun_batch("https://example.com", crawler, config)
        fresh_pages = fresh._pages_crawled

        assert reused_run2_pages == fresh_pages == max_pages, (
            f"reused run2 _pages_crawled={reused_run2_pages}, "
            f"fresh single-run _pages_crawled={fresh_pages}, "
            f"expected both == max_pages={max_pages}"
        )


# ============================================================================
# TEST SUITE: arun_many shared-config (the production-reachable bug)
# ============================================================================

class TestSharedConfigArunMany:
    """Simulates ``crawler.arun_many([seed1, ...], config)`` -- the path
    ``crawl4ai/async_webcrawler.py`` ``arun_many`` takes via a single
    ``primary_cfg`` when a deep-crawl config is passed -- where every seed
    shares the same ``deep_crawl_strategy`` instance off that config."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy_class", [
        BFSDeepCrawlStrategy,
        DFSDeepCrawlStrategy,
        BestFirstCrawlingStrategy,
    ])
    async def test_each_seed_gets_fresh_budget(self, strategy_class):
        """Each seed must crawl ``max_pages`` worth of pages on its own
        fresh budget. Without the fix, the first seed consumes the whole
        shared budget and every subsequent seed crawls 0 pages (or, with
        a higher max_pages, a shrunken budget)."""
        max_pages = 3
        seeds = [
            "https://example.com/seed1",
            "https://example.com/seed2",
            "https://example.com/seed3",
        ]
        strategy = strategy_class(max_depth=1, max_pages=max_pages)
        crawler = make_mock_crawler(num_links=2)
        config = create_mock_config(stream=False)

        per_seed_counts = []
        for seed in seeds:
            results = await strategy._arun_batch(seed, crawler, config)
            per_seed_counts.append(len(results))

        assert per_seed_counts == [max_pages] * len(seeds), (
            f"{strategy_class.__name__} arun_many shared-config: per-seed "
            f"results {per_seed_counts} -- expected each seed to crawl "
            f"{max_pages} pages on its own fresh budget"
        )


# ============================================================================
# TEST SUITE: stats isolation across runs
# ============================================================================

class TestStatsIsolation:
    """``stats`` must be a per-run object; without the per-run reset it
    carries ``urls_skipped`` and ``start_time`` across runs and contaminates
    per-run stats."""

    @pytest.mark.asyncio
    async def test_urls_skipped_reset_per_run_bfs(self):
        """``stats.urls_skipped`` must reset between runs, not accumulate."""
        strategy = BFSDeepCrawlStrategy(
            max_depth=1, max_pages=10,
            filter_chain=FilterChain([_RejectAllFilter()]),
        )
        crawler = make_mock_crawler(num_links=3)
        config = create_mock_config(stream=False)

        # Run 1: every discovered child is rejected by the filter, driving
        # urls_skipped up by 3 (one per child link on the start page).
        await strategy._arun_batch("https://example.com", crawler, config)
        skipped_after_run1 = strategy.stats.urls_skipped
        assert skipped_after_run1 > 0, (
            "precondition: filter must drop some children in run 1"
        )

        # Without the fix: skipped accumulates -- after run 2, it equals
        # 2 * skipped_after_run1. With the fix: per-run stats reset.
        await strategy._arun_batch("https://example.com", crawler, config)
        assert strategy.stats.urls_skipped == skipped_after_run1, (
            f"stats.urls_skipped leaked across runs: "
            f"after run 2 = {strategy.stats.urls_skipped}, "
            f"expected same as after run 1 ({skipped_after_run1})"
        )

    @pytest.mark.asyncio
    async def test_start_time_refreshes_per_run_bfs(self):
        """stats.start_time must refresh at the start of each run; without
        the per-run stats reset, run 2 keeps run 1's stats.start_time (the
        exact same datetime object)."""
        strategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=10)
        crawler = make_mock_crawler(num_links=2)
        config = create_mock_config(stream=False)

        await strategy._arun_batch("https://example.com", crawler, config)
        start_time_run1 = strategy.stats.start_time

        await strategy._arun_batch("https://example.com", crawler, config)
        start_time_run2 = strategy.stats.start_time

        assert start_time_run2 is not start_time_run1, (
            f"stats.start_time carried over the same datetime object across "
            f"runs ({start_time_run1}); stats object did not reset per run"
        )


# ============================================================================
# TEST SUITE: resume path is unaffected by the per-run reset
# ============================================================================

class TestResumeNotAffected:
    """When ``resume_state`` is provided, the strategy must overwrite
    ``_pages_crawled`` from the checkpoint -- NOT reset it to 0 -- otherwise
    resuming a partially-completed crawl would silently re-crawl the
    already-counted pages."""

    @pytest.mark.asyncio
    async def test_resume_overrides_pages_crawled_from_checkpoint_bfs(self):
        resume_state = {
            "visited": ["https://example.com/"],
            "pending": [
                {"url": "https://example.com/child1",
                 "parent_url": "https://example.com/"},
            ],
            "depths": {
                "https://example.com/": 0,
                "https://example.com/child1": 1,
            },
            "pages_crawled": 2,  # checkpoint says 2 pages already crawled
        }
        strategy = BFSDeepCrawlStrategy(
            max_depth=2, max_pages=5, resume_state=resume_state,
        )
        crawler = make_mock_crawler(num_links=0)
        config = create_mock_config(stream=False)

        await strategy._arun_batch("https://example.com", crawler, config)

        # Resume branch sets _pages_crawled = 2 from the checkpoint, then
        # the resume run pops `child1` from `pending` and crawls it (+1).
        # If the per-run reset incorrectly wrote 0 here, we'd see 1.
        assert strategy._pages_crawled == 3, (
            f"resume did not honor checkpoint pages_crawled: "
            f"got {strategy._pages_crawled}, expected 3 (2 checkpoint + 1)"
        )

    @pytest.mark.asyncio
    async def test_resume_overrides_pages_crawled_from_checkpoint_dfs(self):
        resume_state = {
            "visited": ["https://example.com/"],
            "stack": [
                {"url": "https://example.com/child1",
                 "parent_url": "https://example.com/", "depth": 1},
            ],
            "depths": {
                "https://example.com/": 0,
                "https://example.com/child1": 1,
            },
            "pages_crawled": 2,
            "dfs_seen": ["https://example.com/"],
        }
        strategy = DFSDeepCrawlStrategy(
            max_depth=2, max_pages=5, resume_state=resume_state,
        )
        crawler = make_mock_crawler(num_links=0)
        config = create_mock_config(stream=False)

        await strategy._arun_batch("https://example.com", crawler, config)

        assert strategy._pages_crawled == 3, (
            f"resume did not honor checkpoint pages_crawled: "
            f"got {strategy._pages_crawled}, expected 3 (2 checkpoint + 1)"
        )

    @pytest.mark.asyncio
    async def test_resume_overrides_pages_crawled_from_checkpoint_bff(self):
        resume_state = {
            "visited": ["https://example.com/"],
            "queue_items": [
                {"score": 0.0, "depth": 1,
                 "url": "https://example.com/child1",
                 "parent_url": "https://example.com/"},
            ],
            "depths": {
                "https://example.com/": 0,
                "https://example.com/child1": 1,
            },
            "pages_crawled": 2,
        }
        strategy = BestFirstCrawlingStrategy(
            max_depth=2, max_pages=5, resume_state=resume_state,
        )
        crawler = make_mock_crawler(num_links=0)
        config = create_mock_config(stream=True)

        async for _ in strategy._arun_stream(
            "https://example.com", crawler, config,
        ):
            pass

        assert strategy._pages_crawled == 3, (
            f"resume did not honor checkpoint pages_crawled: "
            f"got {strategy._pages_crawled}, expected 3 (2 checkpoint + 1)"
        )


# ============================================================================
# TEST SUITE: cancellation + reuse still works (no regression)
# ============================================================================

class TestCancellationReuseNoRegression:
    """The existing ``test_strategy_reuse_after_cancellation`` only worked
    because its first run was cancelled before any page was crawled. Pin
    the reuse contract when the first run DID crawl some pages before being
    cancelled, exercising the cancellation reset and the new
    ``_pages_crawled`` reset together."""

    @pytest.mark.asyncio
    async def test_reuse_after_cancellation_past_first_page_bfs(self):
        max_pages = 10
        captured_pages = 0
        cancel_active = True

        async def maybe_cancel():
            return cancel_active and captured_pages >= 3

        async def track_pages(state):
            nonlocal captured_pages
            captured_pages = state.get("pages_crawled", 0)

        strategy = BFSDeepCrawlStrategy(
            max_depth=2, max_pages=max_pages,
            should_cancel=maybe_cancel,
            on_state_change=track_pages,
        )
        crawler = make_mock_crawler(num_links=3)
        config = create_mock_config(stream=False)

        # Run 1: cancellation fires after the level-1 batch completes
        # (>= 3 pages captured), leaving _pages_crawled > 0.
        results1 = await strategy._arun_batch(
            "https://example.com", crawler, config,
        )
        assert strategy.cancelled is True
        first_run_pages = strategy._pages_crawled
        assert first_run_pages > 0, "precondition: run 1 crawled some pages"
        assert len(results1) == first_run_pages

        # Run 2: NO cancellation. With the fix the budget is fresh; without
        # the fix the stale counter leaves (max_pages - first_run_pages) units
        # of budget, returning that many pages instead of max_pages.
        cancel_active = False
        captured_pages = 0
        results2 = await strategy._arun_batch(
            "https://example.com", crawler, config,
        )
        assert strategy.cancelled is False
        assert len(results2) == max_pages, (
            f"run 2 returned {len(results2)} pages; expected {max_pages} "
            f"(fresh budget). The stale-budget bug would return "
            f"{max_pages - first_run_pages} (max_pages - first_run_pages)"
        )
        assert strategy._pages_crawled == max_pages
