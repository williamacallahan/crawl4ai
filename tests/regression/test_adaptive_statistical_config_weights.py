"""
Regression tests for ``StatisticalStrategy.calculate_confidence`` sub-metric
weight honoring in ``crawl4ai/adaptive_crawler.py``.

``calculate_confidence`` MUST combine ``coverage`` / ``consistency`` /
``saturation`` using the user-configured ``AdaptiveConfig.coverage_weight`` /
``consistency_weight`` / ``saturation_weight`` fields, not the hardcoded
``0.4 / 0.3 / 0.3`` constants. The original adaptive-crawling commit
(``1a73fb60``) shipped both the configurable weight fields (with ``validate()``
enforcement summing to 1.0) and the hardcoded ``0.4 / 0.3 / 0.3`` defaults in
``calculate_confidence``, so user tuning of the sub-metric weights was
silently ignored. With default ``AdaptiveConfig()`` (weights ``0.4 / 0.3 /
0.3`` matching the hardcoded formula) the bug was invisible, so the tests
below pin both the configured-weight path and the default-config no-regression
path.

Runs fully offline: directly instantiates ``StatisticalStrategy`` /
``CrawlState`` and feeds real ``CrawlResult``s through the real
``update_state`` / ``calculate_confidence`` / ``should_stop`` code paths. No
network, no LLM, no sentence-transformers, no browser.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from crawl4ai import AdaptiveConfig, CrawlResult, MarkdownGenerationResult
from crawl4ai.adaptive_crawler import (
    AdaptiveCrawler,
    StatisticalStrategy,
    CrawlState,
)
from crawl4ai.models import Link


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _md(text: str) -> MarkdownGenerationResult:
    return MarkdownGenerationResult(
        raw_markdown=text,
        markdown_with_citations="",
        references_markdown="",
    )


def _result(url: str, content: str) -> CrawlResult:
    """Build a minimal CrawlResult whose markdown feeds ``update_state``."""
    return CrawlResult(url=url, html="", success=True, markdown=_md(content))


# Two real documents producing the bug-report sub-metrics:
#   coverage    = 1.0000  (every query token appears in both docs)
#   consistency = 0.3333  (Jaccard 4/12 of the two docs' token sets)
#   saturation  = 0.2857  (1 - recent_rate/initial_rate from new_terms_history=7,5)
TWO_DOCS = [
    ("http://x/a", "machine learning docs tutorial introduction overview basics"),
    ("http://x/b", "machine learning docs advanced topics deep neural networks introduction"),
]
QUERY = "machine learning docs"


async def _build_state(strategy: StatisticalStrategy, query: str = QUERY) -> CrawlState:
    """Drive the REAL ``update_state`` with two real ``CrawlResult``s so the
    sub-metrics match a reachable crawl state (not a contrived triple)."""
    state = CrawlState(query=query)
    for url, text in TWO_DOCS:
        r = _result(url, text)
        state.knowledge_base.append(r)
        await strategy.update_state(state, [r])
    return state


# ---------------------------------------------------------------------------
# calculate_confidence honors configured weights (the bug fix)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculate_confidence_uses_configured_weights():
    """``calculate_confidence`` MUST return the configured-weight combination,
    not the hardcoded ``0.4 / 0.3 / 0.3`` formula. With weights
    ``(0.7, 0.15, 0.15)`` the two values diverge, so a single assertion against
    the configured formula pins the fix and guards against a regression that
    re-hardcodes any constant triple."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig(
        coverage_weight=0.7,
        consistency_weight=0.15,
        saturation_weight=0.15,
    )
    strat.config.validate()

    state = await _build_state(strat)

    coverage = strat._calculate_coverage(state)
    consistency = strat._calculate_consistency(state)
    saturation = strat._calculate_saturation(state)
    expected = (
        0.7 * coverage
        + 0.15 * consistency
        + 0.15 * saturation
    )
    hardcoded = 0.4 * coverage + 0.3 * consistency + 0.3 * saturation

    actual = await strat.calculate_confidence(state)

    assert actual == pytest.approx(expected, abs=1e-9), (
        f"expected configured-weight confidence {expected}, got {actual}"
    )
    assert actual != pytest.approx(hardcoded, abs=1e-3), (
        f"confidence must diverge from the hardcoded-weight formula "
        f"({hardcoded}) when custom weights are set; got {actual}"
    )


@pytest.mark.asyncio
async def test_default_config_matches_hardcoded_formula():
    """A default ``AdaptiveConfig()`` (weights ``0.4 / 0.3 / 0.3``) MUST yield
    the same confidence as the legacy hardcoded formula. This is the
    byte-for-byte match that previously masked the bug -- it must continue to
    hold so existing callers see no behavior change when the fix is later
    refactored."""
    strat = StatisticalStrategy()
    assert strat.config.coverage_weight == 0.4
    assert strat.config.consistency_weight == 0.3
    assert strat.config.saturation_weight == 0.3

    state = await _build_state(strat)

    coverage = strat._calculate_coverage(state)
    consistency = strat._calculate_consistency(state)
    saturation = strat._calculate_saturation(state)
    hardcoded = 0.4 * coverage + 0.3 * consistency + 0.3 * saturation

    actual = await strat.calculate_confidence(state)
    assert actual == pytest.approx(hardcoded, abs=1e-12)


@pytest.mark.asyncio
async def test_standalone_strategy_has_default_config():
    """A directly-constructed ``StatisticalStrategy()`` MUST expose a default
    ``AdaptiveConfig()`` so ``calculate_confidence`` does not raise
    ``AttributeError`` when called without an explicit config assignment.
    Guards the standalone-construction pattern used by ``tests/adaptive/
    test_confidence_debug.py`` and by users who instantiate the strategy
    directly."""
    strat = StatisticalStrategy()
    assert isinstance(strat.config, AdaptiveConfig)
    state = await _build_state(strat)
    confidence = await strat.calculate_confidence(state)
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# User-visible impact: should_stop decision flips for tuned thresholds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_should_stop_uses_tuned_confidence_to_flips_termination():
    """The reported impact: with custom weights ``(0.7, 0.15, 0.15)`` and
    ``confidence_threshold=0.75``, the tuned confidence trips the threshold
    and the REAL ``should_stop`` returns ``True``, where the buggy hardcoded
    confidence (``0.5857``) would fall short and return ``False``.
    ``pending_links`` is populated so the "no links left" guard does not fire;
    this isolates the confidence-threshold guard and proves the fix flips the
    user-visible stop/continue decision in the bug-report interval."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig(
        coverage_weight=0.7,
        consistency_weight=0.15,
        saturation_weight=0.15,
        confidence_threshold=0.75,
    )
    strat.config.validate()

    state = await _build_state(strat)
    state.pending_links = [Link(href="http://x/c", text="machine learning docs more")]

    tuned_confidence = await strat.calculate_confidence(state)
    state.metrics["confidence"] = tuned_confidence
    assert await strat.should_stop(state, strat.config) is True

    # Counterfactual: had the bug been present, the hardcoded confidence
    # (0.5857) would NOT trip the 0.75 threshold, so should_stop would be
    # False -- i.e. the tuned early stop would be missed.
    buggy_confidence = 0.4 * strat._calculate_coverage(state) \
                       + 0.3 * strat._calculate_consistency(state) \
                       + 0.3 * strat._calculate_saturation(state)
    state.metrics["confidence"] = buggy_confidence
    assert await strat.should_stop(state, strat.config) is False


# ---------------------------------------------------------------------------
# _create_strategy now passes config to StatisticalStrategy (mirrors the
# existing EmbeddingStrategy pattern)
# ---------------------------------------------------------------------------

def test_create_strategy_passes_config_to_statistical_strategy():
    """``AdaptiveCrawler._create_strategy("statistical")`` MUST attach the
    crawler's ``AdaptiveConfig`` to the returned strategy, mirroring the
    existing ``EmbeddingStrategy`` pattern. Historically this branch returned
    ``StatisticalStrategy()`` without setting ``config``; this guard prevents
    that regression, which would resurface the bug for users who construct
    ``AdaptiveCrawler(config=cfg)`` and call ``calculate_confidence`` /
    ``is_sufficient`` without first running ``digest``."""
    config = AdaptiveConfig(
        coverage_weight=0.7,
        consistency_weight=0.15,
        saturation_weight=0.15,
    )
    config.validate()
    crawler = AdaptiveCrawler(crawler=object(), config=config)

    strategy = crawler._create_strategy("statistical")
    assert isinstance(strategy, StatisticalStrategy)
    assert strategy.config is config
