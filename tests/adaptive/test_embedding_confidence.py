"""
Regression tests for EmbeddingStrategy confidence reporting.

Bug: `EmbeddingStrategy.calculate_confidence()` stores its coverage score under
``state.metrics['coverage_score']`` and returns it. During the crawl loop the
returned value is written to ``state.metrics['confidence']`` (correct).

However, after the crawl loop `AdaptiveCrawler.digest()` calls
``EmbeddingStrategy.get_quality_confidence()`` and overwrites
``state.metrics['confidence']`` with its return value. The old
`get_quality_confidence()` read the *legacy* ``'learning_score'`` key, which the
active code never sets, so it always fell back to ``0.0``. Every embedding-strategy
crawl therefore reported a corrupted, coverage-independent confidence (validated
crawls always showed 0.7; non-validated always showed 0.0) regardless of the
actual coverage score.

The fix makes `get_quality_confidence()` read the ``'coverage_score'`` key that
`calculate_confidence()` stores. These tests guard the key alignment and the
post-loop overwrite path against re-regression.

Runs fully offline: builds numpy embeddings directly and exercises the real
production methods (no network, no LLM, no sentence-transformers).
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from crawl4ai import AdaptiveConfig
from crawl4ai.adaptive_crawler import CrawlState, EmbeddingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy() -> EmbeddingStrategy:
    """Build an EmbeddingStrategy wired to a default AdaptiveConfig.

    Mirrors AdaptiveCrawler._create_strategy('embedding') so the config-dependent
    defaults inside get_quality_confidence() are realistic.
    """
    strategy = EmbeddingStrategy()
    strategy.config = AdaptiveConfig(strategy="embedding")
    return strategy


def _state_with_embeddings(query_embeddings, kb_embeddings) -> CrawlState:
    state = CrawlState(query="test query")
    state.query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    state.kb_embeddings = np.asarray(kb_embeddings, dtype=np.float32)
    return state


# Embedding fixtures. Dimension 4 keeps the math human-verifiable.
_KB = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=np.float32)

# High-coverage queries: both match a KB vector exactly -> best sim 1.0 each
# -> coverage_score = 1.0
_Q_HIGH = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]], dtype=np.float32)

# Low-coverage queries: q0 is orthogonal to every KB doc (best sim 0.0),
# q1 aligns with the third KB doc (best sim ~0.707) -> coverage ~0.354 (< 0.4)
_Q_LOW = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Writer-side key alignment: calculate_confidence stores 'coverage_score'
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculate_confidence_stores_under_coverage_score():
    """calculate_confidence must store its result under 'coverage_score' (the key
    get_quality_confidence reads) and return the same value. Guards against a
    silent rename of the stored key, which is exactly what caused the bug."""
    strategy = _make_strategy()
    state = _state_with_embeddings(_Q_HIGH, _KB)

    score = await strategy.calculate_confidence(state)

    assert score == pytest.approx(1.0)
    assert state.metrics['coverage_score'] == pytest.approx(1.0)
    # The legacy key must NOT be set by the active code path.
    assert 'learning_score' not in state.metrics


# ---------------------------------------------------------------------------
# Reader-side key alignment: get_quality_confidence reads 'coverage_score'
# ---------------------------------------------------------------------------

def test_get_quality_confidence_reads_coverage_score_not_learning_score():
    """get_quality_confidence must read 'coverage_score', not the legacy
    'learning_score' key. Directly guards the bug: setting the legacy key must
    have no effect; only 'coverage_score' drives the result."""
    strategy = _make_strategy()

    # Case A: legacy key set, coverage_score absent -> behaves as coverage 0.0
    state_a = CrawlState(query="test")
    state_a.metrics['learning_score'] = 0.95  # would falsely boost if read
    state_a.metrics['validation_confidence'] = 0.0
    strategy._validation_passed = False
    assert strategy.get_quality_confidence(state_a) == pytest.approx(0.0)

    # Case B: coverage_score set, learning_score left at 0 -> uses coverage_score
    state_b = CrawlState(query="test")
    state_b.metrics['coverage_score'] = 0.85
    state_b.metrics['learning_score'] = 0.0  # legacy must be ignored
    state_b.metrics['validation_confidence'] = 0.0
    strategy._validation_passed = False
    assert strategy.get_quality_confidence(state_b) == pytest.approx(0.85 * 0.8)


# ---------------------------------------------------------------------------
# User-facing symptom: different coverage -> different confidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_distinguishes_coverage_quality():
    """Two crawls with materially different coverage must report materially
    different confidence. Before the fix both reported 0.7 (validated) because
    learning_score was always 0.0 -> < 0.4 -> quality_min floor."""
    strategy = _make_strategy()
    high_state = _state_with_embeddings(_Q_HIGH, _KB)
    low_state = _state_with_embeddings(_Q_LOW, _KB)

    # Let real calculate_confidence populate coverage_score for each.
    await strategy.calculate_confidence(high_state)
    await strategy.calculate_confidence(low_state)
    # Simulate a validated crawl for both.
    for st in (high_state, low_state):
        strategy._validation_passed = True
        st.metrics['validation_confidence'] = 0.6

    high_conf = strategy.get_quality_confidence(high_state)
    low_conf = strategy.get_quality_confidence(low_state)

    # High coverage (1.0 > 0.7) -> quality_max (0.95). Low coverage (~0.354
    # < 0.4) -> quality_min (0.7). They must differ.
    assert high_conf == pytest.approx(strategy.config.embedding_quality_max_confidence)
    assert low_conf == pytest.approx(strategy.config.embedding_quality_min_confidence)
    assert high_conf > low_conf


# ---------------------------------------------------------------------------
# Mechanism: the digest() post-loop overwrite must not corrupt confidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_digest_post_loop_does_not_corrupt_confidence():
    """Reproduce the exact post-loop sequence from AdaptiveCrawler.digest() and
    assert the final 'confidence' metric reflects the coverage score rather than
    the buggy coverage-independent constant.

    Before the fix this sequence overwrote the correct in-loop confidence with
    a value derived from the default 0.0 (since 'learning_score' was never set).
    """
    strategy = _make_strategy()
    state = _state_with_embeddings(_Q_HIGH, _KB)
    strategy._validation_passed = True
    state.metrics['validation_confidence'] = 0.6

    # In-loop confidence (correct)
    in_loop_confidence = await strategy.calculate_confidence(state)
    state.metrics['confidence'] = in_loop_confidence
    assert state.metrics['confidence'] == pytest.approx(1.0)

    # Exact digest() post-loop branch for EmbeddingStrategy:
    #   learning_score = await calculate_confidence(state)
    #   state.metrics['confidence'] = get_quality_confidence(state)
    await strategy.calculate_confidence(state)  # repopulates coverage_score
    assert isinstance(strategy, EmbeddingStrategy)  # mirrors the isinstance check
    state.metrics['confidence'] = strategy.get_quality_confidence(state)

    # The post-loop value must be driven by the real coverage score (1.0 ->
    # quality_max 0.95), NOT the buggy 0.7 constant.
    assert state.metrics['confidence'] == pytest.approx(
        strategy.config.embedding_quality_max_confidence)
    assert state.metrics['confidence'] != pytest.approx(
        strategy.config.embedding_quality_min_confidence)
    assert state.metrics['coverage_score'] == pytest.approx(1.0)
    assert 'learning_score' not in state.metrics


# ---------------------------------------------------------------------------
# Script runner (matches the style of the other tests/adaptive/*.py files)
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("EmbeddingStrategy confidence regression tests")
    print("=" * 60)

    await test_calculate_confidence_stores_under_coverage_score()
    print("PASS: test_calculate_confidence_stores_under_coverage_score")
    test_get_quality_confidence_reads_coverage_score_not_learning_score()
    print("PASS: test_get_quality_confidence_reads_coverage_score_not_learning_score")
    await test_confidence_distinguishes_coverage_quality()
    print("PASS: test_confidence_distinguishes_coverage_quality")
    await test_digest_post_loop_does_not_corrupt_confidence()
    print("PASS: test_digest_post_loop_does_not_corrupt_confidence")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
