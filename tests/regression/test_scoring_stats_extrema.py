"""
Regression guard: ``ScoringStats.get_min`` / ``get_max`` must return the true
running extrema, not the running average.

Commit ``dad592c`` ("2025 feb alpha 1", PR #685) collapsed the dual
``ScoringStats`` + ``FastScoringStats`` hierarchy in
``crawl4ai/deep_crawling/scorers.py`` into a single class. The lazy
``FastScoringStats`` implementation it kept initialised ``_min_score`` /
``_max_score`` to ``None``, only tracked min/max inside ``update()`` when
those fields were already non-``None`` (which they never were until the
first ``get_min`` / ``get_max`` call), and seeded the field in those
accessors with ``total_score / urls_scored`` (the *average*) instead of
the extremum. The observable consequence was
``get_min() == get_max() == get_average()`` for any input, and scores
recorded before the first accessor call were permanently lost.

These tests pin the fixed contract so a future change cannot silently
re-introduce either the "extrema == average" regression or the
"early-accessor call corrupts tracking" regression, and verify the fix
through the only exported surface (``URLScorer.stats``).
"""

import pytest

from crawl4ai.deep_crawling.scorers import (
    CompositeScorer,
    KeywordRelevanceScorer,
    ScoringStats,
)


def test_no_updates_returns_zero_for_all_accessors():
    # The empty-state contract: all accessors must return 0.0 (matching
    # get_average) and must NOT leak the float('inf') / float('-inf')
    # sentinels used internally.
    s = ScoringStats()
    assert s._urls_scored == 0
    assert s.get_average() == 0.0
    assert s.get_min() == 0.0
    assert s.get_max() == 0.0
    assert s.get_min() == s.get_max() == s.get_average()


def test_distinct_scores_min_and_max_are_extrema_not_average():
    # The exact reproduction from the bug report: 0.1 and 0.9 must yield
    # min=0.1, max=0.9, average=0.5 -- NOT min=max=average=0.5.
    s = ScoringStats()
    s.update(0.1)
    s.update(0.9)
    assert s.get_average() == pytest.approx(0.5)
    assert s.get_min() == pytest.approx(0.1)
    assert s.get_max() == pytest.approx(0.9)
    assert s.get_min() < s.get_average() < s.get_max()


def test_extrema_independent_of_insertion_order():
    # Extrema must be the same regardless of the order in which scores
    # are fed in (guards against any "first value wins" regression).
    for order in ([0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1],
                 [0.3, 0.5, 0.1, 0.4, 0.2]):
        s = ScoringStats()
        for v in order:
            s.update(v)
        assert s._urls_scored == 5
        assert s.get_min() == pytest.approx(0.1)
        assert s.get_max() == pytest.approx(0.5)
        assert s.get_average() == pytest.approx(0.3)


def test_get_min_called_early_does_not_corrupt_subsequent_updates():
    # The second reproduction from the bug report: a lazy accessor that
    # seeds with the average loses the true extremum observed before the
    # first call. With the fix, calling get_min() early must NOT seed
    # _min_score with the running average.
    s = ScoringStats()
    s.update(0.5)
    s.update(0.9)
    assert s.get_min() == pytest.approx(0.5)
    s.update(0.6)   # greater than the true minimum
    s.update(0.95)  # new high
    assert s.get_min() == pytest.approx(0.5)
    assert s.get_max() == pytest.approx(0.95)


def test_keyword_scorer_stats_min_max_average_end_to_end():
    # The end-to-end reproduction: the only way a user reaches ScoringStats
    # is via the exported URLScorer.stats property. Verify the fix is
    # observable through that surface.
    sc = KeywordRelevanceScorer(keywords=["cat", "dog", "fish"])
    assert sc.score("http://x/cat") == pytest.approx(1.0 / 3.0)
    assert sc.score("http://x/cat-dog-fish") == pytest.approx(1.0)
    assert sc.stats is sc.stats  # same instance, not rebuilt per access

    st = sc.stats
    assert st._urls_scored == 2
    assert st.get_average() == pytest.approx((1.0 / 3.0 + 1.0) / 2.0)
    assert st.get_min() == pytest.approx(1.0 / 3.0)
    assert st.get_max() == pytest.approx(1.0)


def test_composite_scorer_stats_track_extrema():
    # CompositeScorer wires its own ScoringStats via self.stats.update()
    # in CompositeScorer.score(); verify the extrema are correct there too.
    keyword = KeywordRelevanceScorer(keywords=["python", "blog"])
    composite = CompositeScorer([keyword], normalize=True)
    composite.score("https://example.com/python-blog")    # 1.0
    composite.score("https://example.com/other")          # 0.0
    composite.score("https://example.com/python-only")    # 0.5

    st = composite.stats
    assert st._urls_scored == 3
    assert st.get_min() == pytest.approx(0.0)
    assert st.get_max() == pytest.approx(1.0)
    assert st.get_average() == pytest.approx(0.5)


def test_get_average_contract_unchanged():
    # The fix is scoped to min/max; get_average must not regress. It must
    # remain total/urls_scored (0.0 for empty).
    s = ScoringStats()
    assert s.get_average() == 0.0
    s.update(2.0)
    s.update(4.0)
    s.update(6.0)
    assert s.get_average() == pytest.approx(4.0)
