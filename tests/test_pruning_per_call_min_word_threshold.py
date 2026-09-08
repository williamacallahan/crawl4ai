"""
Tests for the PruningContentFilter.filter_content() per-call ``min_word_threshold``
parameter.

Earlier, ``filter_content(self, html, min_word_threshold=None)`` accepted the
per-call argument but silently ignored it: the pruning logic read only the
constructor value (``self.min_word_threshold``). These tests guard against that
regression by verifying the per-call value is honored, overrides the
constructor value when supplied, falls back to the constructor value when
``None``, and does not mutate instance state.
"""
import pytest

from crawl4ai.content_filter_strategy import PruningContentFilter


BUG_REPORT_HTML = """
<html><head><title>Test Page</title></head><body>
  <article>
    <h1>Main Article Title</h1>
    <p>This is a long enough paragraph with plenty of words to survive any reasonable pruning threshold setting for testing purposes here.</p>
    <p>shorty</p>
    <p>tiny</p>
  </article>
</body></html>
"""

# Paragraphs with deliberately separated word counts (30, 12, 1) so a per-call
# threshold can split them deterministically.
MIXED_WORD_COUNT_HTML = """
<html><head><title>Mixed</title></head><body>
  <article>
    <p id="p30">alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu one two three four</p>
    <p id="p12">alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima</p>
    <p id="p1">solo</p>
  </article>
</body></html>
"""


def _joined(blocks):
    return "\n".join(blocks)


class TestPerCallMinWordThreshold:
    def test_bug_report_reproduction_fixed(self):
        """The exact bug-report scenario: per-call N must behave like constructor N."""
        res_per_call = PruningContentFilter(
            threshold=0.48, threshold_type="fixed"
        ).filter_content(BUG_REPORT_HTML, min_word_threshold=50)
        res_constructor = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=50
        ).filter_content(BUG_REPORT_HTML)

        assert "shorty" not in _joined(res_per_call)
        assert "tiny" not in _joined(res_per_call)
        assert res_per_call == res_constructor

    def test_per_call_is_selective(self):
        """A per-call threshold between the long (21-word) and short (1-word)
        paragraphs must keep the long one and drop the short ones."""
        res = PruningContentFilter(threshold=0.48, threshold_type="fixed").filter_content(
            BUG_REPORT_HTML, min_word_threshold=10
        )
        combined = _joined(res)
        assert "long enough paragraph" in combined
        assert "shorty" not in combined
        assert "tiny" not in combined

    @pytest.mark.parametrize("threshold_type", ["fixed", "dynamic"])
    def test_per_call_equals_constructor_for_same_N(self, threshold_type):
        """Per-call N produces byte-identical output to constructor N for both
        threshold types — the core contract of the fix."""
        res_per_call = PruningContentFilter(
            threshold=0.48, threshold_type=threshold_type
        ).filter_content(MIXED_WORD_COUNT_HTML, min_word_threshold=12)
        res_constructor = PruningContentFilter(
            threshold=0.48, threshold_type=threshold_type, min_word_threshold=12
        ).filter_content(MIXED_WORD_COUNT_HTML)
        assert res_per_call == res_constructor

    def test_per_call_overrides_constructor(self):
        """When both are supplied, the per-call value wins (here: strict 31 over
        lenient 2), matching a filter constructed directly with 31."""
        f = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=2
        )
        res = f.filter_content(MIXED_WORD_COUNT_HTML, min_word_threshold=31)
        expected = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=31
        ).filter_content(MIXED_WORD_COUNT_HTML)
        assert res == expected
        assert "solo" not in _joined(res)

    def test_per_call_none_falls_back_to_constructor(self):
        """Explicit per-call None must fall back to the constructor value
        (no regression to the pre-fix constructor-only path)."""
        f = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=50
        )
        res_none = f.filter_content(BUG_REPORT_HTML, min_word_threshold=None)
        res_default = f.filter_content(BUG_REPORT_HTML)
        assert res_none == res_default
        assert "shorty" not in _joined(res_none)

    def test_constructor_value_not_mutated_by_per_call(self):
        """The instance attribute must not be mutated by a per-call override —
        the same filter reused with different per-call values stays consistent."""
        f = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=31
        )
        assert f.min_word_threshold == 31
        f.filter_content(MIXED_WORD_COUNT_HTML, min_word_threshold=5)
        assert f.min_word_threshold == 31

    def test_preserve_classes_bypass_per_call_word_gate(self):
        """A preserved node with few words must survive a strict per-call
        threshold (the preserve short-circuit runs before the word gate)."""
        html = (
            "<html><body><article>"
            "<p>long paragraph with enough words to survive score pruning comfortably here</p>"
            '<div class="keep-me">solo</div>'
            "</article></body></html>"
        )
        res = PruningContentFilter(
            threshold=0.48, threshold_type="fixed", preserve_classes=["keep-me"]
        ).filter_content(html, min_word_threshold=50)
        assert "solo" in _joined(res)

    def test_per_call_honored_under_dynamic_threshold(self):
        """The per-call argument must also work for threshold_type='dynamic'."""
        res_per_call = PruningContentFilter(
            threshold=0.45, threshold_type="dynamic"
        ).filter_content(MIXED_WORD_COUNT_HTML, min_word_threshold=31)
        res_constructor = PruningContentFilter(
            threshold=0.45, threshold_type="dynamic", min_word_threshold=31
        ).filter_content(MIXED_WORD_COUNT_HTML)
        assert res_per_call == res_constructor
        assert "solo" not in _joined(res_per_call)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
