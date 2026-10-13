import os, sys
import pytest

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai.content_filter_strategy import PruningContentFilter


@pytest.fixture
def basic_html():
    return """
    <html>
        <body>
            <article>
                <h1>Main Article</h1>
                <p>This is a high-quality paragraph with substantial text content. It contains enough words to pass the threshold and has good text density without too many links. This kind of content should survive the pruning process.</p>
                <div class="sidebar">Low quality sidebar content</div>
                <div class="social-share">Share buttons</div>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def link_heavy_html():
    return """
    <html>
        <body>
            <div class="content">
                <p>Good content paragraph that should remain.</p>
                <div class="links">
                    <a href="#">Link 1</a>
                    <a href="#">Link 2</a>
                    <a href="#">Link 3</a>
                    <a href="#">Link 4</a>
                </div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def mixed_content_html():
    return """
    <html>
        <body>
            <article>
                <h1>Article Title</h1>
                <p class="summary">Short summary.</p>
                <div class="content">
                    <p>Long high-quality paragraph with substantial content that should definitely survive the pruning process. This content has good text density and proper formatting which makes it valuable for retention.</p>
                </div>
                <div class="comments">
                    <p>Short comment 1</p>
                    <p>Short comment 2</p>
                </div>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def link_heavy_mixed_content_html():
    """Link-heavy widget whose anchors are mixed-content (icon span + trailing
    text) and nested under ``<ul><li>`` — the common CMS shape that the
    ``a.string`` + ``find_all("a", recursive=False)`` bug misreads as having
    zero link text, inflating link_density and preventing pruning."""
    return """
    <html>
        <body>
            <article>
                <p>Main article body with substantive relevant content for analysis here,
                with enough words to clear any reasonable minimum threshold for retention of
                meaningful prose.</p>
            </article>
            <div class="widget"><ul>
                <li><a href="/p0"><span class="ico" data-x="1"></span>just three words</a></li>
                <li><a href="/p1"><span class="ico" data-x="1"></span>just three words</a></li>
                <li><a href="/p2"><span class="ico" data-x="1"></span>just three words</a></li>
            </ul></div>
        </body>
    </html>
    """


@pytest.fixture
def direct_mixed_content_html():
    """Link-heavy widget with mixed-content anchors as direct children of the
    scored node. Isolates the ``a.string`` flaw: the anchors are direct
    children (so ``recursive=False`` would still find them), but each wraps an
    icon ``<span>`` plus trailing text, so ``a.string`` is ``None``."""
    return """
    <html>
        <body>
            <article>
                <p>Main article body with substantive relevant content for analysis here,
                with enough words to clear any reasonable minimum threshold for retention of
                meaningful prose.</p>
            </article>
            <div class="widget">
                <a href="/p0"><span class="ico" data-x="1"></span>just three words</a>
                <a href="/p1"><span class="ico" data-x="1"></span>just three words</a>
                <a href="/p2"><span class="ico" data-x="1"></span>just three words</a>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def prose_with_inline_links_html():
    """Substantive prose paragraph containing legitimate inline mixed-content
    links. Used as a regression guard to confirm the link-density fix does not
    over-prune legitimate content that happens to contain anchors."""
    return """
    <html>
        <body>
            <article>
                <p>This is a high-quality paragraph with substantial substantive text
                content that has good text density. It discusses
                <a href="/wiki/Apple">the apple fruit</a> and
                <a href="/wiki/Malus"><span class="icon"></span>its genus Malus</a>
                in considerable detail with many words to ensure it passes thresholds.</p>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def class_id_boilerplate_html():
    """Boilerplate ``<div>`` whose ``class`` AND ``id`` both match the
    negative-pattern regex (``nav`` / ``footer``). It is a ``<div>`` (not a
    ``<nav>`` / ``<footer>`` tag), so ``_remove_unwanted_tags`` does not catch it
    by tag name — only the ``class_id_weight`` penalty can prune it. At the
    default ``threshold=0.48`` its other metrics put it just above the cutoff, so
    the penalty is the deciding factor (bug: kept; fix: pruned)."""
    return """
    <html><body>
    <div class="nav" id="footer"><ul>
    <li><a href="/">Home</a></li>
    <li><a href="/about">About</a></li>
    <li><a href="/contact">Contact</a></li>
    </ul>Menu</div>
    <main><article><p>This is the genuine main article body with sufficient
    substantive words to clear the word threshold and remain in the filtered
    output.</p></article></main>
    </body></html>
    """


@pytest.fixture
def class_id_control_html():
    """Same structure as ``class_id_boilerplate_html`` but with non-boilerplate
    class/id (``content`` / ``story``). Used as a contrast to show the
    ``class_id_weight`` penalty targets only negative-pattern matches and does
    not penalize legitimate containers. In fixed mode this control div survives
    at ``threshold=0.48`` while its boilerplate twin is pruned."""
    return """
    <html><body>
    <div class="content" id="story"><ul>
    <li><a href="/">Home</a></li>
    <li><a href="/about">About</a></li>
    <li><a href="/contact">Contact</a></li>
    </ul>Menu</div>
    <main><article><p>This is the genuine main article body with sufficient
    substantive words to clear the word threshold and remain in the filtered
    output.</p></article></main>
    </body></html>
    """


class TestPruningContentFilter:
    def test_basic_pruning(self, basic_html):
        """Test basic content pruning functionality"""
        filter = PruningContentFilter(min_word_threshold=5)
        contents = filter.filter_content(basic_html)

        combined_content = " ".join(contents).lower()
        assert "high-quality paragraph" in combined_content
        assert "sidebar content" not in combined_content
        assert "share buttons" not in combined_content

    def test_min_word_threshold(self, mixed_content_html):
        """Test minimum word threshold filtering"""
        filter = PruningContentFilter(min_word_threshold=10)
        contents = filter.filter_content(mixed_content_html)

        combined_content = " ".join(contents).lower()
        assert "short summary" not in combined_content
        assert "long high-quality paragraph" in combined_content
        assert "short comment" not in combined_content

    def test_dynamic_threshold_preserves_short_paragraph(self):
        """Dynamic mode relaxes the cutoff for prose tags and text density."""
        html = """<body><article>
            <p>Substantive article prose with enough detail to remain useful
            under a strict cutoff in either pruning mode.</p>
            <p><span>Brief summary.</span></p>
            <div class="sidebar"><a href="/share">Share</a></div>
        </article></body>"""
        fixed = " ".join(PruningContentFilter(
            threshold_type="fixed", threshold=0.9
        ).filter_content(html))
        dynamic = " ".join(PruningContentFilter(
            threshold_type="dynamic", threshold=0.9
        ).filter_content(html))

        assert "Brief summary." not in fixed
        assert "Brief summary." in dynamic
        for content in (fixed, dynamic):
            assert "Substantive article prose" in content
            assert "Share" not in content

    def test_link_density_impact(self, link_heavy_html):
        """Test handling of link-heavy content"""
        filter = PruningContentFilter(threshold_type="dynamic")
        contents = filter.filter_content(link_heavy_html)

        combined_content = " ".join(contents).lower()
        assert "good content paragraph" in combined_content
        assert (
            len([c for c in contents if "href" in c]) < 2
        ), "Should prune link-heavy sections"

    def test_link_density_impact_mixed_content(self, link_heavy_mixed_content_html):
        """Link-heavy widget with mixed-content anchors nested under <ul><li>
        must be pruned at the default fixed threshold.

        Regression test for the link_text_len bug: ``a.string`` returns None for
        anchors with multiple NavigableString children (e.g. icon span + trailing
        text) and ``find_all("a", recursive=False)`` misses anchors nested under
        ``<ul><li>``, so link_text_len undercounts to 0, link_density inflates to
        1.0, and the link-heavy widget survives instead of being pruned.
        """
        filter = PruningContentFilter(threshold_type="fixed", threshold=0.48)
        contents = filter.filter_content(link_heavy_mixed_content_html)

        combined_content = " ".join(contents).lower()
        assert "main article body" in combined_content, "Article body should survive"
        assert "just three words" not in combined_content, (
            "Link-heavy widget with mixed-content anchors should be pruned"
        )

    def test_link_density_impact_mixed_content_dynamic(
        self, link_heavy_mixed_content_html
    ):
        """Link-heavy mixed-content widget must also be pruned under the dynamic
        threshold, where the ``link_ratio > 0.6`` penalty is silenced by the bug
        (link_text_len=0 yields link_ratio=0 for link-bearing nodes)."""
        filter = PruningContentFilter(threshold_type="dynamic", threshold=0.48)
        contents = filter.filter_content(link_heavy_mixed_content_html)

        combined_content = " ".join(contents).lower()
        assert "main article body" in combined_content, "Article body should survive"
        assert "just three words" not in combined_content, (
            "Link-heavy widget with mixed-content anchors should be pruned"
        )

    def test_mixed_content_direct_anchor_pruning(self, direct_mixed_content_html):
        """Direct-child mixed-content anchors must be pruned at a stricter
        threshold. Isolates the ``a.string`` flaw: the anchors are direct
        children (so ``recursive=False`` finds them), but ``a.string`` is None
        because each anchor wraps an icon span plus trailing text."""
        filter = PruningContentFilter(threshold_type="fixed", threshold=0.60)
        contents = filter.filter_content(direct_mixed_content_html)

        combined_content = " ".join(contents).lower()
        assert "main article body" in combined_content, "Article body should survive"
        assert "just three words" not in combined_content, (
            "Link-heavy widget with mixed-content anchors should be pruned"
        )

    def test_inline_links_in_prose_preserved(self, prose_with_inline_links_html):
        """Legitimate inline mixed-content links embedded in substantive prose
        must not be over-pruned by the link-density fix. Regression guard."""
        filter = PruningContentFilter(threshold_type="fixed", threshold=0.48)
        contents = filter.filter_content(prose_with_inline_links_html)

        combined_content = " ".join(contents).lower()
        assert "high-quality paragraph" in combined_content, (
            "Prose with inline links should not be over-pruned"
        )

    def test_tag_importance(self, mixed_content_html):
        """Test tag importance in scoring"""
        filter = PruningContentFilter(threshold_type="dynamic")
        contents = filter.filter_content(mixed_content_html)

        has_article = any("article" in c.lower() for c in contents)
        has_h1 = any("h1" in c.lower() for c in contents)
        assert has_article or has_h1, "Should retain important tags"

    def test_empty_input(self):
        """Test handling of empty input"""
        filter = PruningContentFilter()
        assert filter.filter_content("") == []
        assert filter.filter_content(None) == []

    def test_malformed_html(self):
        """Test handling of malformed HTML"""
        malformed_html = "<div>Unclosed div<p>Nested<span>content</div>"
        filter = PruningContentFilter()
        contents = filter.filter_content(malformed_html)
        assert isinstance(contents, list)

    def test_performance(self, basic_html):
        """Test performance with timer"""
        filter = PruningContentFilter()

        import time

        start = time.perf_counter()
        filter.filter_content(basic_html)
        duration = time.perf_counter() - start

        # Extra strict on performance since you mentioned milliseconds matter
        assert duration < 0.1, f"Processing took too long: {duration:.3f} seconds"

    @pytest.mark.parametrize(
        "threshold,expected_count",
        [
            (0.3, 4),  # Very lenient
            (0.48, 2),  # Default
            (0.7, 1),  # Very strict
        ],
    )
    def test_threshold_levels(self, mixed_content_html, threshold, expected_count):
        """Test different threshold levels"""
        filter = PruningContentFilter(threshold_type="fixed", threshold=threshold)
        contents = filter.filter_content(mixed_content_html)
        assert (
            len(contents) <= expected_count
        ), f"Expected {expected_count} or fewer elements with threshold {threshold}"

    def test_consistent_output(self, basic_html):
        """Test output consistency across multiple runs"""
        filter = PruningContentFilter()
        first_run = filter.filter_content(basic_html)
        second_run = filter.filter_content(basic_html)
        assert first_run == second_run, "Output should be consistent"


class TestClassIdWeightPenalty:
    """Regression tests for the ``class_id_weight`` penalty metric.

    The penalty was previously computed and then discarded by a
    ``max(0, class_score)`` clamp in ``_compute_composite_score``. Because
    ``_compute_class_id_weight`` only ever returns ``0`` or negative values,
    ``max(0, ...)`` was always ``0`` and the metric was a no-op. These tests
    guard against that clamp returning and verify the penalty actually lowers a
    boilerplate node's score so it gets pruned without over-pruning legitimate
    containers.
    """

    def test_class_id_penalty_lowers_composite_score(self):
        """For identical structure, a boilerplate class/id node must score
        strictly lower than a non-boilerplate one. The double-match delta is
        exactly 0.10 (= weight 0.1 * penalty -1.0 / total_weight 1.0) and the
        single-match delta 0.05. Fails on the buggy clamp (all three equal)."""
        from bs4 import BeautifulSoup

        inner = (
            "<ul><li><a href=\"/\">Home</a></li>"
            "<li><a href=\"/about\">About</a></li>"
            "<li><a href=\"/contact\">Contact</a></li></ul>Menu"
        )
        f = PruningContentFilter()

        def score(div_html):
            soup = BeautifulSoup(f"<html><body>{div_html}</body></html>", "lxml")
            div = soup.find("div")
            text_len = len(div.get_text(strip=True))
            tag_len = len(div.encode_contents().decode("utf-8"))
            link_text_len = sum(
                len(a.get_text(strip=True)) for a in div.find_all("a")
            )
            metrics = {
                "node": div,
                "tag_name": div.name,
                "text_len": text_len,
                "tag_len": tag_len,
                "link_text_len": link_text_len,
            }
            return f._compute_composite_score(
                metrics, text_len, tag_len, link_text_len
            )

        control = score(f'<div class="content" id="story">{inner}</div>')
        single = score(f'<div class="nav" id="story">{inner}</div>')
        double = score(f'<div class="nav" id="footer">{inner}</div>')

        assert double < single < control, (
            "penalty must lower score monotonically with match count"
        )
        assert round(control - double, 4) == 0.10, (
            "double-match penalty delta should be 0.10"
        )
        assert round(control - single, 4) == 0.05, (
            "single-match penalty delta should be 0.05"
        )

    def test_class_id_penalty_prunes_double_match_boilerplate(
        self, class_id_boilerplate_html
    ):
        """End-to-end: a <div class="nav" id="footer"> boilerplate block that
        escapes tag-based removal is pruned at the default fixed threshold once
        the class_id_weight penalty is applied. Fails on the buggy clamp."""
        f = PruningContentFilter(threshold_type="fixed", threshold=0.48)
        contents = f.filter_content(class_id_boilerplate_html)
        combined = " ".join(contents).lower()

        boilerplate_kept = any("<div" in b and "Menu" in b for b in contents)
        assert not boilerplate_kept, (
            "class_id_weight penalty was not applied; boilerplate survived"
        )
        assert "main article body" in combined, "main content must survive"

    def test_class_id_penalty_does_not_prune_non_boilerplate_control(
        self, class_id_control_html
    ):
        """Contrast: an identical-structure div with non-boilerplate class/id
        (content / story) is NOT penalized and survives at the same threshold
        where its boilerplate twin is pruned. Guards against the penalty being
        over-broad (false positives on legitimate containers)."""
        f = PruningContentFilter(threshold_type="fixed", threshold=0.48)
        contents = f.filter_content(class_id_control_html)
        combined = " ".join(contents).lower()

        control_div_kept = any("<div" in b and "Menu" in b for b in contents)
        assert control_div_kept, (
            "non-boilerplate control div should survive at 0.48 (no penalty)"
        )
        assert "main article body" in combined, "main content must survive"


if __name__ == "__main__":
    pytest.main([__file__])
