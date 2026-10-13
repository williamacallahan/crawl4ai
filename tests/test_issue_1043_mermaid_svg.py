"""
Tests for issue #1043: Missing Mermaid Flowcharts

Verifies that mermaid SVG diagrams are preserved as text content
during HTML scraping, rather than being stripped entirely.
"""

import pytest
from lxml import html as lhtml
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

_MERMAID_FENCE = "```mermaid"


def _gen_markdown(strategy, html, **scrap_kwargs):
    """Run the default scrape -> markdown pipeline and return raw_markdown.

    Mirrors the production default path: ``LXMLWebScrapingStrategy._scrap``
    with the given kwargs (``keep_data_attributes`` defaults to ``False`` in
    ``CrawlerRunConfig``) feeds ``DefaultMarkdownGenerator.generate_markdown``,
    which is what populates ``CrawlResult.markdown``.
    """
    cleaned = strategy._scrap("http://test.com", html, **scrap_kwargs).get(
        "cleaned_html", ""
    )
    return DefaultMarkdownGenerator().generate_markdown(
        cleaned, base_url="http://test.com"
    ).raw_markdown


@pytest.fixture
def strategy():
    return LXMLWebScrapingStrategy()


def _make_html(body_content: str) -> str:
    return f"<html><body>{body_content}</body></html>"


# -- Mermaid SVG detection and replacement --

FLOWCHART_SVG = """
<div>
    <p>Before diagram</p>
    <svg id="mermaid-abc123" aria-roledescription="flowchart-v2" xmlns="http://www.w3.org/2000/svg">
        <g class="node"><foreignObject><div><span class="nodeLabel">Start</span></div></foreignObject></g>
        <g class="node"><foreignObject><div><span class="nodeLabel">Process Data</span></div></foreignObject></g>
        <g class="node"><foreignObject><div><span class="nodeLabel">End</span></div></foreignObject></g>
        <g class="edgeLabel"><foreignObject><div><span>yes</span></div></foreignObject></g>
    </svg>
    <p>After diagram</p>
</div>
"""

CLASS_DIAGRAM_SVG = """
<div>
    <svg id="mermaid-def456" aria-roledescription="class" xmlns="http://www.w3.org/2000/svg">
        <g class="node"><foreignObject><div><span class="nodeLabel">MyClass</span></div></foreignObject></g>
        <g class="node"><foreignObject><div><span class="nodeLabel">+method() : void</span></div></foreignObject></g>
        <g class="node"><foreignObject><div><span class="nodeLabel">-field : int</span></div></foreignObject></g>
    </svg>
</div>
"""

SEQUENCE_SVG = """
<div>
    <svg id="mermaid-seq789" aria-roledescription="sequence" xmlns="http://www.w3.org/2000/svg">
        <g class="label"><foreignObject><div><span>Alice</span></div></foreignObject></g>
        <g class="label"><foreignObject><div><span>Bob</span></div></foreignObject></g>
        <g class="edgeLabel"><foreignObject><div><span>Hello</span></div></foreignObject></g>
    </svg>
</div>
"""


class TestMermaidSVGDetection:
    """Test that mermaid SVGs are detected by their id prefix."""

    def test_flowchart_svg_detected(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        assert result is not None
        cleaned = result.get("cleaned_html", "")
        assert "Start" in cleaned
        assert "Process Data" in cleaned

    def test_non_mermaid_svg_not_affected(self, strategy):
        """Regular SVGs without mermaid id should be unaffected."""
        html = _make_html("""
            <div>
                <svg id="logo" xmlns="http://www.w3.org/2000/svg">
                    <text>Logo Text</text>
                </svg>
                <p>Content here</p>
            </div>
        """)
        result = strategy._scrap("http://test.com", html)
        assert result is not None

    def test_mermaid_svg_replaced_with_pre_code(self, strategy):
        """Mermaid SVG should be replaced with pre/code block.

        The ``class="language-mermaid"`` signal is what survives the default
        attribute-stripping pass (``class`` is in IMPORTANT_ATTRS); assert it
        precisely rather than the permissive ``"mermaid" in cleaned.lower()``
        which passes from the SVG's own ``id="mermaid-*"`` substring.
        """
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert 'class="language-mermaid"' in cleaned, repr(cleaned)


class TestMermaidMarkdownFenceLabel:
    r"""Regression tests for the ```mermaid fence language label.

    Background: ``content_scraping_strategy.py`` writes ``<pre
    data-language="mermaid"><code class="language-mermaid">…</code></pre>`` for
    each labeled, unwrapped mermaid SVG. The default crawl path runs
    ``remove_unwanted_attributes_fast(keep_data_attributes=False)`` which
    strips ``data-language`` (not in IMPORTANT_ATTRS) but keeps ``class`` (in
    IMPORTANT_ATTRS). ``CustomHTML2Text`` must therefore recover the fence
    language from the surviving ``class="language-mermaid"`` so that
    ``CrawlResult.markdown`` carries a ``\`\`\`mermaid`` fence (not a bare
    ``\`\`\``), as introduced in commit dba38c78.
    """

    def test_default_path_emits_mermaid_fence_label(self, strategy):
        """Default keep_data_attributes=False path labels the fence mermaid."""
        html = _make_html(FLOWCHART_SVG)
        md = _gen_markdown(strategy, html)  # default: keep_data_attributes=False
        assert _MERMAID_FENCE in md, repr(md)

    def test_control_path_still_emits_mermaid_fence_label(self, strategy):
        """The keep_data_attributes=True control path must keep working."""
        html = _make_html(FLOWCHART_SVG)
        md = _gen_markdown(strategy, html, keep_data_attributes=True)
        assert _MERMAID_FENCE in md, repr(md)

    def test_default_and_control_paths_match(self, strategy):
        """Default and control paths produce identical markdown bodies.

        Only the path that recovers the label changes; the fence body (diagram
        type comment + labels) and surrounding content are byte-identical.
        """
        html = _make_html(FLOWCHART_SVG)
        md_default = _gen_markdown(strategy, html)
        md_control = _gen_markdown(strategy, html, keep_data_attributes=True)
        assert md_default == md_control, (repr(md_default), repr(md_control))

    @pytest.mark.parametrize(
        "svg,expected_label",
        [
            (FLOWCHART_SVG, "flowchart-v2"),
            (CLASS_DIAGRAM_SVG, "class"),
            (SEQUENCE_SVG, "sequence"),
        ],
        ids=["flowchart", "class", "sequence"],
    )
    def test_each_diagram_type_emits_mermaid_fence(
        self, strategy, svg, expected_label
    ):
        """All diagram variants produce a ```mermaid fence carrying the
        extracted labels and the diagram-type comment line."""
        html = _make_html(svg)
        md = _gen_markdown(strategy, html)
        assert _MERMAID_FENCE in md, repr(md)
        assert f"%% {expected_label} diagram" in md, repr(md)

    def test_mermaid_inside_pre_does_not_emit_mermaid_fence(self, strategy):
        """A mermaid SVG already wrapped in a <pre> uses the span branch and
        must NOT produce a ```mermaid fence (it rides the outer <pre>), so
        there is no fence-language to lose or fabricate."""
        svg = (
            '<svg id="mermaid-inpre" aria-roledescription="flowchart-v2"'
            ' xmlns="http://www.w3.org/2000/svg">'
            '<g class="node"><foreignObject><div>'
            '<span class="nodeLabel">Start</span></div></foreignObject></g></svg>'
        )
        html = _make_html(f"<pre>code around {svg} more code</pre>")
        md = _gen_markdown(strategy, html)
        assert _MERMAID_FENCE not in md, repr(md)
        assert "Start" in md  # label text still preserved

    def test_labelless_mermaid_svg_emits_no_fence(self, strategy):
        """An SVG with no extractable labels produces no <pre> placeholder
        and hence no fence at all."""
        html = _make_html(
            '<svg id="mermaid-empty" aria-roledescription="flowchart-v2"'
            ' xmlns="http://www.w3.org/2000/svg">'
            '<rect width="10" height="10"/></svg>'
        )
        md = _gen_markdown(strategy, html)
        assert "```" not in md, repr(md)

    def test_multiple_mermaid_svgs_each_get_own_mermaid_fence(self, strategy):
        """Multiple diagrams on one page each produce their own ```mermaid
        fence (the per-SVG <pre> placeholder is independent)."""
        html = _make_html(FLOWCHART_SVG + CLASS_DIAGRAM_SVG)
        md = _gen_markdown(strategy, html)
        assert md.count(_MERMAID_FENCE) == 2, repr(md)
        assert "Start" in md and "MyClass" in md


class TestMermaidFitMarkdownFenceLabel:
    """fit_markdown (generated only when a content_filter is configured) runs
    over the same stripped cleaned_html and must also carry ```mermaid."""

    def test_fit_markdown_carries_mermaid_fence_label(self, strategy):
        PruningContentFilter = pytest.importorskip(
            "crawl4ai.content_filter_strategy"
        ).PruningContentFilter
        html = _make_html(
            "<p>Before diagram with enough surrounding prose to clear the "
            "pruning threshold for the page body.</p>" + FLOWCHART_SVG
            + "<p>After diagram with additional surrounding prose too.</p>"
        )
        cleaned = strategy._scrap("http://test.com", html).get("cleaned_html", "")
        res = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter()
        ).generate_markdown(cleaned, base_url="http://test.com")
        assert _MERMAID_FENCE in res.raw_markdown, repr(res.raw_markdown)
        assert _MERMAID_FENCE in (res.fit_markdown or ""), repr(res.fit_markdown)
        assert "language-mermaid" in (res.fit_html or ""), repr(res.fit_html)


class TestMermaidTextExtraction:
    """Test that text content is correctly extracted from mermaid SVGs."""

    def test_node_labels_extracted(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "Start" in cleaned
        assert "Process Data" in cleaned
        assert "End" in cleaned

    def test_edge_labels_extracted(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "yes" in cleaned

    def test_class_diagram_labels_extracted(self, strategy):
        html = _make_html(CLASS_DIAGRAM_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "MyClass" in cleaned
        assert "+method() : void" in cleaned

    def test_sequence_diagram_labels_extracted(self, strategy):
        html = _make_html(SEQUENCE_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "Alice" in cleaned
        assert "Bob" in cleaned

    def test_duplicate_labels_deduplicated(self, strategy):
        """Same label appearing multiple times should only appear once."""
        html = _make_html("""
            <div>
                <svg id="mermaid-dup" aria-roledescription="flowchart-v2" xmlns="http://www.w3.org/2000/svg">
                    <g class="node"><foreignObject><div><span class="nodeLabel">Repeated</span></div></foreignObject></g>
                    <g class="node"><foreignObject><div><span class="nodeLabel">Repeated</span></div></foreignObject></g>
                    <g class="node"><foreignObject><div><span class="nodeLabel">Unique</span></div></foreignObject></g>
                </svg>
            </div>
        """)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        # Should have Repeated once, not twice
        assert cleaned.count("Repeated") == 1
        assert "Unique" in cleaned


class TestMermaidDiagramType:
    """Test that diagram type is preserved."""

    def test_flowchart_type_preserved(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "flowchart" in cleaned.lower()

    def test_class_type_preserved(self, strategy):
        html = _make_html(CLASS_DIAGRAM_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "class" in cleaned.lower()

    def test_sequence_type_preserved(self, strategy):
        html = _make_html(SEQUENCE_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "sequence" in cleaned.lower()


class TestMermaidSurroundingContent:
    """Test that surrounding content is preserved."""

    def test_text_before_diagram_preserved(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "Before diagram" in cleaned

    def test_text_after_diagram_preserved(self, strategy):
        html = _make_html(FLOWCHART_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "After diagram" in cleaned


class TestMermaidEdgeCases:
    """Test edge cases for mermaid SVG handling."""

    def test_empty_mermaid_svg(self, strategy):
        """SVG with no text content should be handled gracefully."""
        html = _make_html("""
            <div>
                <svg id="mermaid-empty" aria-roledescription="flowchart-v2" xmlns="http://www.w3.org/2000/svg">
                    <rect width="100" height="100"/>
                </svg>
                <p>Content</p>
            </div>
        """)
        result = strategy._scrap("http://test.com", html)
        assert result is not None
        cleaned = result.get("cleaned_html", "")
        assert "Content" in cleaned

    def test_multiple_mermaid_svgs(self, strategy):
        """Multiple mermaid diagrams on one page."""
        html = _make_html(FLOWCHART_SVG + CLASS_DIAGRAM_SVG)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "Start" in cleaned
        assert "MyClass" in cleaned

    def test_mermaid_svg_no_aria(self, strategy):
        """Mermaid SVG without aria-roledescription should use 'diagram' fallback."""
        html = _make_html("""
            <div>
                <svg id="mermaid-noaria" xmlns="http://www.w3.org/2000/svg">
                    <g class="node"><foreignObject><div><span class="nodeLabel">Node A</span></div></foreignObject></g>
                </svg>
            </div>
        """)
        result = strategy._scrap("http://test.com", html)
        cleaned = result.get("cleaned_html", "")
        assert "Node A" in cleaned
        assert "diagram" in cleaned.lower()

    def test_mermaid_svg_malformed_no_crash(self, strategy):
        """Malformed SVG should not crash the scraper."""
        html = _make_html("""
            <div>
                <svg id="mermaid-bad" xmlns="http://www.w3.org/2000/svg">
                </svg>
                <p>Still works</p>
            </div>
        """)
        result = strategy._scrap("http://test.com", html)
        assert result is not None
        cleaned = result.get("cleaned_html", "")
        assert "Still works" in cleaned
