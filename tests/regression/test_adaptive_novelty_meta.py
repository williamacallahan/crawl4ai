"""
Regression tests for ``StatisticalStrategy._calculate_novelty`` meta-field
accessor in ``crawl4ai/adaptive_crawler.py``.

``_calculate_novelty`` MUST read ``description`` / ``keywords`` from the nested
``meta`` subdict of ``head_data`` (the schema produced by ``_parse_head`` in
``crawl4ai/async_url_seeder.py`` and the canonical accessor used by the sibling
``_calculate_relevance`` method and the seeder's own ``_extract_text_context``).
The original adaptive-crawling commit shipped with top-level
``head_data.get('description')`` / ``head_data.get('keywords')`` accessors,
which always returned ``''`` and silently dropped meta terms from the novelty
term set on any page that populated those tags -- perturbing ``rank_links``
ordering.

Runs fully offline: directly instantiates ``StatisticalStrategy`` /
``CrawlState`` and feeds real ``_parse_head`` output through it. No network,
no LLM, no sentence-transformers, no browser.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from crawl4ai import AdaptiveConfig, CrawlResult, MarkdownGenerationResult
from crawl4ai.adaptive_crawler import StatisticalStrategy, CrawlState
from crawl4ai.async_url_seeder import _parse_head
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
    """Build a minimal CrawlResult purely to populate the knowledge base so
    ``_calculate_novelty`` exits its early-return branch."""
    return CrawlResult(url=url, html="", success=True, markdown=_md(content))


def _kb_state(query: str, existing_terms: dict, kb_content: str = "stub") -> CrawlState:
    """Build a CrawlState whose knowledge_base is non-empty (so the buggy
    branch executes) and whose term_frequencies reflect `existing_terms`."""
    state = CrawlState(query=query)
    state.knowledge_base.append(_result("http://seed/", kb_content))
    state.term_frequencies = dict(existing_terms)
    return state


PARSED_HEAD_WITH_META = _parse_head(
    """<!doctype html><html lang="en"><head>
    <title>Python Web Crawling Tutorial - Comprehensive Guide</title>
    <meta name="description" content="Learn asynchronous crawling techniques with Python. Covers asyncio, aiohttp, and best practices for scraping at scale.">
    <meta name="keywords" content="python, web crawling, asyncio, scraping, aiohttp, tutorial">
    </head><body><a href="/tut">Tutorial</a></body></html>"""
)


# ---------------------------------------------------------------------------
# Unit: _calculate_novelty now incorporates nested meta.description/keywords
# ---------------------------------------------------------------------------

def test_novelty_includes_meta_description_and_keywords_terms():
    """The meta description/keywords terms ('scraping', 'aiohttp', 'scale',
    etc.) MUST contribute to the novelty term set. Their terms are NOT in the
    existing knowledge base, so novelty must exceed the value computed from
    only anchor text + page <title>. This guards against the accessor
    regressing back to the top-level ``head_data.get('description')`` /
    ``head_data.get('keywords')`` that always returned ``''``."""
    strat = StatisticalStrategy()
    state = _kb_state(
        query="python asyncio scraping best practices",
        existing_terms={"python": 3, "asyncio": 2, "tutorial": 1,
                        "introduction": 1, "overview": 1},
    )

    link = Link(href="https://example.com/tut", text="Tutorial", title="",
                head_data=PARSED_HEAD_WITH_META)
    novelty_with_meta = strat._calculate_novelty(link, state)

    # Control: same link but with meta description/keywords stripped. This is
    # exactly what the bug effectively did -- always read '' for those fields.
    head_no_meta = {
        "title": PARSED_HEAD_WITH_META["title"],
        "meta": {k: v for k, v in PARSED_HEAD_WITH_META["meta"].items()
                 if k not in ("description", "keywords")},
    }
    link_no_meta = Link(href="https://example.com/tut", text="Tutorial", title="",
                        head_data=head_no_meta)
    novelty_without_meta = strat._calculate_novelty(link_no_meta, state)

    assert novelty_with_meta > novelty_without_meta, (
        f"meta terms should raise novelty: {novelty_with_meta=} "
        f"vs {novelty_without_meta=}"
    )


def test_novelty_unaffected_when_meta_description_and_keywords_absent():
    """No regression for pages that don't populate the meta tags: the nested
    accessor must degrade gracefully to '' (no KeyError, no extra tokens), so
    the score matches what it would have been under the bug ('' vs '')."""
    strat = StatisticalStrategy()
    state = _kb_state(
        query="python asyncio",
        existing_terms={"python": 2, "tutorial": 1},
    )
    head = {"title": "Tutorial", "meta": {}}   # no description/keywords
    link = Link(href="https://x/", text="Tutorial", title="", head_data=head)

    # Only 'tutorial' is in link_text (anchor + <title>); 'tutorial' IS in the
    # KB, so new_terms fraction is 0.
    assert strat._calculate_novelty(link, state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end: rank_links ordering now reflects meta novelty
# ---------------------------------------------------------------------------

def _meta_kb_state():
    """Knowledge base already has {python, asyncio, tutorial, introduction,
    overview, basics}; query is about python+asyncio+scraping."""
    state = CrawlState(query="python asyncio scraping best practices")
    state.knowledge_base.append(_result("http://seed/",
                                         "python asyncio tutorial introduction overview basics"))
    state.term_frequencies = {"python": 3, "asyncio": 2, "tutorial": 1,
                              "introduction": 1, "overview": 1, "basics": 1}
    return state


def _two_identical_links_meta_differs():
    """Two links identical in anchor text, link title, page <title>, and
    contextual_score -- differentiated ONLY by meta content.

      * A's meta is full of terms NEW to the KB and aligned with the query
        (scraping, aiohttp, scale, best, practices).
      * B's meta is full of terms ALREADY in the KB (python, asyncio, tutorial,
        overview, basics, introduction).

    Input order is [B, A] so a stable-sort tie (the bug) selects B first.
    """
    headA = {"title": "Tutorial",
             "meta": {"description": "scraping at scale with aiohttp and best practices",
                      "keywords": "scraping, scale, aiohttp, best, practices"}}
    headB = {"title": "Tutorial",
             "meta": {"description": "python asyncio tutorial overview basics introduction",
                      "keywords": "python, asyncio, tutorial, overview, basics, introduction"}}
    la = Link(href="https://A.example.com/tut", text="Tutorial", title="",
              head_data=headA, contextual_score=0.5)
    lb = Link(href="https://B.example.com/tut", text="Tutorial", title="",
              head_data=headB, contextual_score=0.5)
    return lb, la


@pytest.mark.asyncio
async def test_rank_links_selects_new_vocabulary_link_first():
    """The central impact: with identical anchor/title and identical
    contextual_score, ``rank_links`` must rank the link whose meta introduces
    NEW query-relevant vocabulary (A) above the link whose meta only repeats
    existing KB terms (B). Before the fix both tied at 0.45 and input-order
    tiebreak selected B first; after the fix A scores 0.7125 and B 0.45."""
    strat = StatisticalStrategy()
    state = _meta_kb_state()
    lb, la = _two_identical_links_meta_differs()
    state.pending_links = [lb, la]   # B first to expose a tie-driven flip

    ranked = await strat.rank_links(state, AdaptiveConfig())

    assert [l.href for l, _ in ranked] == [
        "https://A.example.com/tut",
        "https://B.example.com/tut",
    ]
