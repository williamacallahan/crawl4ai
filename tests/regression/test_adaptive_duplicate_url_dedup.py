"""
Regression tests for ``AdaptiveCrawler`` duplicate-URL deduplication.

Bug: discovered links were appended to ``state.pending_links`` with only a
``crawled_urls`` guard, and ``StatisticalStrategy.rank_links`` filtered only on
``crawled_urls``. Identical-href entries accumulated, filled the ``top_k``
slice, and were crawled multiple times concurrently -- each duplicate counted
as a separate document by ``StatisticalStrategy.update_state`` (inflating
``total_documents`` / term and document frequencies and skewing
coverage/consistency/saturation). ``EmbeddingStrategy.rank_links`` already
deduped by href; the statistical path and the append site did not.

Fix:
1. ``StatisticalStrategy.rank_links`` mirrors ``EmbeddingStrategy.rank_links``
   (skip ``crawled_urls`` and a ``seen_urls`` set in the loop).
2. ``AdaptiveCrawler.digest`` tracks ``pending_urls`` and routes every append
   through ``_extend_pending_links`` which dedupes against ``crawled_urls``
   and ``pending_urls``.

Runs offline: monkeypatches ``AdaptiveCrawler._crawl_with_preview`` and
exercises the real ``digest`` / ``_crawl_batch`` / ``rank_links`` /
``update_state`` code paths. No network, no LLM, no sentence-transformers.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from crawl4ai import AdaptiveConfig, CrawlResult, MarkdownGenerationResult
from crawl4ai.models import Link
from crawl4ai.adaptive_crawler import (
    AdaptiveCrawler,
    StatisticalStrategy,
    CrawlState,
)


# Fixtures / helpers

def _md(text: str) -> MarkdownGenerationResult:
    return MarkdownGenerationResult(
        raw_markdown=text, markdown_with_citations="", references_markdown="",
    )


def _result(url: str, success: bool = True, content: str = "alpha beta gamma", links=None):
    return CrawlResult(
        url=url, html="", success=success, markdown=_md(content), links=links or {},
    )


def _link(href: str, contextual_score: float = 0.5) -> Link:
    return Link(href=href, text=href, contextual_score=contextual_score)


def _make_crawler(
    max_depth: int = 1, top_k_links: int = 3,
    confidence_threshold: float = 1.0, min_gain_threshold: float = 0.0,
    max_pages: int = 50,
) -> AdaptiveCrawler:
    config = AdaptiveConfig(
        confidence_threshold=confidence_threshold, max_depth=max_depth,
        max_pages=max_pages, top_k_links=top_k_links,
        min_gain_threshold=min_gain_threshold,
    )
    crawler = AdaptiveCrawler(crawler=object(), config=config)
    assert isinstance(crawler.strategy, StatisticalStrategy)
    return crawler


# StatisticalStrategy.rank_links dedup

@pytest.mark.asyncio
async def test_rank_links_dedupes_duplicate_hrefs_in_pending_links():
    """Duplicate-href entries in ``pending_links`` must collapse to one ranked
    entry (the canonical nav/footer topology that produced the bug)."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig()
    state = CrawlState(query="alpha")
    state.pending_links = [
        _link("http://x/dup", contextual_score=0.9),
        _link("http://x/dup", contextual_score=0.9),
        _link("http://x/dup", contextual_score=0.9),
        _link("http://x/other", contextual_score=0.5),
    ]

    ranked = await strat.rank_links(state, strat.config)

    hrefs = [link.href for link, _ in ranked]
    assert hrefs.count("http://x/dup") == 1
    assert hrefs.count("http://x/other") == 1
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_rank_links_skips_already_crawled_urls_after_dedup():
    """The new ``seen_urls`` dedup must NOT weaken the existing
    ``crawled_urls`` skip-filter."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig()
    state = CrawlState(query="alpha")
    state.crawled_urls = {"http://x/crawled"}
    state.pending_links = [
        _link("http://x/crawled", contextual_score=0.9),
        _link("http://x/crawled", contextual_score=0.9),
        _link("http://x/fresh", contextual_score=0.5),
    ]

    ranked = await strat.rank_links(state, strat.config)

    assert [link.href for link, _ in ranked] == ["http://x/fresh"]


@pytest.mark.asyncio
async def test_rank_links_no_duplicates_no_regression():
    """When all ``pending_links`` hrefs are already unique, ``rank_links``
    returns all of them ranked by score (pre-fix behavior preserved)."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig()
    state = CrawlState(query="alpha")
    state.pending_links = [
        _link("http://x/a", contextual_score=0.2),
        _link("http://x/b", contextual_score=0.5),
        _link("http://x/c", contextual_score=0.9),
    ]

    ranked = await strat.rank_links(state, strat.config)
    hrefs = [link.href for link, _ in ranked]

    assert set(hrefs) == {"http://x/a", "http://x/b", "http://x/c"}
    assert hrefs == ["http://x/c", "http://x/b", "http://x/a"]


@pytest.mark.asyncio
async def test_statistical_rank_links_dedup_matches_embedding_strategy_behavior():
    """The dedup shape in ``StatisticalStrategy.rank_links`` mirrors the
    existing ``EmbeddingStrategy.rank_links`` dedup, closing the asymmetry
    from the original feature commit (1a73fb60)."""
    strat = StatisticalStrategy()
    strat.config = AdaptiveConfig()
    state = CrawlState(query="alpha")
    state.pending_links = [
        _link("http://x/dup", contextual_score=0.5),
        _link("http://x/dup", contextual_score=0.5),
        _link("http://x/unique", contextual_score=0.5),
    ]

    ranked = await strat.rank_links(state, strat.config)

    assert len(ranked) == 2
    assert {link.href for link, _ in ranked} == {"http://x/dup", "http://x/unique"}


# _extend_pending_links helper

@pytest.mark.asyncio
async def test_extend_pending_links_dedupes_against_crawled_and_pending():
    """``_extend_pending_links`` drops hrefs in ``crawled_urls`` or
    ``pending_urls``, appends new unique hrefs, and updates ``pending_urls``
    in place so subsequent calls reject their duplicates."""
    crawler = _make_crawler()
    crawler.state = CrawlState(query="alpha")
    crawler.state.crawled_urls = {"http://x/crawled"}
    crawler.state.pending_links = [_link("http://x/already_pending")]
    pending_urls = {link.href for link in crawler.state.pending_links}

    new_links = [
        _link("http://x/already_pending"),  # pending -> dropped
        _link("http://x/crawled"),          # crawled -> dropped
        _link("http://x/fresh1"),           # new
        _link("http://x/fresh1"),           # now duplicate -> dropped
        _link("http://x/fresh2"),           # new
        _link(""),                          # empty href -> dropped
    ]

    crawler._extend_pending_links(new_links, pending_urls)

    appended = [link.href for link in crawler.state.pending_links]
    assert appended == ["http://x/already_pending", "http://x/fresh1", "http://x/fresh2"]
    assert pending_urls == {"http://x/already_pending", "http://x/fresh1", "http://x/fresh2"}

    crawler._extend_pending_links(
        [_link("http://x/fresh1"), _link("http://x/fresh3")], pending_urls,
    )
    appended = [link.href for link in crawler.state.pending_links]
    assert appended == [
        "http://x/already_pending",
        "http://x/fresh1", "http://x/fresh2", "http://x/fresh3",
    ]


# End-to-end: digest never fetches the same URL twice

def _install_nav_link_preview(crawler, start_url="http://site/start"):
    """Mock site where every page links to every not-yet-crawled URL (nav-link
    topology). Identical-href duplicates arrive from every crawled page; page
    contents varied so relevance scores are not all tied (favorable to the
    bug: identical-href duplicates occupy adjacent top-k slots)."""
    nav_hrefs = [
        "http://site/alpha", "http://site/bravo",
        "http://site/charlie", "http://site/delta",
    ]

    def _links_for(_url):
        return {"internal": [{"href": h, "text": h} for h in nav_hrefs]}

    page_content = {
        start_url: "site alpha bravo charlie delta nav hub",
        "http://site/alpha": "alpha alpha alpha page unique",
        "http://site/bravo": "bravo bravo bravo page unique",
        "http://site/charlie": "charlie charlie charlie page unique",
        "http://site/delta": "delta delta delta page unique",
    }

    async def fake_preview(url, query):
        return _result(
            url=url, success=True,
            content=page_content.get(url, f"{url} default"),
            links=_links_for(url),
        )

    crawler._crawl_with_preview = fake_preview
    return nav_hrefs


@pytest.mark.asyncio
async def test_digest_never_crawls_the_same_url_twice_within_a_batch():
    """The bug: nav links fill the ``top_k`` slice so the same URL is fetched
    N times per batch. Fix: each unique URL fetched at most once across the
    entire digest."""
    crawler = _make_crawler(max_depth=3, top_k_links=3)
    _install_nav_link_preview(crawler)

    fetched = []
    real_preview = crawler._crawl_with_preview

    async def recording_preview(url, query):
        fetched.append(url)
        return await real_preview(url, query)

    crawler._crawl_with_preview = recording_preview

    state = await crawler.digest(start_url="http://site/start", query="alpha bravo charlie delta")

    seen = set()
    for url in fetched:
        assert url not in seen, f"fetched twice: {url} (order: {fetched})"
        seen.add(url)
    assert len(fetched) >= 2
    assert len(fetched) == len(state.crawled_urls)


@pytest.mark.asyncio
async def test_digest_knowledge_base_has_no_duplicate_urls():
    """Every KB entry has a unique url (report had ``len(kb) == 13`` for 7
    unique URLs)."""
    crawler = _make_crawler(max_depth=3, top_k_links=3)
    _install_nav_link_preview(crawler)

    state = await crawler.digest(start_url="http://site/start", query="alpha bravo charlie delta")

    kb_urls = [r.url for r in state.knowledge_base]
    assert len(kb_urls) == len(set(kb_urls)), f"duplicate URLs in KB: {kb_urls}"


@pytest.mark.asyncio
async def test_digest_total_documents_equals_unique_crawled_urls():
    """``total_documents`` must equal ``len(crawled_urls)`` (report had
    ``total_documents == 13`` vs 7)."""
    crawler = _make_crawler(max_depth=3, top_k_links=3)
    _install_nav_link_preview(crawler)

    state = await crawler.digest(start_url="http://site/start", query="alpha bravo charlie delta")

    assert state.total_documents == len(state.crawled_urls)
    assert state.total_documents == len(state.knowledge_base)


@pytest.mark.asyncio
async def test_digest_pending_links_has_no_duplicate_hrefs():
    """``pending_links`` must contain no duplicate hrefs across iterations."""
    crawler = _make_crawler(max_depth=3, top_k_links=3)
    _install_nav_link_preview(crawler)

    state = await crawler.digest(start_url="http://site/start", query="alpha bravo charlie delta")

    hrefs = [link.href for link in state.pending_links]
    assert len(hrefs) == len(set(hrefs)), f"duplicate hrefs: {hrefs}"


@pytest.mark.asyncio
async def test_digest_crawl_order_matches_crawled_urls_under_dedup():
    """``crawl_order`` (from ``result.url``) and ``crawled_urls`` (from
    ``link.href``) agree; no duplicates in ``crawl_order``. Guards the
    batch-alignment invariant from test_adaptive_batch_alignment.py under
    the dedup fix."""
    crawler = _make_crawler(max_depth=3, top_k_links=3)
    _install_nav_link_preview(crawler)

    state = await crawler.digest(start_url="http://site/start", query="alpha bravo charlie delta")

    assert len(state.crawl_order) == len(set(state.crawl_order))
    assert set(state.crawl_order) == state.crawled_urls


@pytest.mark.asyncio
async def test_digest_start_pages_links_to_itself_deduped():
    """If the start page advertises its own URL as an internal link, the
    initial extend MUST NOT re-append it (it is in ``crawled_urls`` after the
    initial crawl). Pre-fix the initial extend never checked ``crawled_urls``."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    start_url = "http://site/start"
    start_links = {"internal": [
        {"href": start_url}, {"href": "http://site/a"}, {"href": "http://site/b"},
    ]}

    async def fake_preview(url, query):
        if url == start_url:
            return _result(start_url, True, content="start alpha", links=start_links)
        return _result(url, True, content=f"{url} alpha")

    crawler._crawl_with_preview = fake_preview

    state = await crawler.digest(start_url=start_url, query="alpha")

    hrefs = [link.href for link in state.pending_links]
    assert start_url not in hrefs
    assert hrefs.count("http://site/a") == 1
    assert hrefs.count("http://site/b") == 1


@pytest.mark.asyncio
async def test_digest_start_pages_duplicate_internal_links_deduped():
    """If the start page HTML lists the same internal link twice (common on
    real sites), the initial extend collapses them to one entry. Pre-fix the
    initial extend called ``pending_links.extend(...)`` with no dedup at all."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    start_url = "http://site/start"
    start_links = {"internal": [
        {"href": "http://site/dup"}, {"href": "http://site/dup"},
        {"href": "http://site/dup"}, {"href": "http://site/only_once"},
    ]}

    async def fake_preview(url, query):
        if url == start_url:
            return _result(start_url, True, content="start alpha", links=start_links)
        return _result(url, True, content=f"{url} alpha")

    crawler._crawl_with_preview = fake_preview

    state = await crawler.digest(start_url=start_url, query="alpha")

    hrefs = [link.href for link in state.pending_links]
    assert hrefs.count("http://site/dup") == 1
    assert hrefs.count("http://site/only_once") == 1
    assert len(hrefs) == 2


@pytest.mark.asyncio
async def test_digest_resume_dedupes_preexisting_duplicates_in_pending_links():
    """When resuming from a state file built by an older (buggy) version,
    ``pending_links`` may already contain duplicate hrefs. ``rank_links``
    independently dedupes its output, so the resumed crawl cannot fetch the
    same URL twice within a batch."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    crawler.state = CrawlState(
        crawled_urls={"http://site/already"},
        knowledge_base=[],
        pending_links=[
            _link("http://site/x"), _link("http://site/x"), _link("http://site/x"),
            _link("http://site/y"),
        ],
        query="alpha",
        metrics={},
    )

    ranked = await crawler.strategy.rank_links(crawler.state, crawler.config)
    hrefs = [link.href for link, _ in ranked]
    assert hrefs.count("http://site/x") == 1
    assert hrefs.count("http://site/y") == 1
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_digest_dedup_does_not_reappend_crawled_urls_after_batch():
    """Per-batch loop processes ``(link, result)`` pairs in order, adding
    ``link.href`` to ``crawled_urls`` before extracting its result's links.
    The dedup must ensure a just-crawled URL's href is not re-appended even
    while other crawled pairs in the same batch also list it. Pre-fix this
    produced ``[a, b, c, b, c, c]`` for a 3-link batch whose results each
    linked to a, b, c."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    start_url = "http://site/start"
    start_links = {"internal": [
        {"href": "http://site/a"}, {"href": "http://site/b"}, {"href": "http://site/c"},
    ]}

    async def fake_preview(url, query):
        if url == start_url:
            return _result(start_url, True, content="start alpha", links=start_links)
        return _result(url, True, content=f"{url} unique content alpha", links={"internal": [
            {"href": "http://site/a"}, {"href": "http://site/b"}, {"href": "http://site/c"},
        ]})

    crawler._crawl_with_preview = fake_preview

    state = await crawler.digest(start_url=start_url, query="alpha")

    hrefs = [link.href for link in state.pending_links]
    assert len(hrefs) == len(set(hrefs)), f"duplicate hrefs after batch: {hrefs}"
    assert set(hrefs) == {"http://site/a", "http://site/b", "http://site/c"}
    assert len(hrefs) == 3
    assert state.crawled_urls == {
        start_url, "http://site/a", "http://site/b", "http://site/c",
    }


@pytest.mark.asyncio
async def test_extend_pending_links_keeps_highest_scoring_duplicate():
    """``_extend_pending_links`` keeps the highest-scoring representative per
    href (within a call and across batches), so a low-scoring first link cannot
    crowd out a higher-scoring later preview before ``rank_links`` applies
    ``min_gain_threshold``."""
    crawler = _make_crawler(min_gain_threshold=0.7)
    crawler.state = CrawlState(query="alpha")
    crawler.state.crawled_urls = set()
    crawler.state.pending_links = []
    pending_urls: set = set()
    crawler._extend_pending_links([
        _link("http://x/dup", contextual_score=0.01),
        _link("http://x/dup", contextual_score=0.9),
        _link("http://x/other", contextual_score=0.5),
    ], pending_urls)

    by_href = {link.href: link for link in crawler.state.pending_links}
    assert set(by_href) == {"http://x/dup", "http://x/other"}
    assert by_href["http://x/dup"].contextual_score == 0.9  # not the first 0.01

    # A later batch re-surfaces a pending href with a higher preview score.
    crawler._extend_pending_links([_link("http://x/dup", contextual_score=0.95)], pending_urls)
    assert len(crawler.state.pending_links) == 2
    dup = next(l for l in crawler.state.pending_links if l.href == "http://x/dup")
    assert dup.contextual_score == 0.95


@pytest.mark.asyncio
async def test_digest_normalizes_resumed_pending_links_keeping_best_preview():
    """Resuming from a state file with duplicate hrefs carrying different preview
    scores, ``digest`` normalizes ``pending_links`` keeping the highest-scoring
    representative per href (not the first), so ``rank_links`` sees the stronger
    preview and the href can clear ``min_gain_threshold``."""
    # With the default weights and an empty knowledge base, ``http://site/x``
    # (contextual_score=0.9) scores 0.95 and ``http://site/y`` (0.5) scores
    # 0.75. A threshold between the two lets the stronger href through so the
    # resumed crawl actually fetches it; a threshold of 1.0 would gate it out
    # and the assertions below would pass without exercising the fetch path.
    crawler = _make_crawler(max_depth=1, top_k_links=3, min_gain_threshold=0.8)
    start_url = "http://site/already"
    state = CrawlState(
        crawled_urls={start_url}, knowledge_base=[],
        pending_links=[
            _link("http://site/x", contextual_score=0.01),
            _link("http://site/x", contextual_score=0.9),
            _link("http://site/y", contextual_score=0.5),
        ],
        query="alpha", metrics={},
    )
    tmp_dir = tempfile.mkdtemp()
    state_path = os.path.join(tmp_dir, "adaptive_state.json")
    state.save(state_path)

    async def fake_preview(url, query):
        return _result(url, True, content=f"{url} alpha")

    fetched = []

    async def recording_preview(url, query):
        fetched.append(url)
        return await fake_preview(url, query)

    crawler._crawl_with_preview = recording_preview
    await crawler.digest(start_url=start_url, query="alpha", resume_from=state_path)

    by_href = {link.href: link for link in crawler.state.pending_links}
    assert set(by_href) == {"http://site/x", "http://site/y"}
    assert by_href["http://site/x"].contextual_score == 0.9
    hrefs = [link.href for link in crawler.state.pending_links]
    assert len(hrefs) == len(set(hrefs))
    # The stronger href is fetched exactly once (rank_links dedupes by href).
    assert fetched.count("http://site/x") == 1


# Script runner (matches other tests/regression/*.py files)

async def main():
    print("=" * 60)
    print("AdaptiveCrawler duplicate-URL dedup regression tests")
    print("=" * 60)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            await fn()
            print(f"PASS: {name}")
    print("\n" + "=" * 60 + "\nALL TESTS PASSED\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
