"""
Regression tests for AdaptiveCrawler batch-crawl alignment.

Bug: ``AdaptiveCrawler._crawl_batch`` historically filtered failed crawls out
of the gathered results, returning a ``List[CrawlResult]`` shorter than the
input ``links_with_scores`` whenever any crawl in the batch failed.
``AdaptiveCrawler.digest`` then zipped that shortened list against the full
``to_crawl`` link list (``zip(new_results, to_crawl)``), pairing every
*successful* result with the wrong link after the first failure. The
consequences were:

1. The failed URL was added to ``state.crawled_urls`` (via the shifted
   ``link.href``) and thus permanently excluded from the crawl frontier --
   never retried even though it never succeeded (lost data).
2. The genuinely-succeeded URL was *omitted* from ``state.crawled_urls``,
   leaving it re-eligible for re-crawling and duplicating entries in
   ``knowledge_base``.
3. ``state.crawl_order`` (built from ``result.url`` in ``update_state``)
   diverged from ``state.crawled_urls`` (built from the bugged
   ``link.href``), producing an observable contradiction.

The fix makes ``_crawl_batch`` return ``(link, result)`` pairs, pairing each
raw ``asyncio.gather`` result (which preserves input order) with its
originating link *before* filtering, so a failed crawl can never be
misattributed to a different link's href. ``digest`` consumes the pairs
directly, eliminating the misaligning ``zip``.

Runs fully offline: monkeypatches ``AdaptiveCrawler._crawl_with_preview`` to
return controlled ``CrawlResult`` outcomes and exercises the real production
methods (``_crawl_batch`` and ``digest``) with a lightweight
``StatisticalStrategy`` (the fix is strategy-agnostic). No network, no LLM,
no sentence-transformers.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from crawl4ai import AdaptiveConfig, CrawlResult, MarkdownGenerationResult
from crawl4ai.models import Link
from crawl4ai.adaptive_crawler import AdaptiveCrawler, StatisticalStrategy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _md(text: str) -> MarkdownGenerationResult:
    return MarkdownGenerationResult(
        raw_markdown=text,
        markdown_with_citations="",
        references_markdown="",
    )


def _result(url: str, success: bool, content: str = "alpha beta gamma", links=None):
    """Build a minimal CrawlResult. ``links`` defaults to no links."""
    return CrawlResult(
        url=url,
        html="",
        success=success,
        markdown=_md(content),
        links=links or {},
    )


def _link(href: str, contextual_score: float = 0.5) -> Link:
    """Build a candidate Link with a positive contextual score so it is
    selected by StatisticalStrategy.rank_links (score > min_gain_threshold)."""
    return Link(href=href, text=href, contextual_score=contextual_score)


def _make_crawler(max_depth: int = 1, top_k_links: int = 3) -> AdaptiveCrawler:
    """Build an AdaptiveCrawler wired to a default StatisticalStrategy and a
    sentinel crawler object (so ``digest`` neither creates nor tears down a
    real AsyncWebCrawler). ``max_depth=1`` runs exactly one batch iteration so
    assertions see a single batch in isolation."""
    config = AdaptiveConfig(
        confidence_threshold=1.0,   # never stop on confidence (we want a batch)
        max_depth=max_depth,        # exactly one iteration
        max_pages=20,               # don't trip the page-limit stop
        top_k_links=top_k_links,
        min_gain_threshold=0.0,     # accept every ranked link
    )
    crawler = AdaptiveCrawler(crawler=object(), config=config)
    assert isinstance(crawler.strategy, StatisticalStrategy)
    return crawler


# ---------------------------------------------------------------------------
# Unit-level: _crawl_batch alignment under partial failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crawl_batch_returns_pairs_for_successful_results_only():
    """_crawl_batch must return (link, result) pairs, one per successful crawl,
    with each pair's link being the *originating* link. A failed result must
    not appear, and the surviving pairs must keep their original link (no
    shift)."""
    crawler = _make_crawler()
    to_crawl = [
        (_link("http://a/"), 0.9),
        (_link("http://b/"), 0.8),  # will fail
        (_link("http://c/"), 0.7),
    ]

    async def fake_preview(url, query):
        if url == "http://b/":
            return _result("http://b/", success=False)
        return _result(url, success=True)

    crawler._crawl_with_preview = fake_preview

    pairs = await crawler._crawl_batch(to_crawl, "query")

    assert len(pairs) == 2
    assert [link.href for link, _ in pairs] == ["http://a/", "http://c/"]
    assert [result.url for _, result in pairs] == ["http://a/", "http://c/"]
    # Crucially, each result is paired with its OWN link, not the next available.
    assert [(link.href, result.url) for link, result in pairs] == [
        ("http://a/", "http://a/"),
        ("http://c/", "http://c/"),
    ]


@pytest.mark.asyncio
async def test_crawl_batch_handles_none_results_and_exceptions():
    """_crawl_with_preview normally swallows exceptions and returns None, but
    an escaping exception would surface via gather(return_exceptions=True).
    Both None and Exception results must be skipped without disturbing the
    alignment of the successful pairs."""
    crawler = _make_crawler()
    to_crawl = [
        (_link("http://a/"), 0.9),  # None (swallowed exception)
        (_link("http://b/"), 0.8),  # raises (escapes to gather)
        (_link("http://c/"), 0.7),  # success
    ]

    async def fake_preview(url, query):
        if url == "http://a/":
            return None
        if url == "http://b/":
            raise RuntimeError("boom")
        return _result(url, success=True)

    crawler._crawl_with_preview = fake_preview

    pairs = await crawler._crawl_batch(to_crawl, "query")

    assert [(link.href, result.url) for link, result in pairs] == [
        ("http://c/", "http://c/"),
    ]


@pytest.mark.asyncio
async def test_crawl_batch_all_succeed_no_regression():
    """Happy path: when every crawl succeeds the returned pairs must align
    1:1 with the input links (no regression for the common case)."""
    crawler = _make_crawler()
    to_crawl = [
        (_link("http://a/"), 0.9),
        (_link("http://b/"), 0.8),
        (_link("http://c/"), 0.7),
    ]

    async def fake_preview(url, query):
        return _result(url, success=True)

    crawler._crawl_with_preview = fake_preview

    pairs = await crawler._crawl_batch(to_crawl, "query")

    assert [(link.href, result.url) for link, result in pairs] == [
        ("http://a/", "http://a/"),
        ("http://b/", "http://b/"),
        ("http://c/", "http://c/"),
    ]


# ---------------------------------------------------------------------------
# End-to-end: digest() marks exactly the successful URLs on partial failure
# ---------------------------------------------------------------------------

def _install_partial_failure_preview(crawler, start_url="http://start/"):
    """Monkeypatch _crawl_with_preview so the start page produces three
    candidate links (a, b, c) and the middle link (b) fails when crawled."""

    start_links = {"internal": [
        {"href": "http://a/"}, {"href": "http://b/"}, {"href": "http://c/"},
    ]}

    async def fake_preview(url, query):
        if url == start_url:
            return _result(start_url, True, content="start alpha beta",
                           links=start_links)
        if url == "http://a/":
            return _result("http://a/", True, content="a alpha")
        if url == "http://b/":
            return _result("http://b/", False, content="")  # middle link FAILS
        if url == "http://c/":
            return _result("http://c/", True, content="c gamma")
        raise AssertionError(f"unexpected url: {url}")

    crawler._crawl_with_preview = fake_preview


@pytest.mark.asyncio
async def test_digest_marks_successful_urls_not_failed_ones():
    """The reported bug: middle link (b) fails. After the batch, exactly the
    successful URLs (start, a, c) must be in crawled_urls; the failed URL (b)
    must NOT be there (so it stays re-eligible for retry)."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    _install_partial_failure_preview(crawler)

    state = await crawler.digest(start_url="http://start/", query="alpha gamma")

    assert state.crawled_urls == {"http://start/", "http://a/", "http://c/"}
    # The failed URL must not be mis-marked as crawled.
    assert "http://b/" not in state.crawled_urls


@pytest.mark.asyncio
async def test_digest_crawl_order_reflects_actual_successful_urls():
    """crawl_order (from result.url) and crawled_urls (from link.href) must
    agree on the set of successful URLs after a partial failure. Before the
    fix they diverged -- the smoking gun in the report."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    _install_partial_failure_preview(crawler)

    state = await crawler.digest(start_url="http://start/", query="alpha gamma")

    # crawl_order reflects result.url (always correct, even before the fix).
    assert state.crawl_order == ["http://start/", "http://a/", "http://c/"]
    # crawled_urls must now agree with crawl_order's successful set.
    assert set(state.crawl_order) == state.crawled_urls


@pytest.mark.asyncio
async def test_digest_failed_url_remains_re_eligible():
    """The failed URL (b) must remain reachable from the frontier: it must not
    be in crawled_urls (which is the sole skip-filter in rank_links). This
    guards the 'lost data' impact from the report."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)
    _install_partial_failure_preview(crawler)

    state = await crawler.digest(start_url="http://start/", query="alpha gamma")

    assert "http://b/" not in state.crawled_urls
    # rank_links is the single mechanism that skips already-crawled URLs; it
    # uses crawled_urls. Confirm b is NOT skipped by feeding it through.
    state.pending_links.append(_link("http://b/"))
    ranked = await crawler.strategy.rank_links(state, crawler.config)
    ranked_hrefs = [link.href for link, _ in ranked]
    assert "http://b/" in ranked_hrefs


@pytest.mark.asyncio
async def test_digest_all_succeed_no_regression():
    """Happy-path regression guard: when every crawl in every batch succeeds,
    digest must mark all URLs crawled and populate the KB in order. This
    guards against the fix breaking the common case."""
    crawler = _make_crawler(max_depth=1, top_k_links=3)

    start_links = {"internal": [
        {"href": "http://a/"}, {"href": "http://b/"}, {"href": "http://c/"},
    ]}

    async def fake_preview(url, query):
        if url == "http://start/":
            return _result("http://start/", True, content="start alpha beta",
                           links=start_links)
        return _result(url, True, content=f"{url} content")

    crawler._crawl_with_preview = fake_preview

    state = await crawler.digest(start_url="http://start/", query="alpha beta")

    assert state.crawled_urls == {"http://start/", "http://a/", "http://b/", "http://c/"}
    assert state.crawl_order == ["http://start/", "http://a/", "http://b/", "http://c/"]
    assert [r.url for r in state.knowledge_base] == [
        "http://start/", "http://a/", "http://b/", "http://c/",
    ]


# ---------------------------------------------------------------------------
# Script runner (matches the style of the other tests/regression/*.py files)
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("AdaptiveCrawler batch-alignment regression tests")
    print("=" * 60)

    await test_crawl_batch_returns_pairs_for_successful_results_only()
    print("PASS: test_crawl_batch_returns_pairs_for_successful_results_only")
    await test_crawl_batch_handles_none_results_and_exceptions()
    print("PASS: test_crawl_batch_handles_none_results_and_exceptions")
    await test_crawl_batch_all_succeed_no_regression()
    print("PASS: test_crawl_batch_all_succeed_no_regression")
    await test_digest_marks_successful_urls_not_failed_ones()
    print("PASS: test_digest_marks_successful_urls_not_failed_ones")
    await test_digest_crawl_order_reflects_actual_successful_urls()
    print("PASS: test_digest_crawl_order_reflects_actual_successful_urls")
    await test_digest_failed_url_remains_re_eligible()
    print("PASS: test_digest_failed_url_remains_re_eligible")
    await test_digest_all_succeed_no_regression()
    print("PASS: test_digest_all_succeed_no_regression")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
