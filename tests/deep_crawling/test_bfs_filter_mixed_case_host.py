"""
Regression: BFS ``link_discovery`` must filter/score the canonical
(host-lowercased) ``url_key`` while fetching the raw ``base_url``.

Bug (introduced in commit d8e66427): that commit split the single
``base_url = normalize_url_for_deep_crawl(url, source_url)`` (host-lowercased,
used for both dedup and filtering) into two variables -- ``url_key``
(lowercased, for dedup/visited) and a new raw
``base_url = urldefrag(urljoin(source_url, url.strip()))[0]`` (case-preserved,
for fetching) -- in order to preserve request query strings on the outgoing
fetch. It reused the name ``base_url`` for the raw URL, so the pre-existing
``can_process_url(base_url, ...)`` and ``url_scorer.score(base_url, ...)`` calls
silently switched to receiving the case-preserved form, reverting the earlier
fix 43738c9ed which had routed the normalized URL into the filter.

``URLPatternFilter`` matches case-sensitively (no ``re.IGNORECASE``). In the
``prefetch=True`` deep-crawl path, ``quick_extract_links`` (changed by the same
commit) stores the raw href with host case preserved, so a host-bearing
keep-pattern such as ``*://example.com/*`` silently drops same-domain pages
whose hrefs use a mixed/upper-case host (e.g. ``https://Example.com/page``).

The fix restores the pre-commit filter/scorer input contract: feed the
canonical ``url_key`` to ``can_process_url``/``url_scorer.score`` while still
enqueuing the raw ``base_url`` for fetching (preserving the commit's
query-preservation goal).

These tests pin the contract so the footgun cannot silently return. The
central contract test (``test_filter_sees_canonical_but_fetch_uses_raw_url``)
asserts BOTH halves at once -- the visited key is the canonical lowercased
form and the enqueued fetch URL is the raw case-preserved form -- so a future
change that collapses the two variables in either direction fails loudly.
"""

import pytest

from crawl4ai import CrawlResult
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.utils import quick_extract_links


def _make_strategy(patterns=None, scorer=None, **kw) -> BFSDeepCrawlStrategy:
    chain = FilterChain([URLPatternFilter(patterns=patterns)]) if patterns else FilterChain()
    return BFSDeepCrawlStrategy(
        max_depth=kw.get("max_depth", 2),
        max_pages=kw.get("max_pages", 10),
        filter_chain=chain,
        url_scorer=scorer,
        include_external=kw.get("include_external", False),
        score_threshold=kw.get("score_threshold", float("-inf")),
    )


async def _discover(strategy, html, source_url="https://example.com/"):
    """Run ``link_discovery`` on a prefetch-extracted page and return
    ``(discovered_urls, visited, urls_skipped)``."""
    links = quick_extract_links(html, source_url)
    strategy._pages_crawled = 0
    result = CrawlResult(
        url=source_url, html=html, success=True, links=links
    )
    visited, next_level, depths = set(), [], {}
    await strategy.link_discovery(result, source_url, 0, visited, next_level, depths)
    return [u for u, _ in next_level], visited, strategy.stats.urls_skipped


@pytest.mark.asyncio
async def test_filter_sees_canonical_but_fetch_uses_raw_url():
    """The contract in one test: filtering runs against the canonical
    (host-lowercased, query-sorted) ``url_key`` while the enqueued fetch URL
    is the raw ``base_url`` (host case preserved, query order preserved).

    Fails if the filter is fed the raw URL (case-sensitive drop) OR if the
    fetch URL is over-normalized (query order/host case lost).
    """
    html = '<html><body><a href="https://Example.com/page?b=2&a=1">X</a></body></html>'
    strat = _make_strategy(patterns=["*://example.com/*"])
    discovered, visited, _ = await _discover(strat, html)

    assert discovered == ["https://Example.com/page?b=2&a=1"], (
        f"fetch URL lost raw host case / query order: {discovered}"
    )
    assert visited == {"https://example.com/page?a=1&b=2"}, (
        f"visited set is not the canonical key (host-lowercased, query-sorted): "
        f"{visited}"
    )


@pytest.mark.asyncio
async def test_mixed_case_host_not_dropped_by_host_bearing_pattern():
    """The user-facing symptom: a prefetch-extracted href with a mixed-case
    host and a host-bearing keep-pattern is discovered, not silently dropped
    by the case-sensitive filter. Covers both PATH (``*://host/*``) and
    PREFIX (``https://host/*``) pattern categories."""
    html = '<html><body><a href="https://Example.com/page">P</a></body></html>'
    for pattern in ["*://example.com/*", "https://example.com/*"]:
        strat = _make_strategy(patterns=[pattern])
        discovered, _, _ = await _discover(strat, html, "https://example.com/")
        assert discovered, (
            f"host-bearing keep-pattern {pattern!r} dropped mixed-case-host "
            f"href: {discovered}"
        )


@pytest.mark.asyncio
async def test_lowercase_host_still_discovered():
    """Happy-path control: an already-lowercase same-domain href is still
    discovered with a host-bearing keep-pattern (the fix is no more
    permissive than before)."""
    html = '<html><body><a href="https://example.com/page">P</a></body></html>'
    strat = _make_strategy(patterns=["*://example.com/*"])
    discovered, _, _ = await _discover(strat, html, "https://example.com/")
    assert discovered == ["https://example.com/page"]


@pytest.mark.asyncio
async def test_scorer_receives_canonical_url_not_raw():
    """The url_scorer must score the canonical ``url_key`` (host-lowercased),
    not the raw ``base_url``. Isolated with a *case-sensitive*
    ``KeywordRelevanceScorer`` whose keyword is the lowercase host: the raw
    ``https://Example.com/page`` scores 0.0 while the canonical
    ``https://example.com/page`` scores 1.0. With an empty filter chain and
    ``score_threshold=0.5``, the page is kept after the fix and would be
    dropped (0.0 below threshold) if the scorer received the raw URL."""
    scorer = KeywordRelevanceScorer(keywords=["example.com"], case_sensitive=True)
    assert scorer.score("https://Example.com/page") == 0.0
    assert scorer.score("https://example.com/page") == 1.0

    strat = BFSDeepCrawlStrategy(
        max_depth=2, max_pages=10, filter_chain=FilterChain(),
        url_scorer=scorer, score_threshold=0.5,
    )
    html = '<html><body><a href="https://Example.com/page">P</a></body></html>'
    discovered, _, skipped = await _discover(strat, html, "https://example.com/")

    assert discovered == ["https://Example.com/page"], (
        f"scorer received the raw mixed-case URL (score 0.0 < threshold) and "
        f"dropped the page: {discovered}"
    )
    assert skipped == 0
