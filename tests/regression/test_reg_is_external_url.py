"""Regression tests for ``crawl4ai.utils.is_external_url`` label-boundary
semantics and the downstream link-classification paths that depend on it.

Locks in that lookalike / adjacent registrable domains (e.g.
``notexample.com``) are classified as external when crawling ``example.com``,
while genuine subdomains (``blog.example.com``) and the apex domain remain
internal. The bug was a raw ``endswith(base)`` suffix match with no
leading-dot label boundary, which also bypassed the ``exclude_domains`` /
``exclude_external_links`` removal branches (they only run when
``is_external`` is ``True``) and poisoned the deep-crawl frontier seed list.

A later regression — an unanchored ``.replace("www.", "")`` normalization
on both the URL hostname and the base domain — was silently mangling hosts
whose label ends in ``www`` (e.g. ``mywww.example.com`` ->
``myexample.com``), misclassifying genuine internal subdomains as external.
The fix uses anchored ``removeprefix("www.")`` so only a leading ``www.``
label is stripped.
"""

from types import SimpleNamespace

import pytest

from crawl4ai import CrawlResult, CrawlerRunConfig
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    DFSDeepCrawlStrategy,
)
from crawl4ai.utils import is_external_url, quick_extract_links


# ===================================================================
# is_external_url — direct unit tests (the regression itself)
# ===================================================================

class TestIsExternalUrl:
    """Label-boundary classification of a URL against a base domain.

    ``True`` means external, ``False`` means internal (same-site / subdomain /
    relative).
    """

    @pytest.mark.parametrize(
        "url, base_domain, expected",
        [
            # Lookalikes: distinct registrable domains whose hostname string
            # ends with the base. Must be external (the bug).
            ("http://notexample.com/page", "example.com", True),
            ("http://evilexample.com/page", "example.com", True),
            ("http://cdnexample.com/page", "example.com", True),
            ("http://www.notexample.com/page", "example.com", True),
            ("http://notexample.com:8080/page", "example.com", True),
            ("http://notexample.co.uk/", "example.co.uk", True),
            # Genuine same-site / subdomain / apex: internal (no over-broad fix).
            ("http://example.com/page", "example.com", False),
            ("http://www.example.com/page", "example.com", False),
            ("http://blog.example.com/page", "example.com", False),
            ("http://a.b.example.com/page", "example.com", False),
            ("http://example.com:8080/page", "example.com", False),
            ("http://blog.example.co.uk/p", "example.co.uk", False),
            # Subdomains whose label ends in "www" (a leading "www." must be
            # stripped with an anchored operation, not an unanchored replace,
            # so the host is not mangled). Must be internal — before the fix
            # the unanchored replace turned e.g. "mywww.example.com" into
            # "myexample.com" and misclassified these as external.
            ("http://mywww.example.com/x", "example.com", False),
            ("http://foo.newwww.com/x", "newwww.com", False),
            ("http://www.mywww.example.com/x", "example.com", False),
            # Case-insensitive comparison on both sides.
            ("http://EXAMPLE.com/page", "example.com", False),
            ("http://NOTEXAMPLE.com/page", "example.com", True),
            # Relative URL (no netloc): internal.
            ("/page", "example.com", False),
            # Special schemes: short-circuit to external before the label check.
            ("mailto:a@b.com", "example.com", True),
            ("javascript:void(0)", "example.com", True),
        ],
    )
    def test_classification(self, url, base_domain, expected):
        assert is_external_url(url, base_domain) is expected

    def test_lookalike_vs_subdomain_headline_regression(self):
        """The headline cases from the bug report."""
        assert is_external_url("http://notexample.com/page", "example.com") is True
        assert is_external_url("http://evilexample.com/page", "example.com") is True
        assert is_external_url("http://blog.example.com/page", "example.com") is False
        assert is_external_url("http://example.com/page", "example.com") is False

    def test_mid_hostname_www_label_headline_regression(self):
        """The headline cases from the unanchored-replace bug report.

        A subdomain whose label ends in ``www`` (e.g. ``mywww.example.com``)
        is a genuine internal subdomain of ``example.com``. The unanchored
        ``.replace("www.", "")`` mangled it to ``myexample.com`` and
        misclassified it as external; the anchored ``removeprefix("www.")``
        leaves it intact so the label-boundary check correctly classifies it
        as internal.
        """
        assert is_external_url("https://mywww.example.com/x", "example.com") is False
        assert is_external_url("https://foo.newwww.com/x", "newwww.com") is False
        # Lookalikes must remain external — the fix must not be over-broad.
        assert is_external_url("https://notexample.com/page", "example.com") is True


# ===================================================================
# quick_extract_links — prefetch / domain-mapper seed-list path
# ===================================================================

class TestQuickExtractLinksLookalike:
    """Verify the prefetch-mode consumer routes lookalikes to the external
    bucket (the deep-crawl frontier and domain-mapper seed list both consume
    the ``internal`` bucket)."""

    HTML = """
    <html><body>
        <a href="http://notexample.com/x">lookalike</a>
        <a href="http://blog.example.com/p">subdomain</a>
        <a href="http://example.com/a">apex</a>
        <a href="/rel">relative</a>
        <a href="http://other.com/q">other-external</a>
    </body></html>
    """

    def test_lookalike_routed_to_external_and_seed_list_intact(self):
        result = quick_extract_links(self.HTML, "http://example.com")
        internal_hrefs = [link["href"] for link in result["internal"]]
        external_hrefs = [link["href"] for link in result["external"]]

        # Lookalikes external — before the fix they leaked into internal and
        # poisoned the seed list consumed by the deep-crawl frontier and the
        # domain mapper.
        assert "http://notexample.com/x" in external_hrefs
        assert "http://notexample.com/x" not in internal_hrefs
        # Genuine same-site / subdomain / relative remain internal.
        assert "http://blog.example.com/p" in internal_hrefs
        assert "http://example.com/a" in internal_hrefs
        assert "http://example.com/rel" in internal_hrefs
        # Non-lookalike external link remains external.
        assert "http://other.com/q" in external_hrefs

    def test_mid_hostname_www_label_routed_to_internal(self):
        """A subdomain whose label ends in ``www`` (e.g.
        ``mywww.example.com``) must land in the ``internal`` bucket — before
        the fix the unanchored ``.replace("www.", "")`` in
        ``is_external_url`` mangled the host and ``quick_extract_links``
        misrouted it to ``external``, so the deep-crawl frontier and the
        domain-mapper seed list never visited these internal pages.
        """
        html = (
            '<html><body>'
            '<a href="https://mywww.example.com/x">mid-www subdomain</a>'
            '<a href="https://notexample.com/z">lookalike</a>'
            '</body></html>'
        )
        result = quick_extract_links(html, "https://example.com/")
        internal_hrefs = [link["href"] for link in result["internal"]]
        external_hrefs = [link["href"] for link in result["external"]]

        assert "https://mywww.example.com/x" in internal_hrefs
        assert "https://mywww.example.com/x" not in external_hrefs
        # The lookalike must remain external — the fix must not be over-broad.
        assert "https://notexample.com/z" in external_hrefs
        assert "https://notexample.com/z" not in internal_hrefs


# ===================================================================
# LXMLWebScrapingStrategy — exclude_domains / exclude_external_links bypass
# ===================================================================

class TestScrapingStrategyExcludeDomainsBypass:
    """Verify that listing a lookalike in ``exclude_domains`` actually removes
    it.

    Before the fix, ``is_external_url`` returned ``False`` for lookalikes, so
    the ``exclude_domains`` blocklist branch (inside the ``if is_external:``
    guard in ``content_scraping_strategy._process_element``) was bypassed and
    the lookalike survived in ``result.links["internal"]`` and the cleaned
    HTML. The same applied to ``exclude_external_links=True``.
    """

    HTML = """
    <html><head><title>t</title></head><body>
        <a href="http://notexample.com/x">lookalike</a>
        <a href="http://evilexample.com/y">evil</a>
        <a href="http://blog.example.com/p">subdomain</a>
        <a href="http://mywww.example.com/m">mid-www subdomain</a>
        <a href="http://other.com/q">other-external</a>
        <a href="/rel">relative</a>
    </body></html>
    """

    def _scrap(self, **kwargs):
        from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
        return LXMLWebScrapingStrategy().scrap(
            url="http://example.com", html=self.HTML, **kwargs
        )

    def test_exclude_domains_removes_lookalike(self):
        r = self._scrap(exclude_domains=["notexample.com", "evilexample.com"])
        internal = [link.href for link in r.links.internal]
        external = [link.href for link in r.links.external]

        # Removed entirely from links and from the cleaned HTML.
        assert "http://notexample.com/x" not in internal + external
        assert "http://evilexample.com/y" not in internal + external
        assert "notexample.com" not in r.cleaned_html
        assert "evilexample.com" not in r.cleaned_html
        # Non-blocked external links and genuine same-site links remain.
        assert "http://other.com/q" in external
        assert "http://blog.example.com/p" in internal
        assert "http://mywww.example.com/m" in internal
        assert "http://example.com/rel" in internal

    def test_exclude_external_links_removes_lookalike(self):
        r = self._scrap(exclude_external_links=True)
        internal = [link.href for link in r.links.internal]
        external = [link.href for link in r.links.external]

        assert external == []
        assert "http://notexample.com/x" not in internal
        assert "notexample.com" not in r.cleaned_html
        # Internal same-site links are unaffected. The mid-www subdomain
        # must survive — before the fix the unanchored replace in
        # is_external_url misclassified it as external and it was stripped.
        assert "http://blog.example.com/p" in internal
        assert "http://mywww.example.com/m" in internal
        assert "mywww.example.com" in r.cleaned_html
        assert "http://example.com/rel" in internal


# ===================================================================
# Deep-crawl frontier containment (default, no allowed_domains filter)
# ===================================================================

class TestDeepCrawlFrontierContainment:
    """Verify that when a user opts into a deep crawl with the default
    (empty) filter chain, a lookalike link on the seed page is NOT pulled
    into the next-depth frontier.

    The frontier consumes ``result.links["internal"]``; before the fix the
    lookalike was misclassified internal and fetched at the next depth.
    Mirrors the stub-``AsyncWebCrawler`` pattern from
    ``tests/regression/test_deep_crawl_query_fetch.py``.
    """

    START = "https://example.com/"
    LOOKALIKE = "https://notexample.com/x"
    REAL_SUB = "https://blog.example.com/p"
    MID_WWW_SUB = "https://mywww.example.com/m"

    @pytest.fixture
    def page_links(self):
        html = (
            '<a href="{lk}">lookalike</a>'
            '<a href="{sb}">subdomain</a>'
            '<a href="{mw}">mid-www subdomain</a>'
        ).format(lk=self.LOOKALIKE, sb=self.REAL_SUB, mw=self.MID_WWW_SUB)
        return quick_extract_links(html, self.START)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_type, stream",
        [
            (BFSDeepCrawlStrategy, False),
            (DFSDeepCrawlStrategy, False),
            (BestFirstCrawlingStrategy, True),
        ],
    )
    async def test_lookalike_not_fetched_at_next_depth(
        self, strategy_type, stream, page_links
    ):
        fetched = []

        async def arun_many(urls, config):
            results = []
            for url in urls:
                fetched.append(url)
                page = page_links if url == self.START else {
                    "internal": [],
                    "external": [],
                }
                results.append(
                    CrawlResult(url=url, html="", success=True, links=page)
                )
            if config.stream:
                async def gen():
                    for r in results:
                        yield r
                return gen()
            return results

        strategy = strategy_type(max_depth=2, max_pages=10)
        results = await strategy.arun(
            self.START,
            SimpleNamespace(arun_many=arun_many),
            CrawlerRunConfig(stream=stream),
        )
        if stream:
            results = [r async for r in results]

        # The seed page's internal bucket contains ONLY the genuine
        # subdomains (the mid-www subdomain must be internal too — before
        # the fix the unanchored replace misrouted it to external).
        internal_hrefs = [link["href"] for link in page_links["internal"]]
        assert self.LOOKALIKE not in internal_hrefs
        assert self.REAL_SUB in internal_hrefs
        assert self.MID_WWW_SUB in internal_hrefs

        # The lookalike must never be fetched at any depth.
        assert self.LOOKALIKE not in fetched, (
            f"{strategy_type.__name__} fetched the lookalike: {fetched}"
        )
        # The genuine subdomains are enqueued and fetched.
        assert self.REAL_SUB in fetched
        assert self.MID_WWW_SUB in fetched
        assert self.START in fetched
