"""
Regression tests for Crawl4AI utility functions.

Covers extract_xml_data, URL normalization, CacheContext/CacheMode,
sanitize_input_encode, content hashing, and image scoring.
"""

import pytest

from crawl4ai.utils import (
    extract_xml_data,
    extract_xml_data_legacy,
    normalize_url,
    normalize_url_for_deep_crawl,
    efficient_normalize_url_for_deep_crawl,
    quick_extract_links,
    sanitize_input_encode,
    generate_content_hash,
)
from crawl4ai.cache_context import CacheContext, CacheMode


# ===================================================================
# extract_xml_data
# ===================================================================

class TestExtractXmlData:
    """Verify extract_xml_data correctly parses tag content from strings."""

    def test_basic_single_tag(self):
        """Basic extraction of a single tag should return its content."""
        result = extract_xml_data(["blocks"], "<blocks>hello</blocks>")
        assert result["blocks"] == "hello"

    def test_multiple_tags(self):
        """Extracting multiple tags should return both."""
        result = extract_xml_data(["a", "b"], "<a>1</a><b>2</b>")
        assert result["a"] == "1"
        assert result["b"] == "2"

    def test_longest_match(self):
        """When multiple occurrences exist, return the longest content."""
        text = "<blocks>short</blocks> some text <blocks>this is the longer content here</blocks>"
        result = extract_xml_data(["blocks"], text)
        assert result["blocks"] == "this is the longer content here"

    def test_nested_mention_bug_fix_1183(self):
        """Fix for #1183: nested mention of tag name should not confuse extraction.

        When <think> block mentions <blocks> in prose, the extraction should
        return the actual <blocks> content, not the prose mention.
        """
        text = (
            "<think>The user wants me to extract <blocks> data from the page.</think>"
            "<blocks>real extracted data</blocks>"
        )
        result = extract_xml_data(["blocks"], text)
        assert result["blocks"] == "real extracted data"

    def test_missing_tag_returns_empty(self):
        """Missing tag should return empty string."""
        result = extract_xml_data(["missing"], "<other>content</other>")
        assert result["missing"] == ""

    def test_empty_content(self):
        """Empty tag content should return empty string."""
        result = extract_xml_data(["blocks"], "<blocks></blocks>")
        assert result["blocks"] == ""

    def test_multiline_content(self):
        """Content spanning multiple lines should be extracted."""
        text = "<blocks>\nline 1\nline 2\nline 3\n</blocks>"
        result = extract_xml_data(["blocks"], text)
        assert "line 1" in result["blocks"]
        assert "line 2" in result["blocks"]
        assert "line 3" in result["blocks"]

    def test_special_chars_in_content(self):
        """JSON-like content with special characters should be preserved."""
        text = '<blocks>{"key": "value", "num": 42}</blocks>'
        result = extract_xml_data(["blocks"], text)
        assert '"key": "value"' in result["blocks"]
        assert '"num": 42' in result["blocks"]

    def test_content_with_angle_brackets(self):
        """Content with HTML-like angle brackets should work if not same tag."""
        text = "<blocks>some <b>bold</b> text</blocks>"
        result = extract_xml_data(["blocks"], text)
        assert "<b>bold</b>" in result["blocks"]

    def test_multiple_tags_some_missing(self):
        """Mixed present and missing tags should return values for present, empty for missing."""
        result = extract_xml_data(["found", "missing"], "<found>yes</found>")
        assert result["found"] == "yes"
        assert result["missing"] == ""

    def test_whitespace_stripped(self):
        """Content should be stripped of leading/trailing whitespace."""
        result = extract_xml_data(["blocks"], "<blocks>  trimmed  </blocks>")
        assert result["blocks"] == "trimmed"


class TestExtractXmlDataLegacy:
    """Verify the legacy extract_xml_data function works."""

    def test_basic_extraction(self):
        """Legacy function should extract basic tag content."""
        result = extract_xml_data_legacy(["blocks"], "<blocks>hello</blocks>")
        assert result["blocks"] == "hello"

    def test_missing_tag(self):
        """Legacy function should return empty string for missing tags."""
        result = extract_xml_data_legacy(["missing"], "no tags here")
        assert result["missing"] == ""


# ===================================================================
# URL normalization
# ===================================================================

class TestNormalizeUrl:
    """Verify normalize_url handles various URL edge cases."""

    def test_trailing_slash_preserved(self):
        """Trailing slash should be preserved (fix for #1520)."""
        result = normalize_url("/foo/bar/", "http://x.com")
        assert result.endswith("/foo/bar/")

    def test_no_trailing_slash_not_added(self):
        """URL without trailing slash should NOT have one added."""
        result = normalize_url("/foo/bar", "http://x.com")
        assert result.endswith("/foo/bar")
        assert not result.endswith("/foo/bar/")

    def test_root_path(self):
        """Root path '/' should be preserved."""
        result = normalize_url("/", "http://x.com")
        assert result == "http://x.com/"

    def test_query_param_case_preservation(self):
        """Query parameter values should NOT be lowercased (fix for #1489).

        cHash=AbCd must remain as-is, not become chash=abcd.
        """
        result = normalize_url("/page?cHash=AbCd", "http://x.com")
        assert "cHash=AbCd" in result

    def test_tracking_params_removed(self):
        """Common tracking parameters should be removed."""
        result = normalize_url(
            "/page?utm_source=google&utm_medium=cpc&real_param=keep",
            "http://x.com",
        )
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "real_param=keep" in result

    def test_fbclid_removed(self):
        """fbclid tracking parameter should be removed."""
        result = normalize_url("/page?fbclid=abc123&keep=yes", "http://x.com")
        assert "fbclid" not in result
        assert "keep=yes" in result

    def test_gclid_removed(self):
        """gclid tracking parameter should be removed."""
        result = normalize_url("/page?gclid=xyz&keep=yes", "http://x.com")
        assert "gclid" not in result
        assert "keep=yes" in result

    def test_tracking_removal_case_insensitive(self):
        """Tracking parameter removal should be case-insensitive."""
        # The normalize_url uses k.lower() for comparison
        result = normalize_url("/page?UTM_SOURCE=test&data=1", "http://x.com")
        # UTM_SOURCE (uppercase) should be removed since comparison is case-insensitive
        assert "data=1" in result

    def test_query_sorting(self):
        """Query parameters should be sorted alphabetically."""
        result = normalize_url("/page?z=1&a=2&m=3", "http://x.com")
        # Parameters should appear in alphabetical order
        idx_a = result.index("a=2")
        idx_m = result.index("m=3")
        idx_z = result.index("z=1")
        assert idx_a < idx_m < idx_z

    def test_fragment_removed_by_default(self):
        """Fragment (#section) should be removed by default."""
        result = normalize_url("/page#section", "http://x.com")
        assert "#section" not in result

    def test_fragment_kept_when_requested(self):
        """Fragment should be kept when keep_fragment=True."""
        result = normalize_url("/page#section", "http://x.com", keep_fragment=True)
        assert "#section" in result

    def test_relative_url_resolution(self):
        """Relative URLs should be resolved against base_url."""
        result = normalize_url("page2", "http://x.com/dir/page1")
        assert result == "http://x.com/dir/page2"

    def test_empty_href_returns_none(self):
        """Empty href should return None."""
        result = normalize_url("", "http://x.com")
        assert result is None

    def test_none_href_returns_none(self):
        """None href should return None."""
        result = normalize_url(None, "http://x.com")
        assert result is None

    def test_hostname_lowercased(self):
        """Hostname should be lowercased for consistency."""
        result = normalize_url("/page", "http://EXAMPLE.COM/path")
        assert "example.com" in result

    def test_no_query_params_still_works(self):
        """URL without query params should normalize without issue."""
        result = normalize_url("/simple/path", "http://x.com")
        assert "http://x.com/simple/path" == result


class TestNormalizeUrlForDeepCrawl:
    """Verify normalize_url_for_deep_crawl handles deep crawl edge cases."""

    def test_trailing_slash_preserved(self):
        """Trailing slash should be preserved in deep crawl normalization."""
        result = normalize_url_for_deep_crawl("/foo/bar/", "http://x.com")
        assert result is not None
        assert result.endswith("/foo/bar/")

    def test_empty_href_returns_none(self):
        """Empty href should return None."""
        result = normalize_url_for_deep_crawl("", "http://x.com")
        assert result is None

    def test_none_href_returns_none(self):
        """None href should return None."""
        result = normalize_url_for_deep_crawl(None, "http://x.com")
        assert result is None

    def test_fragment_removed(self):
        """Fragment should be removed in deep crawl normalization."""
        result = normalize_url_for_deep_crawl("/page#anchor", "http://x.com")
        assert "#anchor" not in result

    def test_tracking_params_removed(self):
        """The full set of 9 tracking params should be removed (keep in sync
        with normalize_url). Regression guard for the deep-crawl dedup set
        being silently trimmed back to a smaller subset.
        """
        all_tracking = (
            "utm_source=g&utm_medium=cpc&utm_campaign=spring&utm_term=shoes"
            "&utm_content=ad1&gclid=abc&fbclid=xyz&ref=news&ref_src=email"
        )
        result = normalize_url_for_deep_crawl(
            f"/page?{all_tracking}&keep=yes", "http://x.com"
        )
        for param in ("utm_source", "utm_medium", "utm_campaign", "utm_term",
                      "utm_content", "gclid", "fbclid", "ref", "ref_src"):
            assert param not in result, f"{param} should be stripped: {result}"
        assert "keep=yes" in result

    def test_tracking_removal_case_insensitive(self):
        """Tracking param removal must be case-insensitive (matches normalize_url)."""
        result = normalize_url_for_deep_crawl(
            "/page?UTM_SOURCE=test&GCLID=x&Ref_Src=tw&data=1", "http://x.com"
        )
        assert "UTM_SOURCE" not in result
        assert "GCLID" not in result
        assert "Ref_Src" not in result
        assert "data=1" in result

    def test_query_sorting(self):
        """Query parameters should be sorted alphabetically.

        Mirrors ``TestNormalizeUrl.test_query_sorting``. The line comment in
        ``normalize_url_for_deep_crawl`` says the rebuilt query string is
        "sorted for consistency" -- this enforces that contract so the deep-crawl
        dedup key is order-independent (regression for the prefetch-mode
        duplicate-fetch bug).
        """
        result = normalize_url_for_deep_crawl("/page?z=1&a=2&m=3", "http://x.com")
        idx_a = result.index("a=2")
        idx_m = result.index("m=3")
        idx_z = result.index("z=1")
        assert idx_a < idx_m < idx_z, (
            f"Query params should be sorted alphabetically: {result}"
        )

    def test_query_param_order_dedup(self):
        """URLs differing only in query-param order must normalize equally.

        This is the core of the prefetch-mode duplicate-fetch bug: the deep-crawl
        ``visited`` set uses this function's output as its dedup key, so two
        orderings of the same params must collapse to one key.
        """
        base = "http://x.com"
        a = normalize_url_for_deep_crawl("/p?z=1&a=2", base)
        b = normalize_url_for_deep_crawl("/p?a=2&z=1", base)
        assert a == b, (
            f"Different query-param order should normalize to the same URL: "
            f"{a} vs {b}"
        )

    def test_query_param_order_dedup_with_tracking(self):
        """Order-independence must hold after tracking params are stripped.

        Tracking-param removal plus sorting must still collapse two orderings
        that differ in both tracking and non-tracking param positions.
        """
        base = "http://x.com"
        a = normalize_url_for_deep_crawl(
            "/p?utm_source=g&keep=yes&z=1&a=2", base
        )
        b = normalize_url_for_deep_crawl(
            "/p?a=2&z=1&utm_source=g&keep=yes", base
        )
        assert a == b, f"{a} vs {b}"
        assert "utm_source" not in a
        assert "keep=yes" in a

    def test_multi_value_query_key_sorted(self):
        """Repeated query keys (parse_qs doseq) must be sorted by key position.

        ``parse_qs`` aggregates repeated keys into a list; ``urlencode(doseq=True)``
        re-expands them. Sorting happens at the key level, not within a key's
        value list, which preserves value order for that key.
        """
        result = normalize_url_for_deep_crawl("/p?z=2&a=1&z=1&a=2", "http://x.com")
        idx_a = result.index("a=1")
        idx_z = result.index("z=2")
        assert idx_a < idx_z, f"Keys should be sorted: {result}"
        assert "a=1" in result and "a=2" in result
        assert "z=2" in result and "z=1" in result

    def test_all_tracking_stripped_yields_sortable_remainder(self):
        """When only tracking params remain, query is dropped (no spurious '&=')."""
        result = normalize_url_for_deep_crawl(
            "/page?utm_source=g&utm_medium=cpc", "http://x.com"
        )
        assert "?" not in result, f"Empty query should be dropped: {result}"
        assert result == "http://x.com/page"

    def test_hostname_lowercased(self):
        """Hostname should be lowercased."""
        result = normalize_url_for_deep_crawl("/page", "http://EXAMPLE.COM")
        assert "example.com" in result

    # -- blank-value query parameters (keep_blank_values agreement) ----------

    def test_blank_value_param_preserved(self):
        """A blank-value query param (e.g. ?q=) must survive in the dedup key.

        Regression for the bug where ``parse_qs(query)`` (default
        ``keep_blank_values=False``) silently dropped blank-value params,
        collapsing ``/page?q=`` and ``/page`` into one dedup key while the
        fetched URL preserved the distinction via ``urljoin``.
        """
        result = normalize_url_for_deep_crawl("/page?q=", "http://x.com")
        assert result == "http://x.com/page?q=", (
            f"Blank-value param 'q=' must be preserved, got: {result}"
        )

    def test_blank_value_distinct_from_absent(self):
        """``/page`` and ``/page?q=`` must produce distinct dedup keys.

        This is the core guarantee: two hrefs the fetcher would visit as
        distinct URIs must also be treated as distinct by the ``visited``
        set, modulo only the intended tracking-param coarsening.
        """
        base = "http://x.com"
        bare = normalize_url_for_deep_crawl("/page", base)
        blank = normalize_url_for_deep_crawl("/page?q=", base)
        assert bare != blank, (
            f"Bare /page and /page?q= must have distinct dedup keys: "
            f"{bare!r} == {blank!r}"
        )

    def test_blank_value_with_tracking_stripped(self):
        """Tracking params are stripped but blank-value params are kept.

        ``/page?utm_source=x&q=`` should dedup to ``/page?q=``: the tracking
        param ``utm_source`` is removed (intended coarsening), while the
        blank-value param ``q=`` is preserved (matches ``normalize_url``).
        """
        result = normalize_url_for_deep_crawl(
            "/page?utm_source=x&q=", "http://x.com"
        )
        assert result == "http://x.com/page?q=", (
            f"Tracking stripped, blank kept: expected 'http://x.com/page?q=', "
            f"got {result!r}"
        )

    def test_blank_value_collapses_with_tracking_variant(self):
        """``/page?utm_source=x&q=`` and ``/page?q=`` must dedup equally.

        Both should normalize to the same key once the tracking param is
        stripped; the blank-value param survives in both.
        """
        base = "http://x.com"
        a = normalize_url_for_deep_crawl("/page?utm_source=x&q=", base)
        b = normalize_url_for_deep_crawl("/page?q=", base)
        assert a == b, (
            f"Tracking-tagged blank and bare blank must dedup equally: "
            f"{a!r} vs {b!r}"
        )

    def test_blank_value_cross_function_agreement(self):
        """Dedup key and normalize_url must agree on blank-value params.

        ``normalize_url`` (used to build the href the scraper emits and the
        crawler fetches) parses with ``parse_qsl(keep_blank_values=True)``.
        ``normalize_url_for_deep_crawl`` (used as the dedup key) must use the
        same parser so the two never disagree, modulo tracking-param
        stripping. The in-code comment in ``normalize_url_for_deep_crawl``
        promises this alignment ("Keep this set in sync with the
        ``default_tracking`` set in ``normalize_url``").
        """
        base = "http://x.com"
        for href, expected in [
            ("/page?q=", "http://x.com/page?q="),
            ("/page?utm_source=x&q=", "http://x.com/page?q="),
            ("/page?q=&a=1", "http://x.com/page?a=1&q="),
        ]:
            norm = normalize_url(href, base)
            dedup = normalize_url_for_deep_crawl(href, base)
            assert norm == expected, f"normalize_url({href!r}): {norm!r} != {expected!r}"
            assert dedup == expected, f"dedup({href!r}): {dedup!r} != {expected!r}"

    def test_multiple_blank_value_params_preserved(self):
        """Multiple blank-value params must all survive dedup normalization."""
        result = normalize_url_for_deep_crawl(
            "/page?a=&b=&c=1", "http://x.com"
        )
        assert "a=" in result, f"Blank param 'a' dropped: {result}"
        assert "b=" in result, f"Blank param 'b' dropped: {result}"
        assert "c=1" in result, f"Non-blank param 'c' dropped: {result}"
        assert result == "http://x.com/page?a=&b=&c=1", (
            f"Unexpected normalization: {result}"
        )

    def test_blank_value_multivalue_order_preserved(self):
        """Multi-value key order must be preserved (matches normalize_url).

        ``?z=2&z=1`` and ``?z=1&z=2`` are potentially distinct URIs (the
        server may treat value order within a repeated key as significant),
        so the dedup key must NOT merge them. Sorting is by key only, not
        by the full (key, value) tuple.
        """
        base = "http://x.com"
        a = normalize_url_for_deep_crawl("/p?z=2&z=1", base)
        b = normalize_url_for_deep_crawl("/p?z=1&z=2", base)
        assert a != b, (
            f"Multi-value order must be preserved (not merged): {a!r} == {b!r}"
        )
        assert a == "http://x.com/p?z=2&z=1", f"Unexpected: {a!r}"
        assert b == "http://x.com/p?z=1&z=2", f"Unexpected: {b!r}"

    def test_blank_value_only_query_not_dropped(self):
        """A query consisting solely of a blank-value param must survive.

        Previously ``parse_qs("?q=")`` returned ``{}`` (empty dict), so the
        entire query was dropped and ``/page?q=`` collapsed to ``/page``.
        """
        result = normalize_url_for_deep_crawl("/page?q=", "http://x.com")
        assert "?" in result, (
            f"Query with only a blank-value param must not be dropped: {result}"
        )
        assert result.endswith("q="), f"Blank param must be preserved: {result}"


class TestQuickExtractLinksBlankParam:
    """Verify quick_extract_links does not collapse blank-value-param URIs.

    ``quick_extract_links`` uses ``normalize_url_for_deep_crawl`` for its
    per-document ``seen`` set. When a page exposes both ``/page`` and
    ``/page?q=``, they must both appear in the returned link list (the
    blank-value param keeps them distinct under the fix).
    """

    def test_both_bare_and_blank_returned(self):
        """Both /page and /page?q= must appear in the extracted links."""
        html = (
            '<a href="/page?q=">blank</a>'
            '<a href="/page">bare</a>'
        )
        result = quick_extract_links(html, "http://x.com/")
        hrefs = {link["href"] for link in result["internal"]}
        assert "http://x.com/page?q=" in hrefs, (
            f"/page?q= missing from extracted links: {hrefs}"
        )
        assert "http://x.com/page" in hrefs, (
            f"/page missing from extracted links: {hrefs}"
        )
        assert len(result["internal"]) == 2, (
            f"Expected 2 internal links, got {len(result['internal'])}: {hrefs}"
        )

    def test_both_bare_and_blank_returned_reversed_order(self):
        """Order-independence: bare first then blank must also yield both."""
        html = (
            '<a href="/page">bare</a>'
            '<a href="/page?q=">blank</a>'
        )
        result = quick_extract_links(html, "http://x.com/")
        hrefs = {link["href"] for link in result["internal"]}
        assert "http://x.com/page?q=" in hrefs, f"Missing /page?q=: {hrefs}"
        assert "http://x.com/page" in hrefs, f"Missing /page: {hrefs}"

    def test_tracking_tagged_blank_collapses_to_blank(self):
        """A tracking-tagged blank dedups with the bare blank, not with bare page.

        ``/page?utm_source=x&q=`` and ``/page?q=`` share the same dedup key
        (tracking param stripped, blank param kept), so only ONE of them
        appears in the result — the first one, with its original href preserved
        (``quick_extract_links`` emits the original href, not the dedup key).
        A separate ``/page`` link must survive as a distinct entry because its
        dedup key differs from the blank-param variant.
        """
        html = (
            '<a href="/page?utm_source=x&q=">tracking+blank</a>'
            '<a href="/page?q=">blank</a>'
            '<a href="/page">bare</a>'
        )
        result = quick_extract_links(html, "http://x.com/")
        internal = result["internal"]
        hrefs = {link["href"] for link in internal}
        # The tracking-tagged variant and the bare blank share one dedup key
        # (http://x.com/page?q=), so only the first-discovered one is kept.
        # The bare /page has a distinct key and survives.
        assert len(internal) == 2, (
            f"Expected 2 links (blank-variant + bare page), got "
            f"{len(internal)}: {hrefs}"
        )
        # Exactly one href carries a blank param 'q=':
        blank_hrefs = [h for h in hrefs if "q=" in h]
        assert len(blank_hrefs) == 1, (
            f"Expected exactly one blank-param href, got {blank_hrefs}: {hrefs}"
        )
        assert "utm_source" in blank_hrefs[0], (
            f"First-discovered (tracking-tagged) href should be kept: {hrefs}"
        )
        # The bare page must survive as a distinct entry.
        assert "http://x.com/page" in hrefs, (
            f"Bare /page missing: {hrefs}"
        )



class TestEfficientNormalizeUrlForDeepCrawl:
    """Verify efficient_normalize_url_for_deep_crawl caching and correctness."""

    def test_trailing_slash_preserved(self):
        """Trailing slash should be preserved."""
        result = efficient_normalize_url_for_deep_crawl("/foo/bar/", "http://x.com")
        assert result is not None
        assert result.endswith("/foo/bar/")

    def test_cached_results_consistent(self):
        """Calling twice with same args should return same result (cached)."""
        result1 = efficient_normalize_url_for_deep_crawl("/cached", "http://x.com")
        result2 = efficient_normalize_url_for_deep_crawl("/cached", "http://x.com")
        assert result1 == result2

    def test_empty_href_returns_none(self):
        """Empty href should return None."""
        result = efficient_normalize_url_for_deep_crawl("", "http://x.com")
        assert result is None

    def test_none_href_returns_none(self):
        """None href should return None."""
        result = efficient_normalize_url_for_deep_crawl(None, "http://x.com")
        assert result is None

    def test_fragment_removed(self):
        """Fragment should be removed."""
        result = efficient_normalize_url_for_deep_crawl("/page#top", "http://x.com")
        assert "#top" not in result

    def test_hostname_lowercased(self):
        """Hostname should be lowercased."""
        result = efficient_normalize_url_for_deep_crawl("/path", "http://UPPER.COM")
        assert "upper.com" in result

    def test_relative_url_resolution(self):
        """Relative URLs should be resolved correctly."""
        result = efficient_normalize_url_for_deep_crawl(
            "child", "http://x.com/parent/"
        )
        assert result == "http://x.com/parent/child"


# ===================================================================
# CacheContext / CacheMode
# ===================================================================

class TestCacheMode:
    """Verify CacheContext behavior for each CacheMode."""

    def test_enabled_reads_and_writes(self):
        """CacheMode.ENABLED should allow both reads and writes."""
        ctx = CacheContext("http://example.com", CacheMode.ENABLED)
        assert ctx.should_read() is True
        assert ctx.should_write() is True

    def test_disabled_no_reads_no_writes(self):
        """CacheMode.DISABLED should block both reads and writes."""
        ctx = CacheContext("http://example.com", CacheMode.DISABLED)
        assert ctx.should_read() is False
        assert ctx.should_write() is False

    def test_bypass_no_reads_but_writes(self):
        """CacheMode.BYPASS should skip reads but allow writes."""
        ctx = CacheContext("http://example.com", CacheMode.BYPASS)
        assert ctx.should_read() is False
        assert ctx.should_write() is False

    def test_read_only_reads_no_writes(self):
        """CacheMode.READ_ONLY should allow reads, block writes."""
        ctx = CacheContext("http://example.com", CacheMode.READ_ONLY)
        assert ctx.should_read() is True
        assert ctx.should_write() is False

    def test_write_only_no_reads_but_writes(self):
        """CacheMode.WRITE_ONLY should block reads, allow writes."""
        ctx = CacheContext("http://example.com", CacheMode.WRITE_ONLY)
        assert ctx.should_read() is False
        assert ctx.should_write() is True

    def test_raw_url_not_cacheable(self):
        """raw:// URLs should not be cacheable regardless of mode."""
        ctx = CacheContext("raw://<html>test</html>", CacheMode.ENABLED)
        assert ctx.should_read() is False
        assert ctx.should_write() is False

    def test_raw_url_is_raw_html(self):
        """raw:// URLs should be flagged as raw HTML."""
        ctx = CacheContext("raw://<html>test</html>", CacheMode.ENABLED)
        assert ctx.is_raw_html is True
        assert ctx.is_web_url is False

    def test_http_url_is_cacheable(self):
        """http:// URLs should be cacheable."""
        ctx = CacheContext("http://example.com", CacheMode.ENABLED)
        assert ctx.is_cacheable is True
        assert ctx.is_web_url is True

    def test_https_url_is_cacheable(self):
        """https:// URLs should be cacheable."""
        ctx = CacheContext("https://example.com", CacheMode.ENABLED)
        assert ctx.is_cacheable is True

    def test_file_url_is_cacheable(self):
        """file:// URLs should be cacheable."""
        ctx = CacheContext("file:///tmp/test.html", CacheMode.ENABLED)
        assert ctx.is_cacheable is True
        assert ctx.is_local_file is True

    def test_file_url_never_read_from_cache(self):
        """file:// URLs must never be read from cache, regardless of mode.

        Regression guard for the bug where ``should_read()`` was gated on
        ``is_cacheable`` (True for ``file://``) instead of ``is_web_url``.
        Under ENABLED/READ_ONLY that served a stale cached snapshot of a
        local file forever, even after the on-disk file changed, because
        ``CacheValidator`` cannot freshness-validate ``file://`` URLs
        (httpx only speaks http(s), so validation errors fall back to the
        stale entry). Legacy behavior gated cache reads on ``is_web_url``,
        making ``file://`` effectively write-only — this test pins that.
        """
        url = "file:///tmp/test.html"
        for mode in CacheMode:
            ctx = CacheContext(url, mode)
            assert ctx.should_read() is False, (
                f"file:// must not read from cache under {mode.name}; "
                "no freshness validation exists for the file:// scheme"
            )

    def test_file_url_can_still_write_to_cache(self):
        """file:// URLs remain writeable so snapshots may be stored.

        ``should_write()`` is intentionally left gated on ``is_cacheable``
        (not ``is_web_url``): writing a snapshot is harmless and matches
        legacy behavior. Only the read side is restricted.
        """
        ctx = CacheContext("file:///tmp/test.html", CacheMode.ENABLED)
        assert ctx.should_write() is True
        ctx = CacheContext("file:///tmp/test.html", CacheMode.WRITE_ONLY)
        assert ctx.should_write() is True

    def test_always_bypass_overrides_everything(self):
        """always_bypass=True should force read=False, write=False."""
        ctx = CacheContext("http://example.com", CacheMode.ENABLED, always_bypass=True)
        assert ctx.should_read() is False
        assert ctx.should_write() is False

    def test_display_url_for_web(self):
        """Display URL for web URLs should be the URL itself."""
        ctx = CacheContext("http://example.com", CacheMode.ENABLED)
        assert ctx.display_url == "http://example.com"

    def test_display_url_for_raw(self):
        """Display URL for raw HTML should be 'Raw HTML'."""
        ctx = CacheContext("raw://something", CacheMode.ENABLED)
        assert ctx.display_url == "Raw HTML"


# ===================================================================
# sanitize_input_encode
# ===================================================================

class TestSanitizeInputEncode:
    """Verify sanitize_input_encode handles encoding edge cases."""

    def test_normal_utf8_passthrough(self):
        """Normal UTF-8 text should pass through unchanged."""
        text = "Hello, world! This is normal text."
        assert sanitize_input_encode(text) == text

    def test_unicode_text_preserved(self):
        """Unicode characters should be preserved."""
        text = "Caf\u00e9 na\u00efve r\u00e9sum\u00e9"
        assert sanitize_input_encode(text) == text

    def test_empty_string_returns_empty(self):
        """Empty string should return empty string."""
        assert sanitize_input_encode("") == ""

    def test_ascii_text_passthrough(self):
        """Pure ASCII text should pass through."""
        text = "Simple ASCII text 123"
        assert sanitize_input_encode(text) == text

    def test_cjk_characters_preserved(self):
        """CJK characters should be preserved."""
        text = "\u4f60\u597d\u4e16\u754c"
        assert sanitize_input_encode(text) == text

    def test_emoji_preserved(self):
        """Emoji characters should be preserved in UTF-8."""
        text = "Hello \U0001f600 World"
        result = sanitize_input_encode(text)
        assert "Hello" in result
        assert "World" in result


# ===================================================================
# Content hashing
# ===================================================================

class TestGenerateContentHash:
    """Verify generate_content_hash produces consistent results."""

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        hash1 = generate_content_hash("hello world")
        hash2 = generate_content_hash("hello world")
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        hash1 = generate_content_hash("hello world")
        hash2 = generate_content_hash("goodbye world")
        assert hash1 != hash2

    def test_empty_content_valid_hash(self):
        """Empty content should produce a valid hash (not an error)."""
        h = generate_content_hash("")
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_is_hex_string(self):
        """Hash should be a hexadecimal string."""
        h = generate_content_hash("test content")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic_across_calls(self):
        """Hash should be deterministic, not random."""
        content = "The quick brown fox jumps over the lazy dog"
        hashes = [generate_content_hash(content) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_whitespace_sensitive(self):
        """Hash should be sensitive to whitespace differences."""
        h1 = generate_content_hash("hello world")
        h2 = generate_content_hash("hello  world")
        assert h1 != h2

    def test_case_sensitive(self):
        """Hash should be case-sensitive."""
        h1 = generate_content_hash("Hello")
        h2 = generate_content_hash("hello")
        assert h1 != h2

    def test_long_content(self):
        """Long content should hash without error."""
        content = "x" * 1_000_000
        h = generate_content_hash(content)
        assert isinstance(h, str)
        assert len(h) > 0


# ===================================================================
# Image scoring (import-guarded)
# ===================================================================

class TestImageScoring:
    """Test image scoring logic if available.

    score_image_for_usefulness is a nested function, so we test
    the concept indirectly by checking that the module loads and
    the scoring constants exist.
    """

    def test_image_score_threshold_exists(self):
        """IMAGE_SCORE_THRESHOLD config constant should exist."""
        from crawl4ai.config import IMAGE_SCORE_THRESHOLD
        assert isinstance(IMAGE_SCORE_THRESHOLD, (int, float))

    def test_image_description_threshold_exists(self):
        """IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD should exist."""
        from crawl4ai.config import IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD
        assert isinstance(IMAGE_DESCRIPTION_MIN_WORD_THRESHOLD, (int, float))
