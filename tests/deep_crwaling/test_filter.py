# // File: tests/deep_crawling/test_filters.py
import asyncio
import gc
import re
import warnings
from urllib.parse import urlparse
import pytest
from crawl4ai import (
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
    URLFilter,
)

# Minimal URLFilter base class stub if not already importable directly for tests
# In a real scenario, this would be imported from the library
if not hasattr(URLFilter, '_update_stats'): # Check if it's a basic stub
    class URLFilter: # Basic stub for testing if needed
        def __init__(self, name=None): self.name = name
        def apply(self, url: str) -> bool: raise NotImplementedError
        def _update_stats(self, passed: bool): pass # Mock implementation

# Assume ContentTypeFilter is structured as discussed. If its definition is not fully
# available for direct import in the test environment, a more elaborate stub or direct
# instantiation of the real class (if possible) would be needed.
# For this example, we assume ContentTypeFilter can be imported and used.

class TestContentTypeFilter:
    @pytest.mark.parametrize(
        "url, allowed_types, expected",
        [
            # Existing tests (examples)
            ("http://example.com/page.html", ["text/html"], True),
            ("http://example.com/page.json", ["application/json"], True),
            ("http://example.com/image.png", ["text/html"], False),
            ("http://example.com/document.pdf", ["application/pdf"], True),
            ("http://example.com/page", ["text/html"], True), # No extension: cannot judge, always allowed
            ("http://example.com/page.unknown", ["text/html"], False), # Unknown extension
            
            # Tests for PHP extensions
            ("http://example.com/index.php", ["application/x-httpd-php"], True),
            ("http://example.com/script.php3", ["application/x-httpd-php"], True),
            ("http://example.com/legacy.php4", ["application/x-httpd-php"], True),
            ("http://example.com/main.php5", ["application/x-httpd-php"], True),
            ("http://example.com/api.php7", ["application/x-httpd-php"], True),
            ("http://example.com/index.phtml", ["application/x-httpd-php"], True),
            ("http://example.com/source.phps", ["application/x-httpd-php-source"], True),

            # Test rejection of PHP extensions
            ("http://example.com/index.php", ["text/html"], False),
            ("http://example.com/script.php3", ["text/plain"], False),
            ("http://example.com/source.phps", ["application/x-httpd-php"], False), # Mismatch MIME
            ("http://example.com/source.php", ["application/x-httpd-php-source"], False), # Mismatch MIME for .php

            # Test case-insensitivity of extensions in URL
            ("http://example.com/PAGE.HTML", ["text/html"], True),
            ("http://example.com/INDEX.PHP", ["application/x-httpd-php"], True),
            ("http://example.com/SOURCE.PHPS", ["application/x-httpd-php-source"], True),

            # Test case-insensitivity of allowed_types
            ("http://example.com/index.php", ["APPLICATION/X-HTTPD-PHP"], True),

            # Query strings / fragments must not become part of the extension
            ("http://example.com/guide.html?highlight=keyword", ["text/html"], True),
            ("http://example.com/doc.pdf#page=2", ["text/html"], False),
        ],
    )
    def test_apply(self, url, allowed_types, expected):
        content_filter = ContentTypeFilter(
            allowed_types=allowed_types
        )
        assert content_filter.apply(url) == expected

    @pytest.mark.parametrize(
        "url, expected_extension",
        [
            ("http://example.com/file.html", "html"),
            ("http://example.com/file.tar.gz", "gz"),
            ("http://example.com/path/", ""),
            ("http://example.com/nodot", ""),
            ("http://example.com/.config", "config"), # hidden file with extension
            ("http://example.com/path/to/archive.BIG.zip", "zip"), # Case test
            # Query strings, fragments and ';' path params must be stripped
            # before extracting the extension (they previously leaked in and
            # caused false negatives, e.g. "html?highlight=keyword").
            ("http://example.com/guide.html?highlight=keyword", "html"),
            ("http://example.com/manual.pdf#page=5", "pdf"),
            ("http://example.com/photo.jpg;width=100", "jpg"),
        ]
    )
    def test_extract_extension(self, url, expected_extension):
        # Test the static method directly
        assert ContentTypeFilter._extract_extension(url) == expected_extension


class TestURLPatternFilter:
    @pytest.mark.parametrize(
        "pattern, url, expected",
        [
            # Multi-part suffix patterns (the regression: a stored suffix that
            # itself contains a dot must match the filename's dotted tail, not
            # only its last dot-segment).
            ("*.tar.gz", "https://example.com/archive.tar.gz", True),
            ("*.tar.gz", "https://example.com/docs/backup.tar.gz", True),
            ("*.tar.bz2", "https://example.com/a.tar.bz2", True),
            ("*.tar.gz", "https://example.com/a.tgz", False),  # not a .tar.gz
            # Single-part suffix patterns must keep working.
            ("*.pdf", "https://example.com/report.pdf", True),
            ("*.pdf", "https://example.com/page.html", False),
        ],
    )
    def test_suffix_matching(self, pattern, url, expected):
        # apply() is @lru_cache'd, so a fresh instance per case avoids stale
        # results across parametrized runs.
        f = URLPatternFilter(patterns=[pattern])
        assert f.apply(url) is expected


class TestDomainFilter:
    # Regression guard for the port/userinfo/IPv6 stripping bug: a previous
    # "fast" regex `://([^/]+)` returned the full authority (host:port and
    # user:pass@host), which never matched configured allow/block entries.

    @pytest.mark.parametrize(
        "url, expected",
        [
            # Ports must be stripped (the primary regression).
            ("http://example.com:443/page", "example.com"),
            ("http://example.com:8080/page", "example.com"),
            ("https://guce.techcrunch.com:443/consent", "guce.techcrunch.com"),
            ("http://127.0.0.1:443/secret", "127.0.0.1"),
            # Userinfo must be stripped.
            ("http://user:pass@example.com/page", "example.com"),
            # IPv6 brackets and ports must be stripped.
            ("http://[::1]:8000/page", "::1"),
            ("http://[::1]/page", "::1"),
            # Plain hosts unchanged; hostname is already lowercased.
            ("http://example.com/page", "example.com"),
            ("http://EXAMPLE.COM/page", "example.com"),
            ("http://sub.example.com/page", "sub.example.com"),
            # Non-URL / empty input must return "" rather than raising.
            ("", ""),
            ("not a url", ""),
        ],
    )
    def test_extract_domain(self, url, expected):
        assert DomainFilter._extract_domain(url) == expected

    @pytest.mark.parametrize(
        "url, expected",
        [
            # Port-bearing blocked hrefs must be blocked (primary bypass bug).
            ("http://evil.com:443/page", False),
            ("http://evil.com:8080/page", False),
            ("http://sub.evil.com:443/page", False),
            ("http://127.0.0.1:443/secret", False),
            # Port-bearing non-blocked hrefs still pass.
            ("http://good.com:443/page", True),
            ("http://good.com/page", True),
        ],
    )
    def test_blocked_domains_with_port(self, url, expected):
        f = DomainFilter(blocked_domains=["evil.com", "127.0.0.1"])
        assert f.apply(url) is expected

    @pytest.mark.parametrize(
        "url, expected",
        [
            # Port-bearing allowed hrefs must be allowed (over-block bug).
            ("http://example.com:8080/page", True),
            ("http://example.com:443/page", True),
            ("http://sub.example.com:443/page", True),
            # Port-bearing non-allowed hrefs are still rejected.
            ("http://other.com:443/page", False),
            ("http://other.com/page", False),
        ],
    )
    def test_allowed_domains_with_port(self, url, expected):
        f = DomainFilter(allowed_domains=["example.com"])
        assert f.apply(url) is expected


# ---------------------------------------------------------------------------
# Deterministic in-process filter stubs for FilterChain tests (no network).
# ---------------------------------------------------------------------------


class _SyncFilter(URLFilter):
    """A sync filter that returns a fixed boolean and records every call."""

    def __init__(self, name, result):
        super().__init__(name=name)
        self._result = result
        self.calls = []

    def apply(self, url):
        self.calls.append(url)
        self._update_stats(self._result)
        return self._result


class _AsyncFilter(URLFilter):
    """An async filter that returns a fixed boolean and tracks coroutine state.

    ``body_started`` / ``finally_ran`` distinguish whether the coroutine was
    actually awaited (gathered) versus only ``close()``d without running.
    """

    def __init__(self, name, result):
        super().__init__(name=name)
        self._result = result
        self.body_started = False
        self.finally_ran = False

    async def apply(self, url):
        try:
            self.body_started = True
            await asyncio.sleep(0)  # yield once so it's a genuine coroutine
            self._update_stats(self._result)
            return self._result
        finally:
            self.finally_ran = True


class _RaisingSyncFilter(URLFilter):
    """A sync filter that violates the ``apply -> bool`` contract by raising.

    Records every call so tests can confirm the raising filter was actually
    evaluated and that the original exception propagates unchanged through
    the disposal re-raise in ``FilterChain.apply``.

    Raises a *fresh* exception on every call (storing only the type and
    message) rather than retaining an exception instance. This is deliberate:
    a retained exception object keeps its ``__traceback__`` alive, which
    keeps the suspended ``FilterChain.apply`` frame (and its ``tasks`` list
    of collected coroutines) alive, so the coroutines would NOT be collected
    by the in-window ``gc.collect()`` and the "was never awaited" warning
    would slip past ``warnings.catch_warnings(record=True)`` to
    ``sys.unraisablehook``. A fresh exception is released at the end of the
    ``except`` block (PEP 3110), so the coroutines become collectible and any
    leak is captured deterministically.
    """

    def __init__(self, name, exc_type, exc_msg):
        super().__init__(name=name)
        self._exc_type = exc_type
        self._exc_msg = exc_msg
        self.calls = []

    def apply(self, url):
        self.calls.append(url)
        raise self._exc_type(self._exc_msg)


def _assert_no_never_awaited_warnings(recorded):
    leaked = [
        str(w.message)
        for w in recorded
        if issubclass(w.category, RuntimeWarning)
        and "was never awaited" in str(w.message)
    ]
    assert (
        not leaked
    ), f"Expected no un-awaited-coroutine RuntimeWarnings, got: {leaked}"


async def _apply_raising_and_collect_warnings(chain, url, exc_type, exc_match):
    """Await ``chain.apply(url)`` expecting it to raise ``exc_type``.

    The exception is caught and its type/message asserted *inside* the
    ``except`` block while the exception value is still bound. PEP 3110 then
    auto-clears the bound name at the end of the ``except`` block, releasing
    the exception's traceback frame -- which holds ``FilterChain.apply``'s
    ``tasks`` list of collected coroutines. Only after that release is
    ``gc.collect()`` called, *inside* the active ``warnings.catch_warnings``
    recording window, so any un-awaited-coroutine ``RuntimeWarning`` is
    reliably captured in the returned list rather than leaking later to
    ``sys.unraisablehook``.

    (``with pytest.raises(...)`` instead retains the exception in its
    RaisesContext through the ``gc.collect()`` call, keeping the coroutines
    alive and the warning out of the recorded list -- which is why this
    helper uses ``try/except``.)
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            await chain.apply(url)
        except exc_type as e:
            assert re.search(exc_match, str(e)), (
                f"expected {exc_match!r} to match {str(e)!r}"
            )
        gc.collect()
    return caught



class TestFilterChain:
    """Regression coverage for FilterChain.apply over async+sync mixes.

    Prior to the fix, a synchronous filter rejecting *after* one or more
    async filter coroutines had already been collected into ``tasks`` caused
    those coroutines to be abandoned (never awaited / closed). Python emits
    ``RuntimeWarning: coroutine '...' was never awaited`` when they were
    garbage-collected. These tests pin down that the coroutines are now
    properly disposed of via ``.close()`` on the short-circuit path.
    """

    @pytest.mark.asyncio
    async def test_sync_rejection_closes_pending_async_coroutine(self):
        # The exact trigger: async filter before a rejecting sync filter.
        async_ok = _AsyncFilter("async-pass", True)
        sync_reject = _SyncFilter("sync-reject", False)
        chain = FilterChain([async_ok, sync_reject])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.apply("https://example.com/report.pdf")
            gc.collect()

        assert result is False
        # The sync filter was actually evaluated and short-circuited the chain.
        assert sync_reject.calls == ["https://example.com/report.pdf"]
        # The async coroutine was disposed without running its body (cheap
        # short-circuit: no work performed for an already-rejected URL).
        assert async_ok.body_started is False
        # No coroutine leaked -> no RuntimeWarning.
        _assert_no_never_awaited_warnings(caught)
        # Chain-level stats: total=1, passed=0, rejected=1.
        assert chain.stats.total_urls == 1
        assert chain.stats.passed_urls == 0
        assert chain.stats.rejected_urls == 1

    @pytest.mark.asyncio
    async def test_sync_rejection_closes_multiple_pending_async_coroutines(self):
        a1 = _AsyncFilter("async-1", True)
        a2 = _AsyncFilter("async-2", False)
        a3 = _AsyncFilter("async-3", True)
        sync_reject = _SyncFilter("sync-reject", False)
        chain = FilterChain([a1, a2, a3, sync_reject])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.apply("https://example.com/x.zip")
            gc.collect()

        assert result is False
        for f in (a1, a2, a3):
            assert f.body_started is False
        _assert_no_never_awaited_warnings(caught)
        assert chain.stats.rejected_urls == 1
        assert chain.stats.total_urls == 1

    @pytest.mark.asyncio
    async def test_real_content_relevance_filter_short_circuit_no_warning(self):
        # End-to-end against a real async filter (ContentRelevanceFilter) with
        # its network call patched out, to confirm the production async filter
        # path also benefits from the cleanup.
        from crawl4ai.deep_crawling.filters import ContentRelevanceFilter
        from crawl4ai import utils as _c4ai_utils

        async def _fake_peek_html(url):
            return (
                "<html><head><title>AI defence</title>"
                "<meta name='description' content='AI defence systems'>"
                "</head></html>"
            )

        original = _c4ai_utils.HeadPeekr.peek_html
        _c4ai_utils.HeadPeekr.peek_html = _fake_peek_html
        try:
            crf = ContentRelevanceFilter(query="AI defence", threshold=0.0)
            sync_reject = _SyncFilter("sync-reject", False)
            chain = FilterChain([crf, sync_reject])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await chain.apply("https://techcrunch.com/2025/report.pdf")
                gc.collect()

            assert result is False
            # Body never ran because the sync filter short-circuited first.
            _assert_no_never_awaited_warnings(caught)
        finally:
            _c4ai_utils.HeadPeekr.peek_html = original

    # ------------------------------------------------------------------
    # Regression: a trailing sync filter that *raises* (rather than
    # returning False) after async coroutines have already been collected
    # into ``tasks`` must likewise dispose of those coroutines via
    # ``.close()``, instead of leaving them never-awaited. Without the
    # try/except disposal in the collection loop, CPython emits
    # ``RuntimeWarning: coroutine '...' was never awaited`` for each
    # collected coroutine when it is garbage-collected during the unwind.
    #
    # These tests use ``_apply_raising_and_collect_warnings`` (a
    # ``try/except`` that releases the exception before ``gc.collect()``)
    # rather than ``with pytest.raises(...)``: the latter retains the
    # exception/traceback -- and thus the frame holding ``tasks`` -- through
    # ``gc.collect()``, so the leaked warning slips past the recording
    # window to ``sys.unraisablehook`` instead of being captured in
    # ``caught``. The ``try/except`` form makes the failure deterministic.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sync_filter_raising_closes_pending_async_coroutine(self):
        # The commit hardened the `return False` trailing-sync case; the
        # raising case of the same trailing sync filter must likewise dispose
        # of the already-collected async coroutines, not leave them
        # never-awaited.
        async_ok = _AsyncFilter("async-pass", True)
        sync_raising = _RaisingSyncFilter("sync-raising", ValueError, "boom")
        chain = FilterChain([async_ok, sync_raising])

        caught = await _apply_raising_and_collect_warnings(
            chain, "https://example.com/x", ValueError, "boom"
        )

        # The async coroutine was disposed of (close()'d) without running.
        assert async_ok.body_started is False
        assert async_ok.finally_ran is False
        # The raising sync filter was actually evaluated before the unwind.
        assert sync_raising.calls == ["https://example.com/x"]
        # No coroutine leaked -> no RuntimeWarning.
        _assert_no_never_awaited_warnings(caught)

    @pytest.mark.asyncio
    async def test_sync_filter_raising_closes_multiple_pending_async_coroutines(self):
        # Multiple async coroutines collected before the raising sync filter
        # must all be disposed of.
        a1 = _AsyncFilter("async-1", True)
        a2 = _AsyncFilter("async-2", False)
        a3 = _AsyncFilter("async-3", True)
        sync_raising = _RaisingSyncFilter("sync-raising", ValueError, "kaboom")
        chain = FilterChain([a1, a2, a3, sync_raising])

        caught = await _apply_raising_and_collect_warnings(
            chain, "https://example.com/x.zip", ValueError, "kaboom"
        )

        for f in (a1, a2, a3):
            assert f.body_started is False
            assert f.finally_ran is False
        assert sync_raising.calls == ["https://example.com/x.zip"]
        _assert_no_never_awaited_warnings(caught)

    @pytest.mark.asyncio
    async def test_real_content_relevance_filter_raising_sync_no_warning(self):
        # End-to-end against a real async filter (ContentRelevanceFilter)
        # with its network call patched out, followed by a raising sync
        # filter. The real async coroutine must be disposed of via the
        # exception-path cleanup, not left never-awaited.
        from crawl4ai.deep_crawling.filters import ContentRelevanceFilter
        from crawl4ai import utils as _c4ai_utils

        async def _fake_peek_html(url):
            return (
                "<html><head><title>AI defence</title>"
                "<meta name='description' content='AI defence systems'>"
                "</head></html>"
            )

        original = _c4ai_utils.HeadPeekr.peek_html
        _c4ai_utils.HeadPeekr.peek_html = _fake_peek_html
        try:
            crf = ContentRelevanceFilter(query="AI defence", threshold=0.0)
            sync_raising = _RaisingSyncFilter("sync-raising", ValueError, "bad")
            chain = FilterChain([crf, sync_raising])

            caught = await _apply_raising_and_collect_warnings(
                chain, "https://techcrunch.com/2025/report.pdf", ValueError, "bad"
            )

            assert sync_raising.calls == ["https://techcrunch.com/2025/report.pdf"]
            _assert_no_never_awaited_warnings(caught)
        finally:
            _c4ai_utils.HeadPeekr.peek_html = original

