# // File: tests/deep_crawling/test_filters.py
import pytest
from urllib.parse import urlparse
from crawl4ai import ContentTypeFilter, DomainFilter, URLPatternFilter, URLFilter

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
