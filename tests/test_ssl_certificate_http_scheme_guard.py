"""Regression tests for the SSL certificate fetching scheme guard.

Bug: when ``fetch_ssl_certificate`` was enabled, the crawler unconditionally
called ``SSLCertificate.from_url(url)``. For ``http://`` URLs that helper
silently opened a TLS connection to port 443 of the host and returned the
TLS certificate found there - a certificate that was never used for the
actual (plaintext) HTTP request.  This produced misleading security
metadata for unencrypted crawls.

The fix restores the scheme guard in two places (defense in depth):

1. ``AsyncPlaywrightCrawlerStrategy._crawl_web`` only fetches a certificate
   when the crawl URL's scheme is ``https``.
2. ``SSLCertificate.from_url`` itself refuses non-HTTPS URLs so the public
   API cannot be misused the same way again.

These tests are fully offline: ``socket.create_connection`` is patched so
no real network/HTTPS connection is ever opened.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.ssl_certificate import SSLCertificate


# ---------------------------------------------------------------------------
# Unit tests for SSLCertificate.from_url scheme guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com",
        "HTTP://EXAMPLE.com",  # scheme match must be case-insensitive
        "http://example.com:8080",  # custom port must not trigger :443 fetch
        "ftp://example.com",
        "file:///tmp/something.html",
        "ws://example.com",
        "example.com",  # no scheme at all
        "",
    ),
)
def test_from_url_returns_none_for_non_https_schemes(url):
    """Non-HTTPS URLs must not produce a certificate and must not touch the
    network (the scheme check short-circuits before any socket is opened)."""
    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        result = SSLCertificate.from_url(url, timeout=1)

    assert result is None
    mock_sock.assert_not_called()


def test_from_url_attempts_tls_socket_for_https():
    """HTTPS URLs must still proceed to the TLS fetch path (port 443)."""
    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        # The surrounding TLS handshake is irrelevant here; we only assert that
        # the scheme guard let the call through to the network layer.
        SSLCertificate.from_url("https://example.com", timeout=1)

    mock_sock.assert_called_once()
    args, kwargs = mock_sock.call_args
    assert args[0] == ("example.com", 443)
    assert kwargs.get("timeout") == 1


def test_from_url_https_uppercase_scheme_attempts_socket():
    """``HTTPS://`` (mixed case) must be treated as HTTPS and fetch."""
    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        SSLCertificate.from_url("HTTPS://Example.com", timeout=1)

    mock_sock.assert_called_once_with(("Example.com", 443), timeout=1)


# ---------------------------------------------------------------------------
# Call-site tests for AsyncPlaywrightCrawlerStrategy._crawl_web
# ---------------------------------------------------------------------------


def _build_strategy():
    """Build a strategy with a fully mocked browser surface.

    Construction is lightweight (AsyncPlaywrightCrawlerStrategy.__init__ only
    allocates a BrowserManager; it does not start Chromium), so this stays
    offline and browser-free.
    """
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True),
        logger=AsyncLogger(verbose=False),
    )

    page = MagicMock()
    # Stop _crawl_web immediately after the SSL-cert block: the first
    # mandatory await that follows it is the body-visibility wait_for_selector.
    # Raising here isolates the SSL-fetch decision from the rest of the crawl
    # pipeline without exercising real navigation.
    page.wait_for_selector = AsyncMock(side_effect=RuntimeError("STOP_AFTER_SSL"))
    page.close = AsyncMock()
    context = MagicMock()

    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = AsyncMock(return_value=(page, context))
    strategy.browser_manager.release_page_with_context = AsyncMock()
    return strategy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "http://example.com",
        "HTTP://example.com",
        "http://example.com:8080",
        "ftp://example.com",
    ),
)
async def test_crawler_skips_ssl_fetch_for_non_https_urls(url):
    """With fetch_ssl_certificate enabled, non-HTTPS URLs must NOT trigger
    SSLCertificate.from_url (would otherwise return a misleading :443 cert)."""
    strategy = _build_strategy()
    config = CrawlerRunConfig(fetch_ssl_certificate=True, js_only=True)

    with patch(
        "crawl4ai.async_crawler_strategy.SSLCertificate.from_url"
    ) as fake_from_url:
        # Force a non-fake failure path: if the guard is missing, from_url would
        # be called; we assert it never is. The STOP_AFTER_SSL marker proves we
        # actually reached (and passed) the SSL block.
        with pytest.raises(RuntimeError, match="STOP_AFTER_SSL"):
            await strategy._crawl_web(url, config)

    fake_from_url.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "https://example.com",
        "https://example.com:8443",
        "HTTPS://example.com",
    ),
)
async def test_crawler_fetches_ssl_for_https_urls(url):
    """HTTPS URLs must still trigger exactly one SSLCertificate.from_url call
    with the original URL."""
    strategy = _build_strategy()
    config = CrawlerRunConfig(fetch_ssl_certificate=True, js_only=True)

    with patch(
        "crawl4ai.async_crawler_strategy.SSLCertificate.from_url"
    ) as fake_from_url:
        fake_from_url.return_value = MagicMock(name="ssl_cert")
        with pytest.raises(RuntimeError, match="STOP_AFTER_SSL"):
            await strategy._crawl_web(url, config)

    fake_from_url.assert_called_once_with(url)


@pytest.mark.asyncio
async def test_crawler_skips_ssl_fetch_when_feature_disabled():
    """When fetch_ssl_certificate is disabled, from_url is never called -
    regression guard for the disabled-feature happy path (HTTPS URL)."""
    strategy = _build_strategy()
    config = CrawlerRunConfig(fetch_ssl_certificate=False, js_only=True)

    with patch(
        "crawl4ai.async_crawler_strategy.SSLCertificate.from_url"
    ) as fake_from_url:
        with pytest.raises(RuntimeError, match="STOP_AFTER_SSL"):
            await strategy._crawl_web("https://example.com", config)

    fake_from_url.assert_not_called()
