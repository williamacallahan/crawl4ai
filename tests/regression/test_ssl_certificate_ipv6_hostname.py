"""Regression tests for SSL certificate hostname parsing with IPv6 literals.

Bug: ``SSLCertificate.from_url`` extracted the connection hostname with
``urlparse(url).netloc`` followed by ``split(":")[0]``. For a URL whose
authority embeds an RFC 3986 IPv6 literal (e.g. ``https://[2001:db8::1]/``),
``netloc`` is the bracketed form ``[2001:db8::1]`` and ``split(":")[0]``
splits *inside* the address, yielding the corrupt string ``[2001`` (or
``[`` for ``https://[::1]/``). That corrupt value was then passed to
``socket.create_connection`` (and to ``wrap_socket`` as the SNI
``server_hostname``), causing a ``socket.gaierror`` and making ``from_url``
return ``None`` -- so ``CrawlResult.ssl_certificate`` was ``None`` for any
IPv6-literal HTTPS URL even though the Playwright crawl itself succeeded.

The fix uses ``urlparse(url).hostname``, which strips the RFC 3986 brackets
from IPv6 literals (``[2001:db8::1]`` -> ``2001:db8::1``), producing exactly
what ``socket.create_connection`` and the TLS SNI argument expect.

These tests are fully offline: ``socket.create_connection`` is patched so
no real network/HTTPS connection is ever opened.
"""

from unittest.mock import MagicMock, patch

import pytest

from crawl4ai.ssl_certificate import SSLCertificate


@pytest.mark.parametrize(
    "url, expected_host",
    (
        ("https://[2001:db8::1]/", "2001:db8::1"),
        ("https://[2001:db8::1]:8443/", "2001:db8::1"),
        ("https://[::1]/", "::1"),
        ("https://[::1]:443/", "::1"),
        ("https://[fe80::1]/", "fe80::1"),
        ("https://[fe80::1%25eth0]/", "fe80::1%25eth0"),  # zone-id, %-encoded
        (
            "https://[2001:db8:0:0:0:0:0:1]/path?q=1",
            "2001:db8:0:0:0:0:0:1",
        ),
    ),
)
def test_from_url_passes_unbracketed_ipv6_to_socket(url, expected_host):
    """For IPv6-literal URLs the un-bracketed address must reach
    ``socket.create_connection`` as the host (never ``[2001`` or ``[``).

    Guards against re-introduction of manual ``netloc.split(':')[0]`` parsing,
    which corrupts IPv6 literals by splitting inside the bracketed address.
    """
    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        # The TLS handshake after connect is irrelevant for this assertion;
        # short-circuit it so the function returns without a real socket.
        mock_sock.return_value.__enter__.return_value = MagicMock()
        SSLCertificate.from_url(url, timeout=1)

    mock_sock.assert_called_once()
    args, kwargs = mock_sock.call_args
    assert args[0] == (expected_host, 443)
    assert kwargs.get("timeout") == 1


def test_from_url_returns_certificate_for_ipv6_literal_on_success():
    """A successful TLS fetch for an IPv6-literal URL must return a populated
    ``SSLCertificate`` instead of ``None``.

    Before the fix, ``from_url`` returned ``None`` for every IPv6-literal URL
    because the corrupt host triggered a ``gaierror`` before the TLS handshake.
    With the fix, the un-bracketed host reaches the socket layer, the handshake
    completes, and the certificate is parsed and returned -- i.e. the
    user-visible ``CrawlResult.ssl_certificate`` is no longer ``None``.
    """
    fake_cert_binary = b"\x30\x00"  # minimal placeholder ASN.1 blob
    fake_x509 = MagicMock()
    fake_x509.get_subject.return_value.get_components.return_value = [(b"CN", b"host")]
    fake_x509.get_issuer.return_value.get_components.return_value = [(b"CN", b"issuer")]
    fake_x509.get_version.return_value = 3
    fake_x509.get_serial_number.return_value = 1
    fake_x509.get_notBefore.return_value = b"20240101000000Z"
    fake_x509.get_notAfter.return_value = b"20240101000000Z"
    fake_x509.digest.return_value.hex.return_value = "deadbeef"
    fake_x509.get_signature_algorithm.return_value = b"sha256"
    fake_x509.get_extension_count.return_value = 0

    ssock = MagicMock()
    ssock.getpeercert.return_value = fake_cert_binary

    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        mock_sock.return_value.__enter__.return_value = MagicMock()
        with patch(
            "crawl4ai.ssl_certificate.ssl.create_default_context"
        ) as mock_ctx:
            mock_ctx.return_value.wrap_socket.return_value.__enter__.return_value = ssock
            with patch(
                "crawl4ai.ssl_certificate.OpenSSL.crypto.load_certificate",
                return_value=fake_x509,
            ):
                result = SSLCertificate.from_url("https://[2001:db8::1]/", timeout=1)

    # The connection target must be the un-bracketed IPv6 address.
    assert mock_sock.call_args.args[0] == ("2001:db8::1", 443)

    # A certificate object must come back (not None as in the buggy behavior).
    assert result is not None
    assert isinstance(result, SSLCertificate)
    assert result.subject == {"CN": "host"}
    assert result.issuer == {"CN": "issuer"}
    assert result.fingerprint == "deadbeef"
