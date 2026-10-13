"""Regression tests for duplicate RDN attribute preservation in
``SSLCertificate.from_binary`` / ``from_url``.

Bug: ``SSLCertificate.from_binary`` built the ``subject`` and ``issuer``
convenience dicts with ``dict(x509.get_subject().get_components())``.
``get_components()`` returns an ordered list of ``(key, value)`` byte
tuples and may legitimately contain repeated keys -- multiple ``OU`` (CPS
incorporation notices, organizational units), multiple ``O``, multiple
``CN`` -- which are all valid and common in real certificates. Wrapping
that list in ``dict(...)`` silently keeps only the *last* value for each
repeated key, dropping the earlier values with no warning. The truncated
dicts were what ``SSLCertificate.subject`` / ``issuer`` exposed and what
``to_json()`` serialized, so any consumer reading ``cert.subject["OU"]``
(or the REST API's ``ssl_certificate`` JSON field) for a cert with
duplicate RDNs got an incomplete record.

The fix is non-breaking: ``subject`` / ``issuer`` keep their documented
``Dict[str, str]`` shape (last value wins for a repeated attribute), and a
new ``subject_rdn`` / ``issuer_rdn`` property (plus like-named dict keys,
serialized by ``to_json``) exposes the full ordered ``(key, value)`` RDN
component list, preserving every duplicate attribute in certificate order.

The repo's own public ``PUBLIC_CERTIFICATE`` fixture (an Entrust Root CA)
has two ``OU`` components in its subject/issuer, so the data loss was
live-reproducible on the fixture itself: the CPS-incorporation ``OU``
(``www.entrust.net/CPS is incorporated by reference``) was discarded.

These tests are fully offline: the public certificate is embedded (shared
with ``test_ssl_certificate_from_file_from_binary``) and ``from_url`` is
exercised with a fully mocked TLS surface, so no network is needed.
"""

import base64
import hashlib
import json

import pytest
from OpenSSL import crypto as openssl_crypto

from crawl4ai.ssl_certificate import SSLCertificate


# Public Entrust root (Mozilla CA bundle); two OUs in subject/issuer.
PUBLIC_CERTIFICATE = b"""-----BEGIN CERTIFICATE-----
MIIEkTCCA3mgAwIBAgIERWtQVDANBgkqhkiG9w0BAQUFADCBsDELMAkGA1UEBhMC
VVMxFjAUBgNVBAoTDUVudHJ1c3QsIEluYy4xOTA3BgNVBAsTMHd3dy5lbnRydXN0
Lm5ldC9DUFMgaXMgaW5jb3Jwb3JhdGVkIGJ5IHJlZmVyZW5jZTEfMB0GA1UECxMW
KGMpIDIwMDYgRW50cnVzdCwgSW5jLjEtMCsGA1UEAxMkRW50cnVzdCBSb290IENl
cnRpZmljYXRpb24gQXV0aG9yaXR5MB4XDTA2MTEyNzIwMjM0MloXDTI2MTEyNzIw
NTM0MlowgbAxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1FbnRydXN0LCBJbmMuMTkw
NwYDVQQLEzB3d3cuZW50cnVzdC5uZXQvQ1BTIGlzIGluY29ycG9yYXRlZCBieSBy
ZWZlcmVuY2UxHzAdBgNVBAsTFihjKSAyMDA2IEVudHJ1c3QsIEluYy4xLTArBgNV
BAMTJEVudHJ1c3QgUm9vdCBDZXJ0aWZpY2F0aW9uIEF1dGhvcml0eTCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBALaVtkNC+sZtKm9I35RMOVcF7sN5EUFo
Nu3s/poBj6E4KPz3EEZmLk0eGrEaTsbRwJWIsMn/MYszA9u3g3s+IIRe7bJWKKf4
4LlAcTfFy0cOlypowCKVYhXbR9n10Cv/gkvJrT7eTNuQgFA/CYqEAOwwCj0Yzfv9
KlmaI5UXLEWeH25DeW0MXJj+SKfFI0dcXv1u5x609mhF0YaDW6KKjbHjKYD+JXGI
rb68j6xSlkuqUY3kEzEZ6E5Nn9uss2rVvDlUccp6en+Q3X0dgNmBu1kmwhH+5pPi
94DkZfs0Nw4pgHBNrziGLp5/V6+eF67rHMsoIV+2HNjnogQi+dPa2MsCAwEAAaOB
sDCBrTAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zArBgNVHRAEJDAi
gA8yMDA2MTEyNzIwMjM0MlqBDzIwMjYxMTI3MjA1MzQyWjAfBgNVHSMEGDAWgBRo
kORnpKZTgMeGZqTx90tD+4S9bTAdBgNVHQ4EFgQUaJDkZ6SmU4DHhmak8fdLQ/uE
vW0wHQYJKoZIhvZ9B0EABBAwDhsIVjcuMTo0LjADAgSQMA0GCSqGSIb3DQEBBQUA
A4IBAQCT1DCw1wMgKtD5Y+iRDAUgqV8ZyntyTtSx29CW+1RaGSwMCPeyvIWonX9t
O1KzKtvn1ISMY/YPyyYBkVBs9F8U4pN0wBOeMDpQ47RgxRzwIkSNcUesyBrJ6Zua
AGAT/3B+XxFNSRuzFVJ7yVTav52Vr2ua2J7p8eRDjeIRRDq/r72DQnNSi6q7pynP
9WQcCk3RvKqsnyrQ/39/2n3qse0wJcGE2jTSW3iDVuycNsMm4hH2Z0kdkquM++v/
eu6FSqdQgPCnXEqULl8FmTxSQeDNtGPPAUO6nIPcj2A781q0tHuu2guQOHXvgR1m
0vdXcDazv/wor3ElhVsT/h5/WrQ8
-----END CERTIFICATE-----
"""


@pytest.fixture(scope="module")
def x509_cert():
    return openssl_crypto.load_certificate(
        openssl_crypto.FILETYPE_PEM, PUBLIC_CERTIFICATE
    )


@pytest.fixture(scope="module")
def der_bytes(x509_cert):
    return openssl_crypto.dump_certificate(openssl_crypto.FILETYPE_ASN1, x509_cert)


@pytest.fixture(scope="module")
def pem_bytes(x509_cert):
    return openssl_crypto.dump_certificate(openssl_crypto.FILETYPE_PEM, x509_cert)


@pytest.mark.parametrize("format_name", ["DER", "PEM"])
def test_subject_rdn_preserves_duplicate_ous(format_name, request):
    """The fixture's subject has two ``OU`` components. Before the fix,
    ``dict(get_components())`` kept only the last one and silently dropped
    the CPS-incorporation ``OU``. ``subject_rdn`` must preserve *both*
    ``OU`` values in their certificate order."""
    data = request.getfixturevalue(format_name.lower() + "_bytes")
    cert = SSLCertificate.from_binary(data)

    ou_values = [v for k, v in cert.subject_rdn if k == "OU"]
    assert ou_values == [
        "www.entrust.net/CPS is incorporated by reference",
        "(c) 2006 Entrust, Inc.",
    ], f"duplicate OU dropped; got {ou_values!r}"


def test_subject_rdn_matches_raw_get_components_in_order(x509_cert, der_bytes):
    """``subject_rdn`` / ``issuer_rdn`` must equal the decoded
    ``get_components()`` output, element-for-element in order -- the
    authoritative source pyOpenSSL exposes for the certificate Name. This
    guards against any reordering or further collapsing."""
    cert = SSLCertificate.from_binary(der_bytes)
    assert cert.subject_rdn == [
        (k.decode("utf-8"), v.decode("utf-8"))
        for k, v in x509_cert.get_subject().get_components()
    ]
    assert cert.issuer_rdn == [
        (k.decode("utf-8"), v.decode("utf-8"))
        for k, v in x509_cert.get_issuer().get_components()
    ]


def test_subject_dict_still_collapses_to_str_str(der_bytes):
    """The documented ``subject`` schema is ``Dict[str, str]``. The fix must
    not change that (a consumer doing ``cert.subject["OU"].startswith(...)``
    would break if the value became a list). For a repeated attribute the
    last value wins -- this is the documented, backward-compatible shape."""
    cert = SSLCertificate.from_binary(der_bytes)
    assert isinstance(cert.subject, dict)
    assert cert.subject["OU"] == "(c) 2006 Entrust, Inc."
    assert isinstance(cert.subject["OU"], str)
    for value in cert.subject.values():
        assert isinstance(value, str)


def test_to_json_serializes_subject_rdn_and_issuer_rdn(der_bytes):
    """``to_json()`` and the REST API's ``json.dumps(cert)`` path must
    include the full RDN lists so consumers see every duplicate attribute,
    not just the collapsed last value."""
    cert = SSLCertificate.from_binary(der_bytes)
    loaded = json.loads(cert.to_json())
    assert loaded["subject_rdn"] == [
        ["C", "US"],
        ["O", "Entrust, Inc."],
        ["OU", "www.entrust.net/CPS is incorporated by reference"],
        ["OU", "(c) 2006 Entrust, Inc."],
        ["CN", "Entrust Root Certification Authority"],
    ]
    assert loaded["issuer_rdn"] == loaded["subject_rdn"]

    # The dict-subclass instance is directly JSON-serializable without a
    # custom encoder (the path the deployed REST API uses).
    direct = json.loads(json.dumps(cert))
    assert direct["subject_rdn"] == loaded["subject_rdn"]


def test_from_url_preserves_duplicate_ous(der_bytes):
    """``from_url`` delegates to ``from_binary``, so the duplicate-RDN fix
    must reach the URL-fetch path. The TLS surface is fully mocked with the
    real fixture's multi-OU ``get_components()`` output -- mocks that return
    single-valued lists cannot surface this bug."""
    from unittest.mock import MagicMock, patch

    fake_x509 = MagicMock()
    real = openssl_crypto.load_certificate(
        openssl_crypto.FILETYPE_PEM, PUBLIC_CERTIFICATE
    )
    fake_x509.get_subject.return_value.get_components.return_value = (
        real.get_subject().get_components()
    )
    fake_x509.get_issuer.return_value.get_components.return_value = (
        real.get_issuer().get_components()
    )
    fake_x509.get_version.return_value = 3
    fake_x509.get_serial_number.return_value = 1
    fake_x509.get_notBefore.return_value = b"20240101000000Z"
    fake_x509.get_notAfter.return_value = b"20240101000000Z"
    fake_x509.get_signature_algorithm.return_value = b"sha256"
    fake_x509.get_extension_count.return_value = 0

    ssock = MagicMock()
    ssock.getpeercert.return_value = der_bytes

    with patch("crawl4ai.ssl_certificate.socket.create_connection") as mock_sock:
        mock_sock.return_value.__enter__.return_value = MagicMock()
        with patch(
            "crawl4ai.ssl_certificate.ssl.create_default_context"
        ) as mock_ctx:
            mock_ctx.return_value.wrap_socket.return_value.__enter__.return_value = (
                ssock
            )
            with patch(
                "crawl4ai.ssl_certificate.OpenSSL.crypto.load_certificate",
                return_value=fake_x509,
            ), patch(
                "crawl4ai.ssl_certificate.OpenSSL.crypto.dump_certificate",
                return_value=der_bytes,
            ):
                result = SSLCertificate.from_url(
                    "https://example.com", timeout=1
                )

    assert result is not None
    ou_values = [v for k, v in result.subject_rdn if k == "OU"]
    assert ou_values == [
        "www.entrust.net/CPS is incorporated by reference",
        "(c) 2006 Entrust, Inc.",
    ]
    assert result.issuer_rdn == result.subject_rdn


def test_rdn_addition_does_not_change_fingerprint_or_raw_cert(der_bytes, pem_bytes):
    """Adding ``subject_rdn`` / ``issuer_rdn`` to the cert dict must not
    perturb the fingerprint, the normalized DER, or the PEM round-trip --
    those are computed from the DER bytes, not the Name fields."""
    cert = SSLCertificate.from_binary(der_bytes)
    assert cert.fingerprint == hashlib.sha256(der_bytes).hexdigest()
    assert base64.b64decode(cert["raw_cert"]) == der_bytes
    assert cert.to_der() == der_bytes
    assert cert.to_pem() == pem_bytes.decode("utf-8")
