"""Regression tests for ``SSLCertificate.from_file`` and ``from_binary``.

Bug: the public reference docs (`docs/md_v2/advanced/ssl-certificate.md`,
mirrored in `deploy/docker/c4ai-doc-context.md`) advertised two public
constructors -- ``SSLCertificate.from_file(file_path)`` and
``SSLCertificate.from_binary(binary_data)`` -- with copy-paste runnable
examples, but the class in ``crawl4ai/ssl_certificate.py`` never implemented
them. Calling either raised a hard ``AttributeError`` at call time.

``from_url`` cannot substitute for these because it only accepts ``https://``
URLs (returns ``None`` for any other scheme by design), so there was no
one-call entry point on the class for loading a certificate from an on-disk
``.der``/``.pem`` file or from raw bytes captured from another source.

The fix adds both methods to ``SSLCertificate``. These tests are fully
offline: a public certificate is embedded, so no
network or filesystem fixtures beyond a temporary directory are required.
"""

import base64
import hashlib
import json
import re

import pytest

from OpenSSL import crypto as openssl_crypto

from crawl4ai.ssl_certificate import SSLCertificate


# ---------------------------------------------------------------------------
# Fixtures: public certificate (no network or signing key needed)
# ---------------------------------------------------------------------------


# Public Entrust root from Mozilla's CA bundle; no private key is needed.
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
    return openssl_crypto.load_certificate(openssl_crypto.FILETYPE_PEM, PUBLIC_CERTIFICATE)


@pytest.fixture(scope="module")
def der_bytes(x509_cert):
    return openssl_crypto.dump_certificate(openssl_crypto.FILETYPE_ASN1, x509_cert)


@pytest.fixture(scope="module")
def pem_bytes(x509_cert):
    return openssl_crypto.dump_certificate(openssl_crypto.FILETYPE_PEM, x509_cert)


@pytest.fixture
def der_file(tmp_path, der_bytes):
    path = tmp_path / "cert.der"
    path.write_bytes(der_bytes)
    return str(path)


@pytest.fixture
def pem_file(tmp_path, pem_bytes):
    path = tmp_path / "cert.pem"
    path.write_bytes(pem_bytes)
    return str(path)


# ---------------------------------------------------------------------------
# 1. The documented methods exist on the class (the original bug)
# ---------------------------------------------------------------------------


def test_documented_methods_are_present_on_class():
    """The bug manifested as ``AttributeError`` on access; guard against
    re-removing the documented public methods (and against a dict key
    shadowing the attribute on instances, since the class subclasses dict)."""
    for method in ("from_file", "from_binary"):
        assert hasattr(SSLCertificate, method)
        assert callable(getattr(SSLCertificate, method))

    instance = SSLCertificate({"subject": {}})
    assert callable(getattr(instance, "from_file", None))
    assert callable(getattr(instance, "from_binary", None))


# ---------------------------------------------------------------------------
# 2. from_binary parses PEM and DER into a populated SSLCertificate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format_name", ["DER", "PEM"])
def test_from_binary_returns_certificate(x509_cert, format_name, request, der_bytes):
    data = request.getfixturevalue(format_name.lower() + "_bytes")
    cert = SSLCertificate.from_binary(data)

    assert isinstance(cert, SSLCertificate)
    # Inherits from dict and is directly JSON-serializable.
    assert isinstance(cert, dict)
    assert cert.subject == {"C": "US", "O": "Entrust, Inc.", "OU": "(c) 2006 Entrust, Inc.", "CN": "Entrust Root Certification Authority"}
    assert cert.issuer == {"C": "US", "O": "Entrust, Inc.", "OU": "(c) 2006 Entrust, Inc.", "CN": "Entrust Root Certification Authority"}
    assert cert.valid_from == "20061127202342Z"
    assert cert.valid_until == "20261127205342Z"
    # Fingerprint must match an independent SHA-256 of the DER bytes -- not the
    # tautological ``x509.digest("sha256").hex()`` oracle the original test
    # used (that transform is itself the bug: it re-hex-encodes the ASCII
    # bytes of the colon-separated uppercase hex string into a 190-char
    # mangled value). See ``test_fingerprint_is_sha256_lowercase_hex_of_der``
    # below for the full format contract.
    assert cert.fingerprint == hashlib.sha256(der_bytes).hexdigest()
    assert cert["serial_number"] == hex(1164660820)


def test_from_binary_populates_extensions(der_bytes):
    """The extensions loop must run and populate the extensions list -- a
    public root yields its six extension entries."""
    cert = SSLCertificate.from_binary(der_bytes)
    assert isinstance(cert["extensions"], list)
    assert len(cert["extensions"]) == 6
    ext = cert["extensions"][0]
    assert "name" in ext and "value" in ext
    unknown = next(ext for ext in cert["extensions"] if ext["name"] == "UNDEF")
    assert bytes.fromhex(unknown["value"])


def test_from_binary_normalizes_raw_cert_to_der(der_bytes, pem_bytes, x509_cert):
    """Regardless of input format, ``raw_cert`` must be base64-encoded DER
    so that ``to_der``/``to_pem`` round-trip correctly. Storing the original
    input bytes (as a naive wrapper might) would make ``to_der`` return PEM
    text for PEM input and break ``to_pem`` re-parsing.
    """
    canonical_der = openssl_crypto.dump_certificate(
        openssl_crypto.FILETYPE_ASN1, x509_cert
    )
    # __init__ runs _decode_cert_data, which UTF-8-decodes every bytes value
    # to str, so the stored raw_cert is the base64 string (not bytes).
    expected_raw = base64.b64encode(canonical_der).decode("utf-8")

    for data in (der_bytes, pem_bytes):
        cert = SSLCertificate.from_binary(data)
        assert cert["raw_cert"] == expected_raw


def test_from_binary_pem_round_trips_to_der_and_pem(der_bytes, pem_bytes):
    """Starting from PEM input, the DER/PEM exports must still be the
    canonical DER/PEM forms (proves normalization, not echoing the input)."""
    cert = SSLCertificate.from_binary(pem_bytes)
    assert cert.to_der() == der_bytes
    assert cert.to_pem() == pem_bytes.decode("utf-8")


def test_from_binary_returns_none_for_unparseable_bytes(capsys):
    """Garbage input must produce ``None`` (not raise) and print a warning,
    matching ``from_url``'s print-then-return-None convention."""
    result = SSLCertificate.from_binary(b"not a certificate at all")
    assert result is None
    assert capsys.readouterr().out  # a warning was printed


# ---------------------------------------------------------------------------
# 3. from_file reads PEM and DER files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["der", "pem"])
def test_from_file_returns_certificate(fmt, request):
    file_path = request.getfixturevalue(fmt + "_file")
    cert = SSLCertificate.from_file(file_path)

    assert isinstance(cert, SSLCertificate)
    assert cert.subject == {"C": "US", "O": "Entrust, Inc.", "OU": "(c) 2006 Entrust, Inc.", "CN": "Entrust Root Certification Authority"}
    assert cert.issuer == {"C": "US", "O": "Entrust, Inc.", "OU": "(c) 2006 Entrust, Inc.", "CN": "Entrust Root Certification Authority"}


def test_from_file_returns_none_for_missing_file(capsys):
    result = SSLCertificate.from_file("/no/such/path/cert.der")
    assert result is None
    assert capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. fingerprint contract: 64-char lowercase-hex SHA-256 digest of the DER
# ---------------------------------------------------------------------------
#
# The public docs (``docs/md_v2/advanced/ssl-certificate.md`` and the mirrored
# ``deploy/docker/c4ai-doc-context.md``) state:
#
#     5. ``fingerprint`` (str)
#        - The SHA-256 digest (lowercase hex).
#        - E.g. "d14d2e..."
#
# The original implementation computed it as ``x509.digest("sha256").hex()``.
# ``X509.digest("sha256")`` does NOT return the raw 32-byte digest -- it returns
# ASCII bytes holding a *colon-separated uppercase* hex string
# (``b"8D:2C:1B:..."``, 95 bytes). Calling ``bytes.hex()`` re-hex-encodes every
# ASCII byte (each ``:`` literal ``0x3A`` included), producing a 190-character
# mangled string that is not a valid SHA-256 hex digest by any convention.
#
# These tests pin the documented contract using independent oracles
# (``hashlib.sha256(der).hexdigest()`` and the openssl-CLI-emitted form
# normalized to lowercase hex) so the bug cannot silently return.


@pytest.mark.parametrize("format_name", ["DER", "PEM"])
def test_fingerprint_is_sha256_lowercase_hex_of_der(format_name, request, der_bytes, x509_cert):
    """The fingerprint must be the lowercase-hex SHA-256 digest of the DER
    bytes, matching the documented contract.

    Guards against the original ``x509.digest("sha256").hex()`` bug, which
    re-hex-encoded the ASCII of the colon-separated uppercase form and yielded
    a 190-character string. The mixing of length, regex, and two independent
    value oracles ensures a regression cannot pass: the broken value fails
    the length check, the regex check, the ``hashlib`` oracle, and the
    openssl-CLI-normalized oracle simultaneously.
    """
    data = request.getfixturevalue(format_name.lower() + "_bytes")
    cert = SSLCertificate.from_binary(data)

    fp = cert.fingerprint
    assert isinstance(fp, str)

    # Format contract: exactly 64 lowercase hex chars (no colons, no 0x prefix).
    assert len(fp) == 64, f"expected 64-char hex; got {len(fp)}-char {fp!r}"
    assert re.fullmatch(r"[0-9a-f]{64}", fp), f"not lowercase-hex: {fp!r}"
    assert fp == fp.lower(), f"not lowercase: {fp!r}"
    assert ":" not in fp, f"contains colons: {fp!r}"

    # Independent oracle 1: hashlib SHA-256 of the canonical DER bytes. This is
    # exactly what the docs say the value should be (and what a user comparing
    # ``cert.to_json()`` output to ``hashlib.sha256(open(pem,'rb').read())
    # .hexdigest()`` or an audit/secrets-database cross-reference would need).
    assert fp == hashlib.sha256(der_bytes).hexdigest()

    # Independent oracle 2: the colon-separated uppercase form emitted by
    # ``openssl x509 -fingerprint -sha256`` (``X509.digest("sha256")`` in
    # pyOpenSSL), normalized to lowercase-hex. This pins cross-tool parity
    # with the standard openssl fingerprint output.
    openssl_normalized = (
        x509_cert.digest("sha256").decode("ascii").replace(":", "").lower()
    )
    assert fp == openssl_normalized

    # Negative regression: the buggy 190-char transform must NOT be produced.
    buggy = x509_cert.digest("sha256").hex()
    assert fp != buggy, "regression: fingerprint reverted to the buggy 190-char mangling"
    assert len(buggy) == 190  # documents the shape of the old bug


def test_fingerprint_survives_to_json_round_trip(der_bytes):
    """The deployed REST API serializes ``CrawlResult.ssl_certificate`` to
    JSON via the inherited-``dict`` ``json.dumps`` path (see
    ``crawl4ai/models.py:116`` and ``deploy/docker/server.py:971``). The
    fingerprint must survive a JSON round-trip unchanged -- otherwise an
    audit/secrets-database consumer of ``cert.to_json()`` output would receive
    a mismatched or mangled value.
    """
    cert = SSLCertificate.from_binary(der_bytes)
    expected = hashlib.sha256(der_bytes).hexdigest()

    # cert["fingerprint"]: dict-item access path consumers use directly.
    assert cert["fingerprint"] == expected

    # cert.fingerprint: property access path (same underlying dict value).
    assert cert.fingerprint == expected

    # cert.to_json(): the export path the REST API/audit consumers use.
    json_str = cert.to_json()
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert loaded["fingerprint"] == expected

    # The dict subclass instance must also be directly JSON-serializable
    # without a custom encoder (this is the path the REST API uses).
    direct = json.loads(json.dumps(cert))
    assert direct["fingerprint"] == expected


def test_fingerprint_is_independent_of_input_format(der_bytes, pem_bytes):
    """A certificate parsed from PEM bytes and from DER bytes must yield the
    same fingerprint, since the value is a hash of the normalized DER (not
    of the input). A naive implementation hashing the *input* bytes would
    diverge here.
    """
    from_der = SSLCertificate.from_binary(der_bytes).fingerprint
    from_pem = SSLCertificate.from_binary(pem_bytes).fingerprint
    assert from_der == from_pem
    assert re.fullmatch(r"[0-9a-f]{64}", from_der)
