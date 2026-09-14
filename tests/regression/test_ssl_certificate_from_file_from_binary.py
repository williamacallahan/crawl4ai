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
offline: certificates are generated in-memory with ``OpenSSL.crypto``, so no
network or filesystem fixtures beyond a temporary directory are required.
"""

import base64

import pytest

from OpenSSL import crypto as openssl_crypto

from crawl4ai.ssl_certificate import SSLCertificate


# ---------------------------------------------------------------------------
# Fixtures: in-memory self-signed certificate (no network/filesystem needed)
# ---------------------------------------------------------------------------


def _make_x509():
    """Build a real self-signed X.509 certificate with one extension.

    The extension lets us assert the extensions loop in ``from_binary``
    actually runs and populates the ``extensions`` field, rather than
    silently always being an empty list.
    """
    key = openssl_crypto.PKey()
    key.generate_key(openssl_crypto.TYPE_RSA, 2048)

    cert = openssl_crypto.X509()
    cert.set_serial_number(0x1234)
    cert.get_subject().CN = "example.com"
    cert.get_subject().O = "ExampleOrg"
    cert.get_issuer().CN = "Test CA"
    cert.get_issuer().O = "ExampleOrg"
    cert.set_notBefore(b"20240101000000Z")
    cert.set_notAfter(b"20250101000000Z")
    cert.set_pubkey(key)
    cert.add_extensions(
        [openssl_crypto.X509Extension(b"subjectAltName", False, b"DNS:example.com")]
    )
    cert.sign(key, "sha256")
    return cert


@pytest.fixture(scope="module")
def x509_cert():
    return _make_x509()


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
def test_from_binary_returns_certificate(x509_cert, format_name, request):
    data = request.getfixturevalue(format_name.lower() + "_bytes")
    cert = SSLCertificate.from_binary(data)

    assert isinstance(cert, SSLCertificate)
    # Inherits from dict and is directly JSON-serializable.
    assert isinstance(cert, dict)
    assert cert.subject == {"CN": "example.com", "O": "ExampleOrg"}
    assert cert.issuer == {"CN": "Test CA", "O": "ExampleOrg"}
    assert cert.valid_from == "20240101000000Z"
    assert cert.valid_until == "20250101000000Z"
    assert cert.fingerprint == x509_cert.digest("sha256").hex()
    assert cert["serial_number"] == hex(0x1234)


def test_from_binary_populates_extensions(der_bytes):
    """The extensions loop must run and populate the extensions list -- a
    real cert with one SAN extension yields exactly one extension entry."""
    cert = SSLCertificate.from_binary(der_bytes)
    assert isinstance(cert["extensions"], list)
    assert len(cert["extensions"]) == 1
    ext = cert["extensions"][0]
    assert "name" in ext and "value" in ext


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
    assert cert.subject == {"CN": "example.com", "O": "ExampleOrg"}
    assert cert.issuer == {"CN": "Test CA", "O": "ExampleOrg"}


def test_from_file_returns_none_for_missing_file(capsys):
    result = SSLCertificate.from_file("/no/such/path/cert.der")
    assert result is None
    assert capsys.readouterr().out
