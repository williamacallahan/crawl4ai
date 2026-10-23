"""Certificate extensions read as OpenSSL printed them, and one malformed
extension does not lose the certificate (pyOpenSSL 26 removed X509Extension)."""

import datetime

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import AuthorityInformationAccessOID, ExtendedKeyUsageOID, ExtensionOID, NameOID

from crawl4ai.ssl_certificate import SSLCertificate


def _certificate(*extensions):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.test")])
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    builder = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
               .public_key(key.public_key()).serial_number(1)
               .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=30)))
    for extension in extensions:
        builder = builder.add_extension(extension, critical=False)
    return builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


def _values(certificate):
    return {ext["name"]: ext["value"] for ext in certificate["extensions"]}


def test_web_server_extensions_read_as_openssl_text():
    der = _certificate(
        x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
        x509.AuthorityInformationAccess([
            x509.AccessDescription(AuthorityInformationAccessOID.OCSP,
                                   x509.UniformResourceIdentifier("http://o.example")),
            x509.AccessDescription(AuthorityInformationAccessOID.CA_ISSUERS,
                                   x509.UniformResourceIdentifier("http://i.example/ca.crt")),
        ]),
        x509.CertificatePolicies([x509.PolicyInformation(x509.ObjectIdentifier("2.23.140.1.2.1"), None)]),
        x509.CRLDistributionPoints([x509.DistributionPoint(
            [x509.UniformResourceIdentifier("http://c.example/ca.crl")], None, None, None)]),
        x509.UnrecognizedExtension(x509.ObjectIdentifier("2.16.840.1.113730.1.13"), b"\x16\x05hello"),
    )
    values = _values(SSLCertificate.from_binary(der))
    assert values["extendedKeyUsage"] == "TLS Web Server Authentication, TLS Web Client Authentication"
    assert values["authorityInfoAccess"] == (
        "OCSP - URI:http://o.example\nCA Issuers - URI:http://i.example/ca.crt"
    )
    assert values["certificatePolicies"] == "Policy: 2.23.140.1.2.1"
    assert values["crlDistributionPoints"] == "Full Name:\n  URI:http://c.example/ca.crl"
    assert values["nsComment"] == "hello"


def test_malformed_extension_keeps_the_certificate():
    # basicConstraints holding ASN.1 NULL: OpenSSL accepts it, cryptography refuses it.
    der = _certificate(x509.UnrecognizedExtension(ExtensionOID.BASIC_CONSTRAINTS, b"\x05\x00"))
    certificate = SSLCertificate.from_binary(der)
    assert certificate is not None
    assert certificate.subject["CN"] == "example.test"
    assert certificate["extensions"] == []
    assert certificate["extensions_error"]
