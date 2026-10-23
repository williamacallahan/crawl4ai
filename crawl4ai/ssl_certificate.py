"""SSL Certificate class for handling certificate operations."""

import ssl
import socket
import base64
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import unquote, urlparse
import OpenSSL.crypto
from cryptography import x509 as crypto_x509
from cryptography.x509.oid import AuthorityInformationAccessOID, ExtendedKeyUsageOID, ExtensionOID, ObjectIdentifier
from pathlib import Path

_EXTENSION_NAMES = {
    ExtensionOID.SUBJECT_DIRECTORY_ATTRIBUTES: "subjectDirectoryAttributes",
    ExtensionOID.SUBJECT_KEY_IDENTIFIER: "subjectKeyIdentifier",
    ExtensionOID.KEY_USAGE: "keyUsage",
    ExtensionOID.PRIVATE_KEY_USAGE_PERIOD: "privateKeyUsagePeriod",
    ExtensionOID.SUBJECT_ALTERNATIVE_NAME: "subjectAltName",
    ExtensionOID.ISSUER_ALTERNATIVE_NAME: "issuerAltName",
    ExtensionOID.BASIC_CONSTRAINTS: "basicConstraints",
    ExtensionOID.NAME_CONSTRAINTS: "nameConstraints",
    ExtensionOID.CRL_DISTRIBUTION_POINTS: "crlDistributionPoints",
    ExtensionOID.CERTIFICATE_POLICIES: "certificatePolicies",
    ExtensionOID.POLICY_MAPPINGS: "policyMappings",
    ExtensionOID.AUTHORITY_KEY_IDENTIFIER: "authorityKeyIdentifier",
    ExtensionOID.POLICY_CONSTRAINTS: "policyConstraints",
    ExtensionOID.EXTENDED_KEY_USAGE: "extendedKeyUsage",
    ExtensionOID.FRESHEST_CRL: "freshestCRL",
    ExtensionOID.INHIBIT_ANY_POLICY: "inhibitAnyPolicy",
    ExtensionOID.ISSUING_DISTRIBUTION_POINT: "issuingDistributionPoint",
    ExtensionOID.AUTHORITY_INFORMATION_ACCESS: "authorityInfoAccess",
    ExtensionOID.SUBJECT_INFORMATION_ACCESS: "subjectInfoAccess",
    ExtensionOID.OCSP_NO_CHECK: "noCheck",
    ExtensionOID.TLS_FEATURE: "tlsfeature",
    ExtensionOID.CRL_NUMBER: "crlNumber",
    ExtensionOID.DELTA_CRL_INDICATOR: "deltaCRL",
    ExtensionOID.PRECERT_SIGNED_CERTIFICATE_TIMESTAMPS: "ct_precert_scts",
    ExtensionOID.PRECERT_POISON: "ct_precert_poison",
    ExtensionOID.SIGNED_CERTIFICATE_TIMESTAMPS: "ct_cert_scts",
    ExtensionOID.MS_CERTIFICATE_TEMPLATE: "msCertTemplate",
    ExtensionOID.ADMISSIONS: "x509ExtAdmission",
    ObjectIdentifier("2.16.840.1.113730.1.1"): "nsCertType",
    ObjectIdentifier("2.16.840.1.113730.1.13"): "nsComment",
}

# OpenSSL's long names, so extendedKeyUsage reads as it did through pyOpenSSL.
_KEY_PURPOSES = {
    ExtendedKeyUsageOID.SERVER_AUTH: "TLS Web Server Authentication",
    ExtendedKeyUsageOID.CLIENT_AUTH: "TLS Web Client Authentication",
    ExtendedKeyUsageOID.CODE_SIGNING: "Code Signing",
    ExtendedKeyUsageOID.EMAIL_PROTECTION: "E-mail Protection",
    ExtendedKeyUsageOID.TIME_STAMPING: "Time Stamping",
    ExtendedKeyUsageOID.OCSP_SIGNING: "OCSP Signing",
}
_GENERAL_NAME_LABELS = {
    crypto_x509.DNSName: "DNS", crypto_x509.RFC822Name: "email",
    crypto_x509.UniformResourceIdentifier: "URI", crypto_x509.IPAddress: "IP Address",
    crypto_x509.RegisteredID: "Registered ID",
}


def _general_name(name) -> str:
    return f"{_GENERAL_NAME_LABELS.get(type(name), type(name).__name__)}:{name.value}"


def _extension_value(value: crypto_x509.ExtensionType) -> str:
    """Render common values without depending on removed OpenSSL APIs."""
    if isinstance(value, crypto_x509.UnrecognizedExtension):
        raw = value.public_bytes()
        # nsComment is a short DER IA5String; OpenSSL printed its text.
        if value.oid.dotted_string == "2.16.840.1.113730.1.13" and len(raw) > 2 \
                and raw[0] == 0x16 and raw[1] == len(raw) - 2:
            return raw[2:].decode("ascii", "replace")
        return raw.hex()
    if isinstance(value, crypto_x509.ExtendedKeyUsage):
        return ", ".join(_KEY_PURPOSES.get(purpose, purpose.dotted_string) for purpose in value)
    if isinstance(value, crypto_x509.AuthorityInformationAccess):
        methods = {AuthorityInformationAccessOID.OCSP: "OCSP",
                   AuthorityInformationAccessOID.CA_ISSUERS: "CA Issuers"}
        return "\n".join(
            f"{methods.get(entry.access_method, entry.access_method.dotted_string)} - "
            f"{_general_name(entry.access_location)}"
            for entry in value
        )
    if isinstance(value, crypto_x509.CertificatePolicies):
        return "\n".join(f"Policy: {policy.policy_identifier.dotted_string}" for policy in value)
    if isinstance(value, crypto_x509.CRLDistributionPoints):
        return "\n".join(
            "Full Name:\n" + "\n".join(f"  {_general_name(name)}" for name in point.full_name)
            for point in value if point.full_name
        )
    if isinstance(value, (crypto_x509.PrecertificateSignedCertificateTimestamps,
                          crypto_x509.SignedCertificateTimestamps)):
        return "\n".join(
            f"Log ID: {sct.log_id.hex(':').upper()}, Timestamp: {sct.timestamp.isoformat()}"
            for sct in value
        )
    if isinstance(value, crypto_x509.BasicConstraints):
        return "CA:" + str(value.ca).upper() + (
            f", pathlen:{value.path_length}" if value.path_length is not None else ""
        )
    if isinstance(value, crypto_x509.KeyUsage):
        usages = [
            ("digital_signature", "Digital Signature"),
            ("content_commitment", "Non Repudiation"),
            ("key_encipherment", "Key Encipherment"),
            ("data_encipherment", "Data Encipherment"),
            ("key_agreement", "Key Agreement"),
            ("key_cert_sign", "Certificate Sign"),
            ("crl_sign", "CRL Sign"),
        ]
        if value.key_agreement:
            usages += [("encipher_only", "Encipher Only"), ("decipher_only", "Decipher Only")]
        return ", ".join(label for attribute, label in usages if getattr(value, attribute))
    if isinstance(value, (crypto_x509.SubjectAlternativeName, crypto_x509.IssuerAlternativeName)):
        return ", ".join(_general_name(name) for name in value)
    if isinstance(value, crypto_x509.SubjectKeyIdentifier):
        return value.digest.hex(":").upper()
    if isinstance(value, crypto_x509.AuthorityKeyIdentifier) and (
        value.key_identifier is not None
        and value.authority_cert_issuer is None
        and value.authority_cert_serial_number is None
    ):
        return value.key_identifier.hex(":").upper()
    # Other recognized types retain their public, readable representation.
    return str(value)


# === Inherit from dict ===
class SSLCertificate(dict):
    """
    A class representing an SSL certificate, behaving like a dictionary
    for direct JSON serialization. It stores the certificate information internally
    and provides methods for export and property access.

    Inherits from dict, so instances are directly JSON serializable.

    Construction:
        - ``from_url(url, timeout=10)``: fetch and parse the TLS certificate
          presented by an HTTPS endpoint (returns ``None`` for non-HTTPS URLs).
        - ``from_file(file_path)``: load and parse a certificate from a local
          file containing PEM- or DER-encoded certificate data.
        - ``from_binary(binary_data)``: parse a certificate from raw PEM or DER
          bytes (e.g. captured from a socket or another source).
    """

    # Use __slots__ for potential memory optimization if desired, though less common when inheriting dict
    # __slots__ = ("_cert_info",) # If using slots, be careful with dict inheritance interaction

    def __init__(self, cert_info: Dict[str, Any]):
        """
        Initializes the SSLCertificate object.

        Args:
            cert_info (Dict[str, Any]): The raw certificate dictionary.
        """
        # 1. Decode the data (handle bytes -> str)
        decoded_info = self._decode_cert_data(cert_info)

        # 2. Store the decoded info internally (optional but good practice)
        # self._cert_info = decoded_info # You can keep this if methods rely on it

        # 3. Initialize the dictionary part of the object with the decoded data
        super().__init__(decoded_info)

    @staticmethod
    def _decode_cert_data(data: Any) -> Any:
        """Helper method to decode bytes in certificate data."""
        if isinstance(data, bytes):
            try:
                # Try UTF-8 first, fallback to latin-1 for arbitrary bytes
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1") # Or handle as needed, maybe hex representation
        elif isinstance(data, dict):
            return {
                (
                    k.decode("utf-8") if isinstance(k, bytes) else k
                ): SSLCertificate._decode_cert_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [SSLCertificate._decode_cert_data(item) for item in data]
        return data

    @staticmethod
    def _decode_rdn_components(
        components: List[Tuple[bytes, bytes]]
    ) -> List[Tuple[str, str]]:
        """Decode pyOpenSSL RDN ``get_components()`` output into a list of
        ``(str, str)`` tuples, preserving the original order *and* any
        duplicate attribute types.

        ``x509.get_subject().get_components()`` (and the issuer equivalent)
        returns an ordered list of ``(key, value)`` byte tuples and may
        legitimately contain repeated keys (e.g. multiple ``OU`` entries,
        which are common in real certificates -- CPS incorporation notices,
        organizational units, qualification statements). Wrapping that
        list in ``dict(...)`` silently keeps only the last value for each
        repeated key. This helper keeps every entry so the full certificate
        Name is represented without data loss.

        Decoding mirrors :meth:`_decode_cert_data`: UTF-8 first, with a
        latin-1 fallback for attribute values that are not valid UTF-8.
        """
        decoded: List[Tuple[str, str]] = []
        for k, v in components:
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            if isinstance(v, bytes):
                try:
                    value = v.decode("utf-8")
                except UnicodeDecodeError:
                    value = v.decode("latin-1")
            else:
                value = v
            decoded.append((key, value))
        return decoded

    @staticmethod
    def from_url(url: str, timeout: int = 10) -> Optional["SSLCertificate"]:
        """
        Create SSLCertificate instance from a URL. Fetches cert info and initializes.

        Only HTTPS URLs are eligible: an SSL/TLS certificate only exists for TLS
        connections, so fetching a certificate for an HTTP URL (or any non-HTTPS
        scheme) would silently connect to port 443 and return a certificate that
        was never used for the actual request. Returns ``None`` for such URLs.
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() != "https":
                return None
            # Use urlparse's hostname accessor: it correctly strips RFC 3986
            # brackets from IPv6 literals (e.g. '[2001:db8::1]' -> '2001:db8::1'),
            # drops userinfo, and lowercases the host. Scoped IPv6 literals keep
            # an RFC 6874-encoded zone delimiter until it is decoded below.
            # The previous manual `netloc.split(':')[0]` split broke IPv6
            # literals (and userinfo) by splitting inside the address.
            hostname = parsed.hostname
            if hostname is None:
                return None
            if ":" in hostname:
                hostname = unquote(hostname)
            port = 443 if parsed.port is None else parsed.port
            tls_hostname = hostname.split("%", 1)[0] if ":" in hostname else hostname

            context = ssl.create_default_context()
            # Set check_hostname to False and verify_mode to CERT_NONE temporarily
            # for potentially problematic certificates during fetch, but parse the result regardless.
            # context.check_hostname = False
            # context.verify_mode = ssl.CERT_NONE

            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=tls_hostname) as ssock:
                    cert_binary = ssock.getpeercert(binary_form=True)
                    if not cert_binary:
                         print(f"Warning: No certificate returned for {hostname}")
                         return None

                    return SSLCertificate.from_binary(cert_binary)

        except ssl.SSLCertVerificationError as e:
             print(f"SSL Verification Error for {url}: {e}")
             # Decide if you want to proceed or return None based on your needs
             # You might try fetching without verification here if needed, but be cautious.
             return None
        except socket.gaierror:
            print(f"Could not resolve hostname: {hostname}")
            return None
        except socket.timeout:
            print(f"Connection timed out for {url}")
            return None
        except Exception as e:
            print(f"Error fetching/processing certificate for {url}: {e}")
            # Log the full error details if needed: logging.exception("Cert fetch error")
            return None

    @staticmethod
    def from_binary(binary_data: bytes) -> Optional["SSLCertificate"]:
        """
        Create an SSLCertificate instance from raw certificate bytes.

        Accepts either PEM (base64-encoded, ``-----BEGIN CERTIFICATE-----``)
        or DER (binary ASN.1) input. The parsed certificate is normalized to
        DER internally so that the ``raw_cert`` field and the ``to_der`` /
        ``to_pem`` exports are always consistent regardless of the input
        format.

        Args:
            binary_data (bytes): PEM- or DER-encoded certificate bytes.
                ``bytearray`` and ``memoryview`` are also accepted (coerced
                to ``bytes``) to support data captured directly from a socket.

        Returns:
            Optional["SSLCertificate"]: The parsed certificate, or ``None`` if
            the bytes could not be parsed as either PEM or DER (a warning is
            printed in that case, matching ``from_url``'s convention).
        """
        try:
            # OpenSSL.crypto.load_certificate requires ``bytes``; accept the
            # common bytes-like types a caller might have assembled from a
            # socket capture (bytearray/memoryview) by coercing to bytes.
            if isinstance(binary_data, (bytearray, memoryview)):
                binary_data = bytes(binary_data)
            x509 = None
            # Try PEM first, then DER/ASN.1. Each format only parses its own
            # encoding, so the order does not risk a false-positive match.
            for ftype in (OpenSSL.crypto.FILETYPE_PEM, OpenSSL.crypto.FILETYPE_ASN1):
                try:
                    x509 = OpenSSL.crypto.load_certificate(ftype, binary_data)
                    break
                except OpenSSL.crypto.Error:
                    continue
            if x509 is None:
                print("Error: Could not parse certificate data in PEM or DER format")
                return None

            # Normalize to DER so to_der/to_pem round-trip regardless of
            # whether the caller supplied PEM or DER bytes. from_url stores
            # base64(der) too (getpeercert(binary_form=True) returns DER), so
            # this keeps the on-disk/raw-bytes path consistent with it.
            der_bytes = OpenSSL.crypto.dump_certificate(
                OpenSSL.crypto.FILETYPE_ASN1, x509
            )

            subject_components = SSLCertificate._decode_rdn_components(
                x509.get_subject().get_components()
            )
            issuer_components = SSLCertificate._decode_rdn_components(
                x509.get_issuer().get_components()
            )

            cert_info_raw = {
                # ``subject`` / ``issuer`` stay Dict[str, str] to preserve the
                # documented schema (and the existing consumers that read them
                # as single-valued). For repeated attribute types the last
                # value wins. The full ordered RDN component list -- which
                # preserves every duplicate attribute (e.g. multiple OU/O/CN
                # entries) -- is stored alongside in ``subject_rdn`` /
                # ``issuer_rdn`` and exposed via the like-named properties, so
                # no certificate data is silently dropped.
                "subject": dict(subject_components),
                "issuer": dict(issuer_components),
                "subject_rdn": subject_components,
                "issuer_rdn": issuer_components,
                "version": x509.get_version(),
                "serial_number": hex(x509.get_serial_number()),
                "not_before": x509.get_notBefore(),
                "not_after": x509.get_notAfter(),
                "fingerprint": hashlib.sha256(der_bytes).hexdigest(),
                "signature_algorithm": x509.get_signature_algorithm(),
                "raw_cert": base64.b64encode(der_bytes),
            }

            # cryptography parses every extension at once and rejects some that
            # OpenSSL accepts; keep the certificate and say why its list is empty.
            try:
                parsed = x509.to_cryptography().extensions
            except ValueError as error:
                print(f"Certificate extensions could not be parsed: {error}")
                parsed = []
                cert_info_raw["extensions_error"] = str(error)
            cert_info_raw["extensions"] = [
                {"name": _EXTENSION_NAMES.get(ext.oid, "UNDEF"), "value": _extension_value(ext.value)}
                for ext in parsed
            ]

            return SSLCertificate(cert_info_raw)
        except Exception as e:
            print(f"Error processing certificate binary data: {e}")
            return None

    @staticmethod
    def from_file(file_path: str) -> Optional["SSLCertificate"]:
        """
        Create an SSLCertificate instance from a local certificate file.

        The file may contain either PEM- or DER-encoded certificate data;
        parsing is delegated to :meth:`from_binary` after reading the file.

        Args:
            file_path (str): Path to a PEM (``.pem``) or DER (``.der``)
                certificate file.

        Returns:
            Optional["SSLCertificate"]: The parsed certificate, or ``None`` if
            the file could not be read or parsed (a warning is printed in
            that case, matching ``from_url``'s convention).
        """
        try:
            binary_data = Path(file_path).read_bytes()
        except (OSError, IOError) as e:
            print(f"Error reading certificate file {file_path}: {e}")
            return None
        return SSLCertificate.from_binary(binary_data)


    # --- Properties now access the dictionary items directly via self[] ---
    @property
    def issuer(self) -> Dict[str, str]:
        return self.get("issuer", {}) # Use self.get for safety

    @property
    def subject(self) -> Dict[str, str]:
        return self.get("subject", {})

    @property
    def issuer_rdn(self) -> List[Tuple[str, str]]:
        """The issuer Name as an ordered list of ``(key, value)`` RDN
        tuples, preserving duplicate attribute types in their certificate
        order (e.g. multiple ``OU`` entries). Use this when the
        :attr:`issuer` dict would collapse repeated attributes and drop
        data. Each value is a decoded ``str``."""
        return self.get("issuer_rdn", [])

    @property
    def subject_rdn(self) -> List[Tuple[str, str]]:
        """The subject Name as an ordered list of ``(key, value)`` RDN
        tuples, preserving duplicate attribute types in their certificate
        order (e.g. multiple ``OU`` entries). Use this when the
        :attr:`subject` dict would collapse repeated attributes and drop
        data. Each value is a decoded ``str``."""
        return self.get("subject_rdn", [])

    @property
    def valid_from(self) -> str:
        return self.get("not_before", "")

    @property
    def valid_until(self) -> str:
        return self.get("not_after", "")

    @property
    def fingerprint(self) -> str:
        return self.get("fingerprint", "")

    # --- Export methods can use `self` directly as it is the dict ---
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """Export certificate as JSON."""
        # `self` is already the dictionary we want to serialize
        json_str = json.dumps(self, indent=2, ensure_ascii=False)
        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")
            return None
        return json_str

    def to_pem(self, filepath: Optional[str] = None) -> Optional[str]:
        """Export certificate as PEM."""
        try:
            # Decode the raw_cert (which should be string due to _decode)
            raw_cert_bytes = base64.b64decode(self.get("raw_cert", ""))
            x509 = OpenSSL.crypto.load_certificate(
                OpenSSL.crypto.FILETYPE_ASN1, raw_cert_bytes
            )
            pem_data = OpenSSL.crypto.dump_certificate(
                OpenSSL.crypto.FILETYPE_PEM, x509
            ).decode("utf-8")

            if filepath:
                Path(filepath).write_text(pem_data, encoding="utf-8")
                return None
            return pem_data
        except Exception as e:
             print(f"Error converting to PEM: {e}")
             return None

    def to_der(self, filepath: Optional[str] = None) -> Optional[bytes]:
        """Export certificate as DER."""
        try:
            # Decode the raw_cert (which should be string due to _decode)
            der_data = base64.b64decode(self.get("raw_cert", ""))
            if filepath:
                Path(filepath).write_bytes(der_data)
                return None
            return der_data
        except Exception as e:
             print(f"Error converting to DER: {e}")
             return None

    # Optional: Add __repr__ for better debugging
    def __repr__(self) -> str:
        subject_cn = self.subject.get('CN', 'N/A')
        issuer_cn = self.issuer.get('CN', 'N/A')
        return f"<SSLCertificate Subject='{subject_cn}' Issuer='{issuer_cn}'>"
