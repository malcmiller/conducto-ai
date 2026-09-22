"""Application-owned TLS/mTLS configuration and certificate selection.

Certificates and trust roots are always supplied explicitly by the
application; nothing here reads undocumented global files or environment
state during import, and there is no production path that disables hostname
or chain verification.
"""

from __future__ import annotations

import os
import ssl
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .errors import TLSConfigurationError

_MIN_SUPPORTED_VERSION = ssl.TLSVersion.TLSv1_2


@dataclass(frozen=True, slots=True)
class ClientCertificate:
    """Application-owned client certificate material for mTLS.

    Attributes:
        certificate_pem: PEM-encoded leaf certificate, and optionally its
            chain, presented to the peer.
        private_key_pem: PEM-encoded private key matching the certificate.
        password: Optional password protecting ``private_key_pem``.
    """

    certificate_pem: bytes
    private_key_pem: bytes
    password: bytes | None = None

    def __post_init__(self) -> None:
        """Validate that certificate and key material were supplied."""
        if not self.certificate_pem or not self.private_key_pem:
            raise ValueError("certificate_pem and private_key_pem are required")


@dataclass(frozen=True, slots=True)
class TLSPolicy:
    """Explicit, application-owned TLS/mTLS policy.

    Attributes:
        trusted_ca_pem: PEM-encoded trust roots used for chain validation.
        client_certificate: Optional client certificate presented for mTLS.
        minimum_version: Minimum negotiated TLS protocol version.
    """

    trusted_ca_pem: bytes
    client_certificate: ClientCertificate | None = None
    minimum_version: ssl.TLSVersion = _MIN_SUPPORTED_VERSION

    def __post_init__(self) -> None:
        """Reject missing trust roots or a TLS floor below 1.2."""
        if not self.trusted_ca_pem:
            raise ValueError("trusted_ca_pem is required")
        if self.minimum_version < _MIN_SUPPORTED_VERSION:
            raise ValueError("minimum_version must be at least TLS 1.2")


@contextmanager
def _private_temp_file(data: bytes) -> Iterator[str]:
    """Write ``data`` to a private temporary file and yield its path."""
    fd, path = tempfile.mkstemp()
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_client_ssl_context(policy: TLSPolicy) -> ssl.SSLContext:
    """Build a client :class:`ssl.SSLContext` that always verifies the peer.

    Hostname verification and certificate-chain validation are always
    enabled; there is no parameter that can disable them.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = policy.minimum_version
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    try:
        context.load_verify_locations(cadata=policy.trusted_ca_pem.decode("ascii"))
    except (ssl.SSLError, UnicodeDecodeError, ValueError) as error:
        raise TLSConfigurationError("trusted CA material is invalid") from error
    if policy.client_certificate is not None:
        _load_client_certificate(context, policy.client_certificate)
    return context


def build_server_ssl_context(policy: TLSPolicy) -> ssl.SSLContext:
    """Build a server :class:`ssl.SSLContext` requiring client mTLS.

    A server context always requires and verifies a client certificate
    against the configured trust roots; this is the only supported mode.
    """
    if policy.client_certificate is None:
        raise TLSConfigurationError("server TLS policy requires a server certificate")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = policy.minimum_version
    context.verify_mode = ssl.CERT_REQUIRED
    try:
        context.load_verify_locations(cadata=policy.trusted_ca_pem.decode("ascii"))
    except (ssl.SSLError, UnicodeDecodeError, ValueError) as error:
        raise TLSConfigurationError("trusted CA material is invalid") from error
    _load_client_certificate(context, policy.client_certificate)
    return context


def _load_client_certificate(context: ssl.SSLContext, certificate: ClientCertificate) -> None:
    """Load a certificate/key pair into ``context`` via private temp files."""
    with (
        _private_temp_file(certificate.certificate_pem) as cert_path,
        _private_temp_file(certificate.private_key_pem) as key_path,
    ):
        key_pw = certificate.password.decode("ascii") if certificate.password else None
        try:
            context.load_cert_chain(cert_path, key_path, key_pw)
        except ssl.SSLError as error:
            raise TLSConfigurationError("client certificate or key is invalid") from error
