"""Deterministic TLS/mTLS configuration contract tests."""

from __future__ import annotations

import datetime
import ssl

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from conducto.transport import ClientCertificate, TLSConfigurationError, TLSPolicy
from conducto.transport.tls import build_client_ssl_context, build_server_ssl_context


def _self_signed_ca() -> tuple[bytes, bytes, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem, key


def test_tls_policy_requires_trust_roots() -> None:
    with pytest.raises(ValueError):
        TLSPolicy(trusted_ca_pem=b"")


def test_tls_policy_rejects_pre_tls_1_2() -> None:
    ca_pem, _, _ = _self_signed_ca()
    with pytest.raises(ValueError):
        TLSPolicy(trusted_ca_pem=ca_pem, minimum_version=ssl.TLSVersion.TLSv1_1)


def test_client_certificate_requires_material() -> None:
    with pytest.raises(ValueError):
        ClientCertificate(certificate_pem=b"", private_key_pem=b"key")


def test_build_client_ssl_context_always_verifies_peer() -> None:
    ca_pem, _, _ = _self_signed_ca()
    policy = TLSPolicy(trusted_ca_pem=ca_pem)
    context = build_client_ssl_context(policy)
    assert context.check_hostname is True
    assert context.verify_mode is ssl.CERT_REQUIRED


def test_build_client_ssl_context_with_client_certificate() -> None:
    ca_pem, ca_key_pem, _ = _self_signed_ca()
    certificate = ClientCertificate(certificate_pem=ca_pem, private_key_pem=ca_key_pem)
    policy = TLSPolicy(trusted_ca_pem=ca_pem, client_certificate=certificate)
    context = build_client_ssl_context(policy)
    assert context.verify_mode is ssl.CERT_REQUIRED


def test_build_client_ssl_context_rejects_invalid_ca() -> None:
    policy = TLSPolicy(trusted_ca_pem=b"not-a-valid-pem")
    with pytest.raises(TLSConfigurationError):
        build_client_ssl_context(policy)


def test_build_server_ssl_context_requires_server_certificate() -> None:
    ca_pem, _, _ = _self_signed_ca()
    policy = TLSPolicy(trusted_ca_pem=ca_pem)
    with pytest.raises(TLSConfigurationError):
        build_server_ssl_context(policy)


def test_build_server_ssl_context_requires_client_certificates() -> None:
    ca_pem, ca_key_pem, _ = _self_signed_ca()
    certificate = ClientCertificate(certificate_pem=ca_pem, private_key_pem=ca_key_pem)
    policy = TLSPolicy(trusted_ca_pem=ca_pem, client_certificate=certificate)
    context = build_server_ssl_context(policy)
    assert context.verify_mode is ssl.CERT_REQUIRED
