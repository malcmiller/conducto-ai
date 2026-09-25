"""Tests for the shipped A2A identity resolver adapters."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest

from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2AIdentityResolver,
    JWTBearerIdentityResolver,
    StaticTokenIdentityResolver,
    TransportIdentityResolver,
)
from conducto.security import (
    AudiencePolicy,
    CertificatePolicy,
    InvalidSignatureTokenError,
    IssuerPolicy,
    Principal,
    ScopePolicy,
    TokenValidationError,
    TrustPolicy,
    ValidatedIdentity,
)
from conducto.testing import AllowAllIdentityResolver
from conducto.transport import AuthenticationError
from conducto.transport.auth import MTLSPeerIdentity


def _request(headers: dict[str, str] | None = None) -> A2AAuthenticationRequest:
    return A2AAuthenticationRequest(
        task_id="task-1",
        context_id="context-1",
        message_id="message-1",
        request_id="request-1",
        correlation_id="correlation-1",
        headers=headers or {},
    )


def _principal() -> Principal:
    return Principal(
        subject_id="client",
        issuer="local",
        audience="agent",
        scopes=frozenset({"read"}),
    )


def _policy() -> TrustPolicy:
    return TrustPolicy(
        version="policy-1",
        issuer=IssuerPolicy(issuer="https://issuer.example"),
        audience=AudiencePolicy(audiences=frozenset({"agent"})),
        scopes=ScopePolicy(allowed_scopes=frozenset({"read"})),
    )


def test_static_token_identity_resolver_authenticates_without_echoing_token() -> None:
    token = "secret-token"
    identity = StaticTokenIdentityResolver({token: _principal()})(
        _request({"Authorization": f"bearer {token}"})
    )
    assert identity.authorization.principal == _principal()
    assert token not in repr(identity)


@pytest.mark.parametrize("headers", [None, {"authorization": "Bearer wrong-token"}])
def test_static_token_identity_resolver_rejects_credentials_without_echoing(
    headers: dict[str, str] | None,
) -> None:
    token = "secret-token"
    with pytest.raises(AuthenticationError) as error:
        StaticTokenIdentityResolver({token: _principal()})(_request(headers))
    assert token not in str(error.value)


def test_jwt_bearer_identity_resolver_maps_validated_identity() -> None:
    validator = Mock()
    validated = ValidatedIdentity(
        subject="subject",
        issuer="https://issuer.example",
        audience=("agent",),
        scopes=frozenset({"read"}),
        actor_chain=(),
        expires_at=2_000,
        not_before=1_000,
        policy_version="policy-1",
        claims={"roles": ["operator"]},
    )
    validator.validate.return_value = validated
    resolver = JWTBearerIdentityResolver(validator, _policy())  # type: ignore[arg-type]

    result = resolver(_request({"authorization": "bearer opaque-token"}))

    assert result.authorization.principal.subject_id == "subject"
    assert result.authorization.principal.roles == frozenset({"operator"})
    assert result.authorization.task_id == "task-1"
    validator.validate.assert_called_once_with("opaque-token", policy=_policy())


def test_jwt_bearer_identity_resolver_propagates_typed_validation_failure() -> None:
    validator = Mock()
    validator.validate.side_effect = InvalidSignatureTokenError("token signature is invalid")
    resolver = JWTBearerIdentityResolver(validator, _policy())  # type: ignore[arg-type]

    with pytest.raises(TokenValidationError) as error:
        resolver(_request({"authorization": "Bearer credential"}))
    assert "credential" not in str(error.value)


def test_jwt_bearer_identity_resolver_rejects_missing_header() -> None:
    resolver = JWTBearerIdentityResolver(Mock(), _policy())

    with pytest.raises(AuthenticationError):
        resolver(_request())


def test_transport_identity_resolver_delegates_and_supports_mtls_peer() -> None:
    class Validator:
        def validate(self, token: str, *, policy: TrustPolicy) -> ValidatedIdentity:
            assert token == "token"
            return ValidatedIdentity(
                subject="subject",
                issuer="https://issuer.example",
                audience=("agent",),
                scopes=frozenset({"read"}),
                actor_chain=(),
                expires_at=2_000,
                not_before=1_000,
                policy_version=policy.version,
            )

    resolver = TransportIdentityResolver(
        Validator(),
        _policy(),
        mtls_identity_extractor=lambda _request: MTLSPeerIdentity("client"),
    )
    result = asyncio.run(resolver(_request({"authorization": "bearer token"})))
    assert isinstance(result, A2AAuthenticatedIdentity)
    assert result.authorization.principal.subject_id == "subject"


def test_transport_identity_resolver_rejects_missing_header() -> None:
    resolver = TransportIdentityResolver(Mock(), _policy())

    with pytest.raises(AuthenticationError):
        asyncio.run(resolver(_request()))


def test_transport_identity_resolver_supports_mtls_only() -> None:
    resolver = TransportIdentityResolver(
        Mock(),
        replace(_policy(), certificate=CertificatePolicy(trusted_ca_pem=b"ca-pem")),
        mtls_identity_extractor=lambda _request: MTLSPeerIdentity("client"),
    )

    result = asyncio.run(resolver(_request()))

    assert result.authorization.principal.subject_id == "client"
    assert result.authorization.principal.issuer == "mtls"


def test_allow_all_fixture_is_only_exported_from_testing() -> None:
    import conducto
    import conducto.a2a

    assert AllowAllIdentityResolver
    assert not hasattr(conducto.a2a, "AllowAllIdentityResolver")
    assert not hasattr(conducto, "AllowAllIdentityResolver")


def test_resolvers_conform_to_a2a_identity_protocol() -> None:
    static_resolver: A2AIdentityResolver = StaticTokenIdentityResolver({"token": _principal()})
    jwt_resolver: A2AIdentityResolver = JWTBearerIdentityResolver(Mock(), _policy())  # type: ignore[arg-type]
    transport_resolver: A2AIdentityResolver = TransportIdentityResolver(Mock(), _policy())
    test_resolver: A2AIdentityResolver = AllowAllIdentityResolver()
    assert static_resolver
    assert jwt_resolver
    assert transport_resolver
    assert test_resolver
