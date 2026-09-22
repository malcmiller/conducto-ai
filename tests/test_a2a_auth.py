"""Deterministic authenticated A2A request-boundary contract tests."""

from __future__ import annotations

import asyncio

import pytest

from conducto.security import (
    AudiencePolicy,
    CertificatePolicy,
    ClockSkewPolicy,
    InMemoryAuditSink,
    IssuerPolicy,
    ScopeAttenuationError,
    ScopePolicy,
    TokenValidationError,
    TrustPolicy,
    ValidatedIdentity,
)
from conducto.security.audit import AuditEmitter
from conducto.transport import AuthenticationError, MTLSPeerIdentity
from conducto.transport.auth import (
    BEARER_PREFIX,
    authenticate_incoming_request,
    build_authorization_header,
    build_delegated_token_request,
)

ISSUER = "https://issuer.example"
AUDIENCE = "agent-b"
_TOKEN_HEADER = f"{BEARER_PREFIX}opaque-token"


class _StaticValidator:
    def __init__(self, identity: ValidatedIdentity | None = None, error: Exception | None = None):
        self._identity = identity
        self._error = error

    def validate(self, token: str, *, policy: TrustPolicy) -> ValidatedIdentity:
        if self._error is not None:
            raise self._error
        assert self._identity is not None
        return self._identity


def _identity(**overrides: object) -> ValidatedIdentity:
    defaults: dict[str, object] = dict(
        subject="workload-a",
        issuer=ISSUER,
        audience=(AUDIENCE,),
        scopes=frozenset({"read"}),
        actor_chain=(),
        expires_at=2_000,
        not_before=1_000,
        policy_version="policy-1",
    )
    defaults.update(overrides)
    return ValidatedIdentity(**defaults)  # type: ignore[arg-type]


def _policy(**overrides: object) -> TrustPolicy:
    defaults: dict[str, object] = dict(
        version="policy-1",
        issuer=IssuerPolicy(issuer=ISSUER),
        audience=AudiencePolicy(audiences=frozenset({AUDIENCE})),
        scopes=ScopePolicy(allowed_scopes=frozenset({"read", "write"})),
        clock_skew=ClockSkewPolicy(),
    )
    defaults.update(overrides)
    return TrustPolicy(**defaults)  # type: ignore[arg-type]


def test_authenticate_incoming_request_succeeds() -> None:
    validator = _StaticValidator(identity=_identity())

    async def run() -> None:
        context = await authenticate_incoming_request(
            authorization_header=_TOKEN_HEADER,
            validator=validator,
            policy=_policy(),
            task_id="task-1",
            correlation_id="corr-1",
        )
        assert context.principal.subject_id == "workload-a"
        assert context.principal.scopes == frozenset({"read"})
        assert context.task_id == "task-1"

    asyncio.run(run())


def test_authenticate_incoming_request_rejects_missing_header() -> None:
    validator = _StaticValidator(identity=_identity())

    async def run() -> None:
        with pytest.raises(AuthenticationError):
            await authenticate_incoming_request(
                authorization_header=None,
                validator=validator,
                policy=_policy(),
                task_id="task-1",
                correlation_id="corr-1",
            )

    asyncio.run(run())


def test_authenticate_incoming_request_rejects_non_bearer_scheme() -> None:
    validator = _StaticValidator(identity=_identity())

    async def run() -> None:
        with pytest.raises(AuthenticationError):
            await authenticate_incoming_request(
                authorization_header="Basic abcdef",
                validator=validator,
                policy=_policy(),
                task_id="task-1",
                correlation_id="corr-1",
            )

    asyncio.run(run())


def test_authenticate_incoming_request_propagates_validation_error() -> None:
    validator = _StaticValidator(error=TokenValidationError("invalid"))

    async def run() -> None:
        with pytest.raises(TokenValidationError):
            await authenticate_incoming_request(
                authorization_header=_TOKEN_HEADER,
                validator=validator,
                policy=_policy(),
                task_id="task-1",
                correlation_id="corr-1",
            )

    asyncio.run(run())


def test_authenticate_incoming_request_requires_mtls_when_policy_mandates_it() -> None:
    validator = _StaticValidator(identity=_identity())
    policy = _policy(certificate=CertificatePolicy(trusted_ca_pem=b"ca-pem"))

    async def run() -> None:
        with pytest.raises(AuthenticationError):
            await authenticate_incoming_request(
                authorization_header=_TOKEN_HEADER,
                validator=validator,
                policy=policy,
                task_id="task-1",
                correlation_id="corr-1",
                mtls_peer=None,
            )

    asyncio.run(run())


def test_authenticate_incoming_request_mtls_and_bearer_are_independent() -> None:
    """A trusted mTLS peer does not bypass bearer-token validation."""
    validator = _StaticValidator(error=TokenValidationError("invalid"))
    policy = _policy(certificate=CertificatePolicy(trusted_ca_pem=b"ca-pem"))

    async def run() -> None:
        with pytest.raises(TokenValidationError):
            await authenticate_incoming_request(
                authorization_header=_TOKEN_HEADER,
                validator=validator,
                policy=policy,
                task_id="task-1",
                correlation_id="corr-1",
                mtls_peer=MTLSPeerIdentity(subject_common_name="svc-a"),
            )

    asyncio.run(run())


def test_authenticate_incoming_request_rejects_untrusted_peer_common_name() -> None:
    validator = _StaticValidator(identity=_identity())
    policy = _policy(
        certificate=CertificatePolicy(
            trusted_ca_pem=b"ca-pem", allowed_subject_common_names=frozenset({"svc-a"})
        )
    )

    async def run() -> None:
        with pytest.raises(AuthenticationError):
            await authenticate_incoming_request(
                authorization_header=_TOKEN_HEADER,
                validator=validator,
                policy=policy,
                task_id="task-1",
                correlation_id="corr-1",
                mtls_peer=MTLSPeerIdentity(subject_common_name="svc-untrusted"),
            )

    asyncio.run(run())


def test_authenticate_incoming_request_emits_audit_events() -> None:
    validator = _StaticValidator(identity=_identity())
    emitter = AuditEmitter(InMemoryAuditSink())

    async def run() -> None:
        await authenticate_incoming_request(
            authorization_header=_TOKEN_HEADER,
            validator=validator,
            policy=_policy(),
            task_id="task-1",
            correlation_id="corr-1",
            audit=emitter,
        )

    asyncio.run(run())
    assert emitter.sink.events  # type: ignore[attr-defined]


def test_build_delegated_token_request_attenuates_scopes() -> None:
    identity = _identity(scopes=frozenset({"read", "write"}))

    async def run() -> None:
        request = await build_delegated_token_request(
            identity=identity,
            destination_audience="agent-c",
            requested_scopes=frozenset({"read"}),
            destination_allowed_scopes=frozenset({"read", "write"}),
            subject_token="subject-token",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        )
        assert request.scope == frozenset({"read"})
        assert request.audience == "agent-c"

    asyncio.run(run())


def test_build_delegated_token_request_rejects_broader_scope() -> None:
    identity = _identity(scopes=frozenset({"read"}))

    async def run() -> None:
        with pytest.raises(ScopeAttenuationError):
            await build_delegated_token_request(
                identity=identity,
                destination_audience="agent-c",
                requested_scopes=frozenset({"write"}),
                destination_allowed_scopes=frozenset({"read", "write"}),
                subject_token="subject-token",
                subject_token_type="urn:ietf:params:oauth:token-type:jwt",
            )

    asyncio.run(run())


def test_build_authorization_header() -> None:
    assert build_authorization_header("abc") == f"{BEARER_PREFIX}abc"
    with pytest.raises(ValueError):
        build_authorization_header("")
