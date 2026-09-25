"""Authenticated A2A request boundary: incoming validation and outgoing delegation.

This module wires the narrow contracts in ``conducto.security.trust`` and
``conducto.security.tokens`` to the transport boundary without owning
credentials, provider clients, or global registries. mTLS workload
authentication and OAuth delegated authorization are treated as independent
checks: passing one never substitutes for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

from conducto.security.audit import (
    AuditCategory,
    AuditDecision,
    AuditEmitter,
    AuditEvent,
    AuditEventName,
    AuditOutcome,
    AuditSeverity,
)
from conducto.security.context import AuthorizationContext, Principal
from conducto.security.tokens import (
    ScopeAttenuationError,
    TokenExchangeRequest,
    TokenValidationError,
    TokenValidator,
    ValidatedIdentity,
    attenuate_scopes,
)
from conducto.security.trust import TrustPolicy

from .errors import AuthenticationError

MAX_AUTHORIZATION_HEADER_SIZE = 16_384
BEARER_PREFIX = "Bearer "


@dataclass(frozen=True, slots=True)
class MTLSPeerIdentity:
    """Minimal validated mTLS peer fact required to gate a request.

    Attributes:
        subject_common_name: Peer certificate subject common name, as
            validated by the TLS layer's chain and hostname checks.
    """

    subject_common_name: str

    def __post_init__(self) -> None:
        """Validate the peer subject was supplied."""
        if not self.subject_common_name:
            raise ValueError("subject_common_name is required")


async def authenticate_incoming_request(
    *,
    authorization_header: str | None,
    validator: TokenValidator,
    policy: TrustPolicy,
    task_id: str,
    correlation_id: str,
    mtls_peer: MTLSPeerIdentity | None = None,
    audit: AuditEmitter | None = None,
    trace_headers: dict[str, str] | None = None,
) -> AuthorizationContext:
    """Authenticate one inbound A2A request under an optional remote trace context.

    Args:
        authorization_header: Bounded bearer authorization header.
        validator: Application-owned token validator.
        policy: Trust policy for token and mTLS checks.
        task_id: Domain task identifier, distinct from any trace identifier.
        correlation_id: Conducto correlation identifier for logs and audit.
        mtls_peer: Optional TLS peer identity already validated by the server.
        audit: Optional mandatory audit emitter.
        trace_headers: Optional W3C trace context headers supplied by the transport.

    Returns:
        The immutable authorization context for the inbound request.
    """
    from conducto.core.telemetry import (
        SPAN_A2A_SERVER,
        extract_trace_context,
        start_span,
    )

    extracted = extract_trace_context(trace_headers or {})
    with start_span(
        SPAN_A2A_SERVER,
        kind="server",
        remote_context=extracted.context,
        attributes={
            "conducto.protocol": "a2a",
            "conducto.transport": "jsonrpc",
            "conducto.task.id": task_id,
            "conducto.correlation_id": correlation_id,
            "conducto.invalid_remote_context": extracted.invalid_remote_context,
        },
    ) as span:
        try:
            context = await _authenticate_incoming_request(
                authorization_header=authorization_header,
                validator=validator,
                policy=policy,
                task_id=task_id,
                correlation_id=correlation_id,
                mtls_peer=mtls_peer,
                audit=audit,
            )
        except (AuthenticationError, TokenValidationError) as error:
            span.set_outcome(
                "denied", reason=getattr(error, "reason_code", "authentication_failed")
            )
            raise
        span.set_outcome("success")
        return context


async def _authenticate_incoming_request(
    *,
    authorization_header: str | None,
    validator: TokenValidator,
    policy: TrustPolicy,
    task_id: str,
    correlation_id: str,
    mtls_peer: MTLSPeerIdentity | None = None,
    audit: AuditEmitter | None = None,
) -> AuthorizationContext:
    """Authenticate one inbound A2A request and build its ``AuthorizationContext``.

    Raises:
        AuthenticationError: If mTLS or bearer-token authentication fails.
        TokenValidationError: If the bearer token itself is invalid.
    """
    certificate_policy = policy.certificate
    if certificate_policy is not None and certificate_policy.require_client_certificate:
        if mtls_peer is None:
            await _audit(
                audit,
                AuditEventName.MTLS_REJECTED,
                AuditOutcome.REJECTED,
                "missing_client_certificate",
                task_id,
                correlation_id,
            )
            raise AuthenticationError("mTLS client certificate is required")
        allowed = certificate_policy.allowed_subject_common_names
        if allowed and mtls_peer.subject_common_name not in allowed:
            await _audit(
                audit,
                AuditEventName.MTLS_REJECTED,
                AuditOutcome.REJECTED,
                "untrusted_client_certificate",
                task_id,
                correlation_id,
            )
            raise AuthenticationError("mTLS client certificate is not trusted")
        await _audit(
            audit,
            AuditEventName.MTLS_AUTHENTICATED,
            AuditOutcome.SUCCESS,
            "mtls_verified",
            task_id,
            correlation_id,
        )

        if not authorization_header:
            return AuthorizationContext(
                principal=Principal(
                    subject_id=mtls_peer.subject_common_name,
                    issuer="mtls",
                    audience=tuple(sorted(policy.audience.audiences)),
                ),
                task_id=task_id,
                correlation_id=correlation_id,
                policy_metadata={
                    "trust_policy_version": policy.version,
                    "authentication_method": "mtls",
                },
            )

    if not authorization_header or len(authorization_header) > MAX_AUTHORIZATION_HEADER_SIZE:
        await _audit(
            audit,
            AuditEventName.TOKEN_VALIDATION_REJECTED,
            AuditOutcome.REJECTED,
            "missing_authorization_header",
            task_id,
            correlation_id,
        )
        raise AuthenticationError("a bearer authorization header is required")
    if authorization_header[: len(BEARER_PREFIX)].lower() != BEARER_PREFIX.lower():
        await _audit(
            audit,
            AuditEventName.TOKEN_VALIDATION_REJECTED,
            AuditOutcome.REJECTED,
            "unsupported_authorization_scheme",
            task_id,
            correlation_id,
        )
        raise AuthenticationError(f"only the {BEARER_PREFIX.strip()} scheme is supported")
    token = authorization_header[len(BEARER_PREFIX) :]

    try:
        identity = validator.validate(token, policy=policy)
    except TokenValidationError as error:
        await _audit(
            audit,
            AuditEventName.TOKEN_VALIDATION_REJECTED,
            AuditOutcome.REJECTED,
            error.reason_code,
            task_id,
            correlation_id,
        )
        raise

    await _audit(
        audit,
        AuditEventName.TOKEN_VALIDATED,
        AuditOutcome.SUCCESS,
        "token_validated",
        task_id,
        correlation_id,
        subject_id=identity.subject,
        issuer=identity.issuer,
        audience=identity.audience,
        policy_version=identity.policy_version,
    )

    principal = Principal(
        subject_id=identity.subject,
        issuer=identity.issuer,
        audience=identity.audience,
        claims=identity.claims,
        scopes=identity.scopes,
    )
    return AuthorizationContext(
        principal=principal,
        task_id=task_id,
        correlation_id=correlation_id,
        policy_metadata={
            "trust_policy_version": identity.policy_version,
            "actor_chain": identity.actor_chain,
        },
    )


async def build_delegated_token_request(
    *,
    identity: ValidatedIdentity,
    destination_audience: str,
    requested_scopes: frozenset[str],
    destination_allowed_scopes: frozenset[str],
    subject_token: str,
    subject_token_type: str,
    actor_token: str | None = None,
    actor_token_type: str | None = None,
    task_id: str = "",
    correlation_id: str = "",
    audit: AuditEmitter | None = None,
) -> TokenExchangeRequest:
    """Build a scope-attenuated RFC 8693 token-exchange request for a nested call.

    Outgoing delegated authority is always a subset of both the incoming
    validated identity's scopes and the destination's allowed scopes.

    Raises:
        ScopeAttenuationError: If ``requested_scopes`` would broaden authority.
    """
    from conducto.core.telemetry import SPAN_AUTH_EXCHANGE, start_span

    with start_span(
        SPAN_AUTH_EXCHANGE,
        attributes={
            "conducto.task.id": task_id,
            "conducto.correlation_id": correlation_id,
            "conducto.outbound_audience": destination_audience,
        },
    ) as span:
        try:
            scopes = attenuate_scopes(
                incoming_scopes=identity.scopes,
                requested_scopes=requested_scopes,
                destination_allowed_scopes=destination_allowed_scopes,
            )
        except ScopeAttenuationError as error:
            await _audit(
                audit,
                AuditEventName.DELEGATION_REJECTED,
                AuditOutcome.REJECTED,
                error.reason_code,
                task_id,
                correlation_id,
            )
            span.set_outcome("denied", reason=error.reason_code)
            raise
        await _audit(
            audit,
            AuditEventName.DELEGATION_ATTENUATED,
            AuditOutcome.SUCCESS,
            "scope_attenuated",
            task_id,
            correlation_id,
        )
        span.set_outcome("success")
        return TokenExchangeRequest(
            subject_token=subject_token,
            subject_token_type=subject_token_type,
            audience=destination_audience,
            scope=scopes,
            actor_token=actor_token,
            actor_token_type=actor_token_type,
        )


def build_authorization_header(access_token: str) -> str:
    """Build a bounded ``Bearer`` authorization header for an outgoing request."""
    if not access_token:
        raise ValueError("access_token is required")
    header = f"{BEARER_PREFIX}{access_token}"
    if len(header) > MAX_AUTHORIZATION_HEADER_SIZE:
        raise ValueError("authorization header exceeds the configured size limit")
    return header


async def _audit(
    audit: AuditEmitter | None,
    event_name: AuditEventName,
    outcome: AuditOutcome,
    reason_code: str,
    task_id: str,
    correlation_id: str,
    *,
    subject_id: str = "",
    issuer: str = "",
    audience: str | tuple[str, ...] = "",
    policy_version: str = "",
) -> None:
    if audit is None:
        return
    from conducto.core.telemetry import current_trace_ids

    trace_ids = current_trace_ids()
    decision = AuditDecision.ALLOW if outcome == AuditOutcome.SUCCESS else AuditDecision.DENY
    severity = AuditSeverity.INFO if outcome == AuditOutcome.SUCCESS else AuditSeverity.WARNING
    await audit.emit(
        AuditEvent(
            event_name=event_name,
            category=AuditCategory.AUTHORIZATION,
            decision=decision,
            outcome=outcome,
            reason_code=reason_code,
            subject_id=subject_id,
            issuer=issuer,
            audience=audience,
            task_id=task_id,
            correlation_id=correlation_id,
            trace_id=trace_ids.trace_id if trace_ids is not None else "",
            span_id=trace_ids.span_id if trace_ids is not None else "",
            policy_version=policy_version,
            severity=severity,
        )
    )
