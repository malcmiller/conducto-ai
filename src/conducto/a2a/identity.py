"""Reference identity resolvers for inbound A2A requests."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from conducto.security import (
    AuthorizationContext,
    JWTBearerTokenValidator,
    Principal,
    TokenValidator,
    TrustPolicy,
    ValidatedIdentity,
)
from conducto.security.audit import AuditEmitter
from conducto.transport.auth import (
    MTLSPeerIdentity,
    authenticate_incoming_request,
)
from conducto.transport.errors import AuthenticationError

from .runtime import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
)


def _authorization_header(request: A2AAuthenticationRequest) -> str | None:
    """Return the authorization header without retaining its credential."""
    for name, value in request.headers.items():
        if name.lower() == "authorization":
            return value
    return None


def _roles(identity: ValidatedIdentity) -> frozenset[str]:
    """Read normalized role claims without treating malformed claims as roles."""
    value = identity.claims.get("roles", identity.claims.get("role", ()))
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, list | tuple | set | frozenset):
        return frozenset(item for item in value if isinstance(item, str))
    return frozenset()


def _principal(identity: ValidatedIdentity) -> Principal:
    """Map a validated token identity to the public principal contract."""
    return Principal(
        subject_id=identity.subject,
        issuer=identity.issuer,
        audience=identity.audience,
        claims=identity.claims,
        roles=_roles(identity),
        scopes=identity.scopes,
    )


@dataclass(frozen=True, slots=True)
class StaticTokenIdentityResolver:
    """Authenticate bearer tokens against an immutable token-to-principal map.

    Args:
        principals: Mapping of bearer tokens to authenticated principals. The
            mapping is copied and frozen during construction.

    Raises:
        AuthenticationError: If the bearer header is missing or invalid.
    """

    principals: Mapping[str, Principal] = field(repr=False)

    def __post_init__(self) -> None:
        """Copy the token map so later application mutation cannot affect auth."""
        if not self.principals:
            raise ValueError("principals must not be empty")
        if any(not token for token in self.principals):
            raise ValueError("principal tokens must not be empty")
        object.__setattr__(self, "principals", MappingProxyType(dict(self.principals)))

    def __call__(self, request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
        """Authenticate one request using constant-time token comparisons."""
        header = _authorization_header(request)
        if header is None or not header.startswith("Bearer "):
            raise AuthenticationError("a bearer authorization header is required")
        token = header[7:]
        matched: Principal | None = None
        for expected, principal in self.principals.items():
            if secrets.compare_digest(token, expected):
                matched = principal
        if matched is None:
            raise AuthenticationError("bearer authentication failed")
        return A2AAuthenticatedIdentity(
            authorization=AuthorizationContext(
                principal=matched,
                task_id=request.task_id,
                correlation_id=request.correlation_id,
            )
        )


@dataclass(frozen=True, slots=True)
class JWTBearerIdentityResolver:
    """Validate JWT bearer tokens and adapt their identity facts to A2A."""

    validator: JWTBearerTokenValidator
    policy: TrustPolicy

    def __call__(self, request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
        """Validate the request bearer token without exposing the raw token."""
        header = _authorization_header(request)
        if header is None or not header.startswith("Bearer "):
            raise AuthenticationError("a bearer authorization header is required")
        identity = self.validator.validate(header[7:], policy=self.policy)
        return A2AAuthenticatedIdentity(
            authorization=AuthorizationContext(
                principal=_principal(identity),
                task_id=request.task_id,
                correlation_id=request.correlation_id,
                policy_metadata={
                    "trust_policy_version": identity.policy_version,
                    "actor_chain": identity.actor_chain,
                },
            )
        )


@dataclass(frozen=True, slots=True)
class TransportIdentityResolver:
    """Compose the shared inbound transport authentication boundary.

    Args:
        validator: Application-owned bearer token validator.
        policy: Trust policy used for bearer and optional mTLS checks.
        mtls_identity_extractor: Optional application-owned extractor that
            returns a verified peer identity for the A2A request.
        audit: Optional audit emitter passed to the transport boundary.
    """

    validator: TokenValidator
    policy: TrustPolicy
    mtls_identity_extractor: (
        Callable[[A2AAuthenticationRequest], MTLSPeerIdentity | None] | None
    ) = field(default=None, repr=False, compare=False)
    audit: AuditEmitter | None = field(default=None, repr=False, compare=False)

    def __call__(self, request: A2AAuthenticationRequest) -> Awaitable[A2AAuthenticatedIdentity]:
        """Authenticate bearer and mTLS requirements with independent checks."""
        mtls_peer = (
            self.mtls_identity_extractor(request)
            if self.mtls_identity_extractor is not None
            else None
        )
        return _transport_identity(self, request, mtls_peer)


async def _transport_identity(
    resolver: TransportIdentityResolver,
    request: A2AAuthenticationRequest,
    mtls_peer: MTLSPeerIdentity | None,
) -> A2AAuthenticatedIdentity:
    """Await the shared transport authenticator and adapt its context."""
    authorization = await authenticate_incoming_request(
        authorization_header=_authorization_header(request),
        validator=resolver.validator,
        policy=resolver.policy,
        task_id=request.task_id,
        correlation_id=request.correlation_id,
        mtls_peer=mtls_peer,
        audit=resolver.audit,
    )
    return A2AAuthenticatedIdentity(authorization=authorization)


__all__ = [
    "JWTBearerIdentityResolver",
    "StaticTokenIdentityResolver",
    "TransportIdentityResolver",
]
