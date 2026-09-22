"""Token acquisition/exchange and bearer-token validation contracts.

This module defines narrow, provider-neutral protocols for OAuth 2.0 Token
Exchange (RFC 8693) and bearer-token validation, an immutable validated
identity, typed failures, and a bounded cache with refresh and anti-stampede
behavior. Provider-specific flows (Azure Identity, MSAL, Authlib-backed OIDC
discovery, and similar) are expected to implement :class:`TokenAcquirer` and
:class:`TokenValidator` behind these same contracts; this module never
imports a specific provider SDK.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256

from .trust import TrustPolicy

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
MAX_TOKEN_SIZE = 16_384
MAX_HEADER_SIZE = 4_096
MAX_ACTOR_CHAIN_DEPTH = 16  # bounded to prevent unbounded delegation-chain parsing
_JWS_ALGORITHMS = frozenset({"RS256", "ES256"})


class TokenError(Exception):
    """Base class for token acquisition and validation failures."""

    reason_code = "token_error"


class TokenAcquisitionError(TokenError):
    """The token-exchange or acquisition call failed before a token was issued."""

    reason_code = "acquisition_failed"


class TokenValidationError(TokenError):
    """Base class for bearer-token validation failures."""

    reason_code = "validation_failed"


class MalformedTokenError(TokenValidationError):
    """The token is not a well-formed compact JWS or exceeds size limits."""

    reason_code = "malformed_token"


class UnsupportedAlgorithmError(TokenValidationError):
    """The token algorithm is not in the trust policy's allowed set."""

    reason_code = "unsupported_algorithm"


class UnknownKeyError(TokenValidationError):
    """The issuer/key identifier is not a currently trusted verification key."""

    reason_code = "unknown_key"


class InvalidSignatureTokenError(TokenValidationError):
    """The token signature failed verification against the trusted key."""

    reason_code = "invalid_signature"


class InvalidIssuerError(TokenValidationError):
    """The token issuer does not match the configured trust policy."""

    reason_code = "invalid_issuer"


class InvalidAudienceError(TokenValidationError):
    """The token audience does not match the destination trust policy."""

    reason_code = "invalid_audience"


class TokenNotYetValidError(TokenValidationError):
    """The token is not valid until a future time outside configured skew."""

    reason_code = "not_yet_valid"


class TokenExpiredError(TokenValidationError):
    """The token has expired outside configured clock-skew leeway."""

    reason_code = "token_expired"


class InvalidScopeError(TokenValidationError):
    """The token is missing one or more scopes required by trust policy."""

    reason_code = "invalid_scope"


class InvalidActorChainError(TokenValidationError):
    """The token actor/delegation chain is malformed or untrusted."""

    reason_code = "invalid_actor_chain"


class ScopeAttenuationError(TokenError):
    """A nested call requested authority broader than its caller or policy."""

    reason_code = "scope_attenuation_denied"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64url(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise MalformedTokenError("malformed token segment")
    try:
        return base64.urlsafe_b64decode(value.encode("ascii") + b"=" * (-len(value) % 4))
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise MalformedTokenError("malformed token segment") from error


@dataclass(frozen=True, slots=True)
class ValidatedIdentity:
    """Immutable identity produced by validating a bearer token.

    Credentials and the raw token are deliberately excluded; only the facts
    required by Story 2 guardrails and audit events are retained.

    Attributes:
        subject: Validated ``sub`` claim.
        issuer: Validated ``iss`` claim.
        audience: Exact audience(s) the token was bound to.
        scopes: Exact granted scopes.
        actor_chain: Ordered delegation chain from original caller to the
            immediate actor, excluding the current subject.
        expires_at: Token expiry as epoch seconds.
        not_before: Token not-before as epoch seconds.
        policy_version: Trust-policy version snapshot used for validation.
        claims: Additional non-sensitive normalized claims.
    """

    subject: str
    issuer: str
    audience: tuple[str, ...]
    scopes: frozenset[str]
    actor_chain: tuple[str, ...]
    expires_at: int
    not_before: int
    policy_version: str
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate required identity facts and freeze mutable collections."""
        if not self.subject or not self.issuer or not self.policy_version:
            raise ValueError("subject, issuer, and policy_version are required")
        if not self.audience:
            raise ValueError("audience is required")
        object.__setattr__(self, "audience", tuple(self.audience))
        object.__setattr__(self, "scopes", frozenset(self.scopes))
        object.__setattr__(self, "actor_chain", tuple(self.actor_chain))
        object.__setattr__(self, "claims", dict(self.claims))


@dataclass(frozen=True, slots=True)
class TokenExchangeRequest:
    """RFC 8693 token-exchange request, provider-neutral.

    Attributes:
        subject_token: Token representing the identity to exchange for.
        subject_token_type: RFC 8693 subject token type identifier.
        audience: Destination audience the exchanged token must be bound to.
        scope: Requested least-privilege scopes.
        actor_token: Optional token representing the acting workload identity.
        actor_token_type: RFC 8693 actor token type identifier, required when
            ``actor_token`` is supplied.
        resource: Optional destination resource identifier.
    """

    subject_token: str
    subject_token_type: str
    audience: str
    scope: frozenset[str] = frozenset()
    actor_token: str | None = None
    actor_token_type: str | None = None
    resource: str | None = None
    grant_type: str = TOKEN_EXCHANGE_GRANT_TYPE

    def __post_init__(self) -> None:
        """Validate required fields and the actor-token/type pairing."""
        if not self.subject_token or not self.subject_token_type or not self.audience:
            raise ValueError("subject_token, subject_token_type, and audience are required")
        if bool(self.actor_token) != bool(self.actor_token_type):
            raise ValueError("actor_token and actor_token_type must be supplied together")
        object.__setattr__(self, "scope", frozenset(self.scope))


@dataclass(frozen=True, slots=True)
class AcquiredToken:
    """Result of a successful token acquisition or exchange.

    Attributes:
        access_token: Opaque bearer token; excluded from ``repr``.
        issued_token_type: RFC 8693 issued token type identifier.
        expires_at: Expiry as epoch seconds, used for cache and refresh timing.
        scope: Scopes actually granted, which may be a subset of what was requested.
    """

    access_token: str = field(repr=False)
    issued_token_type: str
    expires_at: int
    scope: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate the token and expiry, and freeze the granted scope set."""
        if not self.access_token or not self.issued_token_type:
            raise ValueError("access_token and issued_token_type are required")
        object.__setattr__(self, "scope", frozenset(self.scope))


class TokenAcquirer(Protocol):
    """Provider-neutral token acquisition/exchange boundary."""

    async def acquire(self, request: TokenExchangeRequest) -> AcquiredToken:
        """Acquire or exchange a bearer token, raising ``TokenAcquisitionError``."""


class TokenValidator(Protocol):
    """Provider-neutral bearer-token validation boundary."""

    def validate(self, token: str, *, policy: TrustPolicy) -> ValidatedIdentity:
        """Validate a compact bearer token against a trust-policy snapshot."""


class JWKSKeyResolver(Protocol):
    """Application-owned resolver for currently trusted verification keys."""

    def resolve(self, issuer: str, key_id: str) -> rsa.RSAPublicKey | ec.EllipticCurvePublicKey:
        """Return a trusted public key or raise ``UnknownKeyError``."""


class StaticJWKSResolver:
    """Deterministic in-memory resolver for local fixtures and tests."""

    def __init__(
        self,
        keys: Mapping[tuple[str, str], rsa.RSAPublicKey | ec.EllipticCurvePublicKey],
        *,
        revoked: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        """Initialize the resolver from a trusted issuer/key-id key map."""
        self._keys = dict(keys)
        self._revoked = frozenset(revoked)

    def resolve(self, issuer: str, key_id: str) -> rsa.RSAPublicKey | ec.EllipticCurvePublicKey:
        """Resolve an active key, failing closed for unknown or revoked keys."""
        key = (issuer, key_id)
        candidate = self._keys.get(key)
        if candidate is None or key in self._revoked:
            raise UnknownKeyError("token verification key is not trusted")
        return candidate


class Clock(Protocol):
    """Clock used to make token verification deterministic in tests."""

    def now(self) -> float:
        """Return the current time as epoch seconds."""


class _SystemClock:
    def now(self) -> float:
        return time.time()


class JWTBearerTokenValidator:
    """Strict RFC 7519/7515 bearer-token validator using explicit trust policy.

    Algorithms, issuers, audiences, and keys are always resolved from the
    supplied trust policy and resolver, never from the token under validation.
    """

    def __init__(self, resolver: JWKSKeyResolver, *, clock: Clock | None = None) -> None:
        """Configure the trusted key resolver and optional deterministic clock."""
        self._resolver = resolver
        self._clock = clock or _SystemClock()

    def validate(self, token: str, *, policy: TrustPolicy) -> ValidatedIdentity:
        """Validate a compact JWS bearer token against ``policy``."""
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_SIZE:
            raise MalformedTokenError("token is malformed or exceeds size limits")
        parts = token.split(".")
        if len(parts) != 3:
            raise MalformedTokenError("token must be a compact JWS")
        header_b, payload_b, signature_b = parts
        if len(header_b) > MAX_HEADER_SIZE:
            raise MalformedTokenError("token header exceeds size limits")
        header = self._decode_json(header_b)
        payload = self._decode_json(payload_b)
        signature = _unb64url(signature_b)

        algorithm = header.get("alg")
        key_id = header.get("kid")
        if not isinstance(algorithm, str) or not isinstance(key_id, str) or not key_id:
            raise MalformedTokenError("token header is missing alg or kid")
        if algorithm not in _JWS_ALGORITHMS:
            raise UnsupportedAlgorithmError("token algorithm is not supported")
        if algorithm not in policy.issuer.allowed_algorithms:
            raise UnsupportedAlgorithmError("token algorithm is not permitted by trust policy")

        issuer = payload.get("iss")
        if issuer != policy.issuer.issuer:
            raise InvalidIssuerError("token issuer is not trusted")

        key = self._resolver.resolve(issuer, key_id)
        signing_input = f"{header_b}.{payload_b}".encode("ascii")
        self._verify_signature(algorithm, key, signing_input, signature)

        return self._validate_claims(payload, policy)

    def _decode_json(self, segment: str) -> dict[str, Any]:
        raw = _unb64url(segment)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MalformedTokenError("token segment is not valid JSON") from error
        if not isinstance(value, dict):
            raise MalformedTokenError("token segment must be a JSON object")
        return value

    def _verify_signature(
        self,
        algorithm: str,
        key: rsa.RSAPublicKey | ec.EllipticCurvePublicKey,
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        try:
            if algorithm == "RS256":
                if not isinstance(key, rsa.RSAPublicKey):
                    raise UnknownKeyError("resolved key type does not match token algorithm")
                key.verify(signature, signing_input, padding.PKCS1v15(), SHA256())
            else:
                if not isinstance(key, ec.EllipticCurvePublicKey):
                    raise UnknownKeyError("resolved key type does not match token algorithm")
                if len(signature) != 64:
                    raise MalformedTokenError("malformed ES256 signature")
                r = int.from_bytes(signature[:32], "big")
                s = int.from_bytes(signature[32:], "big")
                key.verify(encode_dss_signature(r, s), signing_input, ec.ECDSA(SHA256()))
        except InvalidSignature as error:
            raise InvalidSignatureTokenError("token signature is invalid") from error

    def _validate_claims(self, claims: Mapping[str, Any], policy: TrustPolicy) -> ValidatedIdentity:
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise MalformedTokenError("token subject is missing")

        audience_claim = claims.get("aud")
        if isinstance(audience_claim, str):
            audiences: tuple[str, ...] = (audience_claim,)
        elif isinstance(audience_claim, list | tuple) and all(
            isinstance(item, str) for item in audience_claim
        ):
            audiences = tuple(audience_claim)
        else:
            raise MalformedTokenError("token audience claim is invalid")
        if not audiences or not policy.audience.audiences.intersection(audiences):
            raise InvalidAudienceError("token audience does not match trust policy")

        for field_name in ("iat", "nbf", "exp"):
            value = claims.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise MalformedTokenError("token timestamps are invalid")

        now = self._clock.now()
        skew = policy.clock_skew.leeway_seconds
        not_before = int(claims["nbf"])
        expires_at = int(claims["exp"])
        if not_before > now + skew:
            raise TokenNotYetValidError("token is not yet valid")
        if expires_at < now - skew:
            raise TokenExpiredError("token has expired")
        if expires_at <= not_before:
            raise MalformedTokenError("token lifetime is invalid")

        scope_claim = claims.get("scope", "")
        scopes = frozenset(scope_claim.split()) if isinstance(scope_claim, str) else frozenset()
        if not policy.scopes.required_scopes.issubset(scopes):
            raise InvalidScopeError("token is missing required scopes")
        if not scopes.issubset(policy.scopes.allowed_scopes):
            raise InvalidScopeError("token grants scopes outside destination policy")

        actor_chain = self._read_actor_chain(claims)

        return ValidatedIdentity(
            subject=subject,
            issuer=str(claims["iss"]),
            audience=audiences,
            scopes=scopes,
            actor_chain=actor_chain,
            expires_at=expires_at,
            not_before=not_before,
            policy_version=policy.version,
            claims={
                k: v
                for k, v in claims.items()
                if k not in {"iss", "aud", "sub", "iat", "nbf", "exp", "scope", "act"}
            },
        )

    def _read_actor_chain(self, claims: Mapping[str, Any]) -> tuple[str, ...]:
        chain: list[str] = []
        actor = claims.get("act")
        seen = 0
        while actor is not None:
            seen += 1
            if seen > MAX_ACTOR_CHAIN_DEPTH or not isinstance(actor, Mapping):
                raise InvalidActorChainError("token actor chain is malformed")
            sub = actor.get("sub")
            if not isinstance(sub, str) or not sub:
                raise InvalidActorChainError("token actor chain is malformed")
            chain.append(sub)
            actor = actor.get("act")
        return tuple(chain)


def attenuate_scopes(
    *,
    incoming_scopes: frozenset[str],
    requested_scopes: frozenset[str],
    destination_allowed_scopes: frozenset[str],
) -> frozenset[str]:
    """Return the least-privilege scope set for a nested delegated call.

    The result is always a subset of both the caller's incoming authority and
    the destination's allowed scopes; requesting anything broader is denied.

    Raises:
        ScopeAttenuationError: If ``requested_scopes`` is not a subset of the
            incoming authority and destination policy.
    """
    if not requested_scopes.issubset(incoming_scopes):
        raise ScopeAttenuationError("requested scopes exceed incoming delegated authority")
    if not requested_scopes.issubset(destination_allowed_scopes):
        raise ScopeAttenuationError("requested scopes exceed destination trust policy")
    return frozenset(requested_scopes)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    token: AcquiredToken
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)


class TokenCache:
    """Bounded, expiry-aware cache keyed by all security-relevant inputs.

    Concurrent callers sharing one cache key await a single in-flight
    acquisition instead of triggering a refresh stampede. Expired entries are
    never served; callers always observe a fresh acquisition instead.
    """

    def __init__(self, *, max_entries: int = 1024, refresh_skew_seconds: int = 30) -> None:
        """Configure the maximum cache size and pre-expiry refresh skew."""
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if refresh_skew_seconds < 0:
            raise ValueError("refresh_skew_seconds must not be negative")
        self._max_entries = max_entries
        self._refresh_skew = refresh_skew_seconds
        self._entries: OrderedDict[tuple[Any, ...], _CacheEntry] = OrderedDict()
        self._guard = asyncio.Lock()

    @staticmethod
    def make_key(
        *,
        issuer: str,
        client_id: str,
        subject: str,
        actor_chain: tuple[str, ...],
        audience: str,
        scopes: frozenset[str],
        policy_version: str,
    ) -> tuple[Any, ...]:
        """Build a cache key from every security-relevant acquisition input."""
        return (
            issuer,
            client_id,
            subject,
            actor_chain,
            audience,
            tuple(sorted(scopes)),
            policy_version,
        )

    async def get_or_acquire(
        self,
        key: tuple[Any, ...],
        acquire: Callable[[], Awaitable[AcquiredToken]],
        *,
        clock: Clock | None = None,
    ) -> AcquiredToken:
        """Return a cached, unexpired token or perform one bounded acquisition.

        Args:
            key: Cache key produced by :meth:`make_key`.
            acquire: An async, zero-argument callable performing acquisition.
            clock: Optional deterministic clock for expiry evaluation.
        """
        now = (clock or _SystemClock()).now()
        async with self._guard:
            entry = self._entries.get(key)
            if entry is not None and entry.token.expires_at - self._refresh_skew > now:
                self._entries.move_to_end(key)
                return entry.token
            if entry is None:
                entry = _CacheEntry(token=_EXPIRED_PLACEHOLDER)
                self._entries[key] = entry
            lock = entry.lock
        async with lock:
            current = self._entries.get(key)
            now = (clock or _SystemClock()).now()
            if (
                current is not None
                and current.token is not _EXPIRED_PLACEHOLDER
                and current.token.expires_at - self._refresh_skew > now
            ):
                return current.token
            token = await acquire()
            async with self._guard:
                self._entries[key] = _CacheEntry(token=token)
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            return token


_EXPIRED_PLACEHOLDER = AcquiredToken(
    access_token="placeholder", issued_token_type=ACCESS_TOKEN_TYPE, expires_at=0
)
