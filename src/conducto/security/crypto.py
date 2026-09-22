"""Standards-based ES256 approval-token signing and verification."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.hazmat.primitives.hashes import SHA256

from .errors import (
    ApprovalTokenBindingError,
    ApprovalTokenError,
    ApprovalTokenExpiredError,
    InvalidApprovalSignatureError,
    MalformedApprovalTokenError,
    PrematureApprovalTokenError,
    UnknownApprovalKeyError,
    UnsupportedApprovalTokenError,
)

TOKEN_TYPE = "conducto.approval+jwt;v1"
TOKEN_VERSION = "1"
ES256 = "ES256"
MAX_TOKEN_SIZE = 16_384
MAX_LIFETIME_SECONDS = 900
MAX_CLOCK_SKEW_SECONDS = 30


class ApprovalClock(Protocol):
    """Clock used to make token verification deterministic."""

    def now(self) -> datetime:
        """Return the current UTC-aware time."""


class ApprovalTokenSigner(Protocol):
    """Signer boundary that keeps private key custody application-owned."""

    def sign(self, claims: Mapping[str, Any]) -> str:
        """Return a compact JWS for the supplied claims."""


class ApprovalKeyResolver(Protocol):
    """Application-owned resolver for active, trusted verification keys."""

    def resolve(self, issuer: str, key_id: str) -> ec.EllipticCurvePublicKey:
        """Return an active public key or raise ``UnknownApprovalKeyError``."""


class ApprovalVerificationAuditHook(Protocol):
    """Synchronous safe observer for cryptographic verification outcomes.

    Hooks receive no token, claims, signature, key, or exception details.
    ``ApprovalTokenService`` can bridge these outcomes to ``AuditEmitter``.
    """

    def record(self, *, verified: bool, reason_code: str) -> None:
        """Record one verification outcome using a stable reason code."""


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) % 4 == 1:
        raise MalformedApprovalTokenError("malformed approval token")
    try:
        raw = value.encode("ascii")
        if any(char in raw for char in b"+/="):
            raise ValueError
        decoded = base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise MalformedApprovalTokenError("malformed approval token") from error
    if _b64(decoded) != value:
        raise MalformedApprovalTokenError("malformed approval token")
    return decoded


def _json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode()


def _object(value: bytes) -> dict[str, Any]:
    duplicates = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicates
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                duplicates = True
            result[key] = item
        return result

    try:
        parsed = json.loads(value.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MalformedApprovalTokenError("malformed approval token") from error
    if duplicates or not isinstance(parsed, dict) or _json(parsed) != value:
        raise MalformedApprovalTokenError("malformed approval token")
    return parsed


@dataclass(frozen=True, slots=True)
class ES256Signer:
    """Reference ES256 signer for the pinned compact JWS contract.

    Attributes:
        issuer: Trusted issuer placed in the token claims.
        key_id: Application-owned active signing key identifier.
        private_key: Private key retained by the application, never serialized.
    """

    issuer: str
    key_id: str
    private_key: ec.EllipticCurvePrivateKey

    def __post_init__(self) -> None:
        """Reject keys that are not P-256 private keys."""
        if not isinstance(self.private_key, ec.EllipticCurvePrivateKey) or not isinstance(
            self.private_key.curve, ec.SECP256R1
        ):
            raise ValueError("ES256 requires a P-256 private key")

    def sign(self, claims: Mapping[str, Any]) -> str:
        """Sign claims as a compact RFC 7515 JWS."""
        header = {"alg": ES256, "kid": self.key_id, "typ": TOKEN_TYPE}
        protected = _b64(_json(header))
        payload = _b64(_json(dict(claims)))
        der = self.private_key.sign(f"{protected}.{payload}".encode("ascii"), ec.ECDSA(SHA256()))
        r, s = decode_dss_signature(der)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return f"{protected}.{payload}.{_b64(signature)}"


class StaticApprovalKeyResolver:
    """Small application-owned resolver supporting active and revoked keys."""

    def __init__(
        self,
        keys: Mapping[tuple[str, str], ec.EllipticCurvePublicKey],
        *,
        revoked: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        """Initialize a resolver from trusted public keys."""
        self._keys = dict(keys)
        self._revoked = revoked

    def resolve(self, issuer: str, key_id: str) -> ec.EllipticCurvePublicKey:
        """Resolve an active P-256 key, failing closed for unknown keys."""
        key = (issuer, key_id)
        candidate = self._keys.get(key)
        if candidate is None or key in self._revoked:
            raise UnknownApprovalKeyError("approval key is not trusted")
        if not isinstance(candidate, ec.EllipticCurvePublicKey) or not isinstance(
            candidate.curve, ec.SECP256R1
        ):
            raise UnknownApprovalKeyError("approval key is not trusted")
        return candidate


class ES256Verifier:
    """Strict verifier for Conducto approval compact JWS tokens."""

    def __init__(
        self,
        resolver: ApprovalKeyResolver,
        *,
        issuer: str,
        audience: str,
        clock: ApprovalClock,
        max_lifetime_seconds: int = MAX_LIFETIME_SECONDS,
        clock_skew_seconds: int = MAX_CLOCK_SKEW_SECONDS,
        audit_hook: ApprovalVerificationAuditHook | None = None,
    ) -> None:
        """Configure a trusted issuer, audience, time bounds, and key resolver."""
        if max_lifetime_seconds <= 0 or max_lifetime_seconds > MAX_LIFETIME_SECONDS:
            raise ValueError("invalid maximum token lifetime")
        if clock_skew_seconds < 0 or clock_skew_seconds > MAX_CLOCK_SKEW_SECONDS:
            raise ValueError("invalid clock skew")
        self._resolver = resolver
        self._issuer = issuer
        self._audience = audience
        self._clock = clock
        self._max_lifetime = max_lifetime_seconds
        self._skew = clock_skew_seconds
        self._audit_hook = audit_hook

    def verify(self, token: str) -> dict[str, Any]:
        """Verify and return claims without exposing token material in failures."""
        try:
            claims = self._verify(token)
        except ApprovalTokenError as error:
            if self._audit_hook is not None:
                self._audit_hook.record(verified=False, reason_code=error.reason_code)
            raise
        if self._audit_hook is not None:
            self._audit_hook.record(verified=True, reason_code="signature_verified")
        return claims

    def _verify(self, token: str) -> dict[str, Any]:
        """Perform strict verification without exposing token material."""
        if not isinstance(token, str) or len(token) > MAX_TOKEN_SIZE:
            raise MalformedApprovalTokenError("malformed approval token")
        parts = token.split(".")
        if len(parts) != 3:
            raise MalformedApprovalTokenError("malformed approval token")
        header_bytes, payload_bytes, signature = (
            _unb64(parts[0]),
            _unb64(parts[1]),
            _unb64(parts[2]),
        )
        header = _object(header_bytes)
        if set(header) != {"alg", "kid", "typ"}:
            raise MalformedApprovalTokenError("protected header is invalid")
        if header["alg"] != ES256 or header["typ"] != TOKEN_TYPE:
            raise UnsupportedApprovalTokenError("approval token algorithm or type is unsupported")
        if not isinstance(header["kid"], str) or not header["kid"]:
            raise MalformedApprovalTokenError("protected header is invalid")
        if len(signature) != 64:
            raise MalformedApprovalTokenError("malformed approval signature")
        claims = _object(payload_bytes)
        if claims.get("iss") != self._issuer:
            raise ApprovalTokenBindingError("approval token issuer is invalid")
        key = self._resolver.resolve(self._issuer, header["kid"])
        r = int.from_bytes(signature[:32], "big")
        s = int.from_bytes(signature[32:], "big")
        try:
            key.verify(
                encode_dss_signature(r, s),
                f"{parts[0]}.{parts[1]}".encode("ascii"),
                ec.ECDSA(SHA256()),
            )
        except InvalidSignature as error:
            raise InvalidApprovalSignatureError("approval signature is invalid") from error
        self._validate_claims(claims)
        return claims

    def _validate_claims(self, claims: Mapping[str, Any]) -> None:
        required = {
            "iss",
            "aud",
            "sub",
            "iat",
            "nbf",
            "exp",
            "jti",
            "challenge_id",
            "task_id",
            "agent_id",
            "capability_id",
            "decision",
            "required_role",
            "policy_version",
            "ver",
        }
        if set(claims) != required or claims["ver"] != TOKEN_VERSION:
            raise UnsupportedApprovalTokenError("approval token claims are unsupported")
        string_claims = (
            "iss",
            "aud",
            "sub",
            "jti",
            "challenge_id",
            "task_id",
            "agent_id",
            "capability_id",
            "required_role",
        )
        if claims["aud"] != self._audience or not all(
            isinstance(claims[key], str) and claims[key] for key in string_claims
        ):
            raise ApprovalTokenBindingError("approval token binding is invalid")
        if not isinstance(claims["policy_version"], str) or not claims["policy_version"]:
            raise ApprovalTokenBindingError("approval policy version is invalid")
        if claims["decision"] not in ("approve", "deny"):
            raise ApprovalTokenBindingError("approval decision is invalid")
        if any(
            isinstance(claims[key], bool) or not isinstance(claims[key], int)
            for key in ("iat", "nbf", "exp")
        ):
            raise MalformedApprovalTokenError("approval timestamps are invalid")
        now = int(self._clock.now().astimezone(UTC).timestamp())
        if claims["nbf"] > now + self._skew:
            raise PrematureApprovalTokenError("approval token is not yet valid")
        if claims["exp"] < now - self._skew:
            raise ApprovalTokenExpiredError("approval token has expired")
        if claims["exp"] <= claims["iat"] or claims["exp"] - claims["iat"] > self._max_lifetime:
            raise ApprovalTokenExpiredError("approval token lifetime is invalid")
        if claims["nbf"] > claims["exp"]:
            raise MalformedApprovalTokenError("approval timestamps are invalid")
