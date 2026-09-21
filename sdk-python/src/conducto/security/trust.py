"""Immutable trust-policy snapshots for OAuth and mTLS transport security.

These snapshots are provider-neutral configuration facts. They never hold live
provider clients, credentials, or mutable global state; a policy is bound once
per run so configuration refresh cannot mutate in-flight authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_ALGORITHMS = frozenset({"RS256", "ES256"})
_MAX_CLOCK_SKEW_SECONDS = 300


@dataclass(frozen=True, slots=True)
class IssuerPolicy:
    """Trusted token issuer and the signature algorithms it may use.

    Attributes:
        issuer: Exact trusted issuer (``iss``) value.
        allowed_algorithms: Non-empty set of algorithms accepted for this issuer.
            Never derived from the token being validated.
    """

    issuer: str
    allowed_algorithms: frozenset[str] = _DEFAULT_ALGORITHMS

    def __post_init__(self) -> None:
        """Validate the issuer and freeze the allowed-algorithm set."""
        if not self.issuer:
            raise ValueError("issuer is required")
        algorithms = frozenset(self.allowed_algorithms)
        if not algorithms or any(not algorithm for algorithm in algorithms):
            raise ValueError("allowed_algorithms must be a non-empty set of algorithm names")
        object.__setattr__(self, "allowed_algorithms", algorithms)


@dataclass(frozen=True, slots=True)
class AudiencePolicy:
    """Destination audiences a validated token must be bound to.

    Attributes:
        audiences: Non-empty set of accepted audience values.
    """

    audiences: frozenset[str]

    def __post_init__(self) -> None:
        """Validate and freeze the audience set."""
        audiences = frozenset(self.audiences)
        if not audiences or any(not audience for audience in audiences):
            raise ValueError("audiences must be a non-empty set of audience values")
        object.__setattr__(self, "audiences", audiences)


@dataclass(frozen=True, slots=True)
class ScopePolicy:
    """Destination scope policy used to attenuate delegated authority.

    Attributes:
        allowed_scopes: Scopes the destination is willing to accept at all.
        required_scopes: Scopes that must be present in every validated token.
    """

    allowed_scopes: frozenset[str]
    required_scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Freeze scope sets and validate required scopes are permitted."""
        allowed = frozenset(self.allowed_scopes)
        required = frozenset(self.required_scopes)
        if not required.issubset(allowed):
            raise ValueError("required_scopes must be a subset of allowed_scopes")
        object.__setattr__(self, "allowed_scopes", allowed)
        object.__setattr__(self, "required_scopes", required)


@dataclass(frozen=True, slots=True)
class ClockSkewPolicy:
    """Bounded clock-skew leeway applied to token timestamp validation.

    Attributes:
        leeway_seconds: Non-negative, bounded leeway applied to ``nbf``/``exp``.
    """

    leeway_seconds: int = 30

    def __post_init__(self) -> None:
        """Reject unbounded or negative clock-skew configuration."""
        if not (0 <= self.leeway_seconds <= _MAX_CLOCK_SKEW_SECONDS):
            raise ValueError("leeway_seconds must be between 0 and 300")


@dataclass(frozen=True, slots=True)
class CertificatePolicy:
    """Application-owned mTLS trust configuration for the receiving boundary.

    Attributes:
        trusted_ca_pem: PEM-encoded trust roots used to validate peer chains.
        require_client_certificate: Whether workload mTLS authentication is
            mandatory in addition to bearer-token validation.
        allowed_subject_common_names: Optional allow-list of peer certificate
            subject common names; empty means any chain-valid peer is accepted.
    """

    trusted_ca_pem: bytes
    require_client_certificate: bool = True
    allowed_subject_common_names: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate trust roots are present and freeze the allow-list."""
        if not self.trusted_ca_pem:
            raise ValueError("trusted_ca_pem is required")
        object.__setattr__(
            self, "allowed_subject_common_names", frozenset(self.allowed_subject_common_names)
        )


@dataclass(frozen=True, slots=True)
class TrustPolicy:
    """Immutable, versioned trust-policy snapshot bound to one run.

    Attributes:
        version: Opaque trust-policy version bound to every validated identity
            produced under this snapshot; used to isolate token caches across
            configuration changes.
        issuer: Trusted issuer and algorithm policy.
        audience: Accepted destination audiences.
        scopes: Destination scope policy.
        clock_skew: Bounded timestamp-validation leeway.
        certificate: mTLS trust configuration, when transport mTLS is enforced.
    """

    version: str
    issuer: IssuerPolicy
    audience: AudiencePolicy
    scopes: ScopePolicy
    clock_skew: ClockSkewPolicy = field(default_factory=ClockSkewPolicy)
    certificate: CertificatePolicy | None = None

    def __post_init__(self) -> None:
        """Validate the policy version is a non-empty opaque identifier."""
        if not self.version:
            raise ValueError("version is required")
