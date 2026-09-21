"""Deterministic trust-policy snapshot contract tests."""

import pytest

from conducto.security import (
    AudiencePolicy,
    CertificatePolicy,
    ClockSkewPolicy,
    IssuerPolicy,
    ScopePolicy,
    TrustPolicy,
)


def make_policy(**overrides: object) -> TrustPolicy:
    defaults: dict[str, object] = dict(
        version="policy-1",
        issuer=IssuerPolicy(issuer="https://issuer.example"),
        audience=AudiencePolicy(audiences=frozenset({"agent-b"})),
        scopes=ScopePolicy(allowed_scopes=frozenset({"read", "write"})),
    )
    defaults.update(overrides)
    return TrustPolicy(**defaults)  # type: ignore[arg-type]


def test_trust_policy_requires_version() -> None:
    with pytest.raises(ValueError):
        make_policy(version="")


def test_issuer_policy_requires_algorithms() -> None:
    with pytest.raises(ValueError):
        IssuerPolicy(issuer="https://issuer.example", allowed_algorithms=frozenset())


def test_audience_policy_requires_non_empty() -> None:
    with pytest.raises(ValueError):
        AudiencePolicy(audiences=frozenset())


def test_scope_policy_required_must_be_subset_of_allowed() -> None:
    with pytest.raises(ValueError):
        ScopePolicy(allowed_scopes=frozenset({"read"}), required_scopes=frozenset({"write"}))


def test_clock_skew_policy_bounds() -> None:
    with pytest.raises(ValueError):
        ClockSkewPolicy(leeway_seconds=-1)
    with pytest.raises(ValueError):
        ClockSkewPolicy(leeway_seconds=301)
    assert ClockSkewPolicy(leeway_seconds=30).leeway_seconds == 30


def test_certificate_policy_requires_trust_roots() -> None:
    with pytest.raises(ValueError):
        CertificatePolicy(trusted_ca_pem=b"")
    policy = CertificatePolicy(trusted_ca_pem=b"pem", allowed_subject_common_names=["svc-a"])
    assert policy.allowed_subject_common_names == frozenset({"svc-a"})


def test_trust_policy_snapshot_is_immutable() -> None:
    policy = make_policy()
    with pytest.raises(AttributeError):
        policy.version = "policy-2"  # type: ignore[misc]
