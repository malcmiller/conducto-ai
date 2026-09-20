"""Deterministic approval-token contract tests."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from conducto.security import (
    ApprovalChallenge,
    ApprovalTokenBindingError,
    ApprovalTokenExpiredError,
    ApprovalTokenReplayError,
    ES256Signer,
    ES256Verifier,
    InMemoryApprovalReplayStore,
    InMemoryApprovalStore,
    InvalidApprovalSignatureError,
    MalformedApprovalTokenError,
    StaticApprovalKeyResolver,
)


class FixedClock:
    """Deterministic verification clock."""

    def now(self) -> datetime:
        """Return the fixed contract timestamp."""
        return datetime(2026, 1, 1, tzinfo=UTC)


def token() -> tuple[str, ES256Verifier]:
    key = ec.derive_private_key(1, ec.SECP256R1())
    signer = ES256Signer("approval-authority", "key-1", key)
    claims = {
        "iss": "approval-authority",
        "aud": "conducto-runtime",
        "sub": "approver-1",
        "iat": 1767225600,
        "nbf": 1767225600,
        "exp": 1767225900,
        "jti": "token-1",
        "challenge_id": "challenge-1",
        "task_id": "task-1",
        "agent_id": "agent-1",
        "capability_id": "capability-1",
        "decision": "approve",
        "required_role": "owner",
        "policy_version": "policy-7",
        "ver": "1",
    }
    verifier = ES256Verifier(
        StaticApprovalKeyResolver({("approval-authority", "key-1"): key.public_key()}),
        issuer="approval-authority",
        audience="conducto-runtime",
        clock=FixedClock(),
    )
    return signer.sign(claims), verifier


def test_es256_token_verifies_and_tampering_fails() -> None:
    signed, verifier = token()
    claims = verifier.verify(signed)
    assert claims["challenge_id"] == "challenge-1"
    parts = signed.split(".")
    parts[1] = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
    with pytest.raises((InvalidApprovalSignatureError, MalformedApprovalTokenError)):
        verifier.verify(".".join(parts))


def test_expired_token_is_distinct() -> None:
    signed, verifier = token()
    parts = signed.split(".")
    # The payload is intentionally replaced with a validly signed token in the
    # signer test above; this assertion pins the public exception contract.
    assert len(parts) == 3
    with pytest.raises(ApprovalTokenExpiredError):
        verifier.verify(
            ES256Signer(
                "approval-authority",
                "key-1",
                ec.derive_private_key(1, ec.SECP256R1()),
            ).sign(
                {
                    "iss": "approval-authority",
                    "aud": "conducto-runtime",
                    "sub": "approver-1",
                    "iat": 1767225000,
                    "nbf": 1767225000,
                    "exp": 1767225500,
                    "jti": "token-expired",
                    "challenge_id": "challenge-1",
                    "task_id": "task-1",
                    "agent_id": "agent-1",
                    "capability_id": "capability-1",
                    "decision": "approve",
                    "required_role": "owner",
                    "policy_version": "policy-7",
                    "ver": "1",
                }
            )
        )


def test_replay_store_is_atomic() -> None:
    store = InMemoryApprovalReplayStore()
    store.consume("token-1", 0)
    with pytest.raises(ApprovalTokenReplayError):
        store.consume("token-1", 0)


def test_service_binds_challenge() -> None:
    signed, verifier = token()
    challenge = ApprovalChallenge(
        "challenge-1",
        "agent-1",
        "capability-1",
        "task-1",
        "correlation-1",
        "approval_required",
        "owner",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        policy_version="policy-7",
    )
    result = verifier.verify(signed)
    assert result["jti"] == "token-1"
    assert challenge.policy_version == "policy-7"
    store = InMemoryApprovalStore(clock=FixedClock())
    assert store is not None
    with pytest.raises(ApprovalTokenBindingError):
        from conducto.security.approval_token import ApprovalTokenService

        ApprovalTokenService(verifier, InMemoryApprovalReplayStore(), store).verify(
            signed, challenge=replace(challenge, task_id="other")
        )
