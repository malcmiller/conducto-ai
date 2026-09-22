"""Focused deterministic tests for authorization and approval guardrails."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from conducto.security import (
    ApprovalDecision,
    AuthorizationContext,
    InMemoryApprovalStore,
    InvalidApprovalStateError,
    Principal,
    SecurityPipeline,
    require_approval,
    require_scope,
)
from conducto.security.errors import (
    ApprovalExpiredError,
    ApprovalRequiredError,
    PolicyEvaluationError,
)


def auth(*, scopes: frozenset[str] = frozenset()) -> AuthorizationContext:
    return AuthorizationContext(
        Principal("subject", "issuer", "audience", scopes=scopes),
        task_id="task",
        correlation_id="correlation",
    )


def test_scope_and_missing_context_fail_closed() -> None:
    @require_scope("a")
    def protected() -> str:
        return "executed"

    pipeline = SecurityPipeline()
    assert pipeline.check(protected, auth(scopes=frozenset({"A"})), {}).error is not None
    assert pipeline.check(protected, None, {}).error is not None


def test_stacked_approval_requirements_need_all_roles() -> None:
    @require_approval("reviewer")
    @require_approval("owner")
    def protected() -> str:
        return "executed"

    store = InMemoryApprovalStore()
    pipeline = SecurityPipeline(store, identifiers=lambda: "approval")
    result = pipeline.check(protected, auth(), {}, agent_id="agent", capability_id="capability")
    assert result.challenge is not None
    challenge = result.challenge
    first = ApprovalDecision("approval", True, challenge.created_at, "owner", role="owner")
    with pytest.raises(ApprovalRequiredError):
        asyncio.run(
            pipeline.resume(
                first,
                lambda: "executed",
                agent_id="agent",
                capability_id="capability",
                context=auth(),
            )
        )
    second = ApprovalDecision("approval", True, challenge.created_at, "reviewer", role="reviewer")
    assert (
        asyncio.run(
            pipeline.resume(
                second,
                lambda: "executed",
                agent_id="agent",
                capability_id="capability",
                context=auth(),
            )
        )
        == "executed"
    )


def test_unknown_and_expired_approval_ids_are_typed() -> None:
    store = InMemoryApprovalStore()
    with pytest.raises(InvalidApprovalStateError):
        store.get("missing")

    now = datetime.now(UTC)
    from conducto.security.approval import ApprovalChallenge

    store.create(
        ApprovalChallenge(
            "expired",
            "agent",
            "capability",
            "task",
            "correlation",
            "approval_required",
            "owner",
            now - timedelta(seconds=2),
            now - timedelta(seconds=1),
        )
    )
    with pytest.raises(ApprovalExpiredError):
        store.get("expired")


def test_policy_callback_failure_is_explicit() -> None:
    def failing_policy(_context: AuthorizationContext, _arguments: object) -> bool:
        raise RuntimeError("policy failed")

    @require_approval("owner", condition=failing_policy)
    def protected() -> None:
        pass

    result = SecurityPipeline().check(protected, auth(), {})
    assert isinstance(result.error, PolicyEvaluationError)
