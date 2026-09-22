"""Golden fixtures pinning A2A task projection for every invocation result."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from a2a.types.a2a_pb2 import Role, TaskState

from conducto.a2a import A2A_PROTOCOL_VERSION, invocation_result_to_task
from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationBudgetExhausted,
    InvocationCancelled,
    InvocationDelegationFailure,
    InvocationFailure,
    InvocationInternalFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTargetUnavailable,
    InvocationTimeout,
    InvocationValidationFailure,
)
from conducto.security import ApprovalChallenge

pytestmark = pytest.mark.golden

FIXTURE = Path(__file__).parent / "fixtures" / "a2a" / "a2a_result_mapping.json"
CORRELATION_ID = "correlation-1"


def _results() -> dict[str, InvocationResult]:
    """Return one sanitized example of every public invocation result family."""
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "InvocationSuccess": InvocationSuccess(CORRELATION_ID, {"total": 12}),
        "InvocationValidationFailure": InvocationValidationFailure(
            CORRELATION_ID,
            ({"loc": ("units",), "msg": "sensitive detail"},),
        ),
        "InvocationTargetNotFound": InvocationTargetNotFound(
            CORRELATION_ID,
            "Billing Agent",
            "quote",
        ),
        "InvocationTimeout": InvocationTimeout(CORRELATION_ID, 0.5),
        "InvocationCancelled": InvocationCancelled(CORRELATION_ID),
        "InvocationFailure": InvocationFailure(
            CORRELATION_ID,
            "Capability execution failed",
            RuntimeError("sensitive detail"),
        ),
        "InvocationInternalFailure": InvocationInternalFailure(
            CORRELATION_ID,
            "internal_error",
            RuntimeError("sensitive detail"),
        ),
        "InvocationApprovalRequired": InvocationApprovalRequired(
            CORRELATION_ID,
            ApprovalChallenge(
                "approval-1",
                "Billing Agent",
                "quote",
                "task-1",
                CORRELATION_ID,
                "approval_required",
                "reviewer",
                moment,
                moment,
            ),
        ),
        "InvocationAuthorizationFailure": InvocationAuthorizationFailure(
            CORRELATION_ID,
            "missing_scope",
        ),
        "InvocationAuditFailure": InvocationAuditFailure(
            CORRELATION_ID,
            "audit_unavailable",
        ),
        "InvocationBindingFailure": InvocationBindingFailure(
            CORRELATION_ID,
            "invalid_binding",
        ),
        "InvocationStaleBinding": InvocationStaleBinding(
            CORRELATION_ID,
            "Billing Agent",
            "quote",
        ),
        "InvocationTargetUnavailable": InvocationTargetUnavailable(
            CORRELATION_ID,
            "Billing Agent",
            "quote",
            "draining",
        ),
        "InvocationSchemaMismatch": InvocationSchemaMismatch(
            CORRELATION_ID,
            "Billing Agent",
            "quote",
        ),
        "InvocationBudgetExhausted": InvocationBudgetExhausted(
            CORRELATION_ID,
            "calls",
        ),
        "InvocationDelegationFailure": InvocationDelegationFailure(
            CORRELATION_ID,
            "cycle_detected",
            (("Billing Agent", "quote"),),
        ),
    }


@pytest.fixture(scope="module")
def mapping_fixture() -> dict[str, Any]:
    """Return the checked-in A2A result mapping fixture."""
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_mapping_fixture_covers_every_invocation_result(
    mapping_fixture: dict[str, Any],
) -> None:
    """The fixture remains exhaustive when the public result union changes."""
    families = {family.__name__ for family in get_args(InvocationResult)}
    covered = {entry["result"] for entry in mapping_fixture["outcomes"]}

    assert mapping_fixture["profile"]["protocolVersion"] == A2A_PROTOCOL_VERSION
    assert families == covered == set(_results())


def test_every_result_maps_to_its_pinned_sanitized_task(
    mapping_fixture: dict[str, Any],
) -> None:
    """Task status, reason, message, and artifact behavior are deterministic."""
    results = _results()

    for entry in mapping_fixture["outcomes"]:
        task = invocation_result_to_task(
            results[entry["result"]],
            task_id="task-1",
            context_id="context-1",
        )

        assert TaskState.Name(task.status.state) == entry["state"], entry["result"]
        assert task.metadata["reason"] == entry["reason"], entry["result"]
        assert task.metadata["correlationId"] == CORRELATION_ID
        assert task.status.message.role == Role.ROLE_AGENT
        assert task.status.message.parts[0].text == entry["message"]
        assert bool(task.artifacts) is entry["hasArtifact"]
        assert "sensitive detail" not in str(task)
        if task.artifacts:
            assert json.loads(task.artifacts[0].parts[0].text) == {"total": 12}
