"""Golden fixtures pinning Conducto's MCP result and schema projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import pytest

from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationBudgetExhausted,
    InvocationCancelled,
    InvocationDelegationFailure,
    InvocationFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTargetUnavailable,
    InvocationTimeout,
    InvocationValidationFailure,
    UnsupportedReturnValueError,
)
from conducto.mcp import (
    MCP_EXTRA,
    MCP_PROTOCOL_VERSION,
    MCP_PYTHON_SDK_PACKAGE,
    MCP_PYTHON_SDK_VERSION,
    RESULT_PROPERTY,
    TOOL_NAME_SEPARATOR,
    McpSchemaProjectionError,
    invocation_result_to_tool_outcome,
    project_input_schema,
)
from conducto.security.approval import ApprovalChallenge

pytestmark = pytest.mark.golden

FIXTURES = Path(__file__).parent / "fixtures" / "mcp"
CORRELATION_ID = "correlation-1"
MAX_BYTES = 65_536


def _challenge() -> ApprovalChallenge:
    """Return a deterministic approval challenge for mapping coverage."""
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return ApprovalChallenge(
        approval_id="approval-1",
        agent_id="Billing Agent",
        capability_id="refund",
        task_id="task-1",
        correlation_id=CORRELATION_ID,
        reason_code="policy",
        required_role="approver",
        created_at=moment,
        expires_at=moment,
    )


def _results() -> dict[str, InvocationResult]:
    """Return one instance of every public invocation result family."""
    return {
        "InvocationSuccess": InvocationSuccess(CORRELATION_ID, {"total": 12}),
        "InvocationValidationFailure": InvocationValidationFailure(
            CORRELATION_ID, ({"loc": ("units",), "msg": "secret detail"},)
        ),
        "InvocationTargetNotFound": InvocationTargetNotFound(
            CORRELATION_ID, "Billing Agent", "quote"
        ),
        "InvocationTimeout": InvocationTimeout(CORRELATION_ID, 0.5),
        "InvocationCancelled": InvocationCancelled(CORRELATION_ID),
        "InvocationFailure": InvocationFailure(
            CORRELATION_ID, "Capability execution failed", RuntimeError("secret detail")
        ),
        "InvocationFailure.UnsupportedReturnValueError": InvocationFailure(
            CORRELATION_ID,
            "unsupported",
            UnsupportedReturnValueError("secret detail"),
        ),
        "InvocationApprovalRequired": InvocationApprovalRequired(CORRELATION_ID, _challenge()),
        "InvocationAuthorizationFailure": InvocationAuthorizationFailure(
            CORRELATION_ID, "missing_scope"
        ),
        "InvocationAuditFailure": InvocationAuditFailure(CORRELATION_ID, "audit_unavailable"),
        "InvocationBindingFailure": InvocationBindingFailure(CORRELATION_ID, "forged_binding"),
        "InvocationStaleBinding": InvocationStaleBinding(CORRELATION_ID, "Billing Agent", "quote"),
        "InvocationTargetUnavailable": InvocationTargetUnavailable(
            CORRELATION_ID, "Billing Agent", "quote", "draining"
        ),
        "InvocationSchemaMismatch": InvocationSchemaMismatch(
            CORRELATION_ID, "Billing Agent", "quote"
        ),
        "InvocationBudgetExhausted": InvocationBudgetExhausted(CORRELATION_ID, "invocations"),
        "InvocationDelegationFailure": InvocationDelegationFailure(
            CORRELATION_ID, "cycle", (("Billing Agent", "quote"),)
        ),
    }


@pytest.fixture(scope="module")
def mapping_fixture() -> dict[str, Any]:
    """Return the checked-in MCP result mapping fixture."""
    return json.loads((FIXTURES / "mcp_result_mapping.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema_fixture() -> dict[str, Any]:
    """Return the checked-in MCP schema subset fixture."""
    return json.loads((FIXTURES / "mcp_schema_subset.json").read_text(encoding="utf-8"))


def test_mapping_fixture_records_pinned_mcp_profile(mapping_fixture: dict[str, Any]) -> None:
    profile = mapping_fixture["profile"]

    assert profile["package"] == MCP_PYTHON_SDK_PACKAGE
    assert profile["version"] == MCP_PYTHON_SDK_VERSION
    assert profile["extra"] == MCP_EXTRA
    assert profile["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert profile["resultProperty"] == RESULT_PROPERTY
    assert profile["toolNameSeparator"] == TOOL_NAME_SEPARATOR


def test_mapping_fixture_covers_every_public_result_family(
    mapping_fixture: dict[str, Any],
) -> None:
    covered = {entry["result"] for entry in mapping_fixture["outcomes"]}
    families = {family.__name__ for family in get_args(InvocationResult)}

    assert families.issubset(covered)
    assert covered == set(_results())


def test_every_result_family_maps_to_its_pinned_mcp_outcome(
    mapping_fixture: dict[str, Any],
) -> None:
    results = _results()

    for entry in mapping_fixture["outcomes"]:
        outcome = invocation_result_to_tool_outcome(results[entry["result"]])
        structured = (
            None if outcome.structured_content is None else dict(outcome.structured_content)
        )

        assert outcome.is_error is entry["isError"], entry["result"]
        assert outcome.reason_code == entry["reasonCode"], entry["result"]
        assert outcome.message == entry["message"], entry["result"]
        assert structured == entry["structuredContent"], entry["result"]
        assert outcome.correlation_id == mapping_fixture["correlationId"]
        assert "secret detail" not in outcome.message


def test_supported_schema_fixtures_project_to_pinned_projections(
    schema_fixture: dict[str, Any],
) -> None:
    for case in schema_fixture["supported"]:
        projected = project_input_schema(case["schema"], label=case["case"], max_bytes=MAX_BYTES)

        assert projected == case["projection"], case["case"]


def test_rejected_schema_fixtures_fail_with_typed_errors(schema_fixture: dict[str, Any]) -> None:
    for case in schema_fixture["rejected"]:
        with pytest.raises(McpSchemaProjectionError, match=case["error"]):
            project_input_schema(case["schema"], label=case["case"], max_bytes=MAX_BYTES)
