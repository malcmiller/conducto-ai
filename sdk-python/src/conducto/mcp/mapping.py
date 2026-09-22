"""Deterministic mapping from Conducto invocation results to MCP tool results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

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

from .schema import RESULT_PROPERTY

SUCCESS_REASON_CODE = "ok"
"""Reason code reported for a successful capability invocation."""

_FAILURE_MESSAGES: Mapping[str, str] = {
    "invalid_arguments": "The tool arguments did not match the capability schema.",
    "target_not_found": "The requested capability is not available.",
    "timeout": "The capability exceeded its execution deadline.",
    "cancelled": "The capability invocation was cancelled.",
    "capability_failure": "The capability failed during execution.",
    "unsupported_result": "The capability result cannot be represented as MCP content.",
    "approval_required": "The capability requires approval before it can execute.",
    "authorization_denied": "The caller is not authorized to invoke this capability.",
    "audit_unavailable": "Mandatory audit evidence could not be recorded.",
    "binding_rejected": "The capability binding was rejected.",
    "binding_stale": "The capability binding is no longer valid.",
    "target_unavailable": "The capability target is not currently available.",
    "schema_mismatch": "The capability schema changed after the tool was exported.",
    "budget_exhausted": "The invocation budget for this capability is exhausted.",
    "delegation_rejected": "The delegation request was rejected.",
}


@dataclass(frozen=True, slots=True)
class McpToolOutcome:
    """Transport-neutral MCP tool result projected from an invocation result.

    Attributes:
        is_error: Whether MCP reports the outcome as a non-success tool result.
        reason_code: Stable Conducto reason code for the mapped outcome.
        message: Safe, deterministic text content for MCP clients.
        correlation_id: Conducto correlation identifier for the invocation.
        structured_content: Structured content published on success only.

    Notes:
        Messages never contain Python exceptions, tracebacks, arguments,
        approval content, credentials, bindings, or internal endpoints.
    """

    is_error: bool
    reason_code: str
    message: str
    correlation_id: str
    structured_content: Mapping[str, Any] | None = None


def invocation_result_to_tool_outcome(result: InvocationResult) -> McpToolOutcome:
    """Map one public invocation result family to its MCP tool outcome.

    Args:
        result: Outcome returned by ``Runtime.invoke()``.

    Returns:
        The safe MCP tool outcome for the result family.
    """
    match result:
        case InvocationSuccess():
            payload = {RESULT_PROPERTY: result.value}
            return McpToolOutcome(
                is_error=False,
                reason_code=SUCCESS_REASON_CODE,
                message=json.dumps(payload, ensure_ascii=True, sort_keys=True),
                correlation_id=result.correlation_id,
                structured_content=payload,
            )
        case InvocationValidationFailure():
            return _failure("invalid_arguments", result.correlation_id)
        case InvocationTargetNotFound():
            return _failure("target_not_found", result.correlation_id)
        case InvocationTimeout():
            return _failure("timeout", result.correlation_id)
        case InvocationCancelled():
            return _failure("cancelled", result.correlation_id)
        case InvocationFailure():
            reason = (
                "unsupported_result"
                if isinstance(result.exception, UnsupportedReturnValueError)
                else "capability_failure"
            )
            return _failure(reason, result.correlation_id)
        case InvocationApprovalRequired():
            return _failure(
                "approval_required",
                result.correlation_id,
                detail=f"Approval reference: {result.challenge.approval_id}.",
            )
        case InvocationAuthorizationFailure():
            return _failure("authorization_denied", result.correlation_id)
        case InvocationAuditFailure():
            return _failure("audit_unavailable", result.correlation_id)
        case InvocationBindingFailure():
            return _failure("binding_rejected", result.correlation_id)
        case InvocationStaleBinding():
            return _failure("binding_stale", result.correlation_id)
        case InvocationTargetUnavailable():
            return _failure("target_unavailable", result.correlation_id)
        case InvocationSchemaMismatch():
            return _failure("schema_mismatch", result.correlation_id)
        case InvocationBudgetExhausted():
            return _failure("budget_exhausted", result.correlation_id)
        case InvocationDelegationFailure():
            return _failure("delegation_rejected", result.correlation_id)
        case _:  # pragma: no cover - exhaustiveness guard
            assert_never(result)


def _failure(reason_code: str, correlation_id: str, *, detail: str | None = None) -> McpToolOutcome:
    """Build a non-success outcome from a mapped reason code."""
    message = _FAILURE_MESSAGES[reason_code]
    return McpToolOutcome(
        is_error=True,
        reason_code=reason_code,
        message=message if detail is None else f"{message} {detail}",
        correlation_id=correlation_id,
    )


__all__ = ["SUCCESS_REASON_CODE", "McpToolOutcome", "invocation_result_to_tool_outcome"]
