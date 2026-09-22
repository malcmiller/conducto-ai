"""Map typed child invocation outcomes into bounded, credential-free model results."""

from ..invocation_results import (
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
)
from ._fallback import ToolResultStatus
from ._models import ToolResultEnvelope


def _to_tool_result(call_id: str, result: InvocationResult) -> ToolResultEnvelope:
    if isinstance(result, InvocationSuccess):
        return ToolResultEnvelope(call_id, ToolResultStatus.SUCCESS, data=result.value)
    if isinstance(result, InvocationValidationFailure):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.INVALID_ARGUMENTS,
            data={"error_count": len(result.errors)},
        )
    if isinstance(result, InvocationApprovalRequired):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.APPROVAL_REQUIRED,
            reason_code="approval_required",
        )
    if isinstance(result, (InvocationAuthorizationFailure, InvocationAuditFailure)):
        return ToolResultEnvelope(call_id, ToolResultStatus.DENIED, reason_code=result.reason_code)
    if isinstance(result, InvocationStaleBinding):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.STALE_TARGET,
            reason_code="stale_binding",
        )
    if isinstance(
        result,
        (
            InvocationTargetNotFound,
            InvocationTargetUnavailable,
            InvocationSchemaMismatch,
            InvocationBindingFailure,
        ),
    ):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.UNAVAILABLE,
            reason_code=type(result).__name__,
        )
    if isinstance(result, InvocationTimeout):
        return ToolResultEnvelope(call_id, ToolResultStatus.TIMEOUT, reason_code="timeout")
    if isinstance(result, InvocationCancelled):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.CANCELLATION,
            reason_code="cancelled",
        )
    if isinstance(result, InvocationBudgetExhausted):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.BUDGET_REJECTED,
            reason_code=result.budget,
        )
    if isinstance(result, InvocationDelegationFailure):
        status = (
            ToolResultStatus.CYCLE_REJECTED
            if result.reason_code == "cycle_detected"
            else ToolResultStatus.DEPTH_REJECTED
        )
        return ToolResultEnvelope(call_id, status, reason_code=result.reason_code)
    if isinstance(result, InvocationFailure):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.EXECUTION_FAILURE,
            reason_code="execution_failure",
        )
    raise TypeError(f"Unsupported invocation result: {type(result).__name__}")
