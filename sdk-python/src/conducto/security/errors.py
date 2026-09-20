"""Stable fail-closed authorization and approval errors."""

from __future__ import annotations


class SecurityError(Exception):
    """Base class for security pipeline failures."""

    reason_code = "security_error"


class MissingAuthorizationContextError(SecurityError):
    """No complete authenticated execution context was supplied."""

    reason_code = "missing_context"


class InsufficientScopeError(SecurityError):
    """The principal lacks one or more exact required scopes."""

    reason_code = "insufficient_scope"


class ApprovalRequiredError(SecurityError):
    """Execution requires a pending human approval."""

    reason_code = "approval_required"


class AuthorizationDeniedError(SecurityError):
    """Authorization or approval was denied."""

    reason_code = "denied"


class ApprovalExpiredError(SecurityError):
    """An approval challenge expired."""

    reason_code = "expired"


class TaskCanceledError(SecurityError):
    """The bound task was canceled."""

    reason_code = "canceled"


class InvalidApprovalStateError(SecurityError):
    """An approval lifecycle transition is invalid or stale."""

    reason_code = "invalid_state_transition"


class PolicyEvaluationError(SecurityError):
    """An approval condition failed and execution was denied."""

    reason_code = "policy_failure"
