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


class ApprovalTokenError(SecurityError):
    """Base class for approval-token verification failures."""


class MalformedApprovalTokenError(ApprovalTokenError):
    """The compact JWS is malformed, oversized, or non-canonical."""

    reason_code = "malformed_token"


class UnsupportedApprovalTokenError(ApprovalTokenError):
    """The token contract version or algorithm is unsupported."""

    reason_code = "unsupported_token"


class UnknownApprovalKeyError(ApprovalTokenError):
    """The issuer/key identifier is unknown, inactive, or revoked."""

    reason_code = "unknown_key"


class InvalidApprovalSignatureError(ApprovalTokenError):
    """The JWS signature is not valid for the trusted key."""

    reason_code = "invalid_signature"


class PrematureApprovalTokenError(ApprovalTokenError):
    """The token is not yet valid."""

    reason_code = "premature_token"


class ApprovalTokenExpiredError(ApprovalTokenError):
    """The token has expired or exceeds its permitted lifetime."""

    reason_code = "token_expired"


class ApprovalTokenBindingError(ApprovalTokenError):
    """The token does not match the intended invocation."""

    reason_code = "token_binding"


class ApprovalTokenRoleError(ApprovalTokenError):
    """The token's approver role is not required by the challenge."""

    reason_code = "role_mismatch"


class ApprovalTokenReplayError(ApprovalTokenError):
    """The token identifier was already consumed."""

    reason_code = "replay"
