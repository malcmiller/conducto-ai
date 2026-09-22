"""Public A2A 1.0 contracts and Conducto result mapping helpers.

Importing this package does not import Starlette or construct an ASGI server;
the optional ASGI host adapter is resolved lazily and requires the
``a2a-server`` package extra.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from a2a.types.a2a_pb2 import Artifact, Message, Part, Role, Task, TaskState
from google.protobuf.struct_pb2 import Struct

from conducto.core.a2a_profile import (
    A2A_JSONRPC_BINDING,
    A2A_NORMATIVE_COMMIT,
    A2A_NORMATIVE_PROTO_SHA256,
    A2A_NORMATIVE_SOURCE,
    A2A_NORMATIVE_TAG,
    A2A_PROTOCOL_RELEASE,
    A2A_PROTOCOL_VERSION,
    A2A_PYTHON_SDK_PACKAGE,
    A2A_PYTHON_SDK_VERSION,
    CONDUCTO_PARAMETER_EXTENSION_URI,
    MAX_ARTIFACT_PARTS,
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_PARTS,
    MAX_METADATA_BYTES,
    MAX_TASK_ARTIFACTS,
    SUPPORTED_JSONRPC_METHODS,
    SUPPORTED_MEDIA_TYPES,
    SUPPORTED_REQUIRED_EXTENSIONS,
    TERMINAL_TASK_STATES,
    A2AProtocolError,
    parse_agent_card,
    parse_message,
    parse_task,
    validate_jsonrpc_method,
    validate_task_transition,
)
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

from .errors import A2ADependencyError, A2AServerError
from .factory import create_a2a_app
from .handler import A2ACancellableRequestHandler, A2ARequestContext, A2ARequestHandler
from .profile import A2A_SERVER_EXTRA, require_a2a_server_dependency
from .runtime import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2ACapabilityBinding,
    A2AIdentityResolver,
    A2ARuntimeHandler,
)

if TYPE_CHECKING:
    from .asgi import A2AASGI

_LAZY_EXPORTS = {
    "A2AASGI": "conducto.a2a.asgi",
}


def __getattr__(name: str) -> Any:
    """Resolve the Starlette-backed ASGI host adapter only when it is requested.

    Args:
        name: Attribute requested from this package.

    Returns:
        The lazily imported attribute.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
        A2ADependencyError: If the optional Starlette dependency is unavailable.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def invocation_result_to_task(result: InvocationResult, *, task_id: str, context_id: str) -> Task:
    """Map a completed local invocation to a sanitized terminal A2A task.

    Args:
        result: Outcome returned by the shared Conducto invocation pipeline.
        task_id: Server-owned A2A task identifier.
        context_id: A2A context identifier shared by follow-up work.

    Returns:
        A terminal A2A task with no local exception details.
    """
    task = Task(id=task_id, context_id=context_id)
    state = TaskState.TASK_STATE_FAILED
    if isinstance(result, InvocationSuccess):
        state = TaskState.TASK_STATE_COMPLETED
        reason = "ok"
        message = "The capability invocation completed."
        task.artifacts.append(
            Artifact(
                artifact_id=f"{task_id}-result",
                name="result",
                parts=[Part(text=json.dumps(result.value, ensure_ascii=True, sort_keys=True))],
            )
        )
    elif isinstance(result, InvocationCancelled):
        state = TaskState.TASK_STATE_CANCELED
        reason = "cancelled"
        message = "The capability invocation was cancelled."
    elif isinstance(result, InvocationTimeout):
        reason = "timeout"
        message = "The capability exceeded its execution deadline."
    elif isinstance(result, InvocationApprovalRequired):
        state = TaskState.TASK_STATE_INPUT_REQUIRED
        reason = "approval_required"
        message = "The capability requires approval before it can execute."
    elif isinstance(result, InvocationValidationFailure):
        state = TaskState.TASK_STATE_REJECTED
        reason = "invalid_arguments"
        message = "The arguments did not match the capability schema."
    elif isinstance(result, InvocationTargetNotFound):
        state = TaskState.TASK_STATE_REJECTED
        reason = "target_not_found"
        message = "The requested capability is not available."
    elif isinstance(result, InvocationAuthorizationFailure):
        state = TaskState.TASK_STATE_REJECTED
        reason = "authorization_denied"
        message = "The caller is not authorized to invoke this capability."
    elif isinstance(result, InvocationAuditFailure):
        reason = "audit_unavailable"
        message = "Mandatory audit evidence could not be recorded."
    elif isinstance(result, InvocationBindingFailure):
        state = TaskState.TASK_STATE_REJECTED
        reason = "binding_rejected"
        message = "The capability binding was rejected."
    elif isinstance(result, InvocationStaleBinding):
        state = TaskState.TASK_STATE_REJECTED
        reason = "binding_stale"
        message = "The capability binding is no longer valid."
    elif isinstance(result, InvocationTargetUnavailable):
        reason = "target_unavailable"
        message = "The capability target is not currently available."
    elif isinstance(result, InvocationSchemaMismatch):
        state = TaskState.TASK_STATE_REJECTED
        reason = "schema_mismatch"
        message = "The capability schema changed after publication."
    elif isinstance(result, InvocationBudgetExhausted):
        state = TaskState.TASK_STATE_REJECTED
        reason = "budget_exhausted"
        message = "The invocation budget is exhausted."
    elif isinstance(result, InvocationDelegationFailure):
        state = TaskState.TASK_STATE_REJECTED
        reason = "delegation_rejected"
        message = "The delegation request was rejected."
    elif isinstance(result, InvocationFailure):
        reason = "capability_failure"
        message = "The capability failed during execution."
    elif isinstance(result, InvocationInternalFailure):
        reason = "internal_error"
        message = "The capability invocation failed internally."
    else:
        reason = "internal_error"
        message = "The capability invocation failed internally."
    task.status.state = state
    task.status.message.CopyFrom(
        Message(
            message_id=f"{task_id}-status",
            role=Role.ROLE_AGENT,
            parts=[Part(text=message)],
            context_id=context_id,
            task_id=task_id,
        )
    )
    metadata: dict[str, Any] = {
        "reason": reason,
        "correlationId": result.correlation_id,
    }
    if result.metadata is not None:
        metadata["invocation"] = result.metadata.to_dict()
    if isinstance(result, InvocationApprovalRequired):
        metadata["approvalId"] = result.challenge.approval_id
    task.metadata.CopyFrom(Struct())
    task.metadata.update(metadata)
    return task


__all__ = [
    "A2A_JSONRPC_BINDING",
    "A2A_NORMATIVE_COMMIT",
    "A2A_NORMATIVE_PROTO_SHA256",
    "A2A_NORMATIVE_SOURCE",
    "A2A_NORMATIVE_TAG",
    "A2A_PROTOCOL_RELEASE",
    "A2A_PROTOCOL_VERSION",
    "A2A_PYTHON_SDK_PACKAGE",
    "A2A_PYTHON_SDK_VERSION",
    "A2A_SERVER_EXTRA",
    "A2AASGI",
    "A2AAuthenticatedIdentity",
    "A2AAuthenticationRequest",
    "A2ACancellableRequestHandler",
    "A2ACapabilityBinding",
    "A2ADependencyError",
    "A2AIdentityResolver",
    "A2AProtocolError",
    "A2ARequestContext",
    "A2ARequestHandler",
    "A2ARuntimeHandler",
    "A2AServerError",
    "CONDUCTO_PARAMETER_EXTENSION_URI",
    "create_a2a_app",
    "invocation_result_to_task",
    "require_a2a_server_dependency",
    "MAX_ARTIFACT_PARTS",
    "MAX_HISTORY_MESSAGES",
    "MAX_MESSAGE_PARTS",
    "MAX_METADATA_BYTES",
    "MAX_TASK_ARTIFACTS",
    "SUPPORTED_JSONRPC_METHODS",
    "SUPPORTED_MEDIA_TYPES",
    "SUPPORTED_REQUIRED_EXTENSIONS",
    "TERMINAL_TASK_STATES",
    "parse_agent_card",
    "parse_message",
    "parse_task",
    "validate_jsonrpc_method",
    "validate_task_transition",
]
