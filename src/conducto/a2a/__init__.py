"""Public A2A 1.0 contracts and Conducto result mapping helpers.

Importing this package does not import Starlette or construct an ASGI server;
the optional ASGI host adapter is resolved lazily and requires the
``a2a-server`` package extra.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from a2a.types.a2a_pb2 import Artifact, Part, Task, TaskState
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
    InvocationCancelled,
    InvocationFailure,
    InvocationResult,
    InvocationSuccess,
    InvocationTimeout,
)

from .errors import A2ADependencyError, A2AServerError
from .handler import A2ARequestHandler
from .profile import A2A_SERVER_EXTRA, require_a2a_server_dependency

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
    metadata: dict[str, str] = {}
    if isinstance(result, InvocationSuccess):
        state = TaskState.TASK_STATE_COMPLETED
        task.artifacts.append(
            Artifact(
                artifact_id=f"{task_id}-result",
                name="result",
                parts=[Part(text=json.dumps(result.value, ensure_ascii=True, sort_keys=True))],
            )
        )
    elif isinstance(result, InvocationCancelled):
        state = TaskState.TASK_STATE_CANCELED
    elif isinstance(result, InvocationTimeout):
        metadata["reason"] = "timeout"
    elif isinstance(result, InvocationApprovalRequired):
        state = TaskState.TASK_STATE_INPUT_REQUIRED
        metadata["reason"] = "approval_required"
    elif isinstance(result, InvocationFailure):
        metadata["reason"] = "capability_failure"
    else:
        metadata["reason"] = type(result).__name__
    task.status.state = state
    if metadata:
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
    "A2ADependencyError",
    "A2AProtocolError",
    "A2ARequestHandler",
    "A2AServerError",
    "CONDUCTO_PARAMETER_EXTENSION_URI",
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
