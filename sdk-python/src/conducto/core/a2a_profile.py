"""Pinned Conducto A2A 1.0 protocol profile and validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from a2a.types.a2a_pb2 import AgentCard, Message, Task
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError
from google.protobuf.message import Message as ProtobufMessage

A2A_PROTOCOL_VERSION = "1.0"
A2A_PROTOCOL_RELEASE = "1.0.0"
A2A_NORMATIVE_SOURCE = "https://github.com/a2aproject/A2A/blob/v1.0.0/specification/a2a.proto"
A2A_NORMATIVE_TAG = "v1.0.0"
A2A_NORMATIVE_COMMIT = "173695755607e884aa9acf8ce4feed90e32727a1"
A2A_NORMATIVE_PROTO_SHA256 = "4b74c0baa923ae0acb55474e548f1d6e5d3f83b80d757b65f8bf3e99a3c2257f"
A2A_PYTHON_SDK_PACKAGE = "a2a-sdk"
A2A_PYTHON_SDK_VERSION = "1.0.3"
A2A_JSONRPC_BINDING = "JSONRPC"
CONDUCTO_PARAMETER_EXTENSION_URI = "https://conducto.ai/a2a/extensions/parameters/v1"

SUPPORTED_MEDIA_TYPES = frozenset({"text/plain"})
SUPPORTED_REQUIRED_EXTENSIONS = frozenset({CONDUCTO_PARAMETER_EXTENSION_URI})
SUPPORTED_JSONRPC_METHODS = frozenset({"message/send", "tasks/get", "tasks/list", "tasks/cancel"})
MAX_MESSAGE_PARTS = 16
MAX_METADATA_BYTES = 4096
MAX_HISTORY_MESSAGES = 32
MAX_TASK_ARTIFACTS = 16
MAX_ARTIFACT_PARTS = 16
TERMINAL_TASK_STATES = frozenset(
    {
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_REJECTED",
    }
)


class A2AProtocolError(ValueError):
    """Raised when an A2A payload is incompatible with Conducto's profile."""


def proto_json_dict(message: ProtobufMessage) -> dict[str, Any]:
    """Return canonical protobuf JSON field names for an official A2A SDK message.

    Args:
        message: Official protobuf message from the pinned A2A SDK.

    Returns:
        A JSON-compatible dictionary using A2A 1.0 lowerCamel field names.
    """
    return MessageToDict(
        message,
        preserving_proto_field_name=False,
        always_print_fields_with_no_presence=True,
    )


def parse_agent_card(payload: Mapping[str, Any]) -> AgentCard:
    """Parse and validate an Agent Card with official A2A SDK contracts.

    Args:
        payload: Candidate A2A 1.0 Agent Card payload.

    Returns:
        The parsed official SDK AgentCard message.

    Raises:
        A2AProtocolError: If the payload is malformed or unsupported.
    """
    if "protocolVersion" in payload:
        raise A2AProtocolError(
            "A2A 1.0 Agent Cards declare protocol versions per supportedInterfaces entry; "
            "0.3 protocolVersion cards are unsupported"
        )
    try:
        card = ParseDict(dict(payload), AgentCard())
    except ParseError as exc:
        raise A2AProtocolError(f"Invalid A2A 1.0 Agent Card: {exc}") from exc
    if not any(
        interface.protocol_binding == A2A_JSONRPC_BINDING
        and interface.protocol_version == A2A_PROTOCOL_VERSION
        for interface in card.supported_interfaces
    ):
        raise A2AProtocolError("Agent Card must advertise JSON-RPC A2A protocol version 1.0")
    validate_required_extensions(proto_json_dict(card))
    return card


def parse_message(payload: Mapping[str, Any]) -> Message:
    """Parse and validate a message with official A2A SDK contracts.

    Args:
        payload: Candidate A2A 1.0 Message payload.

    Returns:
        The parsed official SDK Message.

    Raises:
        A2AProtocolError: If the payload is malformed or exceeds profile limits.
    """
    try:
        message = ParseDict(dict(payload), Message())
    except ParseError as exc:
        raise A2AProtocolError(f"Invalid A2A 1.0 Message: {exc}") from exc
    if len(message.parts) > MAX_MESSAGE_PARTS:
        raise A2AProtocolError(f"Message parts exceed limit {MAX_MESSAGE_PARTS}")
    _validate_metadata_size(payload.get("metadata"))
    for part in proto_json_dict(message).get("parts", []):
        if isinstance(part, Mapping):
            media_type = part.get("mediaType")
            if media_type is not None and media_type not in SUPPORTED_MEDIA_TYPES:
                raise A2AProtocolError(f"Unsupported media type: {media_type}")
    return message


def parse_task(payload: Mapping[str, Any]) -> Task:
    """Parse and validate a task with official A2A SDK contracts.

    Args:
        payload: Candidate A2A 1.0 Task payload.

    Returns:
        The parsed official SDK Task.

    Raises:
        A2AProtocolError: If the payload is malformed or exceeds profile limits.
    """
    try:
        task = ParseDict(dict(payload), Task())
    except ParseError as exc:
        raise A2AProtocolError(f"Invalid A2A 1.0 Task: {exc}") from exc
    if len(task.history) > MAX_HISTORY_MESSAGES:
        raise A2AProtocolError(f"Task history exceeds limit {MAX_HISTORY_MESSAGES}")
    if len(task.artifacts) > MAX_TASK_ARTIFACTS:
        raise A2AProtocolError(f"Task artifacts exceed limit {MAX_TASK_ARTIFACTS}")
    _validate_metadata_size(payload.get("metadata"))
    for artifact in task.artifacts:
        if len(artifact.parts) > MAX_ARTIFACT_PARTS:
            raise A2AProtocolError(f"Artifact parts exceed limit {MAX_ARTIFACT_PARTS}")
    return task


def validate_jsonrpc_method(method: str) -> None:
    """Validate that a JSON-RPC method is implemented by the pinned A2A profile.

    Args:
        method: Candidate A2A JSON-RPC method name.

    Raises:
        A2AProtocolError: If the method is not implemented.
    """
    if method not in SUPPORTED_JSONRPC_METHODS:
        raise A2AProtocolError(f"Unsupported A2A JSON-RPC method: {method}")


def validate_required_extensions(agent_card: Mapping[str, Any]) -> None:
    """Reject unknown required Agent Card extensions.

    Args:
        agent_card: A2A 1.0 Agent Card payload using JSON field names.

    Raises:
        A2AProtocolError: If a required extension is not supported by Conducto.
    """
    capabilities = agent_card.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return
    extensions = capabilities.get("extensions", [])
    if not isinstance(extensions, Sequence) or isinstance(extensions, (str, bytes)):
        raise A2AProtocolError("Agent Card capabilities.extensions must be a list")
    for extension in extensions:
        if not isinstance(extension, Mapping):
            raise A2AProtocolError("Agent Card extension entries must be objects")
        uri = extension.get("uri")
        if extension.get("required") is True and uri not in SUPPORTED_REQUIRED_EXTENSIONS:
            raise A2AProtocolError(f"Unsupported required Agent Card extension: {uri}")


def validate_task_transition(current_state: str, next_state: str) -> None:
    """Validate Conducto's terminal task-state immutability rule.

    Args:
        current_state: Current A2A TaskState enum name.
        next_state: Proposed next A2A TaskState enum name.

    Raises:
        A2AProtocolError: If a terminal task would be resumed or changed.
    """
    if current_state in TERMINAL_TASK_STATES and next_state != current_state:
        raise A2AProtocolError(
            "Terminal A2A tasks are immutable; create a new task in the same context"
        )


def _validate_metadata_size(metadata: Any) -> None:
    if metadata is None:
        return
    encoded = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise A2AProtocolError(f"Metadata exceeds limit {MAX_METADATA_BYTES} bytes")
