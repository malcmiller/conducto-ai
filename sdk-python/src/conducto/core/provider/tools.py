"""Typed tool definitions, calls, results, and snapshot-bound resolution."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ._json import _freeze_json, _thaw_json
from .errors import MalformedStructuredOutputError


@dataclass(frozen=True, slots=True)
class ProviderToolDefinition:
    """Immutable provider-neutral tool definition for one model turn.

    Attributes:
        tool_id: Opaque runtime-issued identifier bound to one toolbox snapshot.
        name: Provider-facing function name for this tool in the current turn.
        description: Safe provider-facing description.
        input_schema: Safe provider-facing JSON Schema for arguments.
    """

    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id cannot be blank")
        if not self.name.strip():
            raise ValueError("name cannot be blank")
        if not self.description.strip():
            raise ValueError("description cannot be blank")
        object.__setattr__(self, "tool_id", self.tool_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "input_schema", _freeze_json(self.input_schema))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON wire representation with its stable opaque ``id``."""
        return {
            "id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": _thaw_json(self.input_schema),
        }


class ProviderToolCallRequest(BaseModel):
    """Provider-returned native tool call before snapshot resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str | None = Field(default=None, min_length=1)
    tool_id: str | None = Field(default=None, min_length=1)
    tool_name: str | None = Field(default=None, min_length=1)
    arguments: Mapping[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so validated calls cannot be mutated in place."""
        return cast(Mapping[str, Any], _freeze_json(value))

    @field_serializer("arguments")
    def serialize_arguments(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize the immutable snapshot using ordinary JSON containers."""
        return cast(dict[str, Any], _thaw_json(value))

    @model_validator(mode="after")
    def validate_reference(self) -> ProviderToolCallRequest:
        """Require exactly one provider tool identifier channel."""
        if (self.tool_id is None) == (self.tool_name is None):
            raise ValueError("Provider tool calls require exactly one of tool_id or tool_name")
        return self


class ProviderToolCall(BaseModel):
    """Normalized provider tool call resolved against one toolbox snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    arguments: Mapping[str, Any]

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so a normalized call cannot be mutated before execution."""
        return cast(Mapping[str, Any], _freeze_json(value))

    @field_serializer("arguments")
    def serialize_arguments(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize the immutable snapshot using ordinary JSON containers."""
        return cast(dict[str, Any], _thaw_json(value))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """One bounded result from a prior tool call supplied to the next turn."""

    call_id: str
    status: str
    result: dict[str, Any]


def _normalize_provider_tools(
    tools: Sequence[ProviderToolDefinition],
) -> tuple[ProviderToolDefinition, ...]:
    """Snapshot typed definitions without coercing legacy mapping payloads."""
    snapshot = tuple(tools)
    if any(not isinstance(tool, ProviderToolDefinition) for tool in snapshot):
        raise TypeError("Provider tool definitions must be ProviderToolDefinition values")
    if len({tool.tool_id for tool in snapshot}) != len(snapshot):
        raise ValueError("Provider tool definitions require unique tool ids")
    return snapshot


def _resolve_provider_tool_call(
    request: ProviderToolCallRequest,
    tools: Sequence[ProviderToolDefinition],
    *,
    request_id: str | None,
) -> ProviderToolCall:
    """Resolve a provider-returned tool reference through one exact toolbox snapshot."""
    if not tools:
        raise MalformedStructuredOutputError(
            "Provider returned tool calls without an advertised tool snapshot"
        )
    if request.tool_id is not None:
        if not any(tool.tool_id == request.tool_id for tool in tools):
            raise MalformedStructuredOutputError("Provider returned an unknown tool id")
        call_id = request.call_id or _synthesize_tool_call_id(request, request.tool_id, request_id)
        return ProviderToolCall(
            call_id=call_id,
            tool_id=request.tool_id,
            arguments=request.arguments,
        )
    matches = [
        tool for tool in tools if (request.tool_name is not None and tool.name == request.tool_name)
    ]
    if not matches:
        raise MalformedStructuredOutputError("Provider returned an unknown tool name")
    if len(matches) != 1:
        raise MalformedStructuredOutputError("Provider returned an ambiguous tool call")
    call_id = request.call_id or _synthesize_tool_call_id(request, matches[0].tool_id, request_id)
    return ProviderToolCall(
        call_id=call_id,
        tool_id=matches[0].tool_id,
        arguments=request.arguments,
    )


def _synthesize_tool_call_id(
    request: ProviderToolCallRequest,
    resolved_tool_id: str,
    request_id: str | None,
) -> str:
    """Create a stable opaque call ID for providers that omit one."""
    raw_reference = request.tool_id or request.tool_name or resolved_tool_id
    arguments = json.dumps(
        _thaw_json(request.arguments), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    seed = f"{request_id or ''}\0{resolved_tool_id}\0{raw_reference}\0{arguments}"
    return f"tool_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"
