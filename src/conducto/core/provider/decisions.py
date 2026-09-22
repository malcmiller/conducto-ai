"""Strict model decisions and deterministic capability routing schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from ._json import _freeze_json, _thaw_json
from .errors import MalformedStructuredOutputError
from .results import ProviderResult
from .structured import StructuredOutputRequest
from .tools import ProviderToolDefinition, _normalize_provider_tools, _resolve_provider_tool_call


class TerminalModelDecision(BaseModel):
    """The sole terminal response permitted for one model decision turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["terminal"]
    response: dict[str, Any]


class ToolCallModelDecision(BaseModel):
    """The sole tool call permitted for one model decision turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_call"]
    call_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    arguments: Mapping[str, Any]

    @field_validator("arguments", mode="after")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so the decision cannot be mutated before execution."""
        return cast(Mapping[str, Any], _freeze_json(value))

    @field_serializer("arguments")
    def serialize_arguments(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize the immutable decision arguments as JSON containers."""
        return cast(dict[str, Any], _thaw_json(value))


ModelDecision: TypeAlias = Annotated[
    TerminalModelDecision | ToolCallModelDecision,
    Field(discriminator="type"),
]


class RoutingSelection(BaseModel):
    """Strict structured capability selection accepted by the orchestrator.

    Attributes:
        agent_id: Published identifier of the selected agent.
        capability_id: Capability name or generated skill identifier.
        arguments: Structured arguments for the selected capability.
    """

    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


def build_routing_schema(
    routing_metadata: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the constrained selection schema used for model routing."""
    schema = RoutingSelection.model_json_schema()
    if routing_metadata is not None:
        branches: list[dict[str, Any]] = []
        for entry in routing_metadata:
            agent_id = entry.get("name")
            if not agent_id:
                continue
            capability_ids = sorted(
                {
                    str(capability_id)
                    for capability in entry.get("capabilities", [])
                    for capability_id in (
                        capability.get("id"),
                        capability.get("name"),
                    )
                    if capability_id
                }
            )
            if not capability_ids:
                continue
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "agent_id": {"const": str(agent_id)},
                        "capability_id": {"enum": capability_ids},
                        "arguments": {"type": "object"},
                    },
                    "required": ["agent_id", "capability_id"],
                }
            )
        schema = {"oneOf": branches}
    return schema


def parse_routing_selection(result: ProviderResult) -> RoutingSelection:
    """Validate and return structured routing output.

    Raises:
        MalformedStructuredOutputError: If a structured output is absent or does
            not satisfy: class:`RoutingSelection`.
    """
    if result.structured is None:
        raise MalformedStructuredOutputError("Provider returned no structured routing output")
    try:
        return RoutingSelection.model_validate(result.structured)
    except ValidationError as error:
        raise MalformedStructuredOutputError(
            "Provider returned malformed structured routing output"
        ) from error


def build_terminal_output_request(
    response_type: type[BaseModel],
) -> StructuredOutputRequest:
    """Build the terminal structured-output schema for one delegation turn.

    Native tool definitions and prior tool results travel through their
    dedicated provider channels rather than through a synthetic top-level
    schema union.
    """
    return StructuredOutputRequest(
        name=response_type.__name__,
        schema=response_type.model_json_schema(),
        required=False,
    )


def parse_model_decision(
    result: ProviderResult,
    *,
    response_type: type[BaseModel],
    tools: Sequence[ProviderToolDefinition] = (),
) -> ModelDecision:
    """Parse exactly one terminal response or one provider-native tool call."""
    if not issubclass(response_type, BaseModel):
        raise TypeError("response_type must be a Pydantic model type")
    if result.structured is not None and result.tool_calls:
        raise MalformedStructuredOutputError(
            "Provider returned both terminal structured output and tool calls"
        )
    if result.structured is None and not result.tool_calls:
        raise MalformedStructuredOutputError("Provider returned no structured model decision")
    if result.structured is not None:
        if not isinstance(result.structured, Mapping):
            raise MalformedStructuredOutputError(
                "Provider returned malformed terminal structured output"
            )
        try:
            return TerminalModelDecision(type="terminal", response=dict(result.structured))
        except ValidationError as error:
            raise MalformedStructuredOutputError(
                "Provider returned malformed structured model decision"
            ) from error
    if len(result.tool_calls) != 1:
        raise MalformedStructuredOutputError(
            "Provider returned multiple tool calls for a single-turn decision"
        )
    normalized = _resolve_provider_tool_call(
        result.tool_calls[0],
        _normalize_provider_tools(tools),
        request_id=result.request_id,
    )
    try:
        return ToolCallModelDecision(
            type="tool_call",
            call_id=normalized.call_id,
            tool_id=normalized.tool_id,
            arguments=normalized.arguments,
        )
    except ValidationError as error:
        raise MalformedStructuredOutputError(
            "Provider returned malformed structured model decision"
        ) from error
