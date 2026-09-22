"""Provider-neutral model contracts and deterministic routing helpers."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, TypeAlias, cast, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .model_config import ModelReference
from .runtime_errors import IncompatibleProviderCapabilitiesError


class MessageContentPart(BaseModel):
    """A provider-neutral content part supported by the current contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text", "json"]
    text: str | None = None
    value: dict[str, Any] | list[Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Ensure a content part has exactly the payload its type requires."""
        if self.type == "text" and (self.text is None or self.value is not None):
            raise ValueError("text content parts require only text")
        if self.type == "json" and (self.value is None or self.text is not None):
            raise ValueError("json content parts require only value")


class ChatMessage(BaseModel):
    """Provider-neutral chat message sent to a model client.

    Attributes:
        role: Conversation role understood by the provider adapter.
        content: Text content for the message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    content: str | tuple[MessageContentPart, ...]

    @field_validator("content")
    @classmethod
    def validate_content(
        cls, value: str | tuple[MessageContentPart, ...] | list[MessageContentPart]
    ) -> str | tuple[MessageContentPart, ...]:
        """Normalize content parts to an immutable tuple."""
        if isinstance(value, list):
            return tuple(value)
        return value


class GenerationOptions(BaseModel):
    """Validated generation settings passed to a provider for one call.

    Attributes:
        model: Provider-native model identifier from the resolved registration.
        temperature: Sampling temperature.
        max_tokens: Optional positive output-token limit.
        timeout: Optional positive timeout in seconds for each attempt.
        retries: Number of safe retry attempts after the initial request.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)
    retries: int = Field(default=0, ge=0)
    stop: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Reject non-finite sampling values."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be finite and non-negative")
        return value


class JsonSchemaDialect(StrEnum):
    """JSON Schema dialects recognized by the provider contract."""

    DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
    DRAFT_07 = "http://json-schema.org/draft-07/schema#"


class SchemaFeature(StrEnum):
    """Schema features a provider may advertise as natively supported."""

    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    CONST = "const"
    REQUIRED = "required"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"
    MIN_ITEMS = "minItems"
    MAX_ITEMS = "maxItems"
    ONE_OF = "oneOf"
    ANY_OF = "anyOf"
    ALL_OF = "allOf"
    REF = "$ref"


@dataclass(frozen=True)
class StructuredOutputRequest:
    """Immutable, native JSON-schema-constrained output request."""

    name: str
    schema: Mapping[str, Any]
    dialect: JsonSchemaDialect = JsonSchemaDialect.DRAFT_2020_12
    required: bool = True
    features: frozenset[SchemaFeature] = frozenset()

    def __init__(
        self,
        name: str,
        schema: Mapping[str, Any] | None = None,
        *,
        json_schema: Mapping[str, Any] | None = None,
        dialect: JsonSchemaDialect = JsonSchemaDialect.DRAFT_2020_12,
        required: bool = True,
        features: frozenset[SchemaFeature] | set[SchemaFeature] = frozenset(),
    ) -> None:
        resolved_schema = schema if schema is not None else json_schema
        if resolved_schema is None:
            raise ValueError("StructuredOutputRequest requires 'schema' or 'json_schema'")
        if not name.strip():
            raise ValueError("Structured output name cannot be empty")
        if not isinstance(resolved_schema, Mapping) or not resolved_schema:
            raise ValueError("Structured output schema cannot be empty")
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "schema", _freeze_json(resolved_schema))
        object.__setattr__(self, "dialect", JsonSchemaDialect(dialect))
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "features", frozenset(features))

    @property
    def json_schema(self) -> Mapping[str, Any]:
        """Return the native JSON schema for this structured-output request.

        Returns:
            The unconstrained serialized schema definition for the request.
        """
        return cast(Mapping[str, Any], _thaw_json(self.schema))


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProviderToolDefinition:
        """Build a typed tool definition from a legacy mapping payload."""
        tool_id = value.get("id", value.get("tool_id"))
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("Provider tool definitions require a non-empty id")
        name = value.get("name", tool_id)
        description = value.get("description", name)
        input_schema = value.get("input_schema", {"type": "object"})
        if not isinstance(name, str):
            raise ValueError("Provider tool definition name must be a string")
        if not isinstance(description, str):
            raise ValueError("Provider tool definition description must be a string")
        if not isinstance(input_schema, Mapping):
            raise ValueError("Provider tool definition input_schema must be a mapping")
        return cls(
            tool_id=tool_id,
            name=name,
            description=description,
            input_schema=input_schema,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a legacy JSON-serializable tool payload."""
        return {
            "id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": _thaw_json(self.input_schema),
        }


def _freeze_json(value: Any) -> Any:
    """Freeze JSON-like contract data without leaking mutable provider state."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return ordinary JSON-compatible dictionaries and lists."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_thaw_json(item) for item in value]
    return value


class Usage(BaseModel):
    """Provider-neutral token usage counters.

    Attributes:
        input_tokens: Tokens consumed by request input.
        output_tokens: Tokens generated by the provider.
        total_tokens: Total tokens reported for the request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        """Reject non-finite provider cost values."""
        if value is not None and not math.isfinite(value):
            raise ValueError("cost must be finite")
        return value


class ProviderToolCallRequest(BaseModel):
    """Provider-returned native tool call before snapshot resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str | None = Field(default=None, min_length=1)
    tool_id: str | None = Field(default=None, min_length=1)
    tool_name: str | None = Field(default=None, min_length=1)
    arguments: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so validated calls cannot be mutated in place."""
        return cast(Mapping[str, Any], _freeze_json(value))

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

    @field_validator("arguments", mode="before")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so a normalized call cannot be mutated before execution."""
        return cast(Mapping[str, Any], _freeze_json(value))


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

    @field_validator("arguments", mode="before")
    @classmethod
    def freeze_arguments(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively freeze arguments so the decision cannot be mutated before execution."""
        return cast(Mapping[str, Any], _freeze_json(value))


ModelDecision: TypeAlias = Annotated[
    TerminalModelDecision | ToolCallModelDecision,
    Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """One bounded result from a prior tool call supplied to the next turn."""

    call_id: str
    status: str
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FakeModelRequest:
    """Safe observation of one request issued to :class:`FakeModel`.

    Attributes:
        message_roles: Roles supplied on the request.
        options: Generation settings observed by the fake provider.
        structured_output: Terminal structured-output schema for the turn.
        tools: Provider-neutral tool definitions for the exact toolbox snapshot.
        tool_results: Prior bounded tool results for the turn.
        effective_deadline: Effective monotonic deadline applied to the turn.
    """

    message_roles: tuple[str, ...]
    options: GenerationOptions
    structured_output: StructuredOutputRequest
    tools: tuple[ProviderToolDefinition, ...] = ()
    tool_results: tuple[ToolResultMessage, ...] = ()
    effective_deadline: float | None = None


class ProviderResult(BaseModel):
    """Normalized completion result returned by a model provider.

    Attributes:
        content: Optional unstructured text response.
        structured: Optional decoded terminal structured response.
        tool_calls: Optional provider-native tool-call requests for this turn.
        usage: Provider-neutral usage counters.
        accepted: Whether the provider accepted the request. Accepted failures
            are not automatically retried.
        request_id: Optional non-secret provider request identifier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    structured: Any = None
    tool_calls: tuple[ProviderToolCallRequest, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    accepted: bool = False
    request_id: str | None = None
    finish_reason: str | None = None
    content_filtered: bool = False

    @property
    def acceptance(self) -> AcceptanceState:
        """Return the explicit acceptance state represented by this result."""
        return AcceptanceState.ACCEPTED if self.accepted else AcceptanceState.ATTEMPTED_NOT_ACCEPTED


class AcceptanceState(StrEnum):
    """Whether a provider accepted work for processing."""

    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED_NOT_ACCEPTED = "attempted_not_accepted"
    ACCEPTED = "accepted"


@dataclass(frozen=True, slots=True)
class ProviderCallContext:
    """Credential-free deadline and cancellation state visible to adapters."""

    deadline: float | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """Safe, credential-free diagnostic suitable for public serialization."""

    category: ProviderFailureCategory
    message: str
    request_id: str | None = None

    def __post_init__(self) -> None:
        """Replace arbitrary caller text with a stable, safe summary."""
        object.__setattr__(self, "message", f"provider failure: {self.category.value}")

    def to_dict(self) -> dict[str, str | None]:
        """Return a redacted diagnostic representation."""
        return {
            "category": self.category.value,
            "message": self.message,
            "request_id": self.request_id,
        }


class ProviderFailureCategory(StrEnum):
    """Stable categories for failures surfaced by provider adapters."""

    CONFIGURATION = "configuration"
    MISSING_DEPENDENCY = "missing_dependency"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    CONTENT_POLICY = "content_policy"
    MALFORMED_OUTPUT = "malformed_output"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"
    INTERNAL = "internal"


class FinishReason(StrEnum):
    """Provider-neutral completion finish metadata."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Capabilities advertised by a registered provider client.

    Attributes:
        structured_output: Supports native schema-constrained output.
        tool_calling: Supports provider-native tool calls.
        context_limit: Maximum advertised context size, or zero when unknown.
        usage_reporting: Returns token usage with results.
    """

    structured_output: bool = False
    tool_calling: bool = False
    context_limit: int = 0
    usage_reporting: bool = False
    streaming: bool = False
    cancellation: bool = False
    output_limit: int | None = None
    schema_dialects: frozenset[JsonSchemaDialect] = frozenset({JsonSchemaDialect.DRAFT_2020_12})
    schema_features: frozenset[SchemaFeature] = frozenset(SchemaFeature)

    def __post_init__(self) -> None:
        """Normalize capability collections and validate numeric limits."""
        if self.context_limit < 0 or (self.output_limit is not None and self.output_limit <= 0):
            raise ValueError("provider context and output limits must be positive")
        object.__setattr__(self, "schema_dialects", frozenset(self.schema_dialects))
        object.__setattr__(self, "schema_features", frozenset(self.schema_features))


def validate_provider_capabilities(
    reference: ModelReference,
    capabilities: ProviderCapabilities,
    required: frozenset[str],
) -> None:
    """Ensure a provider capabilities object satisfies every required capability name.

    Args:
        reference: Model reference the capabilities were resolved for, used only
            for the diagnostic message.
        capabilities: Advertised provider capabilities.
        required: Capability attribute names that must be present and truthy.

    Raises:
        IncompatibleProviderCapabilitiesError: If any required capability is missing.
    """
    unsupported = sorted(
        name
        for name in required
        if not hasattr(capabilities, name) or not bool(getattr(capabilities, name))
    )
    if unsupported:
        raise IncompatibleProviderCapabilitiesError(
            f"Model reference '{reference}' lacks capabilities: {', '.join(unsupported)}"
        )


class ProviderError(RuntimeError):
    """Base provider error with explicit retry and acceptance state."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        accepted: bool = False,
        category: ProviderFailureCategory = ProviderFailureCategory.INTERNAL,
        request_id: str | None = None,
        attempted: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable and not accepted
        self.accepted = accepted
        self.category = category
        self.request_id = request_id
        self.attempted = attempted

    @property
    def acceptance(self) -> AcceptanceState:
        """Return the explicit acceptance state for retry decisions."""
        if self.accepted:
            return AcceptanceState.ACCEPTED
        return (
            AcceptanceState.ATTEMPTED_NOT_ACCEPTED
            if self.attempted
            else AcceptanceState.NOT_ATTEMPTED
        )

    @property
    def diagnostic(self) -> ProviderDiagnostic:
        """Return a safe diagnostic without provider exception details."""
        return ProviderDiagnostic(self.category, self.__class__.__name__, self.request_id)


class ProviderAuthenticationError(ProviderError):
    """Provider rejected or could not get authentication."""

    def __init__(
        self,
        message: str = "Provider authentication failed",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            accepted=accepted,
            category=ProviderFailureCategory.AUTHENTICATION,
            request_id=request_id,
            attempted=attempted,
        )


class ProviderRateLimitError(ProviderError):
    """Provider rejected a request because the rate limit was exceeded."""

    def __init__(
        self,
        message: str = "Provider rate limit exceeded",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            accepted=accepted,
            category=ProviderFailureCategory.RATE_LIMIT,
            request_id=request_id,
            attempted=attempted,
        )


class ProviderContentPolicyError(ProviderError):
    """Provider rejected content under its safety or usage policy."""

    def __init__(
        self,
        message: str = "Provider content policy rejected the request",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            accepted=accepted,
            category=ProviderFailureCategory.CONTENT_POLICY,
            request_id=request_id,
            attempted=attempted,
        )


class ProviderTimeoutError(ProviderError):
    """Provider request exceeded the allowed timeout window."""

    def __init__(
        self,
        message: str = "Provider request timed out",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            accepted=accepted,
            category=ProviderFailureCategory.TIMEOUT,
            request_id=request_id,
            attempted=attempted,
        )


class UnsupportedProviderCapabilityError(ProviderError):
    """Provider cannot satisfy a capability required by the request."""

    def __init__(self, message: str = "Provider capability is unsupported") -> None:
        super().__init__(message, category=ProviderFailureCategory.UNSUPPORTED_CAPABILITY)


class MalformedStructuredOutputError(ProviderError):
    """Provider output did not satisfy the required structured contract."""

    def __init__(
        self,
        message: str = "Provider returned malformed structured output",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        usage: Usage | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            category=ProviderFailureCategory.MALFORMED_OUTPUT,
            accepted=accepted,
            request_id=request_id,
            attempted=attempted,
        )
        self.usage = usage


class ProviderEndpointUnavailableError(ProviderError):
    """The endpoint, deployment, or selected model is unavailable."""

    def __init__(
        self,
        message: str = "Provider is unavailable",
        *,
        accepted: bool = False,
        request_id: str | None = None,
        attempted: bool = True,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            accepted=accepted,
            category=ProviderFailureCategory.UNAVAILABLE,
            request_id=request_id,
            attempted=attempted,
        )


class ProviderCancellationError(ProviderError):
    """The caller cancelled an in-flight provider request."""

    def __init__(self, message: str = "Provider request cancelled") -> None:
        super().__init__(message, category=ProviderFailureCategory.CANCELLATION)


@runtime_checkable
class SynchronouslyClosableProvider(Protocol):
    """Optional structural protocol for clients that close synchronously."""

    def close(self) -> None:
        """Release the client's local resources."""


@runtime_checkable
class AsynchronouslyClosableProvider(Protocol):
    """Optional structural protocol for clients that close asynchronously."""

    async def aclose(self) -> None:
        """Release the client's local resources."""


class ModelProvider(Protocol):
    """Protocol implemented by credential-bearing model clients.

    Implementations are registered with a runtime-owned provider registry.
    Agents and run requests reference them indirectly and never retain their
    credentials or provider-specific configuration.
    """

    capabilities: ProviderCapabilities

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        effective_deadline: float | None = None,
    ) -> ProviderResult:
        """Generate a completion for a provider-neutral chat request.

        Args:
            messages: Conversation history to send to the provider.
            options: Generation controls for model temperature, token caps, and
                retries.
            structured_output: Native JSON schema required for structured output.
            tools: Provider-neutral tools available for this turn.
            tool_results: Bounded results from prior provider-issued tool calls.
            effective_deadline: Optional monotonic deadline that bounds this call.
            call_context: Optional credential-free cancellation/deadline context.

        Returns:
            Normalized provider response content, metadata, and usage counters.
        """


class CancellableModelProvider(ModelProvider, Protocol):
    """Optional provider protocol that receives cancellation context."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        effective_deadline: float | None = None,
        call_context: ProviderCallContext | None = None,
    ) -> ProviderResult:
        """Generate a completion while observing the caller context."""


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


class ModelConfiguration(BaseModel):
    """Immutable provider-neutral model configuration.

    The configuration identifies a provider and its provider model name, plus
    safe request defaults. Credentials, authorization headers, and other
    sensitive provider settings must remain inside the registered client.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None
    timeout: float = Field(default=30.0, gt=0)
    retries: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class FakeModel:
    """Deterministic provider for unit and acceptance tests."""

    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calling=True,
        context_limit=128_000,
        usage_reporting=True,
    )

    def __init__(
        self,
        selection: RoutingSelection | dict[str, Any] | str | Sequence[object] | None = None,
        *,
        usage: Usage | None = None,
        accepted: bool = True,
        script: Sequence[object] | None = None,
    ) -> None:
        if script is not None and selection is not None:
            raise ValueError("FakeModel accepts either selection or script, not both")
        selected_script = script
        if (
            selected_script is None
            and isinstance(selection, Sequence)
            and not isinstance(selection, str)
        ):
            selected_script = selection
        if selection is None and selected_script is None:
            raise ValueError("FakeModel requires a selection or script")
        self.selection = selection
        self._script = tuple(selected_script) if selected_script is not None else None
        self.usage = usage or Usage()
        self.accepted = accepted
        self.calls = 0
        self.requests: list[FakeModelRequest] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        effective_deadline: float | None = None,
        call_context: ProviderCallContext | None = None,
    ) -> ProviderResult:
        """Return the configured fake completion payload.

        Args:
            messages: Conversation history supplied to the fake provider.
            options: Generation settings observed for the request.
            structured_output: Requested structured output contract.
            tools: Provider-neutral tool definitions for this turn.
            tool_results: Prior tool results for this turn.
            effective_deadline: Effective monotonic deadline for this turn.

        Returns:
            The synthetic provider result configured when the fake provider was created.
        """
        del call_context
        self.calls += 1
        normalized_tools = _normalize_provider_tools(tools)
        self.requests.append(
            FakeModelRequest(
                message_roles=tuple(message.role for message in messages),
                options=options,
                structured_output=structured_output,
                tools=normalized_tools,
                tool_results=tuple(tool_results),
                effective_deadline=effective_deadline,
            )
        )
        selection: object
        if self._script is None:
            selection = self.selection
        elif self.calls <= len(self._script):
            selection = self._script[self.calls - 1]
        else:
            raise ProviderError("FakeModel script exhausted")
        if isinstance(selection, ProviderError):
            raise selection
        if isinstance(selection, ProviderResult):
            return selection
        if isinstance(selection, str):
            return ProviderResult(content=selection, usage=self.usage, accepted=self.accepted)
        payload = selection.model_dump() if isinstance(selection, BaseModel) else selection
        if not isinstance(payload, Mapping):
            raise TypeError("FakeModel script entries must be model results or dictionaries")
        return _coerce_fake_provider_result(payload, usage=self.usage, accepted=self.accepted)


def _coerce_fake_provider_result(
    payload: Mapping[str, Any],
    *,
    usage: Usage,
    accepted: bool,
) -> ProviderResult:
    """Translate legacy fake payloads into the separated provider result contract."""
    decision_type = payload.get("type")
    if decision_type == "terminal":
        return ProviderResult(
            structured=payload.get("response"),
            usage=usage,
            accepted=accepted,
        )
    if decision_type == "tool_call":
        call_payload = dict(payload)
        call_payload.pop("type", None)
        return ProviderResult(
            tool_calls=(ProviderToolCallRequest.model_validate(call_payload),),
            usage=usage,
            accepted=accepted,
        )
    return ProviderResult(structured=dict(payload), usage=usage, accepted=accepted)


def build_model_decision_schema(
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
    tools: Sequence[ProviderToolDefinition | Mapping[str, Any]] = (),
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


def _normalize_provider_tools(
    tools: Sequence[ProviderToolDefinition | Mapping[str, Any]],
) -> tuple[ProviderToolDefinition, ...]:
    """Return typed provider tool definitions from public or legacy payloads."""
    normalized: list[ProviderToolDefinition] = []
    for tool in tools:
        if isinstance(tool, ProviderToolDefinition):
            normalized.append(tool)
        elif isinstance(tool, Mapping):
            normalized.append(ProviderToolDefinition.from_mapping(tool))
        else:
            raise TypeError(
                "Provider tool definitions must be ProviderToolDefinition or mapping values"
            )
    return tuple(normalized)


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
        request.arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    seed = f"{request_id or ''}\0{resolved_tool_id}\0{raw_reference}\0{arguments}"
    return f"tool_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"


def validate_provider_contract(
    provider: ModelProvider,
    *,
    structured_output: StructuredOutputRequest,
    tools: Sequence[ProviderToolDefinition | Mapping[str, Any]] = (),
    tool_results: Sequence[ToolResultMessage] = (),
) -> None:
    """Validate provider support before issuing a structured request.

    Raises:
        UnsupportedProviderCapabilityError: If native structured output is
            unavailable.
        ValueError: If the required JSON schema is empty.
    """
    if not provider.capabilities.structured_output:
        raise UnsupportedProviderCapabilityError(
            "Provider does not support required native structured output"
        )
    required_features = set(structured_output.features) or _schema_keywords(
        structured_output.json_schema
    )
    unknown = _unknown_schema_keywords(structured_output.json_schema)
    if unknown:
        raise UnsupportedProviderCapabilityError(
            f"Structured output schema uses unsupported keywords: {sorted(unknown)!r}"
        )
    unsupported = required_features - set(provider.capabilities.schema_features)
    if unsupported:
        raise UnsupportedProviderCapabilityError(
            f"Structured output schema uses unsupported features: {sorted(unsupported)!r}"
        )
    if structured_output.dialect not in provider.capabilities.schema_dialects:
        raise UnsupportedProviderCapabilityError(
            f"Provider does not support JSON Schema dialect {structured_output.dialect}"
        )
    normalized_tools = _normalize_provider_tools(tools)
    if (normalized_tools or tool_results) and not provider.capabilities.tool_calling:
        raise UnsupportedProviderCapabilityError(
            "Provider does not support required native tool calling"
        )


_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "items",
    "required",
    "additionalProperties",
    "enum",
    "const",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "oneOf",
    "anyOf",
    "allOf",
    "$ref",
    "$defs",
    "$schema",
    "title",
    "description",
    "default",
}


def _schema_keywords(schema: Mapping[str, Any]) -> set[SchemaFeature]:
    """Collect schema features used by a JSON Schema document."""
    found: set[SchemaFeature] = set()
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            try:
                found.add(SchemaFeature(value))
            except ValueError:
                pass
        try:
            feature = SchemaFeature(key)
        except ValueError:
            continue
        found.add(feature)
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, Mapping):
                    found.update(_schema_keywords(child))
        elif isinstance(value, Mapping):
            found.update(_schema_keywords(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    found.update(_schema_keywords(item))
    return found


def _unknown_schema_keywords(schema: Mapping[str, Any]) -> set[str]:
    """Find validation keywords outside the deliberately supported subset."""
    unknown: set[str] = set()
    for key, value in schema.items():
        if key not in _SCHEMA_KEYWORDS and not key.startswith("x-"):
            unknown.add(str(key))
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, Mapping):
                    unknown.update(_unknown_schema_keywords(child))
        elif isinstance(value, Mapping):
            unknown.update(_unknown_schema_keywords(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    unknown.update(_unknown_schema_keywords(item))
    return unknown


def validate_structured_output(
    value: Any,
    request: StructuredOutputRequest,
) -> None:
    """Validate decoded provider output against the requested schema.

    This deliberately implements the provider-neutral subset instead of
    depending on an optional JSON Schema package.
    """
    root_schema = request.json_schema

    def check(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
        if "$ref" in schema:
            ref = schema["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
                raise MalformedStructuredOutputError(f"{path} has unsupported $ref")
            target: Any = root_schema
            for part in ref[2:].split("/"):
                if not isinstance(target, Mapping) or part not in target:
                    raise MalformedStructuredOutputError(f"{path} has unresolved $ref")
                target = target[part]
            if not isinstance(target, Mapping):
                raise MalformedStructuredOutputError(f"{path} has invalid $ref")
            check(instance, target, path)
        if "allOf" in schema:
            for branch in schema["allOf"]:
                check(instance, branch, path)
        expected = schema.get("type")
        type_ok = {
            "object": isinstance(instance, Mapping),
            "array": isinstance(instance, list),
            "string": isinstance(instance, str),
            "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
            "integer": isinstance(instance, int) and not isinstance(instance, bool),
            "boolean": isinstance(instance, bool),
            "null": instance is None,
        }
        if isinstance(expected, str) and not type_ok.get(expected, False):
            raise MalformedStructuredOutputError(f"{path} must be {expected}")
        if "enum" in schema and instance not in schema["enum"]:
            raise MalformedStructuredOutputError(f"{path} is not an allowed enum value")
        if "const" in schema and instance != schema["const"]:
            raise MalformedStructuredOutputError(f"{path} does not match const")
        if isinstance(instance, Mapping):
            properties = schema.get("properties", {})
            for key in schema.get("required", ()):
                if key not in instance:
                    raise MalformedStructuredOutputError(f"{path}.{key} is required")
            if schema.get("additionalProperties") is False:
                extra = set(instance) - set(properties)
                if extra:
                    raise MalformedStructuredOutputError(f"{path} has additional properties")
            for key, child in properties.items():
                if key in instance:
                    check(instance[key], child, f"{path}.{key}")
        if isinstance(instance, list):
            if "minItems" in schema and len(instance) < schema["minItems"]:
                raise MalformedStructuredOutputError(f"{path} has too few items")
            if "maxItems" in schema and len(instance) > schema["maxItems"]:
                raise MalformedStructuredOutputError(f"{path} has too many items")
            if isinstance(schema.get("items"), Mapping):
                for index, item in enumerate(instance):
                    check(item, schema["items"], f"{path}[{index}]")
        if isinstance(instance, str):
            if "minLength" in schema and len(instance) < schema["minLength"]:
                raise MalformedStructuredOutputError(f"{path} is too short")
            if "maxLength" in schema and len(instance) > schema["maxLength"]:
                raise MalformedStructuredOutputError(f"{path} is too long")
        for keyword in ("oneOf", "anyOf"):
            if keyword in schema:
                matches = 0
                for branch in schema[keyword]:
                    try:
                        check(instance, branch, path)
                    except MalformedStructuredOutputError:
                        continue
                    matches += 1
                if (keyword == "oneOf" and matches != 1) or (keyword == "anyOf" and matches < 1):
                    raise MalformedStructuredOutputError(f"{path} does not satisfy {keyword}")

    check(value, request.json_schema)


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


async def complete_with_retries(
    provider: ModelProvider,
    messages: Sequence[ChatMessage],
    *,
    options: GenerationOptions,
    structured_output: StructuredOutputRequest,
    deadline: float | None = None,
    tools: Sequence[ProviderToolDefinition | Mapping[str, Any]] = (),
    tool_results: Sequence[ToolResultMessage] = (),
    effective_deadline: float | None = None,
    call_context: ProviderCallContext | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProviderResult:
    """Complete a provider request under timeout and safe-retry rules.

    Retryable provider failures and timeouts are retried only before a request
    is known to have been accepted. Provider capability validation occurs
    before the first request.

    Args:
        provider: Resolved provider client.
        messages: Provider-neutral conversation messages.
        options: Generation, timeout, and retry settings.
        structured_output: Required native structured-output contract.
        deadline: Optional monotonic deadline governing the whole request.
        tools: Provider-neutral tools available for this request.
        tool_results: Bounded results from prior tool calls.
        effective_deadline: Optional tighter monotonic deadline.
        clock: Monotonic clock used for deterministic deadline enforcement.

    Returns:
        The normalized provider result.

    Raises:
        UnsupportedProviderCapabilityError: If structured output is unsupported.
        ProviderTimeoutError: If all permitted attempts time out.
        ProviderError: If a provider failure is not safely retryable or retries
            are exhausted.
    """
    normalized_tools = _normalize_provider_tools(tools)
    tool_aware = bool(normalized_tools) or bool(tool_results)
    if effective_deadline is not None:
        deadline = min(deadline, effective_deadline) if deadline is not None else effective_deadline
    validate_provider_contract(
        provider,
        structured_output=structured_output,
        tools=normalized_tools,
        tool_results=tool_results,
    )
    attempts = options.retries + 1
    for attempt in range(attempts):
        try:
            timeout_for_attempt: float | None = None
            if deadline is not None:
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("Run deadline exceeded")
                timeout_for_attempt = remaining
            if options.timeout is not None:
                timeout_for_attempt = (
                    min(options.timeout, timeout_for_attempt)
                    if timeout_for_attempt is not None
                    else options.timeout
                )
            request_kwargs: dict[str, Any] = {
                "options": options,
                "structured_output": structured_output,
            }
            if tool_aware:
                request_kwargs["tools"] = normalized_tools
                request_kwargs["tool_results"] = tool_results
            if tool_aware or effective_deadline is not None:
                request_kwargs["effective_deadline"] = deadline
            if call_context is not None and provider.capabilities.cancellation:
                request_kwargs["call_context"] = call_context
            completion = provider.complete(messages, **request_kwargs)
            if timeout_for_attempt is not None:
                result = await asyncio.wait_for(completion, timeout_for_attempt)
            else:
                result = await completion
            if result.content_filtered:
                raise ProviderContentPolicyError(
                    accepted=result.accepted,
                    request_id=result.request_id,
                )
            if structured_output.required:
                if result.structured is None:
                    error = MalformedStructuredOutputError(
                        "Provider returned no required structured output",
                        accepted=result.accepted,
                        request_id=result.request_id,
                        usage=result.usage,
                    )
                    raise error
                try:
                    validate_structured_output(result.structured, structured_output)
                except MalformedStructuredOutputError as error:
                    error.usage = result.usage
                    error.accepted = result.accepted
                    error.request_id = result.request_id
                    error.attempted = True
                    raise
            return result
        except TimeoutError as error:
            timeout_error = ProviderTimeoutError()
            if attempt == attempts - 1 or (deadline is not None and deadline <= clock()):
                raise timeout_error from error
            await asyncio.sleep(0)
        except ProviderError as error:
            error.attempted = True
            if not error.retryable or error.accepted or attempt == attempts - 1:
                raise
            await asyncio.sleep(0)
    raise AssertionError("unreachable")
