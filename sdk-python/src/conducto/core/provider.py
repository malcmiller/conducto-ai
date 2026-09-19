"""Provider-neutral model contracts and deterministic routing helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerationOptions(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)
    retries: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class StructuredOutputRequest:
    name: str
    schema: dict[str, Any]

    def __init__(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        resolved_schema = schema if schema is not None else json_schema
        if resolved_schema is None:
            raise ValueError("StructuredOutputRequest requires 'schema' or 'json_schema'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", resolved_schema)

    @property
    def json_schema(self) -> dict[str, Any]:
        return self.schema


class Usage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ProviderResult(BaseModel):
    content: str | None = None
    structured: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    accepted: bool = False
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    structured_output: bool = False
    tool_calling: bool = False
    context_limit: int = 0
    usage_reporting: bool = False


class ProviderError(RuntimeError):
    """Base provider error with explicit retry and acceptance state."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        accepted: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable and not accepted
        self.accepted = accepted


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    def __init__(
        self,
        message: str = "Provider rate limit exceeded",
        *,
        accepted: bool = False,
    ) -> None:
        super().__init__(message, retryable=True, accepted=accepted)


class ProviderContentPolicyError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    def __init__(
        self,
        message: str = "Provider request timed out",
        *,
        accepted: bool = False,
    ) -> None:
        super().__init__(message, retryable=True, accepted=accepted)


class UnsupportedProviderCapabilityError(ProviderError):
    pass


class MalformedStructuredOutputError(ProviderError):
    pass


class ModelProvider(Protocol):
    capabilities: ProviderCapabilities

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
    ) -> ProviderResult:
        ...


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
    """The only model output accepted by the orchestrator."""

    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelConfiguration(BaseModel):
    """Provider-neutral runtime configuration; secrets are never part of it."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None
    timeout: float = Field(default=30.0, gt=0)
    retries: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class FakeModel:
    """Deterministic provider for unit and acceptance tests."""

    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calling=False,
        context_limit=128_000,
        usage_reporting=True,
    )

    def __init__(
        self,
        selection: RoutingSelection | dict[str, Any] | str,
        *,
        usage: Usage | None = None,
        accepted: bool = True,
    ) -> None:
        self.selection = selection
        self.usage = usage or Usage()
        self.accepted = accepted
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
    ) -> ProviderResult:
        _ = (messages, options, structured_output)
        self.calls += 1
        if isinstance(self.selection, str):
            return ProviderResult(
                content=self.selection, usage=self.usage, accepted=self.accepted
            )
        selection = (
            self.selection.model_dump()
            if isinstance(self.selection, RoutingSelection)
            else self.selection
        )
        return ProviderResult(
            structured=selection,
            usage=self.usage,
            accepted=self.accepted,
        )


def validate_provider_contract(
    provider: ModelProvider,
    *,
    structured_output: StructuredOutputRequest,
) -> None:
    if not provider.capabilities.structured_output:
        raise UnsupportedProviderCapabilityError(
            "Provider does not support required native structured output"
        )
    if not structured_output.json_schema:
        raise ValueError("Structured output schema cannot be empty")


def parse_routing_selection(result: ProviderResult) -> RoutingSelection:
    if result.structured is None:
        raise MalformedStructuredOutputError(
            "Provider returned no structured routing output"
        )
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
) -> ProviderResult:
    validate_provider_contract(provider, structured_output=structured_output)
    attempts = options.retries + 1
    for attempt in range(attempts):
        try:
            completion = provider.complete(
                messages,
                options=options,
                structured_output=structured_output,
            )
            if options.timeout is not None:
                return await asyncio.wait_for(completion, options.timeout)
            return await completion
        except TimeoutError as error:
            timeout_error = ProviderTimeoutError()
            if attempt == attempts - 1:
                raise timeout_error from error
            await asyncio.sleep(0)
        except ProviderError as error:
            if not error.retryable or error.accepted or attempt == attempts - 1:
                raise
            await asyncio.sleep(0)
    raise AssertionError("unreachable")
