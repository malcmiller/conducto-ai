"""Provider protocols and capability checks before model dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model_config import ModelReference
from ..runtime_errors import IncompatibleProviderCapabilitiesError
from .configuration import GenerationOptions
from .errors import UnsupportedProviderCapabilityError
from .messages import ChatMessage
from .results import ProviderResult
from .structured import (
    JsonSchemaDialect,
    SchemaFeature,
    StructuredOutputRequest,
    _schema_keywords,
    _unknown_schema_keywords,
)
from .tools import ProviderToolDefinition, ToolResultMessage, _normalize_provider_tools


@dataclass(frozen=True, slots=True)
class ProviderCallContext:
    """Credential-free deadline and cancellation state visible to adapters."""

    deadline: float | None = None
    cancelled: bool = False


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


def validate_provider_contract(
    provider: ModelProvider,
    *,
    structured_output: StructuredOutputRequest,
    tools: Sequence[ProviderToolDefinition] = (),
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
    required_features = set(structured_output.features) | _schema_keywords(
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
