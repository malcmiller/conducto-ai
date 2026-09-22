"""Provider-neutral contracts and execution helpers for runtime and adapters."""

from .configuration import (
    GenerationOptions,
    ModelConfiguration,
)
from .decisions import (
    ModelDecision,
    RoutingSelection,
    TerminalModelDecision,
    ToolCallModelDecision,
    build_routing_schema,
    build_terminal_output_request,
    parse_model_decision,
    parse_routing_selection,
)
from .errors import (
    AcceptanceState,
    MalformedStructuredOutputError,
    ProviderAuthenticationError,
    ProviderCancellationError,
    ProviderContentPolicyError,
    ProviderDiagnostic,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderFailureCategory,
    ProviderRateLimitError,
    ProviderTimeoutError,
    UnsupportedProviderCapabilityError,
)
from .execution import (
    complete_with_retries,
)
from .messages import (
    ChatMessage,
    MessageContentPart,
)
from .protocol import (
    AsynchronouslyClosableProvider,
    CancellableModelProvider,
    ModelProvider,
    ProviderCallContext,
    ProviderCapabilities,
    SynchronouslyClosableProvider,
    validate_provider_capabilities,
    validate_provider_contract,
)
from .results import (
    FinishReason,
    ProviderResult,
    Usage,
)
from .structured import (
    JsonSchemaDialect,
    SchemaFeature,
    StructuredOutputRequest,
    validate_structured_output,
)
from .tools import (
    ProviderToolCall,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    ToolResultMessage,
)

__all__ = [
    "AcceptanceState",
    "AsynchronouslyClosableProvider",
    "CancellableModelProvider",
    "ChatMessage",
    "FinishReason",
    "GenerationOptions",
    "JsonSchemaDialect",
    "MalformedStructuredOutputError",
    "MessageContentPart",
    "ModelConfiguration",
    "ModelDecision",
    "ModelProvider",
    "ProviderAuthenticationError",
    "ProviderCallContext",
    "ProviderCancellationError",
    "ProviderCapabilities",
    "ProviderContentPolicyError",
    "ProviderDiagnostic",
    "ProviderEndpointUnavailableError",
    "ProviderError",
    "ProviderFailureCategory",
    "ProviderRateLimitError",
    "ProviderResult",
    "ProviderTimeoutError",
    "ProviderToolCall",
    "ProviderToolCallRequest",
    "ProviderToolDefinition",
    "RoutingSelection",
    "SchemaFeature",
    "StructuredOutputRequest",
    "SynchronouslyClosableProvider",
    "TerminalModelDecision",
    "ToolCallModelDecision",
    "ToolResultMessage",
    "UnsupportedProviderCapabilityError",
    "Usage",
    "build_terminal_output_request",
    "build_routing_schema",
    "complete_with_retries",
    "parse_model_decision",
    "parse_routing_selection",
    "validate_provider_capabilities",
    "validate_provider_contract",
    "validate_structured_output",
]
