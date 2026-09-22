"""First-party optional provider adapters.

Importing this package does not import optional provider SDKs, read
credentials, or contact provider endpoints. Applications select concrete
adapters explicitly and install their matching package extras.
"""

from .ollama import (
    OLLAMA_DEFAULT_PROFILE,
    OLLAMA_TOOL_CAPABLE_PROFILE,
    OllamaConfigurationError,
    OllamaIncompatibleVersionError,
    OllamaModelNotFoundError,
    OllamaOversizedResponseError,
    OllamaProfile,
    OllamaProvider,
    OllamaProviderFactory,
    OllamaReadinessError,
    ollama_provider_factory,
)

__all__ = [
    "OLLAMA_DEFAULT_PROFILE",
    "OLLAMA_TOOL_CAPABLE_PROFILE",
    "OllamaConfigurationError",
    "OllamaIncompatibleVersionError",
    "OllamaModelNotFoundError",
    "OllamaOversizedResponseError",
    "OllamaProfile",
    "OllamaProvider",
    "OllamaProviderFactory",
    "OllamaReadinessError",
    "ollama_provider_factory",
]
