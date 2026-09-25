"""Provider-neutral failures raised while invoking a capability."""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn

from .provider.errors import MalformedStructuredOutputError, ProviderError
from .runtime_errors import ConductoError


class CapabilityFailureStage(StrEnum):
    """Stage at which a capability invocation failed."""

    INPUT = "input"
    OUTPUT = "output"
    TRANSPORT = "transport"


class CapabilityFailureCode(StrEnum):
    """Stable public classifications for capability failures."""

    INVALID_INPUT = "invalid_capability_input"
    MALFORMED_OUTPUT = "malformed_model_output"
    OUTPUT_INVARIANT = "unsatisfied_output_invariant"
    TRANSPORT = "capability_transport_failure"


class CapabilityError(ConductoError):
    """Base class for provider-neutral capability invocation failures.

    Args:
        capability: Public capability identifier associated with the failure.
        message: Safe detail that does not include provider diagnostics.
    """

    code: CapabilityFailureCode
    stage: CapabilityFailureStage

    def __init__(self, capability: str, message: str) -> None:
        super().__init__(message)
        self.capability = capability


class InvalidCapabilityInputError(CapabilityError, ValueError):
    """Capability arguments do not conform to the published input contract."""

    code = CapabilityFailureCode.INVALID_INPUT
    stage = CapabilityFailureStage.INPUT

    def __init__(self, capability: str, message: str = "Capability input is invalid") -> None:
        super().__init__(capability, f"Capability '{capability}' input is invalid: {message}")


class MalformedModelOutputError(CapabilityError):
    """Model output is absent, malformed, or does not conform to its schema."""

    code = CapabilityFailureCode.MALFORMED_OUTPUT
    stage = CapabilityFailureStage.OUTPUT

    def __init__(self, capability: str, message: str = "Model output is malformed") -> None:
        super().__init__(capability, f"Capability '{capability}' output is malformed: {message}")


class OutputInvariantError(CapabilityError, ValueError):
    """A capability output violates an invariant beyond schema conformance."""

    code = CapabilityFailureCode.OUTPUT_INVARIANT
    stage = CapabilityFailureStage.OUTPUT

    def __init__(self, capability: str, message: str = "Output invariant is unsatisfied") -> None:
        super().__init__(
            capability, f"Capability '{capability}' output invariant failed: {message}"
        )


class CapabilityTransportError(CapabilityError):
    """A model transport failed without exposing provider-specific diagnostics."""

    code = CapabilityFailureCode.TRANSPORT
    stage = CapabilityFailureStage.TRANSPORT

    def __init__(self, capability: str) -> None:
        super().__init__(capability, f"Capability '{capability}' model transport failed")


def map_provider_error(error: BaseException, capability: str) -> CapabilityError | None:
    """Map a provider failure to the capability taxonomy.

    Args:
        error: Provider or structured-output exception observed at the runtime
            boundary.
        capability: Public capability identifier for the active invocation.

    Returns:
        A provider-neutral capability error, or ``None`` when the exception is
        unrelated to provider-backed capability execution.
    """
    if isinstance(error, MalformedStructuredOutputError):
        return MalformedModelOutputError(capability)
    if isinstance(error, ProviderError):
        return CapabilityTransportError(capability)
    return None


def require_capability_error(error: BaseException, capability: str) -> CapabilityError:
    """Return an existing taxonomy error or map a provider error.

    Args:
        error: Exception raised during capability execution.
        capability: Public capability identifier for the active invocation.

    Returns:
        A typed, provider-neutral capability error.

    Raises:
        TypeError: If ``error`` is not a supported capability failure.
    """
    if isinstance(error, CapabilityError):
        return error
    mapped = map_provider_error(error, capability)
    if mapped is not None:
        return mapped
    raise TypeError("exception is not a capability failure")


def raise_invalid_input(capability: str, message: str) -> NoReturn:
    """Raise an invalid-input failure for small adapter boundaries."""
    raise InvalidCapabilityInputError(capability, message)


__all__ = [
    "CapabilityError",
    "CapabilityFailureCode",
    "CapabilityFailureStage",
    "CapabilityTransportError",
    "InvalidCapabilityInputError",
    "MalformedModelOutputError",
    "OutputInvariantError",
    "map_provider_error",
    "require_capability_error",
]
