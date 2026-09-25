"""Stable public errors raised by the Conducto runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider_registry import ProviderCleanupReport


class ConductoError(RuntimeError):
    """Base error for stable Conducto runtime failures."""


class NoActiveRunContextError(ConductoError):
    """Model-backed code was called without an active runtime invocation."""


class UntrustedSystemMessageError(ConductoError, ValueError):
    """A capability supplied a ``system``-role message to a runtime model call.

    The framework composes the sole system-role message from the resolved,
    trusted instruction chain (runtime policy, then agent, then capability
    instructions) so that callers cannot inject, replace, or suppress those
    instructions. Capability code must not construct its own ``system``-role
    :class:`~conducto.core.provider.ChatMessage`; declare persona or
    behavioral text through ``@a2a_agent``/``@a2a_capability`` ``instructions``
    instead.
    """


class ModelResolutionError(ConductoError):
    """Base error raised before model-backed work begins."""


class MissingModelDefaultError(ModelResolutionError):
    """No override or default supplied a model for model-backed work."""


class UnknownModelReferenceError(ModelResolutionError):
    """The selected model reference is not registered with the runtime."""


class IncompatibleProviderCapabilitiesError(ModelResolutionError):
    """The selected provider cannot satisfy the requested capabilities."""


class ModelOverrideDeniedError(ModelResolutionError):
    """Runtime policy rejected a model or provider selection."""


class ProviderUnavailableError(ModelResolutionError):
    """The selected provider is registered but currently unavailable."""


class ProviderRegistrationError(ConductoError):
    """Base error raised while registering provider types, factories, or clients."""


class DuplicateProviderTypeError(ProviderRegistrationError, ValueError):
    """A provider type is already registered and replacement was not requested."""


class DuplicateModelReferenceError(ProviderRegistrationError, ValueError):
    """A model reference is already registered, and a replacement was not requested."""


class UnknownProviderTypeError(ProviderRegistrationError):
    """The referenced provider type has no registered factory."""


class ContradictoryProviderConfigurationError(ProviderRegistrationError, ValueError):
    """Connection configuration was supplied alongside a preconstructed client."""


class ProviderTypeMismatchError(ProviderRegistrationError, ValueError):
    """The model configuration's provider does not match the registered provider type."""


class ProviderFactoryValidationError(ProviderRegistrationError, TypeError):
    """The supplied factory does not satisfy the provider factory protocol."""


class ProviderClientValidationError(ProviderRegistrationError, TypeError):
    """The supplied client does not satisfy the structural provider protocol."""


class ProviderConstructionError(ProviderRegistrationError):
    """A registered factory failed to construct a provider client."""


class StaleProviderConstructionError(ProviderRegistrationError):
    """A concurrent replacement or deregistration invalidated an in-flight construction.

    Raised when a provider client finished construction after the target
    model reference was replaced or deregistered by another thread. The
    stale result is discarded instead of silently overwriting the newer
    binding or resurrecting a deregistered reference.
    """


class ProviderOwnershipError(ProviderRegistrationError, ValueError):
    """One client was registered with conflicting lifecycle ownership declarations."""


class RuntimeClosedError(ConductoError):
    """The runtime or its provider registry has begun shutdown."""


class ProviderShutdownError(ConductoError):
    """Runtime-owned provider cleanup completed with one or more failures.

    Attributes:
        report: Safe aggregate cleanup outcome. It contains provider identities
            and local exception types, never provider exception messages.
    """

    def __init__(self, report: ProviderCleanupReport) -> None:
        """Create a cleanup failure with its safe aggregate report."""
        super().__init__("Provider cleanup did not complete successfully")
        self.report = report
