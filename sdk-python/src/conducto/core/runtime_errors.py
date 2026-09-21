"""Stable public errors raised by the Conducto runtime."""


class ConductoError(RuntimeError):
    """Base error for stable Conducto runtime failures."""


class NoActiveRunContextError(ConductoError):
    """Model-backed code was called without an active runtime invocation."""


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
