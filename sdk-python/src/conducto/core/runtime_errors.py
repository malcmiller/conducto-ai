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
