"""Typed failures raised at the transport boundary."""


class TransportError(RuntimeError):
    """Base class for failures before a remote capability is invoked."""


class DiscoveryError(TransportError):
    """Raised when a remote Agent Card cannot be retrieved safely."""


class CompatibilityError(DiscoveryError):
    """Raised when a card does not implement Conducto's pinned A2A profile."""


class ProtocolError(TransportError):
    """Raised when a remote A2A payload violates the pinned protocol contract."""


class LimitExceededError(TransportError):
    """Raised when an input or response exceeds a configured transport limit."""


class RemoteTaskError(TransportError):
    """Raised when a requested remote task is missing or cannot transition."""


class TLSConfigurationError(TransportError):
    """Raised when application-owned TLS/mTLS configuration is invalid."""


class AuthenticationError(TransportError):
    """Raised when incoming or outgoing request authentication fails."""
