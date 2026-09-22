"""Typed provider failures, acceptance state, and safe diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import Usage


class AcceptanceState(StrEnum):
    """Whether a provider accepted work for processing."""

    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED_NOT_ACCEPTED = "attempted_not_accepted"
    ACCEPTED = "accepted"


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
