"""Typed failures raised by the optional Conducto A2A ASGI server adapter."""

from __future__ import annotations


class A2AServerError(RuntimeError):
    """Base failure raised by the optional A2A ASGI server adapter."""


class A2ADependencyError(A2AServerError):
    """Raised when an optional A2A ASGI server dependency is unavailable."""

    def __init__(self, message: str, *, extra: str) -> None:
        """Describe the missing dependency and the extra that installs it.

        Args:
            message: Actionable description of the missing or incompatible dependency.
            extra: Conducto package extra that installs the missing dependency.
        """
        super().__init__(message)
        self.extra = extra


class A2AHostConfigurationError(A2AServerError, ValueError):
    """Raised when an A2A host security or limit configuration is invalid.

    Notes:
        This failure is raised during construction, never while serving a
        request, so an invalid deployment cannot start and then silently accept
        unbounded or untrusted traffic.
    """


class A2ARequestRejectedError(A2AServerError):
    """Raised when the hardening boundary refuses a request before execution.

    Attributes:
        reason: Stable, credential-free reason code safe for logs and responses.
        status_code: HTTP status the ASGI boundary emits for this rejection.
    """

    def __init__(self, reason: str, *, status_code: int) -> None:
        """Record a safe rejection reason code and its HTTP status.

        Args:
            reason: Stable reason code. It must never contain caller-supplied
                values, credentials, or exception text.
            status_code: HTTP status returned to the caller.
        """
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


class A2APayloadLimitError(A2AServerError):
    """Raised when an accepted payload exceeds a configured A2A profile limit.

    Attributes:
        reason: Stable, credential-free reason code identifying the limit.
    """

    def __init__(self, reason: str) -> None:
        """Record the violated limit as a stable reason code.

        Args:
            reason: Stable reason code naming the exceeded limit.
        """
        super().__init__(reason)
        self.reason = reason


class A2AStartupError(A2AServerError):
    """Raised when bounded A2A host startup fails or exceeds its deadline.

    Attributes:
        reason: Stable, credential-free reason code for the failed startup.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        """Record why startup did not complete without exposing dependency detail.

        Args:
            message: Stable description of the incomplete startup.
            reason: Stable reason code naming the startup failure mode.
        """
        super().__init__(message)
        self.reason = reason


class A2AShutdownError(A2AServerError):
    """Raised when bounded A2A host shutdown cleanup does not fully succeed.

    Attributes:
        reasons: Ordered, credential-free reason codes for each failed step.
    """

    def __init__(self, message: str, *, reasons: tuple[str, ...]) -> None:
        """Aggregate every failed cleanup step without hiding the failure.

        Args:
            message: Stable description of the incomplete shutdown.
            reasons: Ordered reason codes naming each cleanup step that failed.
        """
        super().__init__(message)
        self.reasons = reasons


__all__ = [
    "A2ADependencyError",
    "A2AHostConfigurationError",
    "A2APayloadLimitError",
    "A2ARequestRejectedError",
    "A2AServerError",
    "A2AShutdownError",
    "A2AStartupError",
]
