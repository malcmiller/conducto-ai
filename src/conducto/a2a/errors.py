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


__all__ = ["A2ADependencyError", "A2AServerError"]
