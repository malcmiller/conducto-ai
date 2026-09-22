"""Typed failures raised by the Conducto MCP export adapter."""

from __future__ import annotations


class McpExportError(ValueError):
    """Base failure raised while constructing an MCP export surface."""


class McpPolicyError(McpExportError):
    """Raised when an export policy is malformed or references unknown targets."""


class McpSchemaProjectionError(McpExportError):
    """Raised when a canonical Conducto schema is outside the exported subset."""


class McpNameCollisionError(McpExportError):
    """Raised when two admitted capabilities normalize to one MCP tool name."""


class McpToolNotFoundError(LookupError):
    """Raised when a requested MCP tool name is not exported by this server."""


class McpServerStateError(RuntimeError):
    """Raised when a server operation is requested in an incompatible state."""


class McpDependencyError(RuntimeError):
    """Raised when the optional official MCP SDK dependency is unavailable."""

    def __init__(self, message: str, *, extra: str) -> None:
        """Describe the missing dependency and the extra that installs it.

        Args:
            message: Actionable description of the missing or incompatible SDK.
            extra: Conducto package extra that installs the official MCP SDK.
        """
        super().__init__(message)
        self.extra = extra


__all__ = [
    "McpDependencyError",
    "McpExportError",
    "McpNameCollisionError",
    "McpPolicyError",
    "McpSchemaProjectionError",
    "McpServerStateError",
    "McpToolNotFoundError",
]
