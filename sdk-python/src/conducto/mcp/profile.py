"""Pinned official MCP Python SDK profile and optional-dependency checks."""

from __future__ import annotations

from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .errors import McpDependencyError

MCP_EXTRA = "mcp"
"""Conducto package extra that installs the official MCP Python SDK."""

MCP_PYTHON_SDK_PACKAGE = "mcp"
"""Distribution name of the official MCP Python SDK."""

MCP_PYTHON_SDK_VERSION = "2.2.0"
"""Pinned official MCP Python SDK release supported by this adapter."""

MCP_PYTHON_SDK_SPECIFIER = f"=={MCP_PYTHON_SDK_VERSION}"
"""Version specifier applied to the installed official MCP Python SDK."""

MCP_PROTOCOL_VERSION = "2026-07-28"
"""Latest MCP protocol revision supported by the pinned SDK release.

Notes:
    The official SDK negotiates the effective revision during ``initialize``.
    Conducto neither overrides nor reimplements that negotiation.
"""

MAX_TOOL_NAME_LENGTH = 128
"""Maximum length of an exported MCP tool name."""

MAX_TOOL_DESCRIPTION_LENGTH = 4096
"""Maximum length of an exported, untrusted MCP tool description."""

MAX_TOOL_SCHEMA_BYTES = 65_536
"""Maximum canonical JSON size of one exported MCP tool schema."""

MAX_EXPORTED_TOOLS = 256
"""Maximum number of tools one MCP server instance may export."""


def conducto_server_version() -> str:
    """Return the installed Conducto distribution version advertised by servers.

    Returns:
        The installed ``conducto-ai`` version, or ``"0"`` when Conducto runs
        from a source tree without installed distribution metadata.
    """
    try:
        return metadata.version("conducto-ai")
    except metadata.PackageNotFoundError:
        return "0"


def require_mcp_dependency() -> str:
    """Validate that the pinned official MCP Python SDK is importable.

    Returns:
        The installed official MCP Python SDK version.

    Raises:
        McpDependencyError: If the SDK is absent or outside the pinned range.
    """
    try:
        installed = metadata.version(MCP_PYTHON_SDK_PACKAGE)
        version = Version(installed)
    except (metadata.PackageNotFoundError, InvalidVersion) as error:
        raise McpDependencyError(
            f"The MCP export adapter requires {MCP_PYTHON_SDK_PACKAGE}"
            f"{MCP_PYTHON_SDK_SPECIFIER}; install conducto-ai[{MCP_EXTRA}]",
            extra=MCP_EXTRA,
        ) from error
    if version not in SpecifierSet(MCP_PYTHON_SDK_SPECIFIER):
        raise McpDependencyError(
            f"The MCP export adapter requires {MCP_PYTHON_SDK_PACKAGE}"
            f"{MCP_PYTHON_SDK_SPECIFIER}; install conducto-ai[{MCP_EXTRA}]",
            extra=MCP_EXTRA,
        )
    return installed


__all__ = [
    "MAX_EXPORTED_TOOLS",
    "MAX_TOOL_DESCRIPTION_LENGTH",
    "MAX_TOOL_NAME_LENGTH",
    "MAX_TOOL_SCHEMA_BYTES",
    "MCP_EXTRA",
    "MCP_PROTOCOL_VERSION",
    "MCP_PYTHON_SDK_PACKAGE",
    "MCP_PYTHON_SDK_SPECIFIER",
    "MCP_PYTHON_SDK_VERSION",
    "conducto_server_version",
    "require_mcp_dependency",
]
