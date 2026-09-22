"""Optional MCP projection of canonical Conducto capabilities.

Importing this package does not import the official MCP Python SDK, construct a
server, mutate the environment, launch a subprocess, or perform I/O. The stdio
server adapter is resolved lazily and requires the ``mcp`` package extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import (
    McpDependencyError,
    McpExportError,
    McpNameCollisionError,
    McpPolicyError,
    McpSchemaProjectionError,
    McpServerStateError,
    McpToolNotFoundError,
)
from .export import McpToolDefinition, McpToolExporter
from .mapping import SUCCESS_REASON_CODE, McpToolOutcome, invocation_result_to_tool_outcome
from .naming import TOOL_NAME_SEPARATOR, default_tool_name, normalize_tool_name
from .policy import McpCapabilityQuery, McpExportPolicy, McpExportRule
from .profile import (
    MAX_EXPORTED_TOOLS,
    MAX_TOOL_DESCRIPTION_LENGTH,
    MAX_TOOL_NAME_LENGTH,
    MAX_TOOL_SCHEMA_BYTES,
    MCP_EXTRA,
    MCP_PROTOCOL_VERSION,
    MCP_PYTHON_SDK_PACKAGE,
    MCP_PYTHON_SDK_SPECIFIER,
    MCP_PYTHON_SDK_VERSION,
    conducto_server_version,
    require_mcp_dependency,
)
from .schema import (
    RESULT_PROPERTY,
    SUPPORTED_KEYWORDS,
    SUPPORTED_TYPES,
    project_input_schema,
    project_output_schema,
)

if TYPE_CHECKING:
    from .server import McpStdioServer

_LAZY_EXPORTS = {"McpStdioServer": "conducto.mcp.server"}


def __getattr__(name: str) -> Any:
    """Resolve the SDK-backed server adapter only when it is requested.

    Args:
        name: Attribute requested from this package.

    Returns:
        The lazily imported attribute.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
        McpDependencyError: If the official MCP Python SDK is unavailable.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


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
    "McpCapabilityQuery",
    "McpDependencyError",
    "McpExportError",
    "McpExportPolicy",
    "McpExportRule",
    "McpNameCollisionError",
    "McpPolicyError",
    "McpSchemaProjectionError",
    "McpServerStateError",
    "McpStdioServer",
    "McpToolDefinition",
    "McpToolExporter",
    "McpToolNotFoundError",
    "McpToolOutcome",
    "RESULT_PROPERTY",
    "SUCCESS_REASON_CODE",
    "SUPPORTED_KEYWORDS",
    "SUPPORTED_TYPES",
    "TOOL_NAME_SEPARATOR",
    "conducto_server_version",
    "default_tool_name",
    "invocation_result_to_tool_outcome",
    "normalize_tool_name",
    "project_input_schema",
    "project_output_schema",
    "require_mcp_dependency",
]
