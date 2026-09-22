"""Deterministic MCP tool naming derived from canonical Conducto identity."""

from __future__ import annotations

from .errors import McpExportError

TOOL_NAME_SEPARATOR = "__"
"""Separator joining the normalized agent and capability name segments."""

_ALLOWED_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def normalize_tool_name(value: str) -> str:
    """Normalize one identifier segment into the exported MCP name alphabet.

    Normalization lowercases the value, replaces every character outside
    ``[a-z0-9]`` with ``_``, collapses repeated underscores, and strips
    leading and trailing underscores.

    Args:
        value: Identifier segment to normalize.

    Returns:
        The normalized segment.

    Raises:
        McpExportError: If normalization produces an empty segment.
    """
    lowered = "".join(
        character if character in _ALLOWED_CHARACTERS else "_" for character in value.lower()
    )
    collapsed: list[str] = []
    for character in lowered:
        if character == "_" and (not collapsed or collapsed[-1] == "_"):
            continue
        collapsed.append(character)
    normalized = "".join(collapsed).strip("_")
    if not normalized:
        raise McpExportError(f"Identifier '{value}' has no exportable MCP name characters")
    return normalized


def default_tool_name(agent_id: str, capability_id: str) -> str:
    """Derive the deterministic default MCP name for one capability.

    Args:
        agent_id: Canonical agent identifier.
        capability_id: Canonical capability identifier.

    Returns:
        The normalized ``<agent>__<capability>`` MCP tool name.

    Raises:
        McpExportError: If either identifier has no exportable characters.
    """
    return (
        f"{normalize_tool_name(agent_id)}{TOOL_NAME_SEPARATOR}{normalize_tool_name(capability_id)}"
    )


__all__ = ["TOOL_NAME_SEPARATOR", "default_tool_name", "normalize_tool_name"]
