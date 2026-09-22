"""Tested JSON Schema subset projected from canonical Conducto schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from conducto.core.gateway_models import canonical_json

from .errors import McpSchemaProjectionError

SUPPORTED_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
"""JSON Schema primitive types projected into MCP tool schemas."""

SUPPORTED_KEYWORDS = frozenset(
    {
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "default",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "properties",
        "propertyNames",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)
"""JSON Schema keywords accepted by the exported subset.

Notes:
    ``$defs`` is accepted only at a schema root and is resolved away during
    projection. Every other keyword, including ``allOf``, ``oneOf``, ``not``,
    ``if``/``then``/``else``, and ``patternProperties``, is rejected instead of
    being silently dropped.
"""

RESULT_PROPERTY = "result"
"""Property holding a capability's serialized value in structured content."""


def project_input_schema(
    schema: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Project a canonical capability input schema into the exported subset.

    Args:
        schema: Canonical Conducto input schema for one capability.
        label: Diagnostic label identifying the capability being exported.
        max_bytes: Maximum canonical JSON size accepted for the projection.

    Returns:
        A self-contained JSON Schema object with references resolved.

    Raises:
        McpSchemaProjectionError: If the schema leaves the supported subset or
            exceeds ``max_bytes``.
    """
    projected = _project(schema, definitions=_definitions(schema, label), label=label, stack=())
    if projected.get("type") != "object":
        raise McpSchemaProjectionError(f"{label}: MCP input schemas must describe an object")
    _check_size(projected, label=label, max_bytes=max_bytes)
    return projected


def project_output_schema(
    schema: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Wrap a canonical capability return schema as MCP structured content.

    Args:
        schema: Canonical Conducto return schema for one capability.
        label: Diagnostic label identifying the capability being exported.
        max_bytes: Maximum canonical JSON size accepted for the projection.

    Returns:
        An object schema whose ``result`` property holds the capability value.

    Raises:
        McpSchemaProjectionError: If the schema leaves the supported subset or
            exceeds ``max_bytes``.
    """
    projected = {
        "type": "object",
        "properties": {
            RESULT_PROPERTY: _project(
                schema,
                definitions=_definitions(schema, label),
                label=label,
                stack=(),
            )
        },
        "required": [RESULT_PROPERTY],
        "additionalProperties": False,
    }
    _check_size(projected, label=label, max_bytes=max_bytes)
    return projected


def _definitions(schema: Mapping[str, Any], label: str) -> Mapping[str, Mapping[str, Any]]:
    """Return the root ``$defs`` block used to resolve references."""
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, Mapping) or any(
        not isinstance(value, Mapping) for value in definitions.values()
    ):
        raise McpSchemaProjectionError(f"{label}: $defs must map names to schema objects")
    return {str(name): value for name, value in definitions.items()}


def _project(
    node: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    label: str,
    stack: tuple[str, ...],
) -> dict[str, Any]:
    """Recursively validate and resolve one schema node."""
    if not isinstance(node, Mapping):
        raise McpSchemaProjectionError(f"{label}: schema nodes must be JSON objects")
    unsupported = sorted(set(node) - SUPPORTED_KEYWORDS - {"$defs"})
    if unsupported:
        raise McpSchemaProjectionError(
            f"{label}: unsupported JSON Schema keyword(s) {', '.join(unsupported)}"
        )
    if "$ref" in node:
        return _resolve_reference(node, definitions=definitions, label=label, stack=stack)

    projected: dict[str, Any] = {}
    for keyword, value in sorted(node.items()):
        if keyword == "$defs":
            continue
        projected[keyword] = _project_keyword(
            keyword,
            value,
            definitions=definitions,
            label=label,
            stack=stack,
        )
    _validate_node(projected, label=label)
    return projected


def _project_keyword(
    keyword: str,
    value: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    label: str,
    stack: tuple[str, ...],
) -> Any:
    """Project one keyword value, recursing into nested schema positions."""
    if keyword == "properties":
        if not isinstance(value, Mapping):
            raise McpSchemaProjectionError(f"{label}: properties must be a JSON object")
        return {
            str(name): _project(item, definitions=definitions, label=label, stack=stack)
            for name, item in sorted(value.items())
        }
    if keyword == "items":
        return _project(value, definitions=definitions, label=label, stack=stack)
    if keyword == "additionalProperties":
        if isinstance(value, bool):
            return value
        return _project(value, definitions=definitions, label=label, stack=stack)
    if keyword == "propertyNames":
        projected = _project(value, definitions=definitions, label=label, stack=stack)
        if projected != {"type": "string"}:
            raise McpSchemaProjectionError(
                f"{label}: mapping keys must be strings in exported MCP schemas"
            )
        return projected
    if keyword == "anyOf":
        return _project_any_of(value, definitions=definitions, label=label, stack=stack)
    if keyword == "required":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise McpSchemaProjectionError(f"{label}: required must be a list of property names")
        return [str(item) for item in value]
    if keyword == "enum":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            raise McpSchemaProjectionError(f"{label}: enum must be a non-empty list")
        return list(value)
    return value


def _project_any_of(
    value: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    label: str,
    stack: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Project an ``anyOf`` that expresses at most one optional branch."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise McpSchemaProjectionError(f"{label}: anyOf must be a non-empty list of schemas")
    branches = [
        _project(branch, definitions=definitions, label=label, stack=stack) for branch in value
    ]
    concrete = [branch for branch in branches if branch.get("type") != "null"]
    if len(concrete) > 1:
        raise McpSchemaProjectionError(
            f"{label}: ambiguous union types are not supported by the MCP export subset"
        )
    return branches


def _resolve_reference(
    node: Mapping[str, Any],
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    label: str,
    stack: tuple[str, ...],
) -> dict[str, Any]:
    """Inline one local ``$defs`` reference, rejecting recursive definitions."""
    if set(node) - {"$ref"}:
        raise McpSchemaProjectionError(f"{label}: $ref cannot be combined with other keywords")
    reference = node["$ref"]
    prefix = "#/$defs/"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        raise McpSchemaProjectionError(f"{label}: only local #/$defs references are supported")
    name = reference[len(prefix) :]
    if name not in definitions:
        raise McpSchemaProjectionError(f"{label}: unresolved schema reference '{reference}'")
    if name in stack:
        raise McpSchemaProjectionError(
            f"{label}: recursive schema reference '{reference}' is not supported"
        )
    return _project(
        definitions[name],
        definitions=definitions,
        label=label,
        stack=(*stack, name),
    )


def _validate_node(node: Mapping[str, Any], *, label: str) -> None:
    """Validate the declared type of projected node."""
    declared = node.get("type")
    if declared is None:
        if "anyOf" in node or "enum" in node or "const" in node:
            return
        raise McpSchemaProjectionError(f"{label}: schema nodes require a supported 'type'")
    if not isinstance(declared, str) or declared not in SUPPORTED_TYPES:
        raise McpSchemaProjectionError(f"{label}: unsupported schema type {declared!r}")


def _check_size(schema: Mapping[str, Any], *, label: str, max_bytes: int) -> None:
    """Enforce the configured canonical schema size bound."""
    encoded = canonical_json(schema).encode("utf-8")
    if len(encoded) > max_bytes:
        raise McpSchemaProjectionError(
            f"{label}: projected schema is {len(encoded)} bytes and exceeds the "
            f"configured {max_bytes} byte bound"
        )


__all__ = [
    "RESULT_PROPERTY",
    "SUPPORTED_KEYWORDS",
    "SUPPORTED_TYPES",
    "project_input_schema",
    "project_output_schema",
]
