"""Conservative input/output compatibility for the supported JSON Schema subset."""

from collections.abc import Mapping
from typing import Any

from ..gateway_models import canonical_json


class _UnsupportedSchemaError(ValueError):
    """Internal signal for a schema outside the compatibility subset."""


def _schema_compatible(
    requested: Mapping[str, Any] | None,
    offered: Mapping[str, Any] | None,
    *,
    output: bool,
) -> bool:
    if offered is not None:
        _validate_compatibility_schema(offered)
    if requested is None:
        return True
    if offered is None:
        return False
    _validate_compatibility_schema(requested)
    return _schema_subsumes(requested, offered, output=output)


_SCHEMA_ANNOTATIONS = frozenset(
    {"$comment", "$schema", "default", "description", "examples", "title"}
)
_SCHEMA_STRUCTURAL_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
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
        "oneOf",
        "pattern",
        "prefixItems",
        "properties",
        "required",
        "type",
        "uniqueItems",
    }
)
_EXACT_CONSTRAINTS = frozenset(
    {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "uniqueItems",
    }
)


def _validate_compatibility_schema(schema: Mapping[str, Any]) -> None:
    unsupported = set(schema) - _SCHEMA_ANNOTATIONS - _SCHEMA_STRUCTURAL_KEYS
    if unsupported:
        raise _UnsupportedSchemaError(
            "Unsupported JSON Schema keyword(s): " + ", ".join(sorted(unsupported))
        )
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise _UnsupportedSchemaError("JSON Schema properties must be an object")
    for child in properties.values():
        if not isinstance(child, Mapping):
            raise _UnsupportedSchemaError("JSON Schema property definitions must be objects")
        _validate_compatibility_schema(child)
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, Mapping):
        raise _UnsupportedSchemaError("JSON Schema $defs must be an object")
    for child in definitions.values():
        if not isinstance(child, Mapping):
            raise _UnsupportedSchemaError("JSON Schema definitions must be objects")
        _validate_compatibility_schema(child)
    for keyword in ("items", "additionalProperties"):
        child = schema.get(keyword)
        if child is not None and not isinstance(child, bool | Mapping):
            raise _UnsupportedSchemaError(f"JSON Schema {keyword} must be boolean or an object")
        if isinstance(child, Mapping):
            _validate_compatibility_schema(child)
    prefix_items = schema.get("prefixItems", ())
    if not isinstance(prefix_items, (list, tuple)):
        raise _UnsupportedSchemaError("JSON Schema prefixItems must be an array")
    for child in prefix_items:
        if not isinstance(child, Mapping):
            raise _UnsupportedSchemaError("JSON Schema prefixItems must contain objects")
        _validate_compatibility_schema(child)
    for keyword in ("allOf", "anyOf", "oneOf"):
        alternatives = schema.get(keyword, ())
        if not isinstance(alternatives, (list, tuple)):
            raise _UnsupportedSchemaError(f"JSON Schema {keyword} must be an array")
        for child in alternatives:
            if not isinstance(child, Mapping):
                raise _UnsupportedSchemaError(f"JSON Schema {keyword} must contain objects")
            _validate_compatibility_schema(child)
    required = schema.get("required", ())
    if not isinstance(required, (list, tuple)) or not all(
        isinstance(item, str) for item in required
    ):
        raise _UnsupportedSchemaError("JSON Schema required must be an array of strings")


def _schema_subsumes(
    requested: Mapping[str, Any],
    offered: Mapping[str, Any],
    *,
    output: bool,
) -> bool:
    requested_core = {
        key: value for key, value in requested.items() if key not in _SCHEMA_ANNOTATIONS
    }
    offered_core = {key: value for key, value in offered.items() if key not in _SCHEMA_ANNOTATIONS}
    if canonical_json(requested_core) == canonical_json(offered_core):
        return True
    if any(
        keyword in requested_core or keyword in offered_core
        for keyword in ("$defs", "$ref", "allOf", "anyOf", "oneOf", "prefixItems")
    ):
        return False
    requested_type = requested.get("type")
    offered_type = offered.get("type")
    if requested_type is not None and offered_type != requested_type:
        return False
    if not output and requested_type is None and offered_type is not None:
        return False
    requested_values = _schema_values(requested)
    offered_values = _schema_values(offered)
    if requested_values is not None:
        if offered_values is None:
            return False
        if output and not offered_values.issubset(requested_values):
            return False
        if not output and not requested_values.issubset(offered_values):
            return False
    elif offered_values is not None and not output:
        return False
    for keyword in _EXACT_CONSTRAINTS:
        if output and keyword not in requested:
            continue
        if requested.get(keyword) != offered.get(keyword):
            return False
    requested_properties = requested.get("properties")
    offered_properties = offered.get("properties")
    if isinstance(requested_properties, Mapping):
        if not isinstance(offered_properties, Mapping):
            return False
        property_names = (
            requested_properties if output else set(requested_properties) | set(offered_properties)
        )
        for name in property_names:
            requested_schema = requested_properties.get(name)
            offered_schema = offered_properties.get(name)
            if requested_schema is None:
                if output:
                    continue
                if requested.get("additionalProperties", True) is False:
                    continue
                if offered.get("additionalProperties", True) is False:
                    return False
                continue
            if offered_schema is None:
                if offered.get("additionalProperties", True) is False:
                    return False
                continue
            if not isinstance(requested_schema, Mapping) or not isinstance(offered_schema, Mapping):
                return False
            if not _schema_subsumes(requested_schema, offered_schema, output=output):
                return False
    requested_required = frozenset(requested.get("required", ()))
    offered_required = frozenset(offered.get("required", ()))
    if output and not requested_required.issubset(offered_required):
        return False
    if not output and not offered_required.issubset(requested_required):
        return False
    requested_additional = requested.get("additionalProperties", True)
    offered_additional = offered.get("additionalProperties", True)
    if output and requested_additional is False and offered_additional is not False:
        return False
    if not output and requested_additional is not False and offered_additional is False:
        return False
    if isinstance(requested_additional, Mapping):
        if not isinstance(offered_additional, Mapping):
            return False
        if not _schema_subsumes(requested_additional, offered_additional, output=output):
            return False
    requested_items = requested.get("items")
    offered_items = offered.get("items")
    if isinstance(requested_items, Mapping):
        if not isinstance(offered_items, Mapping):
            return False
        if not _schema_subsumes(requested_items, offered_items, output=output):
            return False
    elif requested_items is not None and requested_items != offered_items:
        return False
    elif not output and requested_items is None and offered_items is not None:
        return False
    return True


def _schema_values(schema: Mapping[str, Any]) -> frozenset[str] | None:
    if "const" in schema:
        return frozenset({canonical_json(schema["const"])})
    values = schema.get("enum")
    if isinstance(values, (list, tuple)):
        return frozenset(canonical_json(value) for value in values)
    return None
