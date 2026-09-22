"""Validate model-selected arguments before resolving a child execution."""

import math
import operator
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from ..gateway_models import thaw_json


def _arguments_match_schema(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> bool:
    """Validate detached JSON values without mutating immutable provider arguments."""
    document = thaw_json(schema)
    return _matches_schema(thaw_json(arguments), document, document)


def _matches_number(value: int | float, schema: Mapping[str, Any]) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return False
    for keyword, violates in (
        ("minimum", operator.lt),
        ("maximum", operator.gt),
        ("exclusiveMinimum", operator.le),
        ("exclusiveMaximum", operator.ge),
    ):
        if keyword in schema:
            bound = schema[keyword]
            if isinstance(bound, bool) or not isinstance(bound, int | float):
                return False
            if isinstance(bound, float) and not math.isfinite(bound):
                return False
            if violates(value, bound):
                return False
    if "multipleOf" in schema:
        multiple = schema["multipleOf"]
        if isinstance(multiple, bool) or not isinstance(multiple, int | float) or multiple <= 0:
            return False
        if isinstance(multiple, float) and not math.isfinite(multiple):
            return False
        # JSON decimal multiples must not inherit binary float remainder error.
        if Fraction(str(value)) % Fraction(str(multiple)):
            return False
    return True


def _matches_schema(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if not reference.startswith("#/$defs/"):
            return False
        definition = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        return isinstance(definition, Mapping) and _matches_schema(value, definition, root)
    if "const" in schema and value != schema["const"]:
        return False
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, str) and value not in enum:
        return False
    for keyword in ("allOf",):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, str):
            if not all(
                isinstance(branch, Mapping) and _matches_schema(value, branch, root)
                for branch in branches
            ):
                return False
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, str):
            matches = sum(
                isinstance(branch, Mapping) and _matches_schema(value, branch, root)
                for branch in branches
            )
            if matches < 1 or (keyword == "oneOf" and matches != 1):
                return False

    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_matches_schema(value, {**schema, "type": item}, root) for item in expected)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in ("integer", "number"):
        number_type = int if expected == "integer" else (int, float)
        if isinstance(value, bool) or not isinstance(value, number_type):
            return False
        return _matches_number(value, schema)
    if expected == "string":
        if not isinstance(value, str):
            return False
        if len(value) < int(schema.get("minLength", 0)):
            return False
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        pattern = schema.get("pattern")
        return not isinstance(pattern, str) or re.search(pattern, value) is not None
    if expected == "array":
        if not isinstance(value, list):
            return False
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        items = schema.get("items")
        return not isinstance(items, Mapping) or all(
            _matches_schema(item, items, root) for item in value
        )
    if expected == "object" or "properties" in schema:
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return False
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, str):
            return False
        if any(name not in value for name in required):
            return False
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            property_schema = properties.get(name)
            if isinstance(property_schema, Mapping):
                if not _matches_schema(item, property_schema, root):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, Mapping) and not _matches_schema(item, additional, root):
                return False
        return True
    if expected is None:
        return True
    return False
