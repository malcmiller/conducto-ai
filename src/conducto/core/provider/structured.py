"""Immutable structured-output requests and supported JSON Schema validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from ._json import _freeze_json, _thaw_json
from .errors import MalformedStructuredOutputError


class JsonSchemaDialect(StrEnum):
    """JSON Schema dialects recognized by the provider contract."""

    DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
    DRAFT_07 = "http://json-schema.org/draft-07/schema#"


class SchemaFeature(StrEnum):
    """Schema features a provider may advertise as natively supported."""

    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    CONST = "const"
    REQUIRED = "required"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"
    MIN_ITEMS = "minItems"
    MAX_ITEMS = "maxItems"
    ONE_OF = "oneOf"
    ANY_OF = "anyOf"
    ALL_OF = "allOf"
    REF = "$ref"


@dataclass(frozen=True)
class StructuredOutputRequest:
    """Immutable, native JSON-schema-constrained output request."""

    name: str
    schema: Mapping[str, Any]
    dialect: JsonSchemaDialect = JsonSchemaDialect.DRAFT_2020_12
    required: bool = True
    features: frozenset[SchemaFeature] = frozenset()

    def __init__(
        self,
        name: str,
        schema: Mapping[str, Any],
        *,
        dialect: JsonSchemaDialect = JsonSchemaDialect.DRAFT_2020_12,
        required: bool = True,
        features: frozenset[SchemaFeature] | set[SchemaFeature] = frozenset(),
    ) -> None:
        if not name.strip():
            raise ValueError("Structured output name cannot be empty")
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("Structured output schema cannot be empty")
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "schema", _freeze_json(schema))
        object.__setattr__(self, "dialect", JsonSchemaDialect(dialect))
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "features", frozenset(features))

    @property
    def json_schema(self) -> Mapping[str, Any]:
        """Return a detached JSON schema copy for provider wire serialization.

        Returns:
            Ordinary JSON dictionaries and lists, independent of the immutable
            request snapshot.
        """
        return cast(Mapping[str, Any], _thaw_json(self.schema))


_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "items",
    "required",
    "additionalProperties",
    "enum",
    "const",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "oneOf",
    "anyOf",
    "allOf",
    "$ref",
    "$defs",
    "$schema",
    "title",
    "description",
    "default",
}


def _schema_keywords(schema: Mapping[str, Any]) -> set[SchemaFeature]:
    """Collect schema features used by a JSON Schema document."""
    found: set[SchemaFeature] = set()
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            try:
                found.add(SchemaFeature(value))
            except ValueError:
                pass
        try:
            feature = SchemaFeature(key)
        except ValueError:
            continue
        found.add(feature)
    for child in _schema_children(schema):
        found.update(_schema_keywords(child))
    return found


def _unknown_schema_keywords(schema: Mapping[str, Any]) -> set[str]:
    """Find validation keywords outside the deliberately supported subset."""
    unknown: set[str] = set()
    for key in schema:
        if key not in _SCHEMA_KEYWORDS and not key.startswith("x-"):
            unknown.add(str(key))
    for child in _schema_children(schema):
        unknown.update(_unknown_schema_keywords(child))
    return unknown


def _schema_children(schema: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Visit schema positions, never literal enum, const, or default payloads."""
    for key, value in schema.items():
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            yield from (child for child in value.values() if isinstance(child, Mapping))
        elif key in {"items", "additionalProperties"} and isinstance(value, Mapping):
            yield value
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, (list, tuple)):
            yield from (child for child in value if isinstance(child, Mapping))


def validate_structured_output(
    value: Any,
    request: StructuredOutputRequest,
) -> None:
    """Validate decoded provider output against the requested schema.

    This deliberately implements the provider-neutral subset instead of
    depending on an optional JSON Schema package.
    """
    root_schema = request.json_schema

    def check(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
        if "$ref" in schema:
            ref = schema["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
                raise MalformedStructuredOutputError(f"{path} has unsupported $ref")
            target: Any = root_schema
            for part in ref[2:].split("/"):
                if not isinstance(target, Mapping) or part not in target:
                    raise MalformedStructuredOutputError(f"{path} has unresolved $ref")
                target = target[part]
            if not isinstance(target, Mapping):
                raise MalformedStructuredOutputError(f"{path} has invalid $ref")
            check(instance, target, path)
        if "allOf" in schema:
            for branch in schema["allOf"]:
                check(instance, branch, path)
        expected = schema.get("type")
        type_ok = {
            "object": isinstance(instance, Mapping),
            "array": isinstance(instance, list),
            "string": isinstance(instance, str),
            "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
            "integer": isinstance(instance, int) and not isinstance(instance, bool),
            "boolean": isinstance(instance, bool),
            "null": instance is None,
        }
        if isinstance(expected, str) and not type_ok.get(expected, False):
            raise MalformedStructuredOutputError(f"{path} must be {expected}")
        if "enum" in schema and instance not in schema["enum"]:
            raise MalformedStructuredOutputError(f"{path} is not an allowed enum value")
        if "const" in schema and instance != schema["const"]:
            raise MalformedStructuredOutputError(f"{path} does not match const")
        if isinstance(instance, Mapping):
            properties = schema.get("properties", {})
            for key in schema.get("required", ()):
                if key not in instance:
                    raise MalformedStructuredOutputError(f"{path}.{key} is required")
            if schema.get("additionalProperties") is False:
                extra = set(instance) - set(properties)
                if extra:
                    raise MalformedStructuredOutputError(f"{path} has additional properties")
            for key, child in properties.items():
                if key in instance:
                    check(instance[key], child, f"{path}.{key}")
        if isinstance(instance, list):
            if "minItems" in schema and len(instance) < schema["minItems"]:
                raise MalformedStructuredOutputError(f"{path} has too few items")
            if "maxItems" in schema and len(instance) > schema["maxItems"]:
                raise MalformedStructuredOutputError(f"{path} has too many items")
            if isinstance(schema.get("items"), Mapping):
                for index, item in enumerate(instance):
                    check(item, schema["items"], f"{path}[{index}]")
        if isinstance(instance, str):
            if "minLength" in schema and len(instance) < schema["minLength"]:
                raise MalformedStructuredOutputError(f"{path} is too short")
            if "maxLength" in schema and len(instance) > schema["maxLength"]:
                raise MalformedStructuredOutputError(f"{path} is too long")
        for keyword in ("oneOf", "anyOf"):
            if keyword in schema:
                matches = 0
                for branch in schema[keyword]:
                    try:
                        check(instance, branch, path)
                    except MalformedStructuredOutputError:
                        continue
                    matches += 1
                if (keyword == "oneOf" and matches != 1) or (keyword == "anyOf" and matches < 1):
                    raise MalformedStructuredOutputError(f"{path} does not satisfy {keyword}")

    check(value, request.json_schema)
