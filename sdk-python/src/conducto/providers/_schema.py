"""Private traversal of adapter-supported JSON Schema subsets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from conducto.core.provider import SchemaFeature


def conducto_schema_features(supported: frozenset[str]) -> frozenset[SchemaFeature]:
    """Map a wire-format feature allowlist to Conducto feature declarations."""
    return frozenset(feature for feature in SchemaFeature if feature.value in supported)


def _schema_nodes(schema: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield schema
    for key, value in schema.items():
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, Mapping):
                    yield from _schema_nodes(child)
        elif key in {"items", "additionalProperties"} and isinstance(value, Mapping):
            yield from _schema_nodes(value)
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    yield from _schema_nodes(item)


def schema_features(schema: Mapping[str, Any], supported: frozenset[str]) -> set[str]:
    """Collect used features while preserving each adapter's supported subset."""
    features: set[str] = set()
    for node in _schema_nodes(schema):
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                features.add(value)
            if key in supported or key in {"oneOf", "anyOf", "allOf"}:
                features.add(key)
    return features


def unknown_schema_keywords(schema: Mapping[str, Any], allowed: frozenset[str]) -> set[str]:
    """Find unsupported keywords without interpreting property names as keywords."""
    return {
        str(key)
        for node in _schema_nodes(schema)
        for key in node
        if key not in allowed and not key.startswith("x-")
    }
