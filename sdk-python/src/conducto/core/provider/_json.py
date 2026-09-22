"""Immutable JSON snapshots shared by provider contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any) -> Any:
    """Freeze JSON-like contract data without leaking mutable provider state."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return ordinary JSON-compatible dictionaries and lists."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_thaw_json(item) for item in value]
    return value
