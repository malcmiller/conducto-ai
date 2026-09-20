"""Canonical serialization helpers for capability results."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from .invocation_results import UnsupportedReturnValueError


def serialize_result(value: Any) -> Any:
    """Convert a capability return value into a JSON-safe structure.

    Args:
        value: Arbitrary result value returned by a capability.

    Returns:
        A JSON-safe value representation accepted by the runtime.

    Raises:
        UnsupportedReturnValueError: If the value cannot be represented safely.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsupportedReturnValueError("Non-finite floats are unsupported")
        return value
    if isinstance(value, enum.Enum):
        return serialize_result(value.value)
    if isinstance(value, BaseModel):
        try:
            return serialize_result(value.model_dump(mode="json"))
        except Exception as error:
            raise UnsupportedReturnValueError(
                f"Could not serialize Pydantic model: {type(value).__name__}"
            ) from error
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return serialize_result(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise UnsupportedReturnValueError("Mapping keys must be strings")
        return {key: serialize_result(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [serialize_result(item) for item in value]
    if isinstance(value, (set, frozenset)):
        serialized = [serialize_result(item) for item in value]
        return sorted(
            serialized,
            key=lambda item: json.dumps(
                item, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [serialize_result(item) for item in value]
    raise UnsupportedReturnValueError(
        f"Unsupported capability return value: {type(value).__name__}"
    )


def freeze_mapping(value: Any) -> Any:
    """Recursively convert mutable mapping/list structures to immutable equivalents.

    Args:
        value: Value to freeze for stable comparisons or log metadata.

    Returns:
        An immutable equivalent of the supplied value.
    """
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_mapping(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_mapping(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_mapping(item) for item in value)
    return value
