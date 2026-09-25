"""Deterministic content identity helpers shared by the lifecycle contracts.

These helpers normalize JSON-safe payloads and derive stable SHA-256 digests so
provisioning fingerprints and content identifiers are reproducible across
processes and runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = ["digest_text", "freeze_payload", "thaw_payload"]


def freeze_payload(value: Any, *, field: str) -> Any:
    """Return an immutable, JSON-safe copy of a lifecycle payload.

    Args:
        value: Mapping, sequence, or scalar supplied by a caller.
        field: Field name used in validation failures.

    Returns:
        A deeply frozen equivalent using read-only mappings and tuples.

    Raises:
        ValueError: If a key is not a string, a float is not finite, or a value
            type is not JSON-safe.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} keys must be strings")
        return MappingProxyType(
            {key: freeze_payload(value[key], field=field) for key in sorted(value)}
        )
    if isinstance(value, list | tuple):
        return tuple(freeze_payload(item, field=field) for item in value)
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        return value
    raise ValueError(f"{field} contains unsupported value {type(value).__name__}")


def thaw_payload(value: Any) -> Any:
    """Return a mutable JSON-serializable copy of a frozen payload.

    Args:
        value: Frozen payload produced by :func:`freeze_payload`.

    Returns:
        An equivalent structure of plain dictionaries, lists, and scalars.
    """
    if isinstance(value, Mapping):
        return {key: thaw_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_payload(item) for item in value]
    return value


def digest_text(prefix: str, *parts: object) -> str:
    """Return a stable prefixed SHA-256 digest over canonical JSON parts.

    Args:
        prefix: Short identifier prefix, such as ``"cfg"`` or ``"doc"``.
        *parts: JSON-safe values contributing to the identity.

    Returns:
        A deterministic identifier of the form ``"<prefix>-<hex>"`` truncated to
        a stable width.
    """
    canonical = json.dumps(
        [thaw_payload(part) for part in parts],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"{prefix}-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"
