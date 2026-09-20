"""Immutable model configuration and metadata helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelReference:
    """Opaque, credential-free reference to a runtime-registered model."""

    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise ValueError("Model reference cannot be empty")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


class ModelResolutionSource(StrEnum):
    """Precedence source that selected the effective model reference."""

    CALL_OVERRIDE = "call_override"
    RUN_OVERRIDE = "run_override"
    AGENT_DEFAULT = "agent_default"
    RUNTIME_DEFAULT = "runtime_default"


class ModelRequirement(StrEnum):
    """Whether an agent or capability requires a model to execute."""

    REQUIRED = "required"
    NONE = "none"


def normalize_reference(value: ModelReference | str | None) -> ModelReference | None:
    """Normalize a string reference while preserving existing value objects."""
    if value is None or isinstance(value, ModelReference):
        return value
    return ModelReference(value)


def freeze_metadata(value: Any) -> Any:
    """Recursively freeze provider-neutral metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_metadata(item) for item in value)
    return value


def thaw_metadata(value: Any) -> Any:
    """Return a JSON-compatible mutable representation of frozen metadata."""
    if isinstance(value, Mapping):
        return {key: thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [thaw_metadata(item) for item in value]
    return value


def validate_metadata(value: Any) -> None:
    """Reject known credential-bearing keys at any metadata depth."""
    sensitive = {"api_key", "authorization", "credential", "credentials", "secret", "token"}
    if isinstance(value, Mapping):
        invalid = {str(key).lower() for key in value} & sensitive
        if invalid:
            raise ValueError(
                f"Sensitive values are not allowed in run metadata: {sorted(invalid)!r}"
            )
        for item in value.values():
            validate_metadata(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            validate_metadata(item)


@dataclass(frozen=True, slots=True)
class AgentModelConfig:
    """Immutable model policy attached to an agent."""

    default_model: ModelReference | str | None = None
    requirement: ModelRequirement = ModelRequirement.NONE
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_model", normalize_reference(self.default_model))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(self.required_capabilities),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Immutable per-run override and policy input."""

    model: ModelReference | str | None = None
    timeout: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    caller: str | None = None
    environment: str | None = None
    cost_tier: str | None = None
    data_classification: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", normalize_reference(self.model))
        if self.timeout is not None and (
                isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout)
                or self.timeout <= 0
        ):
            raise ValueError("Run timeout must be a finite positive number")
        validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable runtime-wide model defaults."""

    default_model: ModelReference | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_model", normalize_reference(self.default_model))
