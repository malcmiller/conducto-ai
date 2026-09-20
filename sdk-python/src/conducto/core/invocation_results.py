"""Public capability invocation and routing result contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from .provider import Usage
from .run_context import InvocationMetadata


@dataclass(frozen=True, slots=True)
class InvocationSuccess:
    correlation_id: str
    value: Any
    usage: Usage = dataclasses.field(default_factory=Usage)
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationValidationFailure:
    correlation_id: str
    errors: tuple[Mapping[str, Any], ...]
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationTargetNotFound:
    correlation_id: str
    agent_id: str
    capability_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationTimeout:
    correlation_id: str
    timeout: float
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationCancelled:
    correlation_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationFailure:
    correlation_id: str
    message: str
    exception: BaseException = dataclasses.field(repr=False, compare=False, hash=False)
    metadata: InvocationMetadata | None = None


InvocationResult: TypeAlias = (
    InvocationSuccess
    | InvocationValidationFailure
    | InvocationTargetNotFound
    | InvocationTimeout
    | InvocationCancelled
    | InvocationFailure
)


@dataclass(frozen=True, slots=True)
class RoutingFailure:
    message: str
    exception: BaseException = dataclasses.field(repr=False, compare=False, hash=False)
    usage: Usage = dataclasses.field(default_factory=Usage)
    retryable: bool = False
    metadata: InvocationMetadata | None = None


class UnsupportedReturnValueError(TypeError):
    """Raised when a capability result cannot be represented safely."""
