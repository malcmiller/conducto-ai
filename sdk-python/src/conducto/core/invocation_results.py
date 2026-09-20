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
    """Envelope for a successful capability invocation.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        value: Serialized return value from the capability.
        usage: Normalized provider usage for the invocation.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    value: Any
    usage: Usage = dataclasses.field(default_factory=Usage)
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationValidationFailure:
    """Envelope for an invocation rejected during argument validation.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        errors: Validation errors produced by the request schema.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    errors: tuple[Mapping[str, Any], ...]
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationTargetNotFound:
    """Envelope for an invocation whose agent or capability does not exist.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        agent_id: Requested agent identifier.
        capability_id: Requested capability identifier.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    agent_id: str
    capability_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationTimeout:
    """Envelope for an invocation that exceeded its timeout budget.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        timeout: Timeout value used for the run.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    timeout: float
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationCancelled:
    """Envelope for a cancelled capability invocation.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationFailure:
    """Envelope for a capability invocation that failed during execution.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        message: Human-readable failure summary.
        exception: The originating exception, when available.
        metadata: Optional invocation metadata captured by the runtime.
    """

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
    """Envelope for a failure during orchestrator-level routing.

    Attributes:
        message: Human-readable failure description.
        exception: The originating exception, when available.
        usage: Usage from the routing model call.
        retryable: Whether the routing failure is safe to retry.
        metadata: Optional invocation metadata captured by the runtime.
    """

    message: str
    exception: BaseException = dataclasses.field(repr=False, compare=False, hash=False)
    usage: Usage = dataclasses.field(default_factory=Usage)
    retryable: bool = False
    metadata: InvocationMetadata | None = None


class UnsupportedReturnValueError(TypeError):
    """Raised when a capability result cannot be represented safely."""
