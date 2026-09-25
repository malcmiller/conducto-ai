"""Public capability invocation and routing result contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from conducto.security.approval import ApprovalChallenge

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
        exception: Typed validation failure retained for local callers.
    """

    correlation_id: str
    errors: tuple[Mapping[str, Any], ...]
    metadata: InvocationMetadata | None = None
    exception: BaseException | None = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )


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
    """Envelope for a canceled capability invocation.

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
        classification: Stable capability failure code, when classified.
    """

    correlation_id: str
    message: str
    exception: BaseException = dataclasses.field(repr=False, compare=False, hash=False)
    metadata: InvocationMetadata | None = None
    classification: str | None = None


@dataclass(frozen=True, slots=True)
class InvocationInternalFailure:
    """Envelope for a sanitized unexpected runtime or adapter failure.

    Attributes:
        correlation_id: Correlation identifier shared with the invocation.
        reason_code: Stable public category with no exception detail.
        exception: Originating exception retained for local diagnostics only.
        metadata: Optional invocation metadata captured by the runtime.
    """

    correlation_id: str
    reason_code: str = "internal_error"
    exception: BaseException = dataclasses.field(
        default_factory=lambda: RuntimeError("internal failure"),
        repr=False,
        compare=False,
        hash=False,
    )
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationApprovalRequired:
    """Envelope returned before protected business logic can execute."""

    correlation_id: str
    challenge: ApprovalChallenge
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationAuthorizationFailure:
    """Envelope for a fail-closed authorization decision."""

    correlation_id: str
    reason_code: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationAuditFailure:
    """Envelope for mandatory audit evidence that could not be delivered."""

    correlation_id: str
    reason_code: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationBindingFailure:
    """Envelope for a forged, foreign-runtime, or expired gateway binding."""

    correlation_id: str
    reason_code: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationStaleBinding:
    """Envelope for a binding invalidated by target removal or replacement."""

    correlation_id: str
    agent_id: str
    capability_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationTargetUnavailable:
    """Envelope for a disabled, draining, or unhealthy target."""

    correlation_id: str
    agent_id: str
    capability_id: str
    reason_code: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationSchemaMismatch:
    """Envelope for a target whose current schema differs from its binding."""

    correlation_id: str
    agent_id: str
    capability_id: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationBudgetExhausted:
    """Envelope for an invocation rejected by an atomic delegation budget."""

    correlation_id: str
    budget: str
    metadata: InvocationMetadata | None = None


@dataclass(frozen=True, slots=True)
class InvocationDelegationFailure:
    """Envelope for a cycle or maximum-depth violation."""

    correlation_id: str
    reason_code: str
    path: tuple[tuple[str, str], ...]
    metadata: InvocationMetadata | None = None


InvocationResult: TypeAlias = (
    InvocationSuccess
    | InvocationValidationFailure
    | InvocationTargetNotFound
    | InvocationTimeout
    | InvocationCancelled
    | InvocationFailure
    | InvocationInternalFailure
    | InvocationApprovalRequired
    | InvocationAuthorizationFailure
    | InvocationAuditFailure
    | InvocationBindingFailure
    | InvocationStaleBinding
    | InvocationTargetUnavailable
    | InvocationSchemaMismatch
    | InvocationBudgetExhausted
    | InvocationDelegationFailure
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
