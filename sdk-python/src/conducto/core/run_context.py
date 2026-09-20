"""Task-local run context, cancellation, and invocation provenance."""

from __future__ import annotations

import asyncio

# noinspection PyPackageRequirements
import contextvars
import math
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from .model_config import (
    ModelReference,
    ModelResolutionSource,
    RunConfig,
    freeze_metadata,
    thaw_metadata,
)
from .provider import Usage
from .runtime_errors import NoActiveRunContextError

if TYPE_CHECKING:
    from conducto.security import AuthorizationContext

    from .gateway import AgentGateway
    from .model_gateway import ModelGatewayCollection
    from .model_resolution import ResolvedModel, _ResolvedModelBinding
    from .registry import AgentRegistry
    from .runtime import Runtime


@dataclass(frozen=True, slots=True)
class CancellationState:
    """Thread-safe cooperative cancellation state."""

    _event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation for the current run."""
        self._event.set()


@dataclass(frozen=True, slots=True)
class DelegationFrame:
    """Stable agent and capability identity in a delegation path."""

    agent_id: str
    capability_id: str


@dataclass(frozen=True, slots=True)
class RemainingDelegationBudget:
    """Immutable point-in-time view of shared delegation budgets."""

    depth: int
    calls: int
    tokens: int | None
    cost: float | None
    time: float | None


class DelegationBudget:
    """Thread-safe shared call, token, and cost reservation ledger."""

    __slots__ = ("_cost", "_calls", "_lock", "_max_depth", "_tokens")

    def __init__(
        self,
        *,
        max_depth: int = 8,
        calls: int = 32,
        tokens: int | None = None,
        cost: float | None = None,
    ) -> None:
        if max_depth < 0 or calls < 0:
            raise ValueError("Delegation depth and call budgets cannot be negative")
        if tokens is not None and tokens < 0:
            raise ValueError("Delegation token budget cannot be negative")
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("Delegation cost budget must be a finite non-negative number")
        self._max_depth = max_depth
        self._calls = calls
        self._tokens = tokens
        self._cost = cost
        self._lock = threading.Lock()

    @property
    def max_depth(self) -> int:
        """Return the immutable configured delegation depth limit."""
        return self._max_depth

    def snapshot(
        self,
        *,
        current_depth: int,
        remaining_time: float | None,
    ) -> RemainingDelegationBudget:
        """Return an immutable snapshot without exposing mutable ledger state."""
        with self._lock:
            return RemainingDelegationBudget(
                depth=max(0, self._max_depth - current_depth),
                calls=self._calls,
                tokens=self._tokens,
                cost=self._cost,
                time=remaining_time,
            )

    def reserve(
        self,
        *,
        calls: int = 1,
        tokens: int = 0,
        cost: float = 0,
    ) -> bool:
        """Atomically reserve configured resources for a child invocation."""
        if calls < 0 or tokens < 0 or cost < 0 or not math.isfinite(cost):
            raise ValueError("Delegation reservations must be finite and non-negative")
        with self._lock:
            if self._calls < calls:
                return False
            if self._tokens is not None and self._tokens < tokens:
                return False
            if self._cost is not None and self._cost < cost:
                return False
            self._calls -= calls
            if self._tokens is not None:
                self._tokens -= tokens
            if self._cost is not None:
                self._cost -= cost
            return True


@dataclass(frozen=True, slots=True)
class ModelPolicyContext:
    """Provider-neutral facts supplied to a runtime policy hook."""

    agent_id: str
    model_reference: ModelReference
    provider: str
    source: ModelResolutionSource
    caller: str | None = None
    environment: str | None = None
    cost_tier: str | None = None
    data_classification: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelPolicy(Protocol):
    """Policy hook that approves or denies a candidate model selection."""

    def __call__(self, context: ModelPolicyContext, /) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModelCallProvenance:
    """Credential-free provenance for one provider request."""

    purpose: str
    model_reference: str
    provider: str
    resolution_source: ModelResolutionSource
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance metadata to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the provenance record.
        """
        return {
            "purpose": self.purpose,
            "model_reference": self.model_reference,
            "provider": self.provider,
            "resolution_source": self.resolution_source.value,
            "usage": self.usage.model_dump(),
        }


def aggregate_usage(calls: Sequence[ModelCallProvenance]) -> Usage:
    """Aggregate provider-neutral usage in call order."""
    return Usage(
        input_tokens=sum(call.usage.input_tokens for call in calls),
        output_tokens=sum(call.usage.output_tokens for call in calls),
        total_tokens=sum(call.usage.total_tokens for call in calls),
    )


class _ModelCallRecorder:
    def __init__(self) -> None:
        self._calls: list[ModelCallProvenance] = []
        self._lock = threading.Lock()

    def append(self, call: ModelCallProvenance) -> None:
        """Record one model call in a call-order sequence."""
        with self._lock:
            self._calls.append(call)

    def snapshot(self) -> tuple[ModelCallProvenance, ...]:
        """Return a snapshot of all recorded model calls."""
        with self._lock:
            return tuple(self._calls)


class _InvocationState:
    def __init__(self) -> None:
        self._active = False
        self._model_tasks: set[asyncio.Task[Any]] = set()
        self._lock = threading.Lock()

    def activate(self) -> None:
        """Activate the current invocation scope."""
        with self._lock:
            self._active = True

    def deactivate(self) -> None:
        """Deactivate the current invocation scope and cancel tracked model tasks."""
        with self._lock:
            self._active = False
            tasks = tuple(self._model_tasks)
            self._model_tasks.clear()
        for task in tasks:
            task.cancel()

    def require_active(self) -> None:
        """Ensure the invocation state is active before model gateway access."""
        with self._lock:
            if not self._active:
                raise NoActiveRunContextError(
                    "Model gateway is only available within its active runtime invocation"
                )

    def begin_model_call(self) -> asyncio.Task[Any]:
        """Register the current async task as an active model call."""
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if current is None:
            raise NoActiveRunContextError(
                "Model gateway requires an active asynchronous runtime invocation"
            )
        with self._lock:
            if not self._active:
                raise NoActiveRunContextError(
                    "Model gateway is only available within its active runtime invocation"
                )
            self._model_tasks.add(current)
        return current

    def end_model_call(self, task: asyncio.Task[Any]) -> None:
        """Unregister a previously tracked model task."""
        with self._lock:
            self._model_tasks.discard(task)


@dataclass(frozen=True, slots=True)
class InvocationMetadata:
    """Credential-free model provenance and usage for a result envelope."""

    run_id: str
    correlation_id: str
    parent_run_id: str | None = None
    delegation_path: tuple[DelegationFrame, ...] = ()
    model_reference: str | None = None
    provider: str | None = None
    resolution_source: ModelResolutionSource | None = None
    usage: Usage = field(default_factory=Usage)
    model_calls: tuple[ModelCallProvenance, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", freeze_metadata(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the invocation metadata to JSON-safe values.

        Returns:
            A dictionary describing the run, model usage, and provenance.
        """
        return {
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "parent_run_id": self.parent_run_id,
            "delegation_path": [
                {"agent_id": frame.agent_id, "capability_id": frame.capability_id}
                for frame in self.delegation_path
            ],
            "model_reference": self.model_reference,
            "provider": self.provider,
            "resolution_source": (
                self.resolution_source.value if self.resolution_source is not None else None
            ),
            "usage": self.usage.model_dump(),
            "model_calls": [call.to_dict() for call in self.model_calls],
            "attributes": thaw_metadata(self.attributes),
        }

    def with_model_calls(
        self,
        *groups: Sequence[ModelCallProvenance],
    ) -> InvocationMetadata:
        """Return a copy of the metadata with an additional model call provenance.

        Args:
            *groups: Sequences of model-call provenance records to append.

        Returns:
            A new metadata object with merged usage and provenance.
        """
        calls = self.model_calls + tuple(call for group in groups for call in group)
        return replace(self, usage=aggregate_usage(calls), model_calls=calls)

    def with_prior_model_calls(
        self,
        calls: Sequence[ModelCallProvenance],
    ) -> InvocationMetadata:
        """Return a copy with earlier model calls prepended to this metadata.

        Args:
            calls: Prior model-call provenance to prepend.

        Returns:
            A new metadata object with combined call history.
        """
        combined = tuple(calls) + self.model_calls
        return replace(self, usage=aggregate_usage(combined), model_calls=combined)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable, task-local execution context with a resolved model snapshot."""

    run_id: str
    correlation_id: str
    model: ResolvedModel | None = field(default=None, repr=False, compare=False)
    timeout: float | None = None
    deadline: float | None = None
    cancellation: CancellationState = field(default_factory=CancellationState)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    parent_run_id: str | None = None
    delegation_path: tuple[DelegationFrame, ...] = ()
    allowed_capabilities: frozenset[str] | None = None
    delegation_budget: DelegationBudget = field(
        default_factory=DelegationBudget,
        repr=False,
        compare=False,
    )
    authorization: AuthorizationContext | None = field(default=None, repr=False, compare=False)
    policy_context: RunConfig = field(default_factory=RunConfig, repr=False, compare=False)
    _runtime: Runtime | None = field(default=None, repr=False, compare=False)
    _agent_registry: AgentRegistry | None = field(default=None, repr=False, compare=False)
    _binding: _ResolvedModelBinding | None = field(default=None, repr=False, compare=False)
    _model_calls: _ModelCallRecorder = field(
        default_factory=_ModelCallRecorder,
        repr=False,
        compare=False,
    )
    _invocation_state: _InvocationState = field(
        default_factory=_InvocationState,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        object.__setattr__(self, "delegation_path", tuple(self.delegation_path))
        if self.allowed_capabilities is not None:
            object.__setattr__(
                self,
                "allowed_capabilities",
                frozenset(self.allowed_capabilities),
            )

    @property
    def model_reference(self) -> ModelReference | None:
        """Return the currently resolved model reference, if any."""
        return self.model.reference if self.model is not None else None

    @property
    def models(self) -> ModelGatewayCollection:
        """Return the model gateway collection bound to this run context.

        Returns:
            A runtime-aware gateway collection for this invocation.

        Raises:
            NoActiveRunContextError: If the current context is detached from a runtime.
        """
        if self._runtime is None:
            raise NoActiveRunContextError("Run context is not attached to a runtime")
        from .model_gateway import ModelGatewayCollection

        return ModelGatewayCollection(self._runtime, self)

    @property
    def gateway(self) -> AgentGateway:
        """Return the local capability gateway scoped to this invocation."""
        self.require_active()
        if self._runtime is None or self._agent_registry is None:
            raise NoActiveRunContextError("Run context has no local agent gateway")
        from .gateway import LocalAgentGateway

        return LocalAgentGateway(self._runtime, self._agent_registry, self)

    def remaining_timeout(self) -> float | None:
        """Return the remaining monotonic timeout budget for this run.

        Returns:
            Seconds remaining before the deadline, or ``None`` when no timeout is set.

        Raises:
            TimeoutError: If the deadline has already expired.
        """
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Run deadline exceeded")
        return remaining

    @property
    def remaining_delegation_budget(self) -> RemainingDelegationBudget:
        """Return immutable remaining depth, call, time, token, and cost budgets."""
        try:
            remaining_time = self.remaining_timeout()
        except TimeoutError:
            remaining_time = 0.0
        return self.delegation_budget.snapshot(
            current_depth=len(self.delegation_path),
            remaining_time=remaining_time,
        )

    def require_active(self) -> None:
        """Ensure this context is active for invocation-scoped model access."""
        self._invocation_state.require_active()
        if _CURRENT_RUN_CONTEXT.get() is not self:
            raise NoActiveRunContextError(
                "Gateway context is not the active task-local runtime invocation"
            )

    def begin_model_call(self) -> asyncio.Task[Any]:
        """Register and return the current task as an active model call."""
        return self._invocation_state.begin_model_call()

    def end_model_call(self, task: asyncio.Task[Any]) -> None:
        """Unregister a completed model call task."""
        self._invocation_state.end_model_call(task)

    def record_model_call(self, call: ModelCallProvenance) -> None:
        """Append provider-call provenance to this invocation."""
        self._model_calls.append(call)

    def model_calls(self) -> tuple[ModelCallProvenance, ...]:
        """Return model provenance recorded before a delegated invocation."""
        return self._model_calls.snapshot()

    def activate_invocation(self) -> None:
        """Activate invocation-scoped model access for this context."""
        self._invocation_state.activate()

    def deactivate_invocation(self) -> None:
        """Deactivate model access and cancel unfinished model calls."""
        self._invocation_state.deactivate()

    def belongs_to(self, runtime: Runtime) -> bool:
        """Return whether this context was created by the given runtime."""
        return self._runtime is runtime

    @property
    def model_binding(self) -> _ResolvedModelBinding | None:
        """Return the runtime-private provider binding for model resolution."""
        return self._binding

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run context to a JSON-friendly dictionary.

        Returns:
            A summary of the active run, model, and cancellation state.
        """
        return {
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "model_reference": str(self.model.reference) if self.model is not None else None,
            "provider": self.model.provider if self.model is not None else None,
            "resolution_source": self.model.source.value if self.model is not None else None,
            "timeout": self.timeout,
            "deadline": self.deadline,
            "cancelled": self.cancellation.cancelled,
            "metadata": thaw_metadata(self.metadata),
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "delegation_path": [
                {"agent_id": frame.agent_id, "capability_id": frame.capability_id}
                for frame in self.delegation_path
            ],
            "remaining_delegation_budget": asdict(self.remaining_delegation_budget),
        }

    def invocation_metadata(self, usage: Usage | None = None) -> InvocationMetadata:
        """Build invocation metadata for a capability or route result.

        Args:
            usage: Optional usage snapshot to attach when no model calls have been logged.

        Returns:
            A normalized metadata object describing the run and model provenance.
        """
        calls = self._model_calls.snapshot()
        return InvocationMetadata(
            run_id=self.run_id,
            correlation_id=self.correlation_id,
            parent_run_id=self.parent_run_id,
            delegation_path=self.delegation_path,
            model_reference=str(self.model.reference) if self.model is not None else None,
            provider=self.model.provider if self.model is not None else None,
            resolution_source=self.model.source if self.model is not None else None,
            usage=aggregate_usage(calls) if calls else (usage or Usage()),
            model_calls=calls,
            attributes=self.metadata,
        )


_CURRENT_RUN_CONTEXT: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "conducto_run_context",
    default=None,
)


def get_run_context() -> RunContext | None:
    """Return the current invocation's immutable context, if any."""
    return _CURRENT_RUN_CONTEXT.get()


def require_run_context() -> RunContext:
    """Return the active run context or raise a stable Conducto error."""
    context = get_run_context()
    if context is None:
        raise NoActiveRunContextError(
            "No active Conducto run context; invoke this capability through Runtime"
        )
    return context


@contextmanager
def use_run_context(context: RunContext) -> Iterator[RunContext]:
    """Activate a context and constrain its model gateways to this scope."""
    context.activate_invocation()
    token = _CURRENT_RUN_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)
        context.deactivate_invocation()


def activate_run_context(context: RunContext) -> contextvars.Token[RunContext | None]:
    """Activate a context and return its restoration token."""
    return _CURRENT_RUN_CONTEXT.set(context)


def deactivate_run_context(token: contextvars.Token[RunContext | None]) -> None:
    """Restore the context represented by an activation token."""
    _CURRENT_RUN_CONTEXT.reset(token)
