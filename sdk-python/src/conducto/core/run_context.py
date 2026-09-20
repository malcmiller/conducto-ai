"""Task-local run context, cancellation, and invocation provenance."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
    from .model_gateway import ModelGatewayCollection
    from .model_resolution import ResolvedModel, _ResolvedModelBinding
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

    def __call__(self, context: ModelPolicyContext) -> bool: ...


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
    policy_context: RunConfig = field(default_factory=RunConfig, repr=False, compare=False)
    _runtime: Runtime | None = field(default=None, repr=False, compare=False)
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
    context._invocation_state.activate()
    token = _CURRENT_RUN_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)
        context._invocation_state.deactivate()


def activate_run_context(context: RunContext) -> contextvars.Token[RunContext | None]:
    """Activate a context and return its restoration token."""
    return _CURRENT_RUN_CONTEXT.set(context)


def deactivate_run_context(token: contextvars.Token[RunContext | None]) -> None:
    """Restore the context represented by an activation token."""
    _CURRENT_RUN_CONTEXT.reset(token)
