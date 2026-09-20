"""Immutable model configuration and per-run resolution contracts."""

from __future__ import annotations

import asyncio
import contextvars
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .logging import MODEL_SELECTED, MODEL_USAGE_RECORDED, emit_event, log_context
from .provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ModelProvider,
    ProviderCapabilities,
    ProviderResult,
    StructuredOutputRequest,
    Usage,
    complete_with_retries,
)

if TYPE_CHECKING:
    from .agent import BaseAgent
    from .invocation import InvocationResult

ModelResponseT = TypeVar("ModelResponseT", bound=BaseModel)


class ConductoError(RuntimeError):
    """Base error for stable Conducto runtime failures."""


class NoActiveRunContextError(ConductoError):
    """Model-backed code was called without an active runtime invocation."""


class ModelResolutionError(ConductoError):
    """Base error raised before model-backed work begins."""


class MissingModelDefaultError(ModelResolutionError):
    """No override or default supplied a model for model-backed work."""


class UnknownModelReferenceError(ModelResolutionError):
    """The selected model reference is not registered with the runtime."""


class IncompatibleProviderCapabilitiesError(ModelResolutionError):
    """The selected provider cannot satisfy the requested capabilities."""


class ModelOverrideDeniedError(ModelResolutionError):
    """Runtime policy rejected a model or provider selection."""


class ProviderUnavailableError(ModelResolutionError):
    """The selected provider is registered but currently unavailable."""


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


def _reference(value: ModelReference | str | None) -> ModelReference | None:
    if value is None or isinstance(value, ModelReference):
        return value
    return ModelReference(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _validate_metadata(value: Any) -> None:
    sensitive = {"api_key", "authorization", "credential", "credentials", "secret", "token"}
    if isinstance(value, Mapping):
        invalid = {str(key).lower() for key in value} & sensitive
        if invalid:
            raise ValueError(
                f"Sensitive values are not allowed in run metadata: {sorted(invalid)!r}"
            )
        for item in value.values():
            _validate_metadata(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_metadata(item)


@dataclass(frozen=True, slots=True)
class AgentModelConfig:
    """Immutable model policy attached to an agent.

    Attributes:
        default_model: Agent-level fallback model reference.
        requirement: Whether model-free execution is allowed.
        required_capabilities: Provider features the resolved model must offer.
    """

    default_model: ModelReference | None = None
    requirement: ModelRequirement = ModelRequirement.NONE
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_model", _reference(self.default_model))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(self.required_capabilities),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Immutable per-run override and policy input.

    Attributes:
        model: Run-level model override. A call-level override takes precedence.
        timeout: Optional finite positive run timeout in seconds.
        metadata: Provider-neutral application metadata. Known credential and
            secret keys are rejected, and nested values are frozen.
        caller: Optional caller identity supplied to the policy hook.
        environment: Optional deployment environment supplied to policy.
        cost_tier: Optional cost classification supplied to policy.
        data_classification: Optional data sensitivity supplied to policy.

    Raises:
        ValueError: If the timeout is invalid, the model reference is empty, or
            metadata contains a recognized sensitive key.
    """

    model: ModelReference | None = None
    timeout: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    caller: str | None = None
    environment: str | None = None
    cost_tier: str | None = None
    data_classification: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _reference(self.model))
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("Run timeout must be a finite positive number")
        _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable runtime-wide configuration.

    Attributes:
        default_model: Last-resort model reference after call, run, and agent
            selections are absent.
    """

    default_model: ModelReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_model", _reference(self.default_model))


@dataclass(frozen=True, slots=True)
class CancellationState:
    """Thread-safe cooperative cancellation state referenced by a run context.

    The containing: class:`RunContext` remains immutable while this state
    object permits a caller to signal cancellation safely across threads.
    """

    _event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation for work using this state."""
        self._event.set()


@dataclass(frozen=True, slots=True)
class ModelPolicyContext:
    """Provider-neutral facts supplied to a runtime policy hook.

    Attributes:
        agent_id: Published identifier of the agent requesting resolution.
        model_reference: Credential-free candidate model reference.
        provider: Public provider identifier for the candidate.
        source: Precedence level that selected the candidate.
        caller: Optional caller identity from: class:`RunConfig`.
        environment: Optional deployment environment.
        cost_tier: Optional cost classification.
        data_classification: Optional data sensitivity classification.
        metadata: Frozen provider-neutral run metadata.
    """

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
class ResolvedModel:
    """Credential-free description of the model selected for a run."""

    reference: ModelReference
    provider: str
    source: ModelResolutionSource


@dataclass(frozen=True, slots=True)
class _ResolvedModelBinding:
    """Runtime-private model binding that owns provider access."""

    model: ResolvedModel
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModelCallProvenance:
    """Credential-free provenance for one provider request."""

    purpose: str
    model_reference: str
    provider: str
    resolution_source: ModelResolutionSource
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible model-call record."""
        return {
            "purpose": self.purpose,
            "model_reference": self.model_reference,
            "provider": self.provider,
            "resolution_source": self.resolution_source.value,
            "usage": self.usage.model_dump(),
        }


def _aggregate_usage(calls: Sequence[ModelCallProvenance]) -> Usage:
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
        with self._lock:
            self._calls.append(call)

    def snapshot(self) -> tuple[ModelCallProvenance, ...]:
        with self._lock:
            return tuple(self._calls)


class _InvocationState:
    def __init__(self) -> None:
        self._active = False
        self._model_tasks: set[asyncio.Task[Any]] = set()
        self._lock = threading.Lock()

    def activate(self) -> None:
        with self._lock:
            self._active = True

    def deactivate(self) -> None:
        with self._lock:
            self._active = False
            tasks = tuple(self._model_tasks)
            self._model_tasks.clear()
        for task in tasks:
            task.cancel()

    def require_active(self) -> None:
        with self._lock:
            if not self._active:
                raise NoActiveRunContextError(
                    "Model gateway is only available within its active runtime invocation"
                )

    def begin_model_call(self) -> asyncio.Task[Any]:
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
        with self._lock:
            self._model_tasks.discard(task)


@dataclass(frozen=True, slots=True)
class InvocationMetadata:
    """Credential-free model provenance and usage for a result envelope.

    Attributes:
        run_id: Unique identifier for the enclosing run.
        correlation_id: Caller-visible correlation identifier.
        model_reference: Effective credential-free model reference, if any.
        provider: Public provider identifier, if a model was resolved.
        resolution_source: Precedence source for the effective model.
        usage: Provider-neutral usage counters.
        model_calls: Ordered model calls made during the workflow.
        attributes: Frozen provider-neutral run metadata.
    """

    run_id: str
    correlation_id: str
    model_reference: str | None = None
    provider: str | None = None
    resolution_source: ModelResolutionSource | None = None
    usage: Usage = field(default_factory=Usage)
    model_calls: tuple[ModelCallProvenance, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without provider secrets."""
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
            "attributes": _thaw(self.attributes),
        }

    def with_model_calls(
        self,
        *groups: Sequence[ModelCallProvenance],
    ) -> InvocationMetadata:
        """Return metadata containing this and additional ordered model calls."""
        calls = self.model_calls + tuple(call for group in groups for call in group)
        return replace(self, usage=_aggregate_usage(calls), model_calls=calls)

    def with_prior_model_calls(
        self,
        calls: Sequence[ModelCallProvenance],
    ) -> InvocationMetadata:
        """Return metadata with earlier workflow model calls prepended."""
        combined = tuple(calls) + self.model_calls
        return replace(
            self,
            usage=_aggregate_usage(combined),
            model_calls=combined,
        )


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    """Provider result paired with credential-free invocation metadata."""

    result: ProviderResult
    metadata: InvocationMetadata


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable, task-local execution context with a resolved model snapshot.

    Attributes:
        run_id: Unique run identifier.
        correlation_id: Caller-provided or generated correlation identifier.
        model: Runtime-only resolved model binding, or ``None`` for
            deterministic work.
        timeout: Optional run timeout in seconds.
        deadline: Optional monotonic deadline computed when the run is created.
        cancellation: Cooperative cancellation state.
        metadata: Frozen provider-neutral metadata.
        agent_id: Published identifier of the executing agent.

    Notes:
        ``to_dict()`` excludes the model client, provider configuration, policy
        inputs, credentials, and other sensitive runtime states.
    """

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
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def model_reference(self) -> ModelReference | None:
        """Return the effective credential-free model reference, if any."""
        return self.model.reference if self.model is not None else None

    @property
    def models(self) -> ModelGatewayCollection:
        """Return the invocation-scoped, policy-aware model gateway."""
        if self._runtime is None:
            raise NoActiveRunContextError("Run context is not attached to a runtime")
        return ModelGatewayCollection(self._runtime, self)

    def remaining_timeout(self) -> float | None:
        """Return the remaining run budget, raising when it has elapsed."""
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Run deadline exceeded")
        return remaining

    def to_dict(self) -> dict[str, Any]:
        """Serialize only provider-neutral, credential-free run information."""
        return {
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "model_reference": str(self.model.reference) if self.model is not None else None,
            "provider": self.model.provider if self.model is not None else None,
            "resolution_source": self.model.source.value if self.model is not None else None,
            "timeout": self.timeout,
            "deadline": self.deadline,
            "cancelled": self.cancellation.cancelled,
            "metadata": _thaw(self.metadata),
            "agent_id": self.agent_id,
        }

    def invocation_metadata(self, usage: Usage | None = None) -> InvocationMetadata:
        """Create safe result metadata from this context and optional usage."""
        calls = self._model_calls.snapshot()
        return InvocationMetadata(
            run_id=self.run_id,
            correlation_id=self.correlation_id,
            model_reference=str(self.model.reference) if self.model is not None else None,
            provider=self.model.provider if self.model is not None else None,
            resolution_source=self.model.source if self.model is not None else None,
            usage=_aggregate_usage(calls) if calls else (usage or Usage()),
            model_calls=calls,
            attributes=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class ModelGatewayCollection:
    """Invocation-scoped entry point for policy-aware model access."""

    _runtime: Runtime = field(repr=False, compare=False)
    _context: RunContext = field(repr=False, compare=False)

    def require(
        self,
        reference: ModelReference | str | None = None,
    ) -> ModelGateway:
        """Return a constrained gateway for the run model or a call override."""
        self._context._invocation_state.require_active()
        resolved = self._runtime._resolve_for_call_binding(self._context, reference)
        if resolved is None:
            raise MissingModelDefaultError(f"Agent '{self._context.agent_id}' requires a model")
        return ModelGateway(self._runtime, self._context, reference)


@dataclass(frozen=True, slots=True)
class ModelGateway:
    """Constrained model API that enforces runtime policy and telemetry."""

    _runtime: Runtime = field(repr=False, compare=False)
    _context: RunContext = field(repr=False, compare=False)
    _reference: ModelReference | str | None = field(default=None, repr=False)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        structured_output: StructuredOutputRequest,
    ) -> ModelCallResult:
        """Complete one policy-aware model request within the run budget."""
        task = self._context._invocation_state.begin_model_call()
        try:
            return await self._runtime.complete(
                self._context,
                messages,
                structured_output=structured_output,
                model=self._reference,
                purpose="capability",
            )
        finally:
            self._context._invocation_state.end_model_call(task)

    async def complete_typed(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_type: type[ModelResponseT],
    ) -> ModelResponseT:
        """Complete and validate one structured response as a Pydantic model."""
        request = StructuredOutputRequest(
            name=response_type.__name__,
            schema=response_type.model_json_schema(),
        )
        call = await self.complete(messages, structured_output=request)
        if call.result.structured is None:
            raise MalformedStructuredOutputError("Provider returned no structured response")
        try:
            return response_type.model_validate(call.result.structured)
        except ValidationError as error:
            raise MalformedStructuredOutputError(
                "Provider returned malformed structured response"
            ) from error


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """Runtime-owned model registration.

    The client and configuration may contain operational provider details and
    are therefore excluded from representations and serialized contexts.
    """

    reference: ModelReference
    provider: str
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)
    available: bool | Callable[[], bool] = field(default=True, repr=False, compare=False)


class ProviderRegistry:
    """Thread-safe runtime-owned registry of model references and clients.

    Provider clients and their credentials remain here. Agents, run requests,
    prompts, cards, logs, and result envelopes use only model references and
    safe provider identifiers.
    """

    def __init__(self) -> None:
        self._registrations: dict[ModelReference, ProviderRegistration] = {}
        self._lock = threading.RLock()

    def register(
        self,
        reference: ModelReference | str,
        client: ModelProvider,
        configuration: ModelConfiguration,
        *,
        available: bool | Callable[[], bool] = True,
        replace: bool = False,
    ) -> None:
        """Register a model reference and its provider client.

        Args:
            reference: Credential-free model reference used by callers.
            client: Provider client retained only by the runtime registry.
            configuration: Immutable, non-secret provider model settings.
            available: Availability flag or probe evaluated during resolution.
            replace: Replace an existing registration for the same reference.

        Raises:
            ValueError: If the reference is empty or already registered without
                ``replace=True``.
        """
        model_reference = _reference(reference)
        assert model_reference is not None
        registration = ProviderRegistration(
            model_reference,
            configuration.provider,
            client,
            configuration,
            available,
        )
        with self._lock:
            if model_reference in self._registrations and not replace:
                raise ValueError(f"Model reference '{model_reference}' is already registered")
            self._registrations[model_reference] = registration

    def resolve(self, reference: ModelReference | str) -> ProviderRegistration:
        """Return an available registration for a model reference.

        Raises:
            UnknownModelReferenceError: If the reference is not registered.
            ProviderUnavailableError: If its availability flag or probe is
                false.
        """
        model_reference = _reference(reference)
        assert model_reference is not None
        with self._lock:
            registration = self._registrations.get(model_reference)
        if registration is None:
            raise UnknownModelReferenceError(f"Unknown model reference '{model_reference}'")
        available = (
            registration.available() if callable(registration.available) else registration.available
        )
        if not available:
            raise ProviderUnavailableError(
                f"Provider for model reference '{model_reference}' is unavailable"
            )
        return registration


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
    """Activate a run context for model calls and capability execution.

    Context variables isolate concurrent asyncio tasks and propagate into
    synchronous capability workers created with ``asyncio.to_thread``.
    """
    context._invocation_state.activate()
    token = _CURRENT_RUN_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)
        context._invocation_state.deactivate()


class Runtime:
    """Own provider clients and create isolated, policy-checked run contexts.

    Resolution precedence is explicit: call override, run override, agent
    default, then runtime default. Resolution and policy checks are complete before
    a model request or capability invocation begins.
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry | None = None,
        config: RuntimeConfig | None = None,
        policy: ModelPolicy | None = None,
    ) -> None:
        """Initialize a runtime.

        Args:
            provider_registry: Registry that owns provider clients.
            config: Immutable runtime-wide defaults.
            policy: Optional authorization-independent approval hook.
        """
        self.provider_registry = provider_registry or ProviderRegistry()
        self.config = config or RuntimeConfig()
        self.policy = policy
        self._execution_locks: dict[tuple[int, str], threading.Lock] = {}
        self._execution_locks_guard = threading.Lock()

    @staticmethod
    def new_correlation_id() -> str:
        """Return a new public correlation identifier."""
        return str(uuid.uuid4())

    def capability_lock(self, agent: BaseAgent, capability_name: str) -> threading.Lock:
        """Return the shared lock for one synchronous capability."""
        key = (id(agent), capability_name)
        with self._execution_locks_guard:
            return self._execution_locks.setdefault(key, threading.Lock())

    async def invoke(
        self,
        agent: BaseAgent,
        capability: str | Callable[..., Any],
        arguments: Mapping[str, Any],
        *,
        timeout: float | None = None,
        correlation_id: str = "",
        model_reference: ModelReference | str | None = None,
        run_config: RunConfig | None = None,
    ) -> InvocationResult:
        """Invoke an agent capability through the shared execution pipeline."""
        from .invocation import invoke_agent

        return await invoke_agent(
            self,
            agent,
            capability,
            arguments,
            timeout=timeout,
            correlation_id=correlation_id,
            model_reference=model_reference,
            run_config=run_config,
        )

    def create_run_context(
        self,
        *,
        agent_id: str,
        agent_config: AgentModelConfig | None = None,
        run_config: RunConfig | None = None,
        call_override: ModelReference | str | None = None,
        correlation_id: str = "",
        run_id: str = "",
        required_capabilities: frozenset[str] = frozenset(),
    ) -> RunContext:
        """Resolve a model and create an immutable per-run context.

        Args:
            agent_id: Published identifier used for policy evaluation.
            agent_config: Immutable agent default and requirement policy.
            run_config: Run-level override, timeout, metadata, and policy facts.
            call_override: Highest-precedence model override.
            correlation_id: Caller identifier, generated when empty.
            run_id: Run identifier, generated when empty.
            required_capabilities: Provider capabilities required by this work.

        Returns:
            An isolated context containing a snapshot of the resolved model.

        Raises:
            MissingModelDefaultError: If model-backed work has no selection.
            UnknownModelReferenceError: If the selected reference is unknown.
            IncompatibleProviderCapabilitiesError: If the provider lacks a
                required capability.
            ModelOverrideDeniedError: If the policy rejects the selection.
            ProviderUnavailableError: If the selected provider is unavailable.
        """
        agent = agent_config or AgentModelConfig()
        run = run_config or RunConfig()
        binding = self._resolve_model_binding(
            agent_id=agent_id,
            agent_config=agent,
            run_config=run,
            call_override=call_override,
            required_capabilities=required_capabilities or agent.required_capabilities,
        )
        timeout = run.timeout
        return RunContext(
            run_id=run_id or str(uuid.uuid4()),
            correlation_id=correlation_id or str(uuid.uuid4()),
            model=binding.model if binding is not None else None,
            timeout=timeout,
            deadline=time.monotonic() + timeout if timeout is not None else None,
            metadata=run.metadata,
            agent_id=agent_id,
            policy_context=run,
            _runtime=self,
            _binding=binding,
        )

    def resolve_model(
        self,
        *,
        agent_id: str,
        agent_config: AgentModelConfig,
        run_config: RunConfig,
        call_override: ModelReference | str | None = None,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> ResolvedModel | None:
        """Resolve the effective model using the documented precedence order.

        Returns:
            A runtime-only model binding, or ``None`` when model-free execution
            is allowed and no model was selected.

        Raises:
            MissingModelDefaultError: If a model is required but absent.
            UnknownModelReferenceError: If the selected reference is unknown.
            IncompatibleProviderCapabilitiesError: If requirements are unmet.
            ModelOverrideDeniedError: If runtime policy denies the candidate.
            ProviderUnavailableError: If the provider is unavailable.
        """
        binding = self._resolve_model_binding(
            agent_id=agent_id,
            agent_config=agent_config,
            run_config=run_config,
            call_override=call_override,
            required_capabilities=required_capabilities,
        )
        return binding.model if binding is not None else None

    def _resolve_model_binding(
        self,
        *,
        agent_id: str,
        agent_config: AgentModelConfig,
        run_config: RunConfig,
        call_override: ModelReference | str | None = None,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> _ResolvedModelBinding | None:
        call_reference = _reference(call_override)
        candidates = (
            (call_reference, ModelResolutionSource.CALL_OVERRIDE),
            (run_config.model, ModelResolutionSource.RUN_OVERRIDE),
            (agent_config.default_model, ModelResolutionSource.AGENT_DEFAULT),
            (self.config.default_model, ModelResolutionSource.RUNTIME_DEFAULT),
        )
        selected = next(((ref, source) for ref, source in candidates if ref is not None), None)
        if selected is None:
            if agent_config.requirement is ModelRequirement.REQUIRED or required_capabilities:
                raise MissingModelDefaultError(f"Agent '{agent_id}' requires a model")
            return None

        reference, source = selected
        registration = self.provider_registry.resolve(reference)
        self._validate_capabilities(
            reference,
            registration.client.capabilities,
            required_capabilities,
        )
        policy_context = ModelPolicyContext(
            agent_id=agent_id,
            model_reference=reference,
            provider=registration.provider,
            source=source,
            caller=run_config.caller,
            environment=run_config.environment,
            cost_tier=run_config.cost_tier,
            data_classification=run_config.data_classification,
            metadata=run_config.metadata,
        )
        if self.policy is not None and not self.policy(policy_context):
            raise ModelOverrideDeniedError(
                f"Model reference '{reference}' is denied for agent '{agent_id}'"
            )
        return _ResolvedModelBinding(
            ResolvedModel(reference, registration.provider, source),
            registration.client,
            registration.configuration,
        )

    def resolve_for_call(
        self,
        context: RunContext,
        override: ModelReference | str | None,
        *,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> ResolvedModel | None:
        """Resolve a one-call override without mutating the run context.

        ``None`` reuses the context model. A non-``None`` override is resolved
        with call-level precedence and evaluated independently by policy.
        """
        binding = self._resolve_for_call_binding(
            context,
            override,
            required_capabilities=required_capabilities,
        )
        return binding.model if binding is not None else None

    def _resolve_for_call_binding(
        self,
        context: RunContext,
        override: ModelReference | str | None,
        *,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> _ResolvedModelBinding | None:
        if override is None:
            binding = context._binding
            if binding is not None:
                self._validate_capabilities(
                    binding.model.reference,
                    binding.client.capabilities,
                    required_capabilities,
                )
            return binding
        return self._resolve_model_binding(
            agent_id=context.agent_id,
            agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
            run_config=context.policy_context,
            call_override=override,
            required_capabilities=required_capabilities,
        )

    async def complete(
        self,
        context: RunContext,
        messages: Sequence[ChatMessage],
        *,
        structured_output: StructuredOutputRequest,
        model: ModelReference | str | None = None,
        purpose: str = "model_call",
    ) -> ModelCallResult:
        """Execute one model call without changing the enclosing run context.

        Args:
            context: Immutable enclosing run context.
            messages: Provider-neutral messages for this call.
            structured_output: Required native structured-output contract.
            model: Optional call-only model override.
            purpose: Credential-free label for workflow provenance.

        Returns:
            The provider result and credential-free call metadata.

        Raises:
            MissingModelDefaultError: If no model is available.
            IncompatibleProviderCapabilitiesError: If structured output is not
                supported.
            ModelOverrideDeniedError: If policy denies a call override.
            ProviderUnavailableError: If the selected provider is unavailable.
            asyncio.CancelledError: If cooperative cancellation was requested.
            TimeoutError: If the enclosing run deadline has elapsed.
            ProviderError: If the provider request fails.
        """
        required = (
            frozenset({"structured_output"}) if structured_output.json_schema else frozenset()
        )
        if context._runtime is not self:
            raise ValueError("Run context belongs to a different runtime")
        binding = self._resolve_for_call_binding(
            context,
            model,
            required_capabilities=required,
        )
        if binding is None:
            raise MissingModelDefaultError(f"Agent '{context.agent_id}' requires a model")
        if context.cancellation.cancelled:
            raise asyncio.CancelledError
        resolved = binding.model

        def _remaining_timeout() -> float | None:
            if context.deadline is None:
                return binding.configuration.timeout
            remaining = context.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Run deadline exceeded")
            base = binding.configuration.timeout
            return min(base, remaining)

        timeout = _remaining_timeout()
        options = GenerationOptions(
            model=binding.configuration.model,
            timeout=timeout,
            retries=binding.configuration.retries,
        )
        with log_context(
            correlation_id=context.correlation_id,
            run_id=context.run_id,
            agent_id=context.agent_id,
        ):
            emit_event(
                MODEL_SELECTED,
                provider=resolved.provider,
                model_reference=str(resolved.reference),
                resolution_source=resolved.source.value,
                outcome="success",
            )
            result = await complete_with_retries(
                binding.client,
                messages,
                options=options,
                structured_output=structured_output,
                deadline=context.deadline,
            )
            emit_event(
                MODEL_USAGE_RECORDED,
                provider=resolved.provider,
                model_reference=str(resolved.reference),
                resolution_source=resolved.source.value,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                total_tokens=result.usage.total_tokens,
                outcome="success",
            )
        model_call = ModelCallProvenance(
            purpose=purpose,
            model_reference=str(resolved.reference),
            provider=resolved.provider,
            resolution_source=resolved.source,
            usage=result.usage,
        )
        context._model_calls.append(model_call)
        metadata = InvocationMetadata(
            run_id=context.run_id,
            correlation_id=context.correlation_id,
            model_reference=str(resolved.reference),
            provider=resolved.provider,
            resolution_source=resolved.source,
            usage=result.usage,
            model_calls=(model_call,),
            attributes=context.metadata,
        )
        return ModelCallResult(result, metadata)

    @staticmethod
    def _validate_capabilities(
        reference: ModelReference,
        capabilities: ProviderCapabilities,
        required: frozenset[str],
    ) -> None:
        unsupported = sorted(
            name
            for name in required
            if not hasattr(capabilities, name) or not bool(getattr(capabilities, name))
        )
        if unsupported:
            raise IncompatibleProviderCapabilitiesError(
                f"Model reference '{reference}' lacks capabilities: {', '.join(unsupported)}"
            )

    @staticmethod
    def activate(context: RunContext) -> contextvars.Token[RunContext | None]:
        """Activate a context and return the token needed to restore it."""
        return _CURRENT_RUN_CONTEXT.set(context)

    @staticmethod
    def deactivate(token: contextvars.Token[RunContext | None]) -> None:
        """Restore the run context represented by a prior activation token."""
        _CURRENT_RUN_CONTEXT.reset(token)
