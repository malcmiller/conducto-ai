"""Runtime facade composing model resolution, contexts, invocation, and lifecycle."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from conducto.security import (
    ApprovalDecision,
    AuthorizationContext,
    InMemoryApprovalStore,
    SecurityPipeline,
)

from .instructions import compose_system_message
from .model_config import (
    AgentModelConfig,
    ModelReference,
    RunConfig,
    RuntimeConfig,
)
from .model_gateway import (
    ModelCallResult,
    complete_model_call,
)
from .model_resolution import (
    ModelResolver,
    ResolvedModel,
    _ResolvedModelBinding,
)
from .provider import (
    ChatMessage,
    ProviderCapabilities,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
)
from .provider_registry import (
    ProviderCleanupReport,
    ProviderRegistry,
)
from .run_context import (
    CancellationState,
    DelegationBudget,
    DelegationFrame,
    ModelPolicy,
    RunContext,
    activate_run_context,
    deactivate_run_context,
)
from .runtime_context import build_run_context
from .runtime_errors import MissingModelDefaultError, RuntimeClosedError
from .runtime_invocation import (
    invoke_capability,
    resume_approved_capability,
    resume_token_capability,
)

if TYPE_CHECKING:
    from .agent import BaseAgent
    from .catalog import AgentCatalog, DeploymentType
    from .gateway import (
        GatewayPolicy,
        GatewayRemotePolicy,
        GatewaySelectionPolicy,
        RemoteGatewayTransport,
    )
    from .invocation_results import InvocationResult
    from .registry import AgentRegistry

__all__ = ["Runtime"]


class Runtime:
    """Compose provider registration, resolution, contexts, and invocation."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry | None = None,
        config: RuntimeConfig | None = None,
        policy: ModelPolicy | None = None,
        security_pipeline: SecurityPipeline | None = None,
        agent_registry: AgentRegistry | None = None,
        gateway_policy: GatewayPolicy | None = None,
        agent_catalog: AgentCatalog | None = None,
        gateway_transport: RemoteGatewayTransport | None = None,
        gateway_remote_policy: GatewayRemotePolicy | None = None,
        gateway_allowed_deployments: frozenset[DeploymentType] | None = None,
        gateway_selection_policy: GatewaySelectionPolicy | None = None,
        gateway_preferred_agents: Mapping[str, str] | None = None,
        gateway_binding_ttl: float = 300.0,
        gateway_max_results: int = 20,
        gateway_max_serialized_bytes: int = 64 * 1024,
        gateway_clock: Callable[[], float] = time.monotonic,
        policy_instructions: Sequence[str] = (),
    ) -> None:
        """Initialize a runtime with its provider, security, and gateway configuration.

        Args:
            provider_registry: Optional provider registry. Defaults to a new registry.
            config: Optional runtime-wide model defaults and policy instructions.
            policy: Optional model selection policy hook.
            security_pipeline: Optional security guardrail and audit pipeline.
            agent_registry: Optional local agent registry. Defaults to a new registry.
            gateway_policy: Optional local capability gateway policy.
            agent_catalog: Optional remote agent catalog for gateway discovery.
            gateway_transport: Optional remote gateway transport, required with
                ``agent_catalog``.
            gateway_remote_policy: Optional remote gateway invocation policy.
            gateway_allowed_deployments: Optional deployment types permitted for
                gateway discovery.
            gateway_selection_policy: Optional gateway target selection policy.
            gateway_preferred_agents: Optional preferred agent bindings by capability.
            gateway_binding_ttl: Time-to-live in seconds for gateway bindings.
            gateway_max_results: Maximum discovery results returned by the gateway.
            gateway_max_serialized_bytes: Maximum serialized gateway payload size.
            gateway_clock: Monotonic clock used for gateway timing.
            policy_instructions: Runtime-owned policy instructions merged into
                ``config.policy_instructions``. Applied first, ahead of any
                agent or capability instructions, in every resolved
                instruction chain. Trusted, framework-level text that callers
                cannot override, suppress, or reorder through invocation
                arguments.

        Raises:
            ValueError: If exactly one of ``agent_catalog`` and
                ``gateway_transport`` is configured.
        """
        self._provider_registry = provider_registry or ProviderRegistry()
        self._model_resolver = ModelResolver(self._provider_registry)
        self.config = config or RuntimeConfig()
        if policy_instructions:
            self.config = dataclasses.replace(
                self.config,
                policy_instructions=self.config.policy_instructions + tuple(policy_instructions),
            )
        self.policy = policy
        self.security_pipeline = security_pipeline or SecurityPipeline(InMemoryApprovalStore())
        if agent_registry is None:
            from .registry import AgentRegistry

            agent_registry = AgentRegistry()
        self.agent_registry = agent_registry
        self.gateway_policy = gateway_policy
        if (agent_catalog is None) != (gateway_transport is None):
            raise ValueError("agent_catalog and gateway_transport must be configured together")
        self.agent_catalog = agent_catalog
        self.gateway_transport = gateway_transport
        self.gateway_remote_policy = gateway_remote_policy
        self.gateway_allowed_deployments = (
            frozenset(gateway_allowed_deployments)
            if gateway_allowed_deployments is not None
            else None
        )
        from .gateway import GatewaySelectionPolicy

        self.gateway_selection_policy = gateway_selection_policy or GatewaySelectionPolicy()
        self.gateway_preferred_agents = dict(gateway_preferred_agents or {})
        self.gateway_binding_ttl = gateway_binding_ttl
        self.gateway_max_results = gateway_max_results
        self.gateway_max_serialized_bytes = gateway_max_serialized_bytes
        self.gateway_clock = gateway_clock
        self._gateway_runtime_id = str(uuid.uuid4())
        self._gateway_secret = os.urandom(32)
        self._gateway_binding_state: dict[str, object] = {}
        self._gateway_binding_state_lock = threading.Lock()
        self._gateway_selection_counters: dict[str, int] = {}
        self._gateway_selection_lock = threading.Lock()
        self._execution_locks: dict[tuple[int, str], threading.Lock] = {}
        self._execution_locks_guard = threading.Lock()
        self._shutdown_task: asyncio.Task[ProviderCleanupReport] | None = None

    def _ensure_open(self) -> None:
        """Raise when a caller attempts to begin work after shutdown."""
        if self._provider_registry.closed:
            raise RuntimeClosedError("Runtime is shut down")

    def gateway_binding_material(
        self,
    ) -> tuple[str, bytes, dict[str, object], threading.Lock, Callable[[], float]]:
        """Return the runtime-scoped material used to issue gateway bindings."""
        return (
            self._gateway_runtime_id,
            self._gateway_secret,
            self._gateway_binding_state,
            self._gateway_binding_state_lock,
            self.gateway_clock,
        )

    def next_gateway_selection_index(self, key: str, size: int) -> int:
        """Return the next deterministic round-robin slot for a gateway key."""
        if size < 1:
            raise ValueError("size must be positive")
        with self._gateway_selection_lock:
            current = self._gateway_selection_counters.get(key, 0)
            self._gateway_selection_counters[key] = current + 1
            return current % size

    async def aclose(
        self,
        *,
        timeout: float | None = 30.0,
        per_client_timeout: float | None = 10.0,
        max_concurrency: int = 4,
    ) -> ProviderCleanupReport:
        """Shut down provider resolution and release runtime-owned clients.

        Concurrent callers observe the same completion report. Cancelling one
        caller does not cancel the shared cleanup operation.
        """
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._provider_registry.aclose(
                    timeout=timeout,
                    per_client_timeout=per_client_timeout,
                    max_concurrency=max_concurrency,
                )
            )
        shutdown_task = self._shutdown_task
        assert shutdown_task is not None
        return await asyncio.shield(shutdown_task)

    async def __aenter__(self) -> Runtime:
        """Enter an async runtime scope."""
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close owned provider clients when leaving an async runtime scope."""
        await self.aclose()

    @property
    def provider_registry(self) -> ProviderRegistry:
        """Return the runtime-owned provider registry."""
        return self._provider_registry

    @provider_registry.setter
    def provider_registry(self, value: ProviderRegistry) -> None:
        """Replace the active provider registry and refresh the model resolver.

        Args:
            value: New registry instance used for model resolution and provider
                lookup.
        """
        self._ensure_open()
        self._provider_registry = value
        self._model_resolver = ModelResolver(value)

    @staticmethod
    def new_correlation_id() -> str:
        """Generate a fresh correlation ID for an invocation or model call."""
        return str(uuid.uuid4())

    def capability_lock(self, agent: BaseAgent, capability_name: str) -> threading.Lock:
        """Return a runtime-scoped lock for one agent capability.

        Args:
            agent: Agent owning the capability.
            capability_name: Name of the ability to serialize.

        Returns:
            A reentrant lock used for capability-level serialization.
        """
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
        authorization: Any = None,
        allowed_capabilities: frozenset[str] | None = None,
        delegation_budget: DelegationBudget | None = None,
        cancellation: CancellationState | None = None,
    ) -> InvocationResult:
        """Invoke a capability through the runtime-owned execution pipeline.

        Args:
            agent: Agent instance that owns the capability.
            capability: Capability name or callable reference.
            arguments: Argument mapping passed to the capability.
            timeout: Optional execution timeout override.
            correlation_id: Optional correlation ID for logs and metadata.
            model_reference: Optional model override for the invocation.
            run_config: Optional run configuration.
            authorization: Optional authenticated authorization context.
            allowed_capabilities: Optional attenuated set of callable capabilities.
            delegation_budget: Optional root delegation limits inherited by child calls.
            cancellation: Optional application-owned cooperative cancellation state.

        Returns:
            A normalized invocation result envelope.
        """
        self._ensure_open()
        return await invoke_capability(
            self,
            agent,
            capability,
            arguments,
            timeout=timeout,
            correlation_id=correlation_id,
            model_reference=model_reference,
            run_config=run_config,
            authorization=authorization,
            allowed_capabilities=allowed_capabilities,
            delegation_budget=delegation_budget,
            cancellation=cancellation,
        )

    async def resume_approval(
        self,
        agent: BaseAgent,
        capability: str | Callable[..., Any],
        arguments: Mapping[str, Any],
        decision: ApprovalDecision,
        *,
        authorization: AuthorizationContext,
        timeout: float | None = None,
        model_reference: ModelReference | str | None = None,
        run_config: RunConfig | None = None,
        allowed_capabilities: frozenset[str] | None = None,
        delegation_budget: DelegationBudget | None = None,
        cancellation: CancellationState | None = None,
    ) -> InvocationResult:
        """Resume one persisted approval through the canonical invocation pipeline.

        Args:
            agent: Agent instance that owns the approved capability.
            capability: Capability name or callable reference.
            arguments: Arguments bound to the approved invocation.
            decision: Authenticated approval decision.
            authorization: Authorization context bound to the challenge.
            timeout: Optional execution timeout override.
            model_reference: Optional model override for the resumed run.
            run_config: Optional immutable run configuration.
            allowed_capabilities: Optional attenuated capability allowlist.
            delegation_budget: Optional root delegation limits.
            cancellation: Optional cooperative cancellation state.

        Returns:
            A normalized invocation result envelope.
        """
        return await resume_approved_capability(
            self,
            agent,
            capability,
            arguments,
            decision,
            authorization=authorization,
            timeout=timeout,
            model_reference=model_reference,
            run_config=run_config,
            allowed_capabilities=allowed_capabilities,
            delegation_budget=delegation_budget,
            cancellation=cancellation,
        )

    async def resume_approval_token(
        self,
        agent: BaseAgent,
        capability: str | Callable[..., Any],
        arguments: Mapping[str, Any],
        token: str,
        *,
        authorization: AuthorizationContext,
    ) -> InvocationResult:
        """Verify a portable approval token before resuming one invocation."""
        return await resume_token_capability(
            self, agent, capability, arguments, token, authorization=authorization
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
        authorization: Any = None,
        allowed_capabilities: frozenset[str] | None = None,
        delegation_budget: DelegationBudget | None = None,
        delegation_frame: DelegationFrame | None = None,
        cancellation: CancellationState | None = None,
        instruction_chain: tuple[str, ...] = (),
        output_contract: Any = None,
    ) -> RunContext:
        """Create a new run context for an invocation.

        Args:
            agent_id: Agent identifier associated with the run.
            agent_config: Optional agent-level model configuration.
            run_config: Optional run-level timeout and metadata configuration.
            call_override: Optional per-call model override.
            correlation_id: Optional correlation identifier.
            run_id: Optional explicit run identifier.
            required_capabilities: Provider capabilities required by the run.
            authorization: Optional authenticated authorization context.
            allowed_capabilities: Optional capability set bounded by the parent context.
            delegation_budget: Root delegation limits when there is no parent context.
            delegation_frame: Optional frame appended to the parent's delegation path.
            cancellation: Optional cancellation state for a root invocation.
            instruction_chain: Resolved, ordered instruction chain -- runtime
                policy, then agent, then capability instructions -- to record
                on the context and compose into its model calls.
            output_contract: Optional structured-output contract for the active
                capability return value.

        Returns:
            A task-local run context associated with this runtime.
        """
        self._ensure_open()
        agent = agent_config or AgentModelConfig()
        run = run_config or RunConfig()
        binding = self._resolve_model_binding(
            agent_id=agent_id,
            agent_config=agent,
            run_config=run,
            call_override=call_override,
            required_capabilities=required_capabilities or agent.required_capabilities,
        )
        return build_run_context(
            self,
            agent_id=agent_id,
            run=run,
            binding=binding,
            correlation_id=correlation_id,
            run_id=run_id,
            authorization=authorization,
            allowed_capabilities=allowed_capabilities,
            delegation_budget=delegation_budget,
            delegation_frame=delegation_frame,
            cancellation=cancellation,
            instruction_chain=instruction_chain,
            output_contract=output_contract,
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
        """Resolve the model for a run without creating a full run context.

        Args:
            agent_id: Agent identifier associated with the model resolution.
            agent_config: Agent-level model policy.
            run_config: Run-level defaults and metadata.
            call_override: Optional call-level model override.
            required_capabilities: Required provider capabilities.

        Returns:
            The selected model descriptor, if any.
        """
        self._ensure_open()
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
        return self._model_resolver.resolve_binding(
            agent_id=agent_id,
            agent_config=agent_config,
            run_config=run_config,
            runtime_config=self.config,
            policy=self.policy,
            call_override=call_override,
            required_capabilities=required_capabilities,
        )

    def resolve_for_call(
        self,
        context: RunContext,
        override: ModelReference | str | None,
        *,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> ResolvedModel | None:
        """Resolve a model for the current call using the active run context."""
        self._ensure_open()
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
        return self._model_resolver.resolve_for_call_binding(
            context,
            override,
            runtime_config=self.config,
            policy=self.policy,
            required_capabilities=required_capabilities,
        )

    async def complete(
        self,
        context: RunContext,
        messages: Sequence[ChatMessage],
        *,
        structured_output: StructuredOutputRequest,
        model: ModelReference | str | None = None,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        required_capabilities: frozenset[str] = frozenset(),
        effective_deadline: float | None = None,
        purpose: str = "model_call",
        clock: Callable[[], float] = time.monotonic,
    ) -> ModelCallResult:
        """Execute a provider completion from the current run context.

        Args:
            context: Active run context attached to this runtime.
            messages: Conversation history for the provider request.
            structured_output: Native structured-output contract.
            model: Optional model override for this request.
            tools: Provider-neutral tool definitions for this turn.
            tool_results: Bounded results from prior tool calls.
            required_capabilities: Additional provider capabilities required by this call.
            effective_deadline: Optional monotonic deadline, capped by the run deadline.
            purpose: Logical purpose name recorded on the call provenance.
            clock: Monotonic clock used for model-call timing.

        Returns:
            The provider result and invocation metadata.

        Raises:
            ValueError: If the run context belongs to a different runtime.
        """
        if not context.belongs_to(self):
            raise ValueError("Run context belongs to a different runtime")
        self._ensure_open()
        required = frozenset(
            {
                "structured_output",
                *(("tool_calling",) if tools or tool_results else ()),
                *required_capabilities,
            }
        )
        binding = self._resolve_for_call_binding(
            context,
            model,
            required_capabilities=required,
        )
        if binding is None:
            raise MissingModelDefaultError(f"Agent '{context.agent_id}' requires a model")
        lease = self._provider_registry.acquire(binding.registration)
        from .run_context import get_run_context

        current = get_run_context()
        invocation_token: contextvars.Token[int] | None = None
        context_token: contextvars.Token[RunContext | None] | None = None
        try:
            if current is not context:
                invocation_token = context.activate_invocation()
                context_token = self.activate(context)
            composed_messages = compose_system_message(messages, context.instruction_chain)
            return await complete_model_call(
                context,
                binding,
                composed_messages,
                structured_output=structured_output,
                tools=tools,
                tool_results=tool_results,
                effective_deadline=effective_deadline,
                purpose=purpose,
                clock=clock,
            )
        finally:
            if context_token is not None:
                self.deactivate(context_token)
            if invocation_token is not None:
                context.deactivate_invocation(invocation_token)
            await lease.release()

    @staticmethod
    def _validate_capabilities(
        reference: ModelReference,
        capabilities: ProviderCapabilities,
        required: frozenset[str],
    ) -> None:
        ModelResolver.validate_capabilities(reference, capabilities, required)

    @staticmethod
    def activate(context: RunContext) -> contextvars.Token[RunContext | None]:
        """Activate a run context in the current task-local context."""
        return activate_run_context(context)

    @staticmethod
    def deactivate(token: contextvars.Token[RunContext | None]) -> None:
        """Restore the previous task-local run context from a token."""
        deactivate_run_context(token)
