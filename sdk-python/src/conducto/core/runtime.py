"""Stable Runtime facade and compatibility re-exports."""

from __future__ import annotations

# noinspection PyPackageRequirements
import contextvars
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from conducto.security import (
    ApprovalDecision,
    AuditDeliveryError,
    AuthorizationContext,
    InMemoryApprovalStore,
    SecurityPipeline,
)
from conducto.security.context import delegate_context
from conducto.security.errors import SecurityError

from .logging import MODEL_SELECTED, MODEL_USAGE_RECORDED, emit_event, log_context
from .model_config import (
    AgentModelConfig,
    ModelReference,
    ModelRequirement,
    ModelResolutionSource,
    ProviderType,
    RunConfig,
    RuntimeConfig,
)
from .model_gateway import (
    ModelCallResult,
    ModelGateway,
    ModelGatewayCollection,
    ModelResponseT,
    complete_model_call,
)
from .model_resolution import (
    ModelResolver,
    ResolvedModel,
    _ResolvedModelBinding,
)
from .provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ModelProvider,
    ProviderCapabilities,
    ProviderResult,
    StructuredOutputRequest,
    ToolResultMessage,
    Usage,
    complete_with_retries,
)
from .provider_registry import (
    DEFAULT_AVAILABILITY_TIMEOUT_SECONDS,
    ModelBindingSnapshot,
    ProviderClientConfig,
    ProviderFactory,
    ProviderOwnership,
    ProviderRegistration,
    ProviderRegistry,
    ProviderRegistrySnapshot,
    ProviderTypeRegistration,
)
from .run_context import (
    CancellationState,
    DelegationBudget,
    DelegationFrame,
    InvocationMetadata,
    ModelCallProvenance,
    ModelPolicy,
    ModelPolicyContext,
    RemainingDelegationBudget,
    RunContext,
    activate_run_context,
    deactivate_run_context,
    get_run_context,
    require_run_context,
    use_run_context,
)
from .runtime_errors import (
    ConductoError,
    ContradictoryProviderConfigurationError,
    DuplicateModelReferenceError,
    DuplicateProviderTypeError,
    IncompatibleProviderCapabilitiesError,
    MissingModelDefaultError,
    ModelOverrideDeniedError,
    ModelResolutionError,
    NoActiveRunContextError,
    ProviderClientValidationError,
    ProviderConstructionError,
    ProviderFactoryValidationError,
    ProviderRegistrationError,
    ProviderTypeMismatchError,
    ProviderUnavailableError,
    StaleProviderConstructionError,
    UnknownModelReferenceError,
    UnknownProviderTypeError,
)

if TYPE_CHECKING:
    from .agent import BaseAgent
    from .gateway import GatewayPolicy
    from .invocation_results import InvocationResult
    from .registry import AgentRegistry

__all__ = [
    "AgentModelConfig",
    "CancellationState",
    "ContradictoryProviderConfigurationError",
    "DEFAULT_AVAILABILITY_TIMEOUT_SECONDS",
    "DelegationBudget",
    "DelegationFrame",
    "ConductoError",
    "DuplicateModelReferenceError",
    "DuplicateProviderTypeError",
    "GenerationOptions",
    "IncompatibleProviderCapabilitiesError",
    "InvocationMetadata",
    "MODEL_SELECTED",
    "MODEL_USAGE_RECORDED",
    "MalformedStructuredOutputError",
    "MissingModelDefaultError",
    "ModelBindingSnapshot",
    "ModelCallProvenance",
    "ModelCallResult",
    "ModelConfiguration",
    "ModelGateway",
    "ModelGatewayCollection",
    "ModelOverrideDeniedError",
    "ModelPolicy",
    "ModelPolicyContext",
    "ModelProvider",
    "ModelResponseT",
    "ModelReference",
    "ModelRequirement",
    "ModelResolutionError",
    "ModelResolutionSource",
    "NoActiveRunContextError",
    "ProviderCapabilities",
    "ProviderClientConfig",
    "ProviderClientValidationError",
    "ProviderConstructionError",
    "ProviderFactory",
    "ProviderFactoryValidationError",
    "ProviderOwnership",
    "ProviderRegistration",
    "ProviderRegistrationError",
    "ProviderRegistry",
    "ProviderRegistrySnapshot",
    "ProviderResult",
    "ProviderType",
    "ProviderTypeMismatchError",
    "ProviderTypeRegistration",
    "ProviderUnavailableError",
    "RemainingDelegationBudget",
    "ResolvedModel",
    "RunConfig",
    "RunContext",
    "Runtime",
    "RuntimeConfig",
    "StaleProviderConstructionError",
    "StructuredOutputRequest",
    "UnknownModelReferenceError",
    "UnknownProviderTypeError",
    "Usage",
    "complete_with_retries",
    "emit_event",
    "get_run_context",
    "log_context",
    "require_run_context",
    "use_run_context",
    "ChatMessage",
]


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
        gateway_preferred_agents: Mapping[str, str] | None = None,
        gateway_binding_ttl: float = 300.0,
        gateway_max_results: int = 20,
        gateway_max_serialized_bytes: int = 64 * 1024,
    ) -> None:
        self._provider_registry = provider_registry or ProviderRegistry()
        self._model_resolver = ModelResolver(self._provider_registry)
        self.config = config or RuntimeConfig()
        self.policy = policy
        self.security_pipeline = security_pipeline or SecurityPipeline(InMemoryApprovalStore())
        if agent_registry is None:
            from .registry import AgentRegistry

            agent_registry = AgentRegistry()
        self.agent_registry = agent_registry
        self.gateway_policy = gateway_policy
        self.gateway_preferred_agents = dict(gateway_preferred_agents or {})
        self.gateway_binding_ttl = gateway_binding_ttl
        self.gateway_max_results = gateway_max_results
        self.gateway_max_serialized_bytes = gateway_max_serialized_bytes
        self._gateway_runtime_id = str(uuid.uuid4())
        self._gateway_secret = os.urandom(32)
        self._execution_locks: dict[tuple[int, str], threading.Lock] = {}
        self._execution_locks_guard = threading.Lock()

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
        authorization_context: Any = None,
        allowed_capabilities: frozenset[str] | None = None,
        delegation_budget: DelegationBudget | None = None,
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
            authorization_context: Alias for ``authorization``.

        Returns:
            A normalized invocation result envelope.
        """
        from .invocation import invoke_agent

        active_context = get_run_context()
        try:
            effective_authorization = delegate_context(
                active_context.authorization if active_context is not None else None,
                authorization if authorization is not None else authorization_context,
            )
        except SecurityError as error:
            from .invocation_results import InvocationAuthorizationFailure

            return InvocationAuthorizationFailure(
                correlation_id or self.new_correlation_id(),
                error.reason_code,
            )
        try:
            return await invoke_agent(
                self,
                agent,
                capability,
                arguments,
                timeout=timeout,
                correlation_id=correlation_id,
                model_reference=model_reference,
                run_config=run_config,
                authorization=effective_authorization,
                security_pipeline=self.security_pipeline,
                allowed_capabilities=allowed_capabilities,
                delegation_budget=delegation_budget,
            )
        except AuditDeliveryError as error:
            from .invocation_results import InvocationAuditFailure

            return InvocationAuditFailure(
                correlation_id or self.new_correlation_id(),
                error.reason_code,
            )
        except SecurityError as error:
            from .invocation_results import InvocationAuthorizationFailure

            return InvocationAuthorizationFailure(
                correlation_id or self.new_correlation_id(),
                error.reason_code,
            )

    async def resume_approval(
        self,
        agent: BaseAgent,
        capability: str | Callable[..., Any],
        arguments: Mapping[str, Any],
        decision: ApprovalDecision,
        *,
        authorization: AuthorizationContext,
    ) -> InvocationResult:
        """Resume one persisted approval-bound invocation exactly once."""
        from .invocation import invoke_agent
        from .invocation_results import InvocationResult

        async def execute() -> InvocationResult:
            return await invoke_agent(
                self,
                agent,
                capability,
                arguments,
                correlation_id=authorization.correlation_id,
                authorization=authorization,
                security_pipeline=self.security_pipeline,
                approved_approval_id=decision.approval_id,
            )

        capability_name = capability if isinstance(capability, str) else ""
        if not isinstance(capability, str):
            requested = getattr(capability, "__func__", capability)
            for name, registered in agent.capabilities.items():
                candidate = getattr(registered.callable, "__func__", registered.callable)
                if candidate is requested:
                    capability_name = name
                    break
        try:
            return cast(
                InvocationResult,
                await self.security_pipeline.resume(
                    decision,
                    execute,
                    agent_id=agent.agent_metadata.name,
                    capability_id=capability_name,
                    context=authorization,
                ),
            )
        except SecurityError as error:
            from .invocation_results import (
                InvocationApprovalRequired,
                InvocationAuthorizationFailure,
            )

            if error.reason_code == "approval_required":
                assert self.security_pipeline.store is not None
                challenge = self.security_pipeline.store.get(decision.approval_id)
                return InvocationApprovalRequired(authorization.correlation_id, challenge)
            return InvocationAuthorizationFailure(
                authorization.correlation_id,
                error.reason_code,
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
        from .invocation import invoke_agent

        if self.security_pipeline.token_service is None:
            from .invocation_results import InvocationAuthorizationFailure

            return InvocationAuthorizationFailure(
                authorization.correlation_id,
                "invalid_state_transition",
            )
        claims = self.security_pipeline.token_service.verifier.verify(token)

        async def execute() -> InvocationResult:
            return await invoke_agent(
                self,
                agent,
                capability,
                arguments,
                correlation_id=authorization.correlation_id,
                authorization=authorization,
                security_pipeline=self.security_pipeline,
                approved_approval_id=claims["challenge_id"],
            )

        try:
            return cast(
                InvocationResult,
                await self.security_pipeline.resume_token(
                    token,
                    execute,
                    context=authorization,
                ),
            )
        except SecurityError as error:
            from .invocation_results import InvocationAuthorizationFailure

            return InvocationAuthorizationFailure(
                authorization.correlation_id,
                error.reason_code,
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

        Returns:
            A task-local run context associated with this runtime.
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
        active_context = get_run_context()
        parent = (
            active_context
            if active_context is not None and active_context.belongs_to(self)
            else None
        )
        effective_timeout: float | None = run.timeout
        requested_deadline = (
            time.monotonic() + effective_timeout if effective_timeout is not None else None
        )
        deadline: float | None
        if parent is not None and parent.deadline is not None:
            deadline = (
                min(requested_deadline, parent.deadline)
                if requested_deadline is not None
                else parent.deadline
            )
            effective_timeout = max(0.0, deadline - time.monotonic())
        else:
            deadline = requested_deadline
        effective_run_id = run_id or (
            str(uuid.uuid4())
            if parent is not None
            else (
                authorization.task_id
                if isinstance(authorization, AuthorizationContext)
                else str(uuid.uuid4())
            )
        )
        if isinstance(authorization, AuthorizationContext) and authorization.correlation_id != (
            correlation_id or authorization.correlation_id
        ):
            raise SecurityError("authorization correlation_id does not match run context")
        parent_allowed = parent.allowed_capabilities if parent is not None else None
        effective_allowed: frozenset[str] | None
        if parent_allowed is not None:
            if allowed_capabilities is None:
                effective_allowed = parent_allowed
            elif not allowed_capabilities.issubset(parent_allowed):
                raise SecurityError("delegated capabilities are broader than their caller")
            else:
                effective_allowed = frozenset(allowed_capabilities)
        else:
            effective_allowed = (
                frozenset(allowed_capabilities) if allowed_capabilities is not None else None
            )
        path = parent.delegation_path if parent is not None else ()
        if delegation_frame is not None:
            path += (delegation_frame,)
        return RunContext(
            run_id=effective_run_id,
            correlation_id=correlation_id or str(uuid.uuid4()),
            model=binding.model if binding is not None else None,
            timeout=effective_timeout,
            deadline=deadline,
            cancellation=parent.cancellation if parent is not None else CancellationState(),
            metadata=run.metadata,
            agent_id=agent_id,
            parent_run_id=parent.run_id if parent is not None else None,
            delegation_path=path,
            allowed_capabilities=effective_allowed,
            delegation_budget=(
                parent.delegation_budget
                if parent is not None
                else (delegation_budget or DelegationBudget())
            ),
            policy_context=run,
            _runtime=self,
            _agent_registry=self.agent_registry,
            _binding=binding,
            authorization=authorization,
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
        tools: Sequence[Mapping[str, Any]] = (),
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

        Returns:
            The provider result and invocation metadata.

        Raises:
            ValueError: If the run context belongs to a different runtime.
        """
        if not context.belongs_to(self):
            raise ValueError("Run context belongs to a different runtime")
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
        return await complete_model_call(
            context,
            binding,
            messages,
            structured_output=structured_output,
            tools=tools,
            tool_results=tool_results,
            effective_deadline=effective_deadline,
            purpose=purpose,
            clock=clock,
        )

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
