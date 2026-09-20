"""Public OrchestratorAgent facade."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from .agent import BaseAgent
from .invocation_results import (
    InvocationResult,
    InvocationSuccess,
    InvocationTargetNotFound,
    RoutingFailure,
)
from .logging import INVOCATION_FAILED, emit_event, log_context
from .provider import (
    ChatMessage,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ModelProvider,
    ProviderError,
    StructuredOutputRequest,
    Usage,
    build_routing_schema,
    parse_routing_selection,
)
from .registry import AgentRegistry, card_url_for, routing_prompt_context
from .runtime import (
    AgentModelConfig,
    IncompatibleProviderCapabilitiesError,
    ModelReference,
    ModelRequirement,
    ProviderRegistry,
    RunConfig,
    Runtime,
    use_run_context,
)

__all__ = ["OrchestratorAgent", "RoutingFailure"]


class OrchestratorAgent(BaseAgent):
    """A local registry of reflected agents for deterministic routing."""

    def __init__(
            self,
            *,
            model_provider: ModelProvider | None = None,
            model_config: ModelConfiguration | None = None,
            model_reference: ModelReference | str | None = None,
            runtime: Runtime | None = None,
    ) -> None:
        if runtime is not None and model_provider is not None:
            raise ValueError("runtime cannot be combined with model_provider")
        if model_provider is not None and model_config is None:
            raise ValueError("model_config is required with model_provider")
        self._legacy_direct_provider = model_provider is not None
        self._registry = AgentRegistry()
        # Preserve existing internal attributes for compatible direct integrations.
        self._registered_agents = self._registry.agents
        self._registered_capabilities = self._registry.capabilities
        self._registry_lock = self._registry.lock
        effective_reference = model_reference or (model_config.model if model_config else None)
        if isinstance(effective_reference, str):
            effective_reference = ModelReference(effective_reference)
        super().__init__(
            model_config=model_config,
            model_reference=effective_reference,
            agent_config=AgentModelConfig(
                default_model=effective_reference,
                requirement=ModelRequirement.REQUIRED,
                required_capabilities=frozenset({"structured_output"}),
            ),
        )
        if runtime is None:
            provider_registry = ProviderRegistry()
            if model_provider is not None and model_config is not None:
                assert effective_reference is not None
                provider_registry.register(effective_reference, model_provider, model_config)
            runtime = Runtime(provider_registry=provider_registry)
        self.runtime = runtime

    async def route(
            self,
            user_input: str,
            *,
            model_provider: ModelProvider | None = None,
            model_config: ModelConfiguration | None = None,
            model_reference: ModelReference | str | None = None,
            run_config: RunConfig | None = None,
            agent_run_config: RunConfig | None = None,
            timeout: float | None = None,
            correlation_id: str = "",
    ) -> InvocationResult | RoutingFailure:
        """Route a user message to the best registered agent capability.

        Args:
            user_input: Natural-language task description to route.
            model_provider: Optional provider override for this routing call.
            model_config: Optional model configuration used with a direct provider.
            model_reference: Optional model name or reference override.
            run_config: Configuration used for the routing run itself.
            agent_run_config: Optional run configuration forwarded to the matched
                agent invocation.
            timeout: Optional per-call timeout override in seconds.
            correlation_id: Correlation identifier propagated to logs and metadata.

        Returns:
            Either the routed invocation result or a structured routing failure.
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        if model_provider is not None and model_config is None:
            raise ValueError("model_config is required with model_provider")
        active_runtime = self.runtime
        call_override = model_reference
        if model_provider is not None and model_config is not None:
            registry = ProviderRegistry()
            call_override = ModelReference(model_config.model)
            registry.register(call_override, model_provider, model_config)
            active_runtime = Runtime(
                provider_registry=registry,
                config=self.runtime.config,
                policy=self.runtime.policy,
            )
        elif model_config is not None:
            call_override = ModelReference(model_config.model)

        effective_run = run_config or RunConfig()
        if timeout is not None:
            effective_run = dataclasses.replace(effective_run, timeout=timeout)
        try:
            context = active_runtime.create_run_context(
                agent_id=self.agent_metadata.name,
                agent_config=self.agent_config,
                run_config=effective_run,
                call_override=call_override,
                correlation_id=correlation_id,
                required_capabilities=frozenset({"structured_output"}),
            )
        except IncompatibleProviderCapabilitiesError as error:
            if model_provider is None and not self._legacy_direct_provider:
                raise
            return RoutingFailure(str(error), error)

        assert context.model is not None
        with (
            use_run_context(context),
            log_context(correlation_id=correlation_id, run_id=context.run_id),
        ):
            routing_metadata = self.get_routing_metadata()
            request = StructuredOutputRequest(
                name="conducto_capability_selection",
                schema=build_routing_schema(routing_metadata),
            )
            messages = (
                ChatMessage(
                    role="system",
                    content=(
                        "Choose one capability from the structured local registry. "
                        "Return only the requested schema."
                    ),
                ),
                ChatMessage(role="user", content=user_input),
                ChatMessage(
                    role="system",
                    content=json.dumps(routing_metadata, sort_keys=True),
                ),
            )
            call = None
            try:
                call = await active_runtime.complete(
                    context,
                    messages,
                    structured_output=request,
                    purpose="routing",
                )
                result = call.result
                selection = parse_routing_selection(result)
            except MalformedStructuredOutputError as error:
                usage = call.result.usage if call is not None else Usage()
                return RoutingFailure(
                    str(error),
                    error,
                    usage=usage,
                    metadata=context.invocation_metadata(usage),
                )
            except ProviderError as error:
                return RoutingFailure(
                    str(error),
                    error,
                    retryable=error.retryable,
                    metadata=context.invocation_metadata(),
                )

            invocation = await self.invoke(
                selection.agent_id,
                selection.capability_id,
                selection.arguments,
                timeout=timeout,
                correlation_id=correlation_id,
                run_config=agent_run_config,
            )
            capability_metadata = invocation.metadata
            if capability_metadata is not None:
                invocation = dataclasses.replace(
                    invocation,
                    metadata=capability_metadata.with_prior_model_calls(call.metadata.model_calls),
                )
            else:
                invocation = dataclasses.replace(invocation, metadata=call.metadata)
            if isinstance(invocation, InvocationSuccess):
                return dataclasses.replace(
                    invocation,
                    usage=result.usage,
                    metadata=invocation.metadata or call.metadata,
                )
            return invocation

    @property
    def registered_agents(self) -> tuple[BaseAgent, ...]:
        """Return all registered agents in sorted name order.

        Returns:
            A tuple of registered agent instances.
        """
        return self._registry.registered_agents()

    @property
    def registered_capabilities(self) -> dict[str, BaseAgent]:
        """Return the live capability-to-agent mapping.

        Returns:
            A dictionary keyed by capability name.
        """
        return self._registry.registered_capabilities()

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        """Alias for the registered agent collection."""
        return self.registered_agents

    @property
    def routing_metadata(self) -> list[dict[str, Any]]:
        """Return structured routing metadata for all registered agents."""
        return self.get_routing_metadata()

    @property
    def routing_prompt_context(self) -> str:
        """Return the textual routing prompt context for the current registry."""
        return self.get_routing_prompt_context()

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._registry)

    def __iter__(self) -> Iterator[BaseAgent]:
        """Iterate over registered agents in sorted order."""
        return iter(self.registered_agents)

    def __contains__(self, agent: object) -> bool:
        """Test whether the registry contains an agent object or name.

        Args:
            agent: An agent instance or agent name to look up.

        Returns:
            ``True`` when the agent is present; otherwise ``False``.
        """
        return self._registry.contains(agent)

    def register_agent(self, agent: BaseAgent, *, replace: bool = False) -> BaseAgent | None:
        """Register an agent in the local routing registry.

        Args:
            agent: Agent instance to register.
            replace: Whether to replace an existing agent with the same name.

        Returns:
            The previous agent instance that was replaced, if any.
        """
        return self._registry.register(agent, replace=replace)

    async def invoke(
            self,
            agent_id: str,
            capability_id: str,
            arguments: Mapping[str, Any],
            *,
            timeout: float | None = None,
            correlation_id: str = "",
            model_reference: ModelReference | str | None = None,
            run_config: RunConfig | None = None,
    ) -> InvocationResult:
        """Invoke one capability on a registered agent.

        Args:
            agent_id: Name of the target agent.
            capability_id: The capability name or skill identifier to invoke.
            arguments: Structured arguments for the target capability.
            timeout: Optional per-invocation timeout override.
            correlation_id: Optional correlation identifier for logs and telemetry.
            model_reference: Optional model override for the call.
            run_config: Optional run-level execution configuration.

        Returns:
            An invocation result envelope describing success or failure.
        """
        agent = self._registry.get(agent_id)
        if agent is None:
            correlation_id = correlation_id or str(uuid.uuid4())
            with log_context(
                    correlation_id=correlation_id,
                    agent_id=agent_id,
                    capability_id=capability_id,
            ):
                emit_event(
                    INVOCATION_FAILED,
                    level=20,
                    outcome="failure",
                    error_category="target_not_found",
                )
            return InvocationTargetNotFound(correlation_id, agent_id, capability_id)
        return await self.runtime.invoke(
            agent,
            capability_id,
            arguments,
            timeout=timeout,
            correlation_id=correlation_id,
            model_reference=model_reference,
            run_config=run_config,
        )

    async def invoke_capability(
            self,
            agent_id: str,
            capability_id: str,
            arguments: Mapping[str, Any],
            *,
            timeout: float | None = None,
            correlation_id: str = "",
            model_reference: ModelReference | str | None = None,
            run_config: RunConfig | None = None,
    ) -> InvocationResult:
        """Alias for :meth: 'invoke` that preserves the capability-oriented API."""
        return await self.invoke(
            agent_id,
            capability_id,
            arguments,
            timeout=timeout,
            correlation_id=correlation_id,
            model_reference=model_reference,
            run_config=run_config,
        )

    def replace_agent(self, agent: BaseAgent) -> BaseAgent | None:
        """Replace an existing agent using the same registration name.

        Args:
            agent: Agent instance to register, replacing any previous registration.

        Returns:
            The previously registered agent, if one existed.
        """
        return self.register_agent(agent, replace=True)

    def remove_agent(self, agent: BaseAgent | str) -> BaseAgent:
        """Remove a registered agent from the registry.

        Args:
            agent: The agent instance or agent name to remove.

        Returns:
            The removed agent instance.
        """
        return self._registry.remove(agent)

    def clear_agents(self) -> None:
        """Remove all locally registered agents."""
        self._registry.clear()

    def get_agent_by_name(self, name: str) -> BaseAgent | None:
        """Return a registered agent by name.

        Args:
            name: Agent identifier to look up.

        Returns:
            The matching agent instance, if present.
        """
        return self._registry.get(name)

    def get_registered_agent_names(self) -> tuple[str, ...]:
        """Return all registered agent names in sorted order.

        Returns:
            A tuple of agent names.
        """
        return self._registry.names()

    def _conflicting_capabilities(self, agent: BaseAgent) -> set[str]:
        """Return capability names that would conflict with a candidate agent."""
        return self._registry.conflicting_capabilities(agent)

    def _remove_agent_mapping(self, agent: BaseAgent) -> None:
        """Remove the internal registry mapping for one agent."""
        self._registry.remove_mapping(agent)

    def get_routing_metadata(self) -> list[dict[str, Any]]:
        """Return routing metadata generated from the live agent registry.

        Returns:
            A list of metadata records suitable for prompt-based routing.
        """
        return self._registry.routing_metadata()

    def discover_agents(self) -> tuple[BaseAgent, ...]:
        """Return currently registered agents without re-discovery.

        Returns:
            Registered agent instances in sorted order.
        """
        return self.registered_agents

    def get_routing_prompt_context(self) -> str:
        """Return prompt-safe routing context built from the current registry."""
        return routing_prompt_context(self.get_routing_metadata())

    def get_routing_context(self) -> str:
        """Alias for the prompt-safe routing context."""
        return self.get_routing_prompt_context()

    @staticmethod
    def _card_url_for(agent: BaseAgent) -> str:
        """Return the synthetic local card URL for an agent."""
        return card_url_for(agent)
