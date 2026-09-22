"""Public OrchestratorAgent facade."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from conducto.security.approval import ApprovalDecision
from conducto.security.context import AuthorizationContext, delegate_context

from .agent import BaseAgent
from .invocation_results import (
    InvocationResult,
    InvocationSuccess,
    InvocationTargetNotFound,
    RoutingFailure,
)
from .logging import INVOCATION_FAILED, emit_event, log_context
from .model_config import AgentModelConfig, ModelReference, ModelRequirement, RunConfig
from .provider import (
    ChatMessage,
    MalformedStructuredOutputError,
    ProviderError,
    StructuredOutputRequest,
    Usage,
    build_routing_schema,
    parse_routing_selection,
)
from .registry import AgentRegistry, routing_prompt_context
from .run_context import get_run_context, use_run_context
from .runtime import Runtime

__all__ = ["OrchestratorAgent"]


class OrchestratorAgent(BaseAgent):
    """Register and invoke local agents, routing with runtime-owned models.

    Model-free capability invocation needs no configuration. Model-based routing
    requires a runtime with registered providers and a resolvable model reference.
    """

    def __init__(
        self,
        *,
        model_reference: ModelReference | str | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        self._registry = AgentRegistry()
        effective_reference = model_reference
        if isinstance(effective_reference, str):
            effective_reference = ModelReference(effective_reference)
        super().__init__(
            model_reference=effective_reference,
            agent_config=AgentModelConfig(
                default_model=effective_reference,
                requirement=ModelRequirement.REQUIRED,
                required_capabilities=frozenset({"structured_output"}),
            ),
        )
        self.runtime = runtime if runtime is not None else Runtime()

    async def route(
        self,
        user_input: str,
        *,
        model_reference: ModelReference | str | None = None,
        run_config: RunConfig | None = None,
        authorization: AuthorizationContext | None = None,
        agent_run_config: RunConfig | None = None,
        timeout: float | None = None,
        correlation_id: str = "",
    ) -> InvocationResult | RoutingFailure:
        """Route a user message to the best registered agent capability.

        Args:
            user_input: Natural-language task description to route.
            model_reference: Optional model name or reference override.
            run_config: Configuration used for the routing run itself.
            authorization: Optional authenticated authorization context.
            agent_run_config: Optional run configuration forwarded to the matched
                agent invocation.
            timeout: Optional per-call timeout override in seconds.
            correlation_id: Correlation identifier propagated to logs and metadata.

        Returns:
            Either the routed invocation result or a structured routing failure.

        Raises:
            ModelResolutionError: If no model resolves or the selected registered
                provider does not satisfy the routing policy and capabilities.
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        effective_run = run_config or RunConfig()
        if timeout is not None:
            effective_run = dataclasses.replace(effective_run, timeout=timeout)
        active_context = get_run_context()
        effective_authorization = delegate_context(
            active_context.authorization if active_context is not None else None,
            authorization,
        )
        context = self.runtime.create_run_context(
            agent_id=self.agent_metadata.name,
            agent_config=self.agent_config,
            run_config=effective_run,
            call_override=model_reference,
            correlation_id=correlation_id,
            required_capabilities=frozenset({"structured_output"}),
            authorization=effective_authorization,
        )

        assert context.model is not None
        with (
            use_run_context(context),
            log_context(correlation_id=correlation_id, run_id=context.run_id),
        ):
            routing_metadata = self.get_routing_metadata()
            request = StructuredOutputRequest(
                name="conducto_capability_selection",
                schema=build_routing_schema(routing_metadata),
                required=False,
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
                call = await self.runtime.complete(
                    context,
                    messages,
                    structured_output=request,
                    purpose="routing",
                )
                result = call.result
                selection = parse_routing_selection(result)
            except MalformedStructuredOutputError as error:
                usage = call.result.usage if call is not None else (error.usage or Usage())
                return RoutingFailure(
                    str(error),
                    error,
                    usage=usage,
                    metadata=context.invocation_metadata(usage),
                )
            except ProviderError as error:
                return RoutingFailure(
                    error.diagnostic.to_dict()["message"] or "Provider failure",
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
                authorization=authorization,
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
            replace: Whether to replace an existing agent with the same name and
                remove other agents whose capability names conflict.

        Returns:
            The previous agent instance that was replaced, if any.
        """
        return self._registry.register(
            agent,
            replace=replace,
            allow_capability_conflicts=False,
        )

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
        authorization: AuthorizationContext | None = None,
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
            authorization: Optional authenticated authorization context.

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
            authorization=authorization,
        )

    async def resume_approval(
        self,
        agent_id: str,
        capability_id: str,
        arguments: Mapping[str, Any],
        decision: ApprovalDecision,
        *,
        authorization: AuthorizationContext,
    ) -> InvocationResult:
        """Resume a persisted approval-bound capability invocation.

        Args:
            agent_id: Name of the target agent.
            capability_id: Capability bound to the pending approval.
            arguments: Arguments bound to the pending approval.
            decision: Authorized decision for the persisted approval challenge.
            authorization: Authenticated context bound to the invocation.

        Returns:
            An invocation result envelope describing the resumed execution.
        """
        agent = self._registry.get(agent_id)
        if agent is None:
            return InvocationTargetNotFound(
                authorization.correlation_id,
                agent_id,
                capability_id,
            )
        return await self.runtime.resume_approval(
            agent,
            capability_id,
            arguments,
            decision,
            authorization=authorization,
        )

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

    def get_routing_metadata(self) -> list[dict[str, Any]]:
        """Return routing metadata generated from the live agent registry.

        Returns:
            A list of metadata records suitable for prompt-based routing.
        """
        return self._registry.routing_metadata()

    def get_routing_prompt_context(self) -> str:
        """Return prompt-safe routing context built from the current registry."""
        return routing_prompt_context(self.get_routing_metadata())
