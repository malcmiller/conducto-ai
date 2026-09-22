"""Model-neutral local capability discovery and invocation gateway."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..gateway_models import (
    BoundCapability,
    CapabilityBinding,
    DiscoveryQuery,
    DiscoveryResult,
    GatewayFailure,
    GatewayFailureCode,
    RegistrationLifecycle,
    SelectionOutcome,
    SelectionStatus,
    ToolDiscoveryResult,
)
from ..invocation_results import (
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationBudgetExhausted,
    InvocationDelegationFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationSuccess,
    InvocationTargetUnavailable,
)
from ..registry import AgentRegistry
from ..run_context import DelegationFrame, RunContext
from ..telemetry import SPAN_GATEWAY_DISCOVER, SPAN_GATEWAY_INVOKE, start_span
from ._bindings import BindingAuthority
from ._contracts import GatewayPolicy
from ._discovery import GatewayAuthorization, _GatewayPolicyEvaluationError, matching_candidates
from ._projection import project_tools
from ._schema import _UnsupportedSchemaError

if TYPE_CHECKING:
    from ..runtime import Runtime


class LocalAgentGateway:
    """In-process adapter over an ``AgentRegistry`` and the shared ``Runtime``."""

    def __init__(
        self,
        runtime: Runtime,
        registry: AgentRegistry,
        context: RunContext,
        *,
        policy: GatewayPolicy | None = None,
        preferred_agents: Mapping[str, str] | None = None,
        binding_ttl: float | None = None,
        max_results: int | None = None,
        max_serialized_bytes: int | None = None,
    ) -> None:
        binding_ttl = runtime.gateway_binding_ttl if binding_ttl is None else binding_ttl
        max_results = runtime.gateway_max_results if max_results is None else max_results
        max_serialized_bytes = (
            runtime.gateway_max_serialized_bytes
            if max_serialized_bytes is None
            else max_serialized_bytes
        )
        if binding_ttl <= 0:
            raise ValueError("Gateway binding TTL must be positive")
        if max_results < 1 or max_serialized_bytes < 1:
            raise ValueError("Gateway result limits must be positive")
        self._runtime = runtime
        self._registry = registry
        self._context = context
        self._authorization = GatewayAuthorization(
            context, policy if policy is not None else runtime.gateway_policy
        )
        self._preferred_agents = dict(
            preferred_agents if preferred_agents is not None else runtime.gateway_preferred_agents
        )
        self._bindings = BindingAuthority(*runtime.gateway_binding_material(), binding_ttl)
        self._max_results = max_results
        self._max_serialized_bytes = max_serialized_bytes

    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Discover authorized compatible candidates from one registry snapshot."""
        with start_span(
            SPAN_GATEWAY_DISCOVER,
            attributes={
                "conducto.agent.id": self._context.agent_id,
                "conducto.correlation_id": self._context.correlation_id,
                "conducto.run.id": self._context.run_id,
                "conducto.transport": "in_process",
            },
        ) as span:
            self._context.require_active()
            try:
                revision, matches, denied, unsupported = matching_candidates(
                    query, self._registry.snapshot(), self._authorization
                )
            except _GatewayPolicyEvaluationError:
                span.set_outcome(
                    "policy_evaluation_failed",
                    reason=GatewayFailureCode.POLICY_EVALUATION_FAILED.value,
                )
                return DiscoveryResult(
                    self._registry.revision,
                    failure=GatewayFailure(
                        GatewayFailureCode.POLICY_EVALUATION_FAILED,
                        "Gateway policy evaluation failed",
                    ),
                )
            except _UnsupportedSchemaError as error:
                span.set_outcome(
                    "validation_failure",
                    reason=GatewayFailureCode.UNSUPPORTED_SCHEMA.value,
                )
                return DiscoveryResult(
                    self._registry.revision,
                    failure=GatewayFailure(
                        GatewayFailureCode.UNSUPPORTED_SCHEMA,
                        str(error),
                    ),
                )
            limit = min(query.limit, self._max_results)
            candidates = tuple(
                BoundCapability(
                    descriptor,
                    self._bindings.issue(descriptor, revision, generation),
                )
                for descriptor, generation in matches[:limit]
            )
            failure = None
            if not candidates:
                if unsupported:
                    code = GatewayFailureCode.UNSUPPORTED_SCHEMA
                    message = "Matching providers use unsupported JSON Schema features"
                else:
                    code = (
                        GatewayFailureCode.DISCOVERY_DENIED
                        if denied
                        else GatewayFailureCode.NO_MATCH
                    )
                    message = (
                        "Capability discovery was denied"
                        if denied
                        else "No eligible capability matched the query"
                    )
                failure = GatewayFailure(code, message)
                span.set_outcome(code.value, reason=code.value)
            else:
                span.set_outcome("success")
            return DiscoveryResult(
                revision,
                candidates,
                failure,
                truncated=len(matches) > limit or (unsupported and bool(candidates)),
            )

    async def lookup(self, agent_id: str, capability_id: str) -> SelectionOutcome:
        """Resolve an exact target without a model call."""
        return await self.select(
            DiscoveryQuery(
                agent_id=agent_id,
                capability_ids=frozenset({capability_id}),
                limit=2,
            )
        )

    async def select(self, query: DiscoveryQuery) -> SelectionOutcome:
        """Select one configured provider or return explicit ambiguity."""
        self._context.require_active()
        try:
            revision, matches, denied, unsupported = matching_candidates(
                query, self._registry.snapshot(), self._authorization
            )
        except _GatewayPolicyEvaluationError:
            failure = GatewayFailure(
                GatewayFailureCode.POLICY_EVALUATION_FAILED,
                "Gateway policy evaluation failed",
            )
            return SelectionOutcome(SelectionStatus.FAILED, failure=failure)
        except _UnsupportedSchemaError as error:
            failure = GatewayFailure(GatewayFailureCode.UNSUPPORTED_SCHEMA, str(error))
            return SelectionOutcome(SelectionStatus.FAILED, failure=failure)
        if not matches:
            if unsupported:
                failure = GatewayFailure(
                    GatewayFailureCode.UNSUPPORTED_SCHEMA,
                    "Matching providers use unsupported JSON Schema features",
                )
                status = SelectionStatus.FAILED
            else:
                failure = GatewayFailure(
                    GatewayFailureCode.DISCOVERY_DENIED if denied else GatewayFailureCode.NO_MATCH,
                    (
                        "Capability discovery was denied"
                        if denied
                        else "No eligible capability matched the query"
                    ),
                )
                status = SelectionStatus.DENIED if denied else SelectionStatus.NO_MATCH
            return SelectionOutcome(status=status, failure=failure)
        if len(matches) == 1:
            descriptor, generation = matches[0]
            return SelectionOutcome(
                SelectionStatus.SELECTED,
                self._bindings.issue(descriptor, revision, generation),
                descriptor,
                truncated=unsupported,
            )

        capability_ids = {descriptor.capability_id for descriptor, _ in matches}
        if len(capability_ids) == 1:
            capability_id = next(iter(capability_ids))
            preferred_agent = self._preferred_agents.get(capability_id)
            if preferred_agent is not None:
                preferred = next(
                    (item for item in matches if item[0].agent_id == preferred_agent),
                    None,
                )
                if preferred is not None:
                    descriptor, generation = preferred
                    return SelectionOutcome(
                        SelectionStatus.SELECTED,
                        self._bindings.issue(descriptor, revision, generation),
                        descriptor,
                        truncated=unsupported,
                    )
        descriptors = tuple(descriptor for descriptor, _ in matches[: self._max_results])
        return SelectionOutcome(
            SelectionStatus.AMBIGUOUS,
            candidates=descriptors,
            failure=GatewayFailure(
                GatewayFailureCode.AMBIGUOUS,
                "Several eligible capabilities matched without a configured selection",
            ),
            truncated=len(matches) > self._max_results or unsupported,
        )

    async def discover_tools(self, query: DiscoveryQuery) -> ToolDiscoveryResult:
        """Project bounded candidates into safe JSON-schema tool descriptors."""
        result = await self.discover(query)
        return project_tools(result, max_serialized_bytes=self._max_serialized_bytes)

    async def invoke(
        self,
        binding: CapabilityBinding,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None = None,
        token_cost: int = 0,
        cost: float = 0,
    ) -> InvocationResult:
        """Revalidate and invoke a binding through ``Runtime.invoke``."""
        with start_span(
            SPAN_GATEWAY_INVOKE,
            attributes={
                "conducto.agent.id": self._context.agent_id,
                "conducto.target_agent.id": getattr(binding, "agent_id", ""),
                "conducto.capability.id": getattr(binding, "capability_id", ""),
                "conducto.correlation_id": self._context.correlation_id,
                "conducto.run.id": self._context.run_id,
                "conducto.transport": "in_process",
            },
        ) as span:
            self._context.require_active()
            correlation_id = self._context.correlation_id
            metadata = self._context.invocation_metadata()
            binding_failure = self._bindings.validate(binding)
            if binding_failure is not None:
                span.set_outcome("binding_failure", reason=binding_failure.value)
                return InvocationBindingFailure(
                    correlation_id,
                    binding_failure.value,
                    metadata,
                )
            frame = DelegationFrame(binding.agent_id, binding.capability_id)
            path = self._context.delegation_path
            if frame in path:
                span.set_outcome(
                    "delegation_failure",
                    reason=GatewayFailureCode.CYCLE_DETECTED.value,
                )
                return InvocationDelegationFailure(
                    correlation_id,
                    GatewayFailureCode.CYCLE_DETECTED.value,
                    tuple((item.agent_id, item.capability_id) for item in path),
                    metadata,
                )
            if self._context.remaining_delegation_budget.depth < 1:
                span.set_outcome(
                    "delegation_failure",
                    reason=GatewayFailureCode.DEPTH_EXCEEDED.value,
                )
                return InvocationDelegationFailure(
                    correlation_id,
                    GatewayFailureCode.DEPTH_EXCEEDED.value,
                    tuple((item.agent_id, item.capability_id) for item in path),
                    metadata,
                )

            (
                agent,
                agent_descriptor,
                lifecycle,
                healthy,
                generation_valid,
                schema_valid,
            ) = self._registry.accept_binding(
                agent_id=binding.agent_id,
                capability_id=binding.capability_id,
                generation=binding.registration_generation,
                schema_digest=binding.schema_digest,
            )
            if agent is None or not generation_valid:
                span.set_outcome("stale_binding", reason="stale_binding")
                return InvocationStaleBinding(
                    correlation_id,
                    binding.agent_id,
                    binding.capability_id,
                    metadata,
                )
            if not schema_valid:
                span.set_outcome("schema_mismatch", reason="schema_mismatch")
                return InvocationSchemaMismatch(
                    correlation_id,
                    binding.agent_id,
                    binding.capability_id,
                    metadata,
                )
            if lifecycle is not RegistrationLifecycle.ACTIVE or not healthy:
                reason = lifecycle.value if lifecycle is not None else "unhealthy"
                if lifecycle is RegistrationLifecycle.ACTIVE and not healthy:
                    reason = "unhealthy"
                span.set_outcome("target_unavailable", reason=reason)
                return InvocationTargetUnavailable(
                    correlation_id,
                    binding.agent_id,
                    binding.capability_id,
                    reason,
                    metadata,
                )
            assert agent_descriptor is not None
            descriptor = next(
                item
                for item in agent_descriptor.capabilities
                if item.capability_id == binding.capability_id
            )
            try:
                authorized = self._authorization.permits(descriptor, check_budget=False)
            except _GatewayPolicyEvaluationError:
                span.set_outcome(
                    "authorization_failure",
                    reason=GatewayFailureCode.POLICY_EVALUATION_FAILED.value,
                )
                return InvocationAuthorizationFailure(
                    correlation_id,
                    GatewayFailureCode.POLICY_EVALUATION_FAILED.value,
                    metadata,
                )
            if not authorized:
                span.set_outcome(
                    "authorization_failure",
                    reason=GatewayFailureCode.DISCOVERY_DENIED.value,
                )
                return InvocationAuthorizationFailure(
                    correlation_id,
                    GatewayFailureCode.DISCOVERY_DENIED.value,
                    metadata,
                )
            try:
                self._context.remaining_timeout()
            except TimeoutError:
                from ..invocation_results import InvocationTimeout

                span.set_outcome("timeout", reason="timeout")
                return InvocationTimeout(correlation_id, 0.0, metadata)
            if not self._context.delegation_budget.reserve(
                calls=1,
                tokens=token_cost,
                cost=cost,
            ):
                span.set_outcome("budget_exhausted", reason="budget_exhausted")
                return InvocationBudgetExhausted(
                    correlation_id,
                    "calls, tokens, or cost",
                    metadata,
                )
            prior_calls = self._context.model_calls()
            result = await self._runtime.invoke(
                agent,
                binding.capability_id,
                arguments,
                timeout=timeout,
                correlation_id=correlation_id,
            )
            if result.metadata is not None and prior_calls:
                result = dataclasses.replace(
                    result,
                    metadata=result.metadata.with_prior_model_calls(prior_calls),
                )
            if isinstance(result, InvocationSuccess):
                span.set_outcome("success")
            else:
                span.set_outcome(type(result).__name__)
            return result
