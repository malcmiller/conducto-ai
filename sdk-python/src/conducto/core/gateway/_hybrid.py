"""Composed local and remote capability discovery and invocation gateway."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..catalog import (
    AgentCatalog,
    AgentInstanceRecord,
    CatalogAgentRecord,
    CatalogCapabilityDescriptor,
    CatalogLifecycleState,
    DeploymentType,
)
from ..gateway_models import (
    BoundCapability,
    CapabilityBinding,
    CapabilityDescriptor,
    DiscoveryQuery,
    DiscoveryResult,
    GatewayFailure,
    GatewayFailureCode,
    SelectionOutcome,
    SelectionStatus,
    ToolDiscoveryResult,
    canonical_json,
)
from ..invocation_results import (
    InvocationAuthorizationFailure,
    InvocationBudgetExhausted,
    InvocationFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationTargetUnavailable,
)
from ..registry import AgentRegistry
from ..run_context import RunContext
from ._contracts import (
    GatewayRemotePolicy,
    GatewaySelectionMode,
    GatewaySelectionPolicy,
    RemoteGatewayTransport,
    RemoteTransportError,
)
from ._discovery import _GatewayPolicyEvaluationError, _version_matches, matching_candidates
from ._local import LocalAgentGateway
from ._projection import project_tools
from ._schema import (
    _schema_compatible,
    _UnsupportedSchemaError,
    _validate_compatibility_schema,
)


@dataclass(frozen=True, slots=True)
class _GatewayMatch:
    descriptor: CapabilityDescriptor
    generation: int
    source: str
    agent_record: CatalogAgentRecord | None = None
    capability_record: CatalogCapabilityDescriptor | None = None


@dataclass(frozen=True, slots=True)
class _RemoteBindingState:
    agent_snapshot: CatalogAgentRecord
    capability_snapshot: CatalogCapabilityDescriptor
    snapshot_revision: int
    selection_policy: GatewaySelectionPolicy
    source: str = "remote"


class HybridAgentGateway(LocalAgentGateway):
    """Compose local registry discovery with catalog-backed remote dispatch."""

    def __init__(
        self,
        runtime: Any,
        registry: AgentRegistry,
        catalog: AgentCatalog,
        transport: RemoteGatewayTransport,
        context: RunContext,
        *,
        policy: Any = None,
        remote_policy: GatewayRemotePolicy | None = None,
        allowed_deployments: frozenset[DeploymentType] | None = None,
        selection_policy: GatewaySelectionPolicy | None = None,
        preferred_agents: Mapping[str, str] | None = None,
        binding_ttl: float | None = None,
        max_results: int | None = None,
        max_serialized_bytes: int | None = None,
    ) -> None:
        super().__init__(
            runtime,
            registry,
            context,
            policy=policy,
            preferred_agents=preferred_agents,
            binding_ttl=binding_ttl,
            max_results=max_results,
            max_serialized_bytes=max_serialized_bytes,
        )
        self._catalog = catalog
        self._transport = transport
        self._remote_policy = (
            remote_policy if remote_policy is not None else runtime.gateway_remote_policy
        )
        self._allowed_deployments = (
            frozenset(allowed_deployments)
            if allowed_deployments is not None
            else runtime.gateway_allowed_deployments
        )
        self._selection_policy = selection_policy or runtime.gateway_selection_policy

    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Discover authorized compatible candidates from local and remote snapshots."""
        self._context.require_active()
        try:
            revision, matches, denied, unsupported, no_eligible = self._matching_candidates(query)
        except _GatewayPolicyEvaluationError:
            return DiscoveryResult(
                max(self._registry.revision, self._catalog.revision),
                failure=GatewayFailure(
                    GatewayFailureCode.POLICY_EVALUATION_FAILED,
                    "Gateway policy evaluation failed",
                ),
            )
        except _UnsupportedSchemaError as error:
            return DiscoveryResult(
                max(self._registry.revision, self._catalog.revision),
                failure=GatewayFailure(
                    GatewayFailureCode.UNSUPPORTED_SCHEMA,
                    str(error),
                ),
            )
        limit = min(query.limit, self._max_results)
        candidates = tuple(
            BoundCapability(
                item.descriptor,
                self._issue_match_binding(item, revision),
            )
            for item in matches[:limit]
        )
        failure = None
        if not candidates:
            if no_eligible:
                failure = GatewayFailure(
                    GatewayFailureCode.NO_ELIGIBLE_INSTANCE,
                    "No eligible runtime instance matched the query",
                )
            elif unsupported:
                failure = GatewayFailure(
                    GatewayFailureCode.UNSUPPORTED_SCHEMA,
                    "Matching providers use unsupported JSON Schema features",
                )
            elif denied:
                failure = GatewayFailure(
                    GatewayFailureCode.DISCOVERY_DENIED,
                    "Capability discovery was denied",
                )
            else:
                failure = GatewayFailure(
                    GatewayFailureCode.NO_MATCH,
                    "No eligible capability matched the query",
                )
        return DiscoveryResult(
            revision,
            candidates,
            failure,
            truncated=len(matches) > limit or (unsupported and bool(candidates)),
        )

    async def select(self, query: DiscoveryQuery) -> SelectionOutcome:
        """Select one local or remote capability deterministically."""
        self._context.require_active()
        try:
            revision, matches, denied, unsupported, no_eligible = self._matching_candidates(query)
        except _GatewayPolicyEvaluationError:
            return SelectionOutcome(
                SelectionStatus.FAILED,
                failure=GatewayFailure(
                    GatewayFailureCode.POLICY_EVALUATION_FAILED,
                    "Gateway policy evaluation failed",
                ),
            )
        except _UnsupportedSchemaError as error:
            return SelectionOutcome(
                SelectionStatus.FAILED,
                failure=GatewayFailure(GatewayFailureCode.UNSUPPORTED_SCHEMA, str(error)),
            )
        if not matches:
            if no_eligible:
                return SelectionOutcome(
                    SelectionStatus.FAILED,
                    failure=GatewayFailure(
                        GatewayFailureCode.NO_ELIGIBLE_INSTANCE,
                        "No eligible runtime instance matched the query",
                    ),
                )
            if unsupported:
                return SelectionOutcome(
                    SelectionStatus.FAILED,
                    failure=GatewayFailure(
                        GatewayFailureCode.UNSUPPORTED_SCHEMA,
                        "Matching providers use unsupported JSON Schema features",
                    ),
                )
            if denied:
                return SelectionOutcome(
                    SelectionStatus.DENIED,
                    failure=GatewayFailure(
                        GatewayFailureCode.DISCOVERY_DENIED,
                        "Capability discovery was denied",
                    ),
                )
            return SelectionOutcome(
                SelectionStatus.NO_MATCH,
                failure=GatewayFailure(
                    GatewayFailureCode.NO_MATCH,
                    "No eligible capability matched the query",
                ),
            )
        if len(matches) == 1:
            match = matches[0]
            return SelectionOutcome(
                SelectionStatus.SELECTED,
                self._issue_match_binding(match, revision),
                match.descriptor,
                truncated=unsupported,
            )
        capability_ids = {item.descriptor.capability_id for item in matches}
        if len(capability_ids) == 1:
            capability_id = next(iter(capability_ids))
            preferred_agent = self._preferred_agents.get(capability_id)
            if preferred_agent is not None:
                preferred = next(
                    (item for item in matches if item.descriptor.agent_id == preferred_agent),
                    None,
                )
                if preferred is not None:
                    return SelectionOutcome(
                        SelectionStatus.SELECTED,
                        self._issue_match_binding(preferred, revision),
                        preferred.descriptor,
                        truncated=unsupported,
                    )
        selected = self._select_match(query, matches)
        if selected is not None:
            return SelectionOutcome(
                SelectionStatus.SELECTED,
                self._issue_match_binding(selected, revision),
                selected.descriptor,
                truncated=unsupported,
            )
        return SelectionOutcome(
            SelectionStatus.AMBIGUOUS,
            candidates=tuple(item.descriptor for item in matches[: self._max_results]),
            failure=GatewayFailure(
                GatewayFailureCode.AMBIGUOUS,
                "Several eligible capabilities matched without a configured selection",
            ),
            truncated=len(matches) > self._max_results or unsupported,
        )

    async def discover_tools(self, query: DiscoveryQuery) -> ToolDiscoveryResult:
        """Project local and remote candidates into bounded model-facing tools."""
        result = await self.discover(query)
        return project_tools(result, max_serialized_bytes=self._max_serialized_bytes)

    async def _invoke_resolved_binding(
        self,
        binding: CapabilityBinding,
        state: object | None,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None,
        token_cost: int,
        cost: float,
    ) -> InvocationResult:
        if isinstance(state, _RemoteBindingState):
            return await self._invoke_remote_binding(
                binding,
                state,
                arguments,
                timeout=timeout,
                token_cost=token_cost,
                cost=cost,
            )
        return await super()._invoke_resolved_binding(
            binding,
            state,
            arguments,
            timeout=timeout,
            token_cost=token_cost,
            cost=cost,
        )

    def _matching_candidates(
        self,
        query: DiscoveryQuery,
    ) -> tuple[int, list[_GatewayMatch], bool, bool, bool]:
        local_revision, local_matches, denied_local, unsupported_local = matching_candidates(
            query,
            self._registry.snapshot(),
            self._authorization,
        )
        remote_revision, remote_matches, denied_remote, unsupported_remote, no_eligible_remote = (
            self._matching_remote_candidates(query)
        )
        matches = [
            _GatewayMatch(descriptor=descriptor, generation=generation, source="local")
            for descriptor, generation in local_matches
        ]
        matches.extend(remote_matches)
        matches.sort(
            key=lambda item: (
                item.descriptor.capability_id,
                item.descriptor.agent_id,
                item.descriptor.agent_version,
                item.descriptor.schema_digest,
                item.source,
            )
        )
        return (
            max(local_revision, remote_revision),
            matches,
            denied_local or denied_remote,
            unsupported_local or unsupported_remote,
            no_eligible_remote,
        )

    def _matching_remote_candidates(
        self,
        query: DiscoveryQuery,
    ) -> tuple[int, list[_GatewayMatch], bool, bool, bool]:
        if query.input_schema is not None:
            _validate_compatibility_schema(query.input_schema)
        if query.output_schema is not None:
            _validate_compatibility_schema(query.output_schema)
        snapshot = self._catalog.snapshot()
        matches: list[_GatewayMatch] = []
        denied = False
        unsupported = False
        no_eligible = False
        for agent in snapshot.agents:
            if query.agent_id is not None and agent.agent_id != query.agent_id:
                continue
            eligible_instances = self._eligible_instances(agent)
            for capability in agent.capabilities:
                descriptor = capability.to_capability_descriptor(
                    agent_id=agent.agent_id,
                    agent_version=capability.version,
                )
                if query.capability_ids and descriptor.capability_id not in query.capability_ids:
                    continue
                if query.tags and not query.tags.issubset(descriptor.tags):
                    continue
                if query.version_constraint and not _version_matches(
                    descriptor.agent_version,
                    query.version_constraint,
                ):
                    continue
                try:
                    input_compatible = _schema_compatible(
                        query.input_schema,
                        descriptor.input_schema,
                        output=False,
                    )
                    output_compatible = _schema_compatible(
                        query.output_schema,
                        descriptor.output_schema,
                        output=True,
                    )
                except _UnsupportedSchemaError:
                    unsupported = True
                    continue
                if not input_compatible or not output_compatible:
                    continue
                if not query.include_approval_required and descriptor.approval_required:
                    continue
                if not eligible_instances:
                    no_eligible = True
                    continue
                if not self._authorization.permits(descriptor):
                    denied = True
                    continue
                if self._remote_policy is not None:
                    try:
                        if not self._remote_policy(self._context, agent, capability):
                            denied = True
                            continue
                    except Exception as error:
                        raise _GatewayPolicyEvaluationError from error
                matches.append(
                    _GatewayMatch(
                        descriptor=descriptor,
                        generation=agent.generation,
                        source="remote",
                        agent_record=agent,
                        capability_record=capability,
                    )
                )
        return snapshot.revision, matches, denied, unsupported, no_eligible

    def _eligible_instances(self, agent: CatalogAgentRecord) -> tuple[AgentInstanceRecord, ...]:
        """Return transport- and deployment-eligible instances in stable order."""
        eligible: list[AgentInstanceRecord] = []
        for instance in sorted(agent.instances, key=lambda item: item.instance_id):
            if (
                self._allowed_deployments is not None
                and instance.deployment_type not in self._allowed_deployments
            ):
                continue
            if instance.transports and not (
                instance.transports & self._transport.supported_transports
            ):
                continue
            eligible.append(instance)
        return tuple(eligible)

    def _issue_match_binding(
        self,
        match: _GatewayMatch,
        revision: int,
    ) -> CapabilityBinding:
        if match.source == "local":
            return self._issue_binding(match.descriptor, revision, match.generation)
        assert match.agent_record is not None
        assert match.capability_record is not None
        return self._bindings.issue(
            match.descriptor,
            revision,
            match.generation,
            state=_RemoteBindingState(
                agent_snapshot=match.agent_record,
                capability_snapshot=match.capability_record,
                snapshot_revision=revision,
                selection_policy=self._selection_policy,
            ),
        )

    def _select_match(
        self,
        query: DiscoveryQuery,
        matches: list[_GatewayMatch],
    ) -> _GatewayMatch | None:
        """Select one eligible candidate according to the configured strategy."""
        mode = self._selection_policy.mode
        if mode is GatewaySelectionMode.AMBIGUOUS:
            return None
        ordered = list(matches)
        if mode is GatewaySelectionMode.LOCAL_PREFERRED:
            ordered.sort(key=lambda item: (item.source != "local", item.descriptor.agent_id))
            return ordered[0]
        if mode is GatewaySelectionMode.REMOTE_PREFERRED:
            ordered.sort(key=lambda item: (item.source != "remote", item.descriptor.agent_id))
            return ordered[0]
        if mode is GatewaySelectionMode.ROUND_ROBIN:
            index = self._runtime.next_gateway_selection_index(
                self._selection_key(query, prefix="candidate"),
                len(ordered),
            )
            return ordered[index]
        if mode is GatewaySelectionMode.STICKY_TASK:
            return ordered[self._sticky_index(ordered, task_scoped=True)]
        if mode is GatewaySelectionMode.STICKY_SESSION:
            return ordered[self._sticky_index(ordered, task_scoped=False)]
        return None

    def _selection_key(self, query: DiscoveryQuery, *, prefix: str) -> str:
        return (
            prefix
            + "\0"
            + canonical_json(
                {
                    "agent_id": query.agent_id,
                    "capability_ids": sorted(query.capability_ids),
                    "tags": sorted(query.tags),
                    "version_constraint": query.version_constraint,
                    "input_schema": query.input_schema,
                    "output_schema": query.output_schema,
                }
            )
        )

    def _sticky_index(self, values: list[Any], *, task_scoped: bool) -> int:
        task_id = (
            self._context.authorization.task_id if self._context.authorization is not None else ""
        )
        key = task_id if task_scoped and task_id else self._context.run_id
        if not task_scoped:
            key = self._context.correlation_id
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(values)

    async def _invoke_remote_binding(
        self,
        binding: CapabilityBinding,
        state: _RemoteBindingState,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None,
        token_cost: int,
        cost: float,
    ) -> InvocationResult:
        correlation_id = self._context.correlation_id
        metadata = self._context.invocation_metadata()
        current = self._catalog.get(binding.agent_id)
        if current is None:
            return InvocationStaleBinding(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                metadata,
            )
        if current.lifecycle is not CatalogLifecycleState.ACTIVE:
            return InvocationTargetUnavailable(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                current.lifecycle.value,
                metadata,
            )
        capability = next(
            (item for item in current.capabilities if item.name == binding.capability_id),
            None,
        )
        if capability is None or current.generation != state.agent_snapshot.generation:
            return InvocationStaleBinding(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                metadata,
            )
        descriptor = capability.to_capability_descriptor(
            agent_id=current.agent_id,
            agent_version=capability.version,
        )
        if descriptor.schema_digest != binding.schema_digest:
            return InvocationSchemaMismatch(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                metadata,
            )
        try:
            authorized = self._authorization.permits(descriptor)
        except _GatewayPolicyEvaluationError:
            return InvocationAuthorizationFailure(
                correlation_id,
                GatewayFailureCode.POLICY_EVALUATION_FAILED.value,
                metadata,
            )
        if not authorized:
            return InvocationAuthorizationFailure(
                correlation_id,
                GatewayFailureCode.DISCOVERY_DENIED.value,
                metadata,
            )
        if self._remote_policy is not None:
            try:
                if not self._remote_policy(self._context, current, capability):
                    return InvocationAuthorizationFailure(
                        correlation_id,
                        GatewayFailureCode.DISCOVERY_DENIED.value,
                        metadata,
                    )
            except Exception:
                return InvocationAuthorizationFailure(
                    correlation_id,
                    GatewayFailureCode.POLICY_EVALUATION_FAILED.value,
                    metadata,
                )
        try:
            self._context.remaining_timeout()
        except TimeoutError:
            from ..invocation_results import InvocationTimeout

            return InvocationTimeout(correlation_id, 0.0, metadata)
        if not self._context.delegation_budget.reserve(
            calls=1,
            tokens=token_cost,
            cost=cost,
        ):
            return InvocationBudgetExhausted(
                correlation_id,
                "calls, tokens, or cost",
                metadata,
            )
        instances = list(self._eligible_instances(current))
        if not instances:
            return InvocationTargetUnavailable(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                GatewayFailureCode.NO_ELIGIBLE_INSTANCE.value,
                metadata,
            )
        ordered_instances = self._order_instances(
            instances,
            current=current,
            capability=capability,
            mode=state.selection_policy.mode,
        )
        prior_calls = self._context.model_calls()
        for index, instance in enumerate(ordered_instances):
            try:
                result = await self._transport.invoke(
                    runtime=self._runtime,
                    context=self._context,
                    agent=current,
                    instance=instance,
                    capability=capability,
                    arguments=arguments,
                    timeout=timeout,
                )
            except RemoteTransportError as error:
                last_error = error
                can_retry = (
                    state.selection_policy.allow_pre_acceptance_failover
                    and not error.acceptance_uncertain
                    and index + 1 < len(ordered_instances)
                )
                if can_retry:
                    continue
                return InvocationFailure(
                    correlation_id,
                    (
                        f"Remote transport boundary '{error.boundary}' failed for "
                        f"{binding.agent_id}:{binding.capability_id}"
                    ),
                    error,
                    metadata,
                )
            if result.metadata is not None and prior_calls:
                result = dataclasses.replace(
                    result,
                    metadata=result.metadata.with_prior_model_calls(prior_calls),
                )
            return result
        return InvocationFailure(
            correlation_id,
            (
                f"Remote transport boundary '{self._transport.boundary_name}' failed for "
                f"{binding.agent_id}:{binding.capability_id}"
            ),
            last_error,
            metadata,
        )

    def _order_instances(
        self,
        instances: list[AgentInstanceRecord],
        *,
        current: CatalogAgentRecord,
        capability: CatalogCapabilityDescriptor,
        mode: GatewaySelectionMode,
    ) -> list[AgentInstanceRecord]:
        ordered = list(instances)
        if mode is GatewaySelectionMode.ROUND_ROBIN:
            index = self._runtime.next_gateway_selection_index(
                f"instance\0{current.agent_id}\0{capability.name}",
                len(ordered),
            )
            return ordered[index:] + ordered[:index]
        if mode is GatewaySelectionMode.STICKY_TASK:
            index = self._sticky_index(ordered, task_scoped=True)
            return ordered[index:] + ordered[:index]
        if mode is GatewaySelectionMode.STICKY_SESSION:
            index = self._sticky_index(ordered, task_scoped=False)
            return ordered[index:] + ordered[:index]
        return ordered
