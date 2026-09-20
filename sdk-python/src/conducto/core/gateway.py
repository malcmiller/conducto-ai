"""Model-neutral local capability discovery and invocation gateway."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import re
import time
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from .gateway_models import (
    BoundCapability,
    CapabilityBinding,
    CapabilityDescriptor,
    DiscoveryQuery,
    DiscoveryResult,
    GatewayFailure,
    GatewayFailureCode,
    RegistrationLifecycle,
    SelectionOutcome,
    SelectionStatus,
    ToolDescriptor,
    ToolDiscoveryResult,
    canonical_json,
)
from .invocation_results import (
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationBudgetExhausted,
    InvocationDelegationFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationTargetUnavailable,
)
from .registry import AgentRegistry
from .run_context import DelegationFrame, RunContext

if TYPE_CHECKING:
    from .runtime import Runtime

GatewayPolicy = Callable[[RunContext, CapabilityDescriptor], bool]


class AgentGateway(Protocol):
    """Async contract consumed by agents for discovery and local dispatch."""

    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult: ...

    async def lookup(self, agent_id: str, capability_id: str) -> SelectionOutcome: ...

    async def select(self, query: DiscoveryQuery) -> SelectionOutcome: ...

    async def discover_tools(self, query: DiscoveryQuery) -> ToolDiscoveryResult: ...

    async def invoke(
        self,
        binding: CapabilityBinding,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None = None,
        token_cost: int = 0,
        cost: float = 0,
    ) -> InvocationResult: ...


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
        binding_ttl = binding_ttl or runtime.gateway_binding_ttl
        max_results = max_results or runtime.gateway_max_results
        max_serialized_bytes = max_serialized_bytes or runtime.gateway_max_serialized_bytes
        if binding_ttl <= 0:
            raise ValueError("Gateway binding TTL must be positive")
        if max_results < 1 or max_serialized_bytes < 1:
            raise ValueError("Gateway result limits must be positive")
        self._runtime = runtime
        self._registry = registry
        self._context = context
        self._policy = policy if policy is not None else runtime.gateway_policy
        self._preferred_agents = dict(
            preferred_agents if preferred_agents is not None else runtime.gateway_preferred_agents
        )
        self._binding_ttl = binding_ttl
        self._max_results = max_results
        self._max_serialized_bytes = max_serialized_bytes

    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Discover authorized compatible candidates from one registry snapshot."""
        self._context.require_active()
        snapshot = self._registry.snapshot()
        matches: list[tuple[CapabilityDescriptor, int]] = []
        denied = False
        for agent in snapshot.agents:
            if agent.lifecycle is not RegistrationLifecycle.ACTIVE or not agent.healthy:
                continue
            if query.agent_id is not None and agent.agent_id != query.agent_id:
                continue
            if query.version_constraint and not _version_matches(
                agent.version,
                query.version_constraint,
            ):
                continue
            for descriptor in agent.capabilities:
                if query.capability_ids and descriptor.capability_id not in query.capability_ids:
                    continue
                if query.tags and not query.tags.issubset(descriptor.tags):
                    continue
                if not _schema_compatible(query.input_schema, descriptor.input_schema):
                    continue
                if not _schema_compatible(query.output_schema, descriptor.output_schema):
                    continue
                if not query.include_approval_required and descriptor.approval_required:
                    continue
                if not self._is_authorized(descriptor):
                    denied = True
                    continue
                matches.append((descriptor, agent.generation))

        matches.sort(
            key=lambda item: (
                item[0].capability_id,
                item[0].agent_id,
                item[0].agent_version,
                item[0].schema_digest,
            )
        )
        limit = min(query.limit, self._max_results)
        candidates = tuple(
            BoundCapability(
                descriptor,
                self._issue_binding(descriptor, snapshot.revision, generation),
            )
            for descriptor, generation in matches[:limit]
        )
        failure = None
        if not candidates:
            code = GatewayFailureCode.DISCOVERY_DENIED if denied else GatewayFailureCode.NO_MATCH
            message = (
                "Capability discovery was denied"
                if denied
                else "No eligible capability matched the query"
            )
            failure = GatewayFailure(code, message)
        return DiscoveryResult(snapshot.revision, candidates, failure)

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
        result = await self.discover(dataclasses.replace(query, limit=self._max_results))
        if not result:
            status = (
                SelectionStatus.DENIED
                if result.failure and result.failure.code is GatewayFailureCode.DISCOVERY_DENIED
                else SelectionStatus.NO_MATCH
            )
            return SelectionOutcome(status=status, failure=result.failure)
        if len(result) == 1:
            candidate = result[0]
            return SelectionOutcome(
                SelectionStatus.SELECTED,
                candidate.binding,
                candidate.descriptor,
            )

        capability_ids = {item.descriptor.capability_id for item in result}
        if len(capability_ids) == 1:
            capability_id = next(iter(capability_ids))
            preferred_agent = self._preferred_agents.get(capability_id)
            if preferred_agent is not None:
                preferred = next(
                    (item for item in result if item.descriptor.agent_id == preferred_agent),
                    None,
                )
                if preferred is not None:
                    return SelectionOutcome(
                        SelectionStatus.SELECTED,
                        preferred.binding,
                        preferred.descriptor,
                    )
        descriptors = tuple(item.descriptor for item in result)
        return SelectionOutcome(
            SelectionStatus.AMBIGUOUS,
            candidates=descriptors,
            failure=GatewayFailure(
                GatewayFailureCode.AMBIGUOUS,
                "Several eligible capabilities matched without a configured selection",
            ),
        )

    async def discover_tools(self, query: DiscoveryQuery) -> ToolDiscoveryResult:
        """Project bounded candidates into safe JSON-schema tool descriptors."""
        result = await self.discover(query)
        tools: list[ToolDescriptor] = []
        serialized_size = 2
        for candidate in result:
            descriptor = candidate.descriptor
            suffix = hashlib.sha256(
                f"{descriptor.agent_id}\0{descriptor.capability_id}".encode()
            ).hexdigest()[:12]
            stem = _tool_slug(f"{descriptor.agent_id}__{descriptor.capability_id}")
            tool = ToolDescriptor(
                tool_id=f"conducto_{suffix}",
                name=f"{stem}_{suffix}",
                description=descriptor.description or descriptor.capability_id,
                input_schema=descriptor.input_schema,
                binding=candidate.binding,
            )
            encoded_size = len(canonical_json(tool.to_dict()).encode("utf-8"))
            if serialized_size + encoded_size > self._max_serialized_bytes:
                if not tools:
                    return ToolDiscoveryResult(
                        result.registry_revision,
                        failure=GatewayFailure(
                            GatewayFailureCode.RESULT_LIMIT_EXCEEDED,
                            "The first tool exceeds the configured serialized-size limit",
                        ),
                    )
                break
            tools.append(tool)
            serialized_size += encoded_size + 1
        return ToolDiscoveryResult(
            result.registry_revision,
            tuple(tools),
            result.failure if not tools else None,
        )

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
        self._context.require_active()
        correlation_id = self._context.correlation_id
        metadata = self._context.invocation_metadata()
        binding_failure = self._validate_binding(binding)
        if binding_failure is not None:
            return InvocationBindingFailure(
                correlation_id,
                binding_failure.value,
                metadata,
            )
        frame = DelegationFrame(binding.agent_id, binding.capability_id)
        path = self._context.delegation_path
        if frame in path:
            return InvocationDelegationFailure(
                correlation_id,
                GatewayFailureCode.CYCLE_DETECTED.value,
                tuple((item.agent_id, item.capability_id) for item in path),
                metadata,
            )
        if len(path) >= self._context.delegation_budget.max_depth:
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
            return InvocationStaleBinding(
                correlation_id,
                binding.agent_id,
                binding.capability_id,
                metadata,
            )
        if not schema_valid:
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
        if not self._is_authorized(descriptor, check_budget=False):
            return InvocationAuthorizationFailure(
                correlation_id,
                GatewayFailureCode.DISCOVERY_DENIED.value,
                metadata,
            )
        try:
            self._context.remaining_timeout()
        except TimeoutError:
            from .invocation_results import InvocationTimeout

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
        return result

    def _is_authorized(
        self,
        descriptor: CapabilityDescriptor,
        *,
        check_budget: bool = True,
    ) -> bool:
        allowed = self._context.allowed_capabilities
        exact_id = f"{descriptor.agent_id}:{descriptor.capability_id}"
        if allowed is not None and not (descriptor.capability_id in allowed or exact_id in allowed):
            return False
        authorization = self._context.authorization
        scopes = authorization.principal.scopes if authorization is not None else frozenset()
        if not set(descriptor.required_scopes).issubset(scopes):
            return False
        if self._policy is not None:
            try:
                if not self._policy(self._context, descriptor):
                    return False
            except Exception:
                return False
        if check_budget:
            budget = self._context.remaining_delegation_budget
            if budget.depth < 1 or budget.calls < 1 or budget.time == 0:
                return False
        frame = DelegationFrame(descriptor.agent_id, descriptor.capability_id)
        return frame not in self._context.delegation_path

    def _issue_binding(
        self,
        descriptor: CapabilityDescriptor,
        revision: int,
        generation: int,
    ) -> CapabilityBinding:
        issued_at = time.monotonic()
        values = {
            "agent_id": descriptor.agent_id,
            "capability_id": descriptor.capability_id,
            "schema_digest": descriptor.schema_digest,
            "registry_revision": revision,
            "registration_generation": generation,
            "runtime_id": self._runtime._gateway_runtime_id,
            "issued_at": issued_at,
            "expires_at": issued_at + self._binding_ttl,
            "nonce": uuid.uuid4().hex,
        }
        signature = hmac.new(
            self._runtime._gateway_secret,
            canonical_json(values).encode(),
            hashlib.sha256,
        ).hexdigest()
        return CapabilityBinding(
            agent_id=descriptor.agent_id,
            capability_id=descriptor.capability_id,
            schema_digest=descriptor.schema_digest,
            registry_revision=revision,
            registration_generation=generation,
            runtime_id=self._runtime._gateway_runtime_id,
            issued_at=issued_at,
            expires_at=issued_at + self._binding_ttl,
            nonce=str(values["nonce"]),
            signature=signature,
        )

    def _validate_binding(self, binding: CapabilityBinding) -> GatewayFailureCode | None:
        if not isinstance(binding, CapabilityBinding):
            return GatewayFailureCode.INVALID_BINDING
        if binding.runtime_id != self._runtime._gateway_runtime_id:
            return GatewayFailureCode.FOREIGN_RUNTIME
        if binding.expires_at <= time.monotonic():
            return GatewayFailureCode.EXPIRED_BINDING
        values = {
            "agent_id": binding.agent_id,
            "capability_id": binding.capability_id,
            "schema_digest": binding.schema_digest,
            "registry_revision": binding.registry_revision,
            "registration_generation": binding.registration_generation,
            "runtime_id": binding.runtime_id,
            "issued_at": binding.issued_at,
            "expires_at": binding.expires_at,
            "nonce": binding.nonce,
        }
        expected = hmac.new(
            self._runtime._gateway_secret,
            canonical_json(values).encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(binding.signature, expected):
            return GatewayFailureCode.INVALID_BINDING
        return None


def _tool_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return (slug or "capability")[:48]


def _version_tuple(value: str) -> tuple[int, int, int, str]:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?([-.+].*)?\s*", value)
    if match is None:
        return (0, 0, 0, value)
    return (
        int(match.group(1)),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
        match.group(4) or "",
    )


def _version_matches(version: str, constraint: str) -> bool:
    actual = _version_tuple(version)
    for clause in constraint.split(","):
        match = re.fullmatch(r"\s*(==|!=|>=|<=|>|<)?\s*(\S+)\s*", clause)
        if match is None:
            return False
        operator = match.group(1) or "=="
        expected = _version_tuple(match.group(2))
        comparisons = {
            "==": actual == expected,
            "!=": actual != expected,
            ">=": actual >= expected,
            "<=": actual <= expected,
            ">": actual > expected,
            "<": actual < expected,
        }
        if not comparisons[operator]:
            return False
    return True


def _schema_compatible(
    requested: Mapping[str, Any] | None,
    offered: Mapping[str, Any] | None,
) -> bool:
    if requested is None:
        return True
    if offered is None:
        return False
    requested_type = requested.get("type")
    offered_type = offered.get("type")
    if requested_type is not None and offered_type != requested_type:
        return False
    requested_properties = requested.get("properties")
    offered_properties = offered.get("properties")
    if isinstance(requested_properties, Mapping):
        if not isinstance(offered_properties, Mapping):
            return False
        for name, schema in requested_properties.items():
            offered_schema = offered_properties.get(name)
            if not isinstance(schema, Mapping) or not isinstance(offered_schema, Mapping):
                return False
            if not _schema_compatible(schema, offered_schema):
                return False
    return True
