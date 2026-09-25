"""Application-owned projection of canonical capabilities into MCP tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from conducto.core.agent import BaseAgent
from conducto.core.data_sources import DataSourceRegistry
from conducto.core.gateway_models import CapabilityDescriptor, freeze_json, thaw_json
from conducto.core.registry import AgentRegistry
from conducto.core.runtime import Runtime
from conducto.security.context import AuthorizationContext, Principal

from .errors import McpExportError, McpNameCollisionError, McpPolicyError, McpToolNotFoundError
from .mapping import McpToolOutcome, invocation_result_to_tool_outcome
from .naming import default_tool_name
from .policy import McpExportPolicy
from .schema import project_input_schema, project_output_schema


@dataclass(frozen=True, slots=True)
class McpToolDefinition:
    """Immutable MCP projection of one canonical Conducto capability.

    Attributes:
        name: Deterministic or explicitly aliased MCP tool name.
        title: Human-readable title derived from canonical identity.
        description: Untrusted capability description, bounded by policy.
        input_schema: Canonical input schema projected into the MCP subset.
        output_schema: Structured-content schema, when a return schema exists.
        agent_id: Canonical agent identifier dispatched on invocation.
        capability_id: Canonical capability identifier dispatched on invocation.
        required_scopes: Scopes a principal must hold to see and call the tool.
        approval_required: Whether the capability declares an approval guardrail.
    """

    name: str
    title: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    agent_id: str
    capability_id: str
    required_scopes: tuple[str, ...] = ()
    approval_required: bool = False

    def __post_init__(self) -> None:
        """Freeze the projected schemas so exported metadata stays immutable."""
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", freeze_json(self.output_schema))
        object.__setattr__(self, "required_scopes", tuple(self.required_scopes))

    def input_schema_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the projected input schema."""
        return dict(thaw_json(self.input_schema))

    def output_schema_dict(self) -> dict[str, Any] | None:
        """Return a mutable copy of the projected output schema, when present."""
        if self.output_schema is None:
            return None
        return dict(thaw_json(self.output_schema))

    def is_eligible_for(self, principal: Principal | None) -> bool:
        """Report whether a principal may discover and invoke this tool.

        Args:
            principal: Immutable stdio principal, or ``None`` when the session
                is anonymous.

        Returns:
            ``True`` when the principal holds every required scope.
        """
        if not self.required_scopes:
            return True
        if principal is None:
            return False
        return set(self.required_scopes).issubset(principal.scopes)


class McpToolExporter:
    """Project policy-admitted capabilities and dispatch through the runtime.

    The exporter owns export policy, deterministic naming, collision handling,
    schema projection, and result mapping. It never reflects over callables,
    revalidates arguments, or reimplements authorization, approval, audit,
    timeout, cancellation, or serialization behavior.
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        policy: McpExportPolicy,
        agents: Sequence[BaseAgent] = (),
        registry: AgentRegistry | None = None,
        data_sources: DataSourceRegistry | None = None,
    ) -> None:
        """Build the immutable exported tool list for one server instance.

        Args:
            runtime: Runtime that executes every admitted capability.
            policy: Immutable default-deny export policy.
            agents: Agent instances exported without a shared registry.
            registry: Registry whose current snapshot supplies canonical
                capability metadata and dispatch targets.
            data_sources: Configured source metadata used when building a
                private registry for ``agents``.

        Raises:
            McpExportError: If neither agents nor a registry are supplied, or
                if a projected tool violates a configured bound.
            McpPolicyError: If an allowlist entry references an unknown target.
            McpNameCollisionError: If two admitted capabilities normalize to
                one MCP tool name without distinct aliases.
            McpSchemaProjectionError: If a canonical schema leaves the
                supported JSON Schema subset.
        """
        if registry is None and not agents:
            raise McpExportError("An MCP exporter requires agents or a registry snapshot source")
        source = (
            registry if registry is not None else _registry_for(agents, data_sources=data_sources)
        )
        self._runtime = runtime
        self._policy = policy
        self._agents: dict[str, BaseAgent] = {}
        snapshot = source.snapshot()
        definitions: dict[str, McpToolDefinition] = {}
        for descriptor in snapshot.agents:
            agent = source.get(descriptor.agent_id)
            if agent is None:  # pragma: no cover - snapshot is taken under the registry lock
                continue
            for capability in descriptor.capabilities:
                definition = self._project(capability)
                if definition is None:
                    continue
                existing = definitions.get(definition.name)
                if existing is not None:
                    raise McpNameCollisionError(
                        f"MCP tool name '{definition.name}' is claimed by "
                        f"{existing.agent_id}:{existing.capability_id} and "
                        f"{definition.agent_id}:{definition.capability_id}; "
                        "configure distinct aliases"
                    )
                definitions[definition.name] = definition
                self._agents[descriptor.agent_id] = agent
        self._verify_allowlist(definitions.values())
        if len(definitions) > policy.max_tools:
            raise McpExportError(
                f"The export policy admits {len(definitions)} tools and exceeds its "
                f"{policy.max_tools} tool bound"
            )
        self._tools = MappingProxyType(dict(sorted(definitions.items())))

    @property
    def policy(self) -> McpExportPolicy:
        """Return the immutable export policy for this exporter."""
        return self._policy

    @property
    def tools(self) -> tuple[McpToolDefinition, ...]:
        """Return every admitted tool in deterministic name order."""
        return tuple(self._tools.values())

    def list_tools(self, principal: Principal | None) -> tuple[McpToolDefinition, ...]:
        """Return the admitted tools a principal may discover.

        Args:
            principal: Immutable stdio principal, or ``None`` for an anonymous
                session.

        Returns:
            Eligible tools in deterministic name order.
        """
        return tuple(tool for tool in self._tools.values() if tool.is_eligible_for(principal))

    def tool(self, name: str) -> McpToolDefinition:
        """Return one exported tool definition by MCP name.

        Args:
            name: Exported MCP tool name.

        Returns:
            The matching tool definition.

        Raises:
            McpToolNotFoundError: If the name is not exported.
        """
        definition = self._tools.get(name)
        if definition is None:
            raise McpToolNotFoundError(f"Unknown MCP tool '{name}'")
        return definition

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: Principal | None = None,
        authorization: AuthorizationContext | None = None,
        task_id: str,
        correlation_id: str = "",
        timeout: float | None = None,
    ) -> McpToolOutcome:
        """Dispatch one MCP tool call through ``Runtime.invoke()``.

        Args:
            name: Exported MCP tool name.
            arguments: Raw MCP arguments, validated by the runtime.
            principal: Immutable stdio principal, or ``None`` for an anonymous
                session that fails closed on protected capabilities.
            authorization: Immutable authorization context supplied by an
                authenticated transport. Its principal must match
                ``principal`` when both are supplied.
            task_id: Stable task identifier for audit and approval lineage.
            correlation_id: Optional correlation identifier for the invocation.
            timeout: Earliest effective deadline for the invocation.

        Returns:
            The mapped MCP tool outcome for the invocation result.

        Raises:
            McpToolNotFoundError: If the tool is not exported or the principal
                is not eligible to invoke it.
            McpExportError: If supplied authorization and principal disagree.
        """
        definition = self.tool(name)
        effective_principal = authorization.principal if authorization is not None else principal
        if (
            principal is not None
            and authorization is not None
            and principal != authorization.principal
        ):
            raise McpExportError("principal and authorization context disagree")
        if not definition.is_eligible_for(effective_principal):
            raise McpToolNotFoundError(f"Unknown MCP tool '{name}'")
        agent = self._agents[definition.agent_id]
        effective_correlation_id = (
            authorization.correlation_id
            if authorization is not None
            else correlation_id or self._runtime.new_correlation_id()
        )
        effective_authorization = authorization or (
            None
            if effective_principal is None
            else AuthorizationContext(
                principal=effective_principal,
                task_id=task_id,
                correlation_id=effective_correlation_id,
            )
        )
        result = await self._runtime.invoke(
            agent,
            definition.capability_id,
            arguments,
            timeout=timeout,
            correlation_id=effective_correlation_id,
            authorization=effective_authorization,
        )
        return invocation_result_to_tool_outcome(result)

    def _project(self, capability: CapabilityDescriptor) -> McpToolDefinition | None:
        """Project one canonical capability descriptor admitted by policy."""
        rule = self._policy.rule_for(capability.agent_id, capability.capability_id)
        if rule is None and not self._admitted_by_query(capability):
            return None
        name = rule.alias if rule is not None and rule.alias else None
        name = name or default_tool_name(capability.agent_id, capability.capability_id)
        label = f"{capability.agent_id}:{capability.capability_id}"
        if len(name) > self._policy.max_name_length:
            raise McpExportError(
                f"{label}: MCP tool name '{name}' exceeds the configured "
                f"{self._policy.max_name_length} character bound"
            )
        description = (capability.description or label).strip()
        if len(description) > self._policy.max_description_length:
            raise McpExportError(
                f"{label}: description exceeds the configured "
                f"{self._policy.max_description_length} character bound"
            )
        input_schema = project_input_schema(
            thaw_json(capability.input_schema),
            label=label,
            max_bytes=self._policy.max_schema_bytes,
        )
        output_schema = (
            None
            if capability.output_schema is None
            else project_output_schema(
                thaw_json(capability.output_schema),
                label=label,
                max_bytes=self._policy.max_schema_bytes,
            )
        )
        return McpToolDefinition(
            name=name,
            title=label,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            agent_id=capability.agent_id,
            capability_id=capability.capability_id,
            required_scopes=capability.required_scopes,
            approval_required=capability.approval_required,
        )

    def _admitted_by_query(self, capability: CapabilityDescriptor) -> bool:
        """Report whether a bounded query admits one capability."""
        query = self._policy.query_for(capability.agent_id)
        if query is None:
            return False
        return query.tags.issubset(capability.tags)

    def _verify_allowlist(self, definitions: Iterable[McpToolDefinition]) -> None:
        """Fail construction when an allowlist entry matched no capability."""
        projected = tuple(definitions)
        exported = {(item.agent_id, item.capability_id) for item in projected}
        for rule in self._policy.rules:
            if rule.target not in exported:
                raise McpPolicyError(
                    f"Export rule {rule.agent_id}:{rule.capability_id} matches no "
                    "registered capability"
                )
        for query in self._policy.queries:
            matches = sum(1 for item in projected if item.agent_id == query.agent_id)
            if matches == 0:
                raise McpPolicyError(
                    f"Export query for agent '{query.agent_id}' matches no registered capability"
                )
            if matches > query.limit:
                raise McpPolicyError(
                    f"Export query for agent '{query.agent_id}' matches {matches} "
                    f"capabilities and exceeds its bound of {query.limit}"
                )


def _registry_for(
    agents: Sequence[BaseAgent], *, data_sources: DataSourceRegistry | None = None
) -> AgentRegistry:
    """Build a private registry so agent exports use canonical descriptors."""
    registry = AgentRegistry(data_sources=data_sources)
    for agent in agents:
        registry.register(agent)
    return registry


__all__ = ["McpToolDefinition", "McpToolExporter"]
