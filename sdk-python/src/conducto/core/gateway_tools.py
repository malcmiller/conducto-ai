"""Model-facing toolbox contracts projected from ``AgentGateway`` discovery.

This module bridges the model-neutral ``conducto.core.gateway`` contract and
immutable bindings in ``conducto.core.gateway_models`` with the execution
loop in ``conducto.core.delegation``. It lets an agent author declare which
capability families an agent may use -- independent of concrete provider
agent IDs -- and lets the runtime project those declarations into an
immutable, bounded, schema-valid toolbox for exactly one model decision
boundary.

Nothing in this module selects a target, invokes a capability, or calls a
model. It only turns already-authorized gateway candidates into safe,
provider-neutral tool definitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .gateway import AgentGateway
from .gateway_models import (
    CapabilityBinding,
    DiscoveryQuery,
    GatewayFailure,
    GatewayFailureCode,
    ToolDescriptor,
    canonical_json,
)
from .provider import ProviderToolDefinition

__all__ = [
    "CapabilityUse",
    "CapabilityUseRequirement",
    "ToolboxPolicy",
    "ToolboxResult",
    "ToolboxSnapshot",
    "ToolboxStatus",
    "ToolboxUseFailure",
    "build_toolbox",
]

# Upper bound on a single declared use's per-query result limit. This keeps
# a capability declaration from requesting an effectively unbounded result
# set regardless of the configured ``ToolboxPolicy`` totals.
_MAX_USE_LIMIT = 100

_DEFAULT_MAX_TOOLS = 20
_DEFAULT_MAX_TOOL_SCHEMA_BYTES = 8 * 1024
_DEFAULT_MAX_TOOL_DESCRIPTION_BYTES = 1024
_DEFAULT_MAX_TOTAL_BYTES = 64 * 1024


class CapabilityUseRequirement(StrEnum):
    """Whether a declared capability use must be satisfiable before a model call."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class CapabilityUse:
    """Immutable declaration of one capability family an agent may use.

    A use is expressed as capability IDs and/or tags plus optional version
    and schema compatibility constraints, never as a concrete provider agent
    ID, endpoint, callable, or credential.

    Attributes:
        capability_ids: Exact capability identifiers eligible for this use.
        tags: Tags every eligible capability must carry.
        version_constraint: Optional semantic version constraint for providers.
        input_schema: Optional required input schema compatibility.
        output_schema: Optional required output schema compatibility.
        limit: Maximum candidates requested for this use from one query.
        requirement: Whether a missing capability fails toolbox construction.
    """

    capability_ids: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    version_constraint: str | None = None
    input_schema: Mapping[str, Any] | None = None
    output_schema: Mapping[str, Any] | None = None
    limit: int = 5
    requirement: CapabilityUseRequirement = CapabilityUseRequirement.OPTIONAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_ids", frozenset(self.capability_ids))
        object.__setattr__(self, "tags", frozenset(self.tags))
        if not self.capability_ids and not self.tags:
            raise ValueError("A capability use must declare capability_ids or tags")
        if any(not capability_id.strip() for capability_id in self.capability_ids):
            raise ValueError("capability_ids cannot contain empty identifiers")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("tags cannot contain empty identifiers")
        if self.version_constraint is not None and not self.version_constraint.strip():
            raise ValueError("version_constraint cannot be blank")
        if self.limit < 1:
            raise ValueError("Capability use limit must be positive")
        if self.limit > _MAX_USE_LIMIT:
            raise ValueError(
                f"Capability use limit cannot exceed {_MAX_USE_LIMIT} (unbounded requests"
                " are not permitted)"
            )

    @property
    def required(self) -> bool:
        """Return whether this use must be satisfied before a model call."""
        return self.requirement is CapabilityUseRequirement.REQUIRED

    def to_query(self, *, include_approval_required: bool = True) -> DiscoveryQuery:
        """Translate this declaration into a Story 3.1 discovery query.

        Args:
            include_approval_required: Whether approval-gated capabilities are eligible.

        Returns:
            A bounded, deterministic ``DiscoveryQuery`` for this use.
        """
        return DiscoveryQuery(
            capability_ids=self.capability_ids,
            tags=self.tags,
            version_constraint=self.version_constraint,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            limit=self.limit,
            include_approval_required=include_approval_required,
        )


@dataclass(frozen=True, slots=True)
class ToolboxPolicy:
    """Immutable declaration of the capability families one agent may use.

    Attributes:
        uses: Declared capability uses, independent of any provider agent ID.
        max_tools: Maximum number of tools across all declared uses.
        max_tool_schema_bytes: Maximum canonical JSON size of one tool's input schema.
        max_tool_description_bytes: Maximum encoded size of one tool's description.
        max_total_bytes: Maximum canonical JSON size of the full projected toolbox.
    """

    uses: tuple[CapabilityUse, ...] = ()
    max_tools: int = _DEFAULT_MAX_TOOLS
    max_tool_schema_bytes: int = _DEFAULT_MAX_TOOL_SCHEMA_BYTES
    max_tool_description_bytes: int = _DEFAULT_MAX_TOOL_DESCRIPTION_BYTES
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "uses", tuple(self.uses))
        if self.max_tools < 1:
            raise ValueError("max_tools must be positive")
        if self.max_tool_schema_bytes < 1 or self.max_tool_description_bytes < 1:
            raise ValueError("Per-tool size limits must be positive")
        if self.max_total_bytes < 1:
            raise ValueError("max_total_bytes must be positive")
        seen: set[tuple[frozenset[str], frozenset[str]]] = set()
        for use in self.uses:
            key = (use.capability_ids, use.tags)
            if key in seen:
                raise ValueError(
                    "Duplicate capability use declaration for the same"
                    f" capability_ids/tags: {sorted(use.capability_ids)!r}/{sorted(use.tags)!r}"
                )
            seen.add(key)


class ToolboxStatus(StrEnum):
    """Typed outcome of projecting declared capability uses into a toolbox."""

    SUCCESS = "success"
    REQUIRED_CAPABILITY_UNAVAILABLE = "required_capability_unavailable"
    DISCOVERY_DENIED = "discovery_denied"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    DUPLICATE_TOOL_ID = "duplicate_tool_id"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclass(frozen=True, slots=True)
class ToolboxSnapshot:
    """Opaque toolbox projection frozen for exactly one model decision boundary.

    Attributes:
        registry_revision: Highest registry revision observed while building this snapshot.
        tools: Bounded, schema-valid, provider-neutral tool descriptors.
    """

    registry_revision: int
    tools: tuple[ToolDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Any:
        return iter(self.tools)

    def __contains__(self, tool_id: str) -> bool:
        return any(tool.tool_id == tool_id for tool in self.tools)

    def resolve(self, tool_id: str) -> CapabilityBinding | None:
        """Return the binding for ``tool_id`` if it belongs to this snapshot.

        Args:
            tool_id: The opaque tool identifier returned by the model.

        Returns:
            The bound capability, or ``None`` if the tool ID is unknown, stale,
            or from another snapshot.
        """
        for tool in self.tools:
            if tool.tool_id == tool_id:
                return tool.binding
        return None

    def as_model_payload(self) -> tuple[dict[str, Any], ...]:
        """Return the safe, prompt-ready tool definitions for this snapshot.

        Returns:
            JSON-serializable tool definitions containing no binding internals.
        """
        return tuple(tool.to_dict() for tool in self.tools)

    def as_provider_tools(self) -> tuple[ProviderToolDefinition, ...]:
        """Return typed provider-neutral tool definitions for this snapshot."""
        return tuple(
            ProviderToolDefinition(
                tool_id=tool.tool_id,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in self.tools
        )


@dataclass(frozen=True, slots=True)
class ToolboxUseFailure:
    """Per-declared-use outcome when an optional capability could not be projected."""

    use: CapabilityUse
    failure: GatewayFailure


@dataclass(frozen=True, slots=True)
class ToolboxResult:
    """Typed outcome of building a toolbox from declared capability uses.

    Attributes:
        status: The distinct outcome of toolbox construction.
        snapshot: The immutable toolbox, present only on success.
        failure: The gateway failure that produced a non-success status.
        skipped: Optional uses that were not satisfied and were omitted.
    """

    status: ToolboxStatus
    snapshot: ToolboxSnapshot | None = None
    failure: GatewayFailure | None = None
    skipped: tuple[ToolboxUseFailure, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "skipped", tuple(self.skipped))

    @property
    def ok(self) -> bool:
        """Return whether toolbox construction succeeded."""
        return self.status is ToolboxStatus.SUCCESS


_REQUIRED_FAILURE_STATUS: Mapping[GatewayFailureCode, ToolboxStatus] = {
    GatewayFailureCode.NO_MATCH: ToolboxStatus.REQUIRED_CAPABILITY_UNAVAILABLE,
    GatewayFailureCode.DISCOVERY_DENIED: ToolboxStatus.DISCOVERY_DENIED,
    GatewayFailureCode.AMBIGUOUS: ToolboxStatus.AMBIGUOUS,
    GatewayFailureCode.UNSUPPORTED_SCHEMA: ToolboxStatus.UNSUPPORTED_SCHEMA,
    GatewayFailureCode.RESULT_LIMIT_EXCEEDED: ToolboxStatus.LIMIT_EXCEEDED,
    GatewayFailureCode.POLICY_EVALUATION_FAILED: ToolboxStatus.REQUIRED_CAPABILITY_UNAVAILABLE,
}


def _status_for_required_failure(failure: GatewayFailure) -> ToolboxStatus:
    return _REQUIRED_FAILURE_STATUS.get(
        failure.code,
        ToolboxStatus.REQUIRED_CAPABILITY_UNAVAILABLE,
    )


async def build_toolbox(
    gateway: AgentGateway,
    policy: ToolboxPolicy,
    *,
    include_approval_required: bool = True,
) -> ToolboxResult:
    """Project declared capability uses into one bounded, immutable toolbox.

    Every candidate has already been filtered for the active caller, authority,
    policy, lifecycle, compatibility, and budget by ``AgentGateway`` discovery.
    This function only aggregates, deduplicates, and bounds the resulting
    provider-neutral tool descriptors; it never selects or invokes a target.

    Args:
        gateway: The invocation-scoped gateway used for bounded discovery.
        policy: The agent's declared capability uses and toolbox limits.
        include_approval_required: Whether approval-gated capabilities are eligible.

    Returns:
        A typed ``ToolboxResult`` describing success, a required-capability
        failure, or a size/collision limit violation.
    """
    if not policy.uses:
        return ToolboxResult(ToolboxStatus.SUCCESS, ToolboxSnapshot(registry_revision=-1))

    tools: list[ToolDescriptor] = []
    skipped: list[ToolboxUseFailure] = []
    revision = -1
    total_bytes = 2  # Account for the enclosing JSON array brackets.

    for use in policy.uses:
        query = use.to_query(include_approval_required=include_approval_required)
        discovery = await gateway.discover_tools(query)
        revision = max(revision, discovery.registry_revision)
        if not discovery.tools:
            failure = discovery.failure or GatewayFailure(
                GatewayFailureCode.NO_MATCH,
                "No eligible capability matched the declared use",
            )
            if use.required:
                return ToolboxResult(_status_for_required_failure(failure), failure=failure)
            skipped.append(ToolboxUseFailure(use, failure))
            continue

        for tool in discovery.tools:
            existing = next((item for item in tools if item.tool_id == tool.tool_id), None)
            if existing is not None:
                if (
                    existing.binding.agent_id != tool.binding.agent_id
                    or existing.binding.capability_id != tool.binding.capability_id
                ):
                    return ToolboxResult(
                        ToolboxStatus.DUPLICATE_TOOL_ID,
                        failure=GatewayFailure(
                            GatewayFailureCode.TOOL_ID_COLLISION,
                            f"Tool ID '{tool.tool_id}' collides across distinct capabilities",
                        ),
                    )
                continue  # Same capability discovered by an overlapping declaration.

            description_bytes = len(tool.description.encode("utf-8"))
            if description_bytes > policy.max_tool_description_bytes:
                return ToolboxResult(
                    ToolboxStatus.LIMIT_EXCEEDED,
                    failure=GatewayFailure(
                        GatewayFailureCode.RESULT_LIMIT_EXCEEDED,
                        f"Tool '{tool.tool_id}' description exceeds the configured size limit",
                    ),
                )
            schema_bytes = len(canonical_json(tool.input_schema).encode("utf-8"))
            if schema_bytes > policy.max_tool_schema_bytes:
                return ToolboxResult(
                    ToolboxStatus.LIMIT_EXCEEDED,
                    failure=GatewayFailure(
                        GatewayFailureCode.RESULT_LIMIT_EXCEEDED,
                        f"Tool '{tool.tool_id}' input schema exceeds the configured size limit",
                    ),
                )
            if len(tools) >= policy.max_tools:
                return ToolboxResult(
                    ToolboxStatus.LIMIT_EXCEEDED,
                    failure=GatewayFailure(
                        GatewayFailureCode.RESULT_LIMIT_EXCEEDED,
                        "Toolbox exceeds the configured maximum tool count",
                    ),
                )
            encoded_size = len(canonical_json(tool.to_dict()).encode("utf-8"))
            # A comma separator is only needed between elements, never after the
            # last one, so only account for it once a prior tool exists.
            separator_size = 1 if tools else 0
            if total_bytes + separator_size + encoded_size > policy.max_total_bytes:
                return ToolboxResult(
                    ToolboxStatus.LIMIT_EXCEEDED,
                    failure=GatewayFailure(
                        GatewayFailureCode.RESULT_LIMIT_EXCEEDED,
                        "Toolbox exceeds the configured maximum serialized size",
                    ),
                )
            total_bytes += separator_size + encoded_size
            tools.append(tool)

    snapshot = ToolboxSnapshot(registry_revision=revision, tools=tuple(tools))
    return ToolboxResult(ToolboxStatus.SUCCESS, snapshot, skipped=tuple(skipped))
