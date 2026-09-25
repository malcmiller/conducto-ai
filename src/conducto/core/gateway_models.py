"""Immutable, transport-neutral contracts for capability discovery and binding."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .decorators import CapabilityPolicyMetadata


def freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Gateway metadata must contain finite JSON numbers")
        return value
    raise TypeError(f"Gateway metadata contains unsupported value {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return mutable JSON-compatible data from a frozen gateway value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a gateway value canonically for hashing and size limits."""
    return json.dumps(
        thaw_json(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class RegistrationLifecycle(StrEnum):
    """Lifecycle state of a local agent registration."""

    ACTIVE = "active"
    DRAINING = "draining"
    DISABLED = "disabled"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class DiscoveryQuery:
    """Caller-neutral filters applied to one coherent registry snapshot."""

    agent_id: str | None = None
    capability_ids: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    version_constraint: str | None = None
    input_schema: Mapping[str, Any] | None = None
    output_schema: Mapping[str, Any] | None = None
    limit: int = 20
    include_approval_required: bool = True

    def __post_init__(self) -> None:
        if self.agent_id is not None and not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if self.limit < 1:
            raise ValueError("Discovery limit must be positive")
        object.__setattr__(self, "capability_ids", frozenset(self.capability_ids))
        object.__setattr__(self, "tags", frozenset(self.tags))
        if self.input_schema is not None:
            object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", freeze_json(self.output_schema))


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Credential-free description of one discoverable capability.

    Attributes:
        policy: Immutable capability governance metadata.
    """

    agent_id: str
    agent_version: str
    capability_id: str
    description: str | None
    tags: frozenset[str]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    schema_digest: str
    required_scopes: tuple[str, ...] = ()
    approval_required: bool = False
    policy: CapabilityPolicyMetadata = CapabilityPolicyMetadata()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", freeze_json(self.output_schema))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe descriptor without execution details."""
        descriptor = {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "capability_id": self.capability_id,
            "description": self.description,
            "tags": sorted(self.tags),
            "input_schema": thaw_json(self.input_schema),
            "output_schema": thaw_json(self.output_schema),
            "schema_digest": self.schema_digest,
            "required_scopes": list(self.required_scopes),
            "approval_required": self.approval_required,
        }
        if not self.policy.is_empty:
            descriptor["policy"] = self.policy.to_dict()
        return descriptor


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """Immutable public metadata for one registered local agent."""

    agent_id: str
    version: str
    description: str | None
    tags: frozenset[str]
    lifecycle: RegistrationLifecycle
    healthy: bool
    generation: int
    capabilities: tuple[CapabilityDescriptor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """Atomic immutable view of local registration metadata."""

    revision: int
    agents: tuple[AgentDescriptor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", tuple(self.agents))

    @property
    def capabilities(self) -> tuple[CapabilityDescriptor, ...]:
        """Return all capabilities in stable agent/capability order."""
        return tuple(capability for agent in self.agents for capability in agent.capabilities)


@dataclass(frozen=True, slots=True)
class CapabilityBinding:
    """Opaque, runtime-bound authority to request one capability invocation."""

    agent_id: str
    capability_id: str
    schema_digest: str
    registry_revision: int
    registration_generation: int
    runtime_id: str
    issued_at: float
    expires_at: float
    nonce: str = field(repr=False)
    signature: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class BoundCapability:
    """A discoverable descriptor paired with its opaque invocation binding."""

    descriptor: CapabilityDescriptor
    binding: CapabilityBinding


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Bounded model-facing tool projection with a stable collision-safe ID."""

    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]
    binding: CapabilityBinding = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))

    def to_dict(self) -> dict[str, Any]:
        """Return the safe model-facing fields, excluding the binding."""
        return {
            "id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": thaw_json(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class ToolDiscoveryResult:
    """Bounded model-facing projection from one registry revision."""

    registry_revision: int
    tools: tuple[ToolDescriptor, ...] = ()
    failure: GatewayFailure | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))

    def __getitem__(self, index: int) -> ToolDescriptor:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Iterator[ToolDescriptor]:
        return iter(self.tools)


class GatewayFailureCode(StrEnum):
    """Stable reason codes for discovery, selection, and binding failures."""

    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    DISCOVERY_DENIED = "discovery_denied"
    STALE_BINDING = "stale_binding"
    INVALID_BINDING = "invalid_binding"
    FOREIGN_RUNTIME = "foreign_runtime"
    EXPIRED_BINDING = "expired_binding"
    TARGET_UNAVAILABLE = "target_unavailable"
    NO_ELIGIBLE_INSTANCE = "no_eligible_instance"
    SCHEMA_MISMATCH = "schema_mismatch"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CYCLE_DETECTED = "cycle_detected"
    DEPTH_EXCEEDED = "depth_exceeded"
    RESULT_LIMIT_EXCEEDED = "result_limit_exceeded"
    POLICY_EVALUATION_FAILED = "policy_evaluation_failed"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    TOOL_ID_COLLISION = "tool_id_collision"


@dataclass(frozen=True, slots=True)
class GatewayFailure:
    """Typed gateway failure that contains no target implementation details."""

    code: GatewayFailureCode
    message: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Bounded candidates from one coherent registry revision."""

    registry_revision: int
    candidates: tuple[BoundCapability, ...] = ()
    failure: GatewayFailure | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def __getitem__(self, index: int) -> BoundCapability:
        return self.candidates[index]

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[BoundCapability]:
        return iter(self.candidates)


class SelectionStatus(StrEnum):
    """Outcome of deterministic gateway selection."""

    SELECTED = "selected"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    DENIED = "denied"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """Explicit deterministic selection result."""

    status: SelectionStatus
    binding: CapabilityBinding | None = None
    descriptor: CapabilityDescriptor | None = None
    candidates: tuple[CapabilityDescriptor, ...] = ()
    failure: GatewayFailure | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
