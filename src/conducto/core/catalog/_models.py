"""Immutable catalog admission entries, discovery records, and typed failures."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..agent_card import is_absolute_http_url
from ..decorators import CapabilityPolicyMetadata
from ..gateway_models import CapabilityDescriptor, canonical_json, freeze_json, thaw_json


class CatalogError(Exception):
    """Base class for governed remote-agent catalog failures."""


class CatalogValidationError(CatalogError):
    """Raised when a catalog entry or Agent Card fails admission validation."""


class CatalogProviderUnavailableError(CatalogError):
    """Raised when a catalog provider cannot be reached, read, or parsed."""


class UnknownCatalogAgentError(CatalogError, KeyError):
    """Raised when an operation references a logical agent not in the catalog."""


class UnknownCatalogInstanceError(CatalogError, KeyError):
    """Raised when an operation references an instance identity not in the catalog."""


class DeploymentType(StrEnum):
    """Declared deployment topology for one catalog instance."""

    IN_PROCESS = "in_process"
    LOCAL_CONTAINER = "local_container"
    REMOTE_CONTAINER = "remote_container"
    FOUNDRY = "foundry"


class CatalogLifecycleState(StrEnum):
    """Lifecycle state of a logical agent's catalog registration."""

    ACTIVE = "active"
    QUARANTINED = "quarantined"
    DISABLED = "disabled"
    REVOKED = "revoked"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class CatalogCapabilityDescriptor:
    """Immutable, credential-free description of one remote capability.

    Attributes:
        capability_id: Stable capability identifier from the Agent Card skill.
        name: Stable Conducto capability name exposed through the gateway.
        description: Human-readable capability description.
        tags: Capability tags used for discovery filtering.
        input_schema: JSON Schema for capability arguments, if published.
        output_schema: JSON Schema for capability results, if published.
        version: Agent Card version this capability was published under.
        modality: Primary input modality (media type) for this capability.
        required_scopes: Scopes required to authorize an invocation.
        approval_required: Whether the destination requires human approval.
        policy: Immutable governance metadata declared by the capability.
    """

    capability_id: str
    name: str
    description: str | None
    tags: frozenset[str]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    version: str
    modality: str
    required_scopes: tuple[str, ...] = ()
    approval_required: bool = False
    policy: CapabilityPolicyMetadata = CapabilityPolicyMetadata()

    def __post_init__(self) -> None:
        """Freeze mutable fields so the descriptor is safe to share."""
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", freeze_json(self.output_schema))
        object.__setattr__(self, "required_scopes", tuple(self.required_scopes))

    def to_capability_descriptor(
        self, *, agent_id: str, agent_version: str
    ) -> CapabilityDescriptor:
        """Project this remote descriptor into a gateway ``CapabilityDescriptor``.

        Args:
            agent_id: Logical agent identity that owns this capability.
            agent_version: Agent version to attribute this capability to.

        Returns:
            A transport-neutral descriptor usable by a future agent gateway.
        """
        digest = hashlib.sha256(
            canonical_json(
                {"input": thaw_json(self.input_schema), "output": thaw_json(self.output_schema)}
            ).encode()
        ).hexdigest()
        return CapabilityDescriptor(
            agent_id=agent_id,
            agent_version=agent_version,
            capability_id=self.name,
            description=self.description,
            tags=self.tags,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            schema_digest=digest,
            required_scopes=self.required_scopes,
            approval_required=self.approval_required,
            policy=self.policy,
        )


@dataclass(frozen=True, slots=True)
class AgentInstanceRecord:
    """Immutable public metadata for one healthy or leased agent instance.

    Attributes:
        instance_id: Stable instance identity; never an endpoint URL.
        deployment_type: Declared deployment topology for this instance.
        agent_card_url: Absolute URL this instance's Agent Card was read from.
        transports: Transport bindings this instance offers.
        healthy: Explicit health flag independent of lease expiration.
        lease_expires_at: Monotonic-clock timestamp the current lease expires.
        last_heartbeat_at: Monotonic-clock timestamp of the last renewal.
        environment: Immutable managed deployment environment, empty for legacy entries.
        deployment_id: Managed deployment attribution, empty for legacy entries.
        provenance: Managed admission provenance, empty for legacy entries.
        subject_id: Authenticated principal that admitted a managed instance.
        issuer: Authentication issuer for the admitting principal.
    """

    instance_id: str
    deployment_type: DeploymentType
    agent_card_url: str
    transports: frozenset[str]
    healthy: bool
    lease_expires_at: float
    last_heartbeat_at: float
    environment: str = ""
    deployment_id: str = ""
    provenance: str = ""
    subject_id: str = ""
    issuer: str = ""

    def __post_init__(self) -> None:
        """Freeze the transport set."""
        object.__setattr__(self, "transports", frozenset(self.transports))

    def is_expired(self, now: float) -> bool:
        """Return whether this instance's lease has expired at ``now``."""
        return now >= self.lease_expires_at


@dataclass(frozen=True, slots=True)
class CatalogAgentRecord:
    """Immutable public metadata for one logical agent's catalog registration.

    Attributes:
        agent_id: Stable logical agent identity, organization-qualified when
            federation is enabled.
        owner: Owning organization or team identifier.
        provenance: Opaque attestation or signing reference for the current
            admitted Agent Card, or ``None`` when none was supplied.
        trust_policy_ref: Opaque reference to the trust policy version this
            registration was evaluated under, or ``None`` when unfederated.
        supported_versions: Protocol/contract versions this agent supports.
        card_digest: Canonical digest of the currently admitted Agent Card.
        capabilities: Immutable capability descriptors indexed from the card.
        lifecycle: Current lifecycle state of this logical registration.
        generation: Monotonic counter incremented on every admitted change.
        instances: All known instances, including expired or unhealthy ones.
    """

    agent_id: str
    owner: str
    provenance: str | None
    trust_policy_ref: str | None
    supported_versions: frozenset[str]
    card_digest: str
    capabilities: tuple[CatalogCapabilityDescriptor, ...]
    lifecycle: CatalogLifecycleState
    generation: int
    instances: tuple[AgentInstanceRecord, ...]

    def __post_init__(self) -> None:
        """Freeze collection fields."""
        object.__setattr__(self, "supported_versions", frozenset(self.supported_versions))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "instances", tuple(self.instances))

    def healthy_instances(self, now: float) -> tuple[AgentInstanceRecord, ...]:
        """Return instances that are healthy and not lease-expired at ``now``."""
        return tuple(
            instance
            for instance in self.instances
            if instance.healthy and not instance.is_expired(now)
        )


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Atomic immutable view of catalog registrations eligible for discovery."""

    revision: int
    agents: tuple[CatalogAgentRecord, ...]

    def __post_init__(self) -> None:
        """Freeze the agent tuple."""
        object.__setattr__(self, "agents", tuple(self.agents))

    @property
    def capabilities(self) -> tuple[CatalogCapabilityDescriptor, ...]:
        """Return all eligible capabilities in stable agent/capability order."""
        return tuple(capability for agent in self.agents for capability in agent.capabilities)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One source-of-truth record offered by a provider for admission.

    Attributes:
        agent_id: Stable logical agent identity to admit or update.
        instance_id: Stable instance identity for this deployment.
        owner: Owning organization or team identifier.
        deployment_type: Declared deployment topology for this instance.
        agent_card_url: Absolute URL the Agent Card was retrieved from.
        agent_card: The already-retrieved Agent Card payload to validate.
        provenance: Opaque attestation or signing reference for the card.
        trust_policy_ref: Opaque trust-policy version reference, required
            for federated entries.
        supported_versions: Protocol/contract versions this instance supports.
        transports: Transport bindings this instance offers.
        lease_seconds: Requested lease duration for this instance's health.
        signature: Detached signature over the canonical Agent Card, required
            when federation policy mandates provenance verification.
    """

    agent_id: str
    instance_id: str
    owner: str
    deployment_type: DeploymentType
    agent_card_url: str
    agent_card: Mapping[str, Any]
    provenance: str | None = None
    trust_policy_ref: str | None = None
    supported_versions: frozenset[str] = frozenset()
    transports: frozenset[str] = frozenset()
    lease_seconds: float = 60.0
    signature: str | None = None

    def __post_init__(self) -> None:
        """Validate required identity fields and freeze mutable collections."""
        if not self.agent_id or not self.agent_id.strip():
            raise CatalogValidationError("agent_id is required")
        if not self.instance_id or not self.instance_id.strip():
            raise CatalogValidationError("instance_id is required")
        if not self.owner or not self.owner.strip():
            raise CatalogValidationError("owner is required")
        if not is_absolute_http_url(self.agent_card_url):
            raise CatalogValidationError("agent_card_url must be an absolute http(s) URL")
        if not math.isfinite(self.lease_seconds) or self.lease_seconds <= 0:
            raise CatalogValidationError("lease_seconds must be finite and positive")
        object.__setattr__(self, "supported_versions", frozenset(self.supported_versions))
        object.__setattr__(self, "transports", frozenset(self.transports))
        object.__setattr__(self, "agent_card", freeze_json(self.agent_card))
