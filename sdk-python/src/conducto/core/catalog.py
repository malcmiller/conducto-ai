"""Governed remote-agent catalog: provider contract and lifecycle state machine.

The catalog is the control plane for independently operated agents. It owns
stable logical and instance identity, ownership, deployment type, Agent Card
provenance, trust policy references, capability indexing, lease-based health,
and lifecycle state (active, quarantined, disabled, revoked, removed).

Runtime transport selection and invocation belong to the agent gateway, not
to the catalog. The catalog only decides which agents and capabilities are
currently eligible for discovery; it never calls a model or a remote
endpoint on the caller's behalf.

A :class:`CatalogProvider` is the pluggable admission source. Static
configuration (:class:`InMemoryCatalogProvider`,
:class:`StaticFileCatalogProvider`) is supported first; a future hosted
registry (for example a Microsoft Entra Agent Registry adapter) implements
the same protocol without changing :class:`AgentCatalog` or gateway code.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .a2a_profile import A2AProtocolError, parse_agent_card
from .agent_card import capability_parameter_map, is_absolute_http_url
from .gateway_models import CapabilityDescriptor, canonical_json, freeze_json, thaw_json
from .logging import (
    CATALOG_AGENT_ADMITTED,
    CATALOG_AGENT_LIFECYCLE_CHANGED,
    CATALOG_DISCOVERY,
    CATALOG_INSTANCE_EXPIRED,
    CATALOG_INSTANCE_LEASE_RENEWED,
    emit_event,
)


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
        description: Human-readable capability description.
        tags: Capability tags used for discovery filtering.
        input_schema: JSON Schema for capability arguments, if published.
        output_schema: JSON Schema for capability results, if published.
        version: Agent Card version this capability was published under.
        modality: Primary input modality (media type) for this capability.
        required_scopes: Scopes required to authorize an invocation.
        approval_required: Whether the destination requires human approval.
    """

    capability_id: str
    description: str | None
    tags: frozenset[str]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    version: str
    modality: str
    required_scopes: tuple[str, ...] = ()
    approval_required: bool = False

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
            capability_id=self.capability_id,
            description=self.description,
            tags=self.tags,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            schema_digest=digest,
            required_scopes=self.required_scopes,
            approval_required=self.approval_required,
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
    """

    instance_id: str
    deployment_type: DeploymentType
    agent_card_url: str
    transports: frozenset[str]
    healthy: bool
    lease_expires_at: float
    last_heartbeat_at: float

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
        if self.lease_seconds <= 0:
            raise CatalogValidationError("lease_seconds must be positive")
        object.__setattr__(self, "supported_versions", frozenset(self.supported_versions))
        object.__setattr__(self, "transports", frozenset(self.transports))
        object.__setattr__(self, "agent_card", freeze_json(self.agent_card))


ProvenanceVerifier = Callable[[CatalogEntry], bool]


class CatalogProvider(Protocol):
    """Contract for any catalog backend that can enumerate admission entries."""

    def list_entries(self) -> Sequence[CatalogEntry]:
        """Return the current set of candidate entries for admission."""
        ...


class InMemoryCatalogProvider:
    """Reference catalog provider backed by an explicit in-memory entry list."""

    def __init__(self, entries: Sequence[CatalogEntry] = ()) -> None:
        """Initialize the provider with an initial, possibly empty, entry set."""
        self._entries: tuple[CatalogEntry, ...] = tuple(entries)

    def list_entries(self) -> tuple[CatalogEntry, ...]:
        """Return the current in-memory entries."""
        return self._entries

    def replace(self, entries: Sequence[CatalogEntry]) -> None:
        """Atomically replace the provider's entries.

        Args:
            entries: The new complete entry set the provider should offer.
        """
        self._entries = tuple(entries)


class StaticFileCatalogProvider:
    """Reference provider that loads admission entries from a static JSON file.

    The file must contain a JSON array of objects with the same fields as
    :class:`CatalogEntry` (``deployment_type`` as a string matching
    :class:`DeploymentType`). This is the static-configuration provider
    described by the catalog provider contract; a hosted registry adapter
    implements :class:`CatalogProvider` the same way without changing
    :class:`AgentCatalog`.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind this provider to a static catalog configuration file path."""
        self._path = Path(path)

    def list_entries(self) -> tuple[CatalogEntry, ...]:
        """Read and parse the configured file into catalog entries.

        Raises:
            CatalogProviderUnavailableError: If the file cannot be read or
                does not contain a valid JSON array of catalog entries.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as error:
            raise CatalogProviderUnavailableError(
                f"Could not read catalog file '{self._path}': {error}"
            ) from error
        try:
            documents = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CatalogProviderUnavailableError(
                f"Catalog file '{self._path}' is not valid JSON: {error}"
            ) from error
        if not isinstance(documents, list):
            raise CatalogProviderUnavailableError(
                f"Catalog file '{self._path}' must contain a JSON array"
            )
        return tuple(_entry_from_document(document) for document in documents)


def _entry_from_document(document: Any) -> CatalogEntry:
    if not isinstance(document, Mapping):
        raise CatalogProviderUnavailableError("Catalog file entries must be JSON objects")
    try:
        deployment_type = DeploymentType(document["deployment_type"])
    except (KeyError, ValueError) as error:
        raise CatalogProviderUnavailableError(
            f"Catalog file entry has an invalid deployment_type: {error}"
        ) from error
    try:
        return CatalogEntry(
            agent_id=document["agent_id"],
            instance_id=document["instance_id"],
            owner=document["owner"],
            deployment_type=deployment_type,
            agent_card_url=document["agent_card_url"],
            agent_card=document["agent_card"],
            provenance=document.get("provenance"),
            trust_policy_ref=document.get("trust_policy_ref"),
            supported_versions=frozenset(document.get("supported_versions", ())),
            transports=frozenset(document.get("transports", ())),
            lease_seconds=float(document.get("lease_seconds", 60.0)),
            signature=document.get("signature"),
        )
    except KeyError as error:
        raise CatalogProviderUnavailableError(
            f"Catalog file entry is missing required field {error}"
        ) from error
    except CatalogValidationError as error:
        raise CatalogProviderUnavailableError(str(error)) from error


@dataclass(frozen=True, slots=True)
class _InstanceState:
    instance_id: str
    deployment_type: DeploymentType
    agent_card_url: str
    transports: frozenset[str]
    healthy: bool
    lease_expires_at: float
    last_heartbeat_at: float

    def to_record(self) -> AgentInstanceRecord:
        return AgentInstanceRecord(
            instance_id=self.instance_id,
            deployment_type=self.deployment_type,
            agent_card_url=self.agent_card_url,
            transports=self.transports,
            healthy=self.healthy,
            lease_expires_at=self.lease_expires_at,
            last_heartbeat_at=self.last_heartbeat_at,
        )

    def is_expired(self, now: float) -> bool:
        """Return whether this instance's lease has expired at ``now``."""
        return now >= self.lease_expires_at


@dataclass(frozen=True, slots=True)
class _AgentState:
    agent_id: str
    owner: str
    provenance: str | None
    trust_policy_ref: str | None
    supported_versions: frozenset[str]
    card_name: str
    card_digest: str
    capabilities: tuple[CatalogCapabilityDescriptor, ...]
    lifecycle: CatalogLifecycleState
    generation: int
    instances: Mapping[str, _InstanceState]

    def to_record(self) -> CatalogAgentRecord:
        return CatalogAgentRecord(
            agent_id=self.agent_id,
            owner=self.owner,
            provenance=self.provenance,
            trust_policy_ref=self.trust_policy_ref,
            supported_versions=self.supported_versions,
            card_digest=self.card_digest,
            capabilities=self.capabilities,
            lifecycle=self.lifecycle,
            generation=self.generation,
            instances=tuple(instance.to_record() for _, instance in sorted(self.instances.items())),
        )


class AgentCatalog:
    """Own the governed remote-agent catalog control plane.

    The catalog validates admission, indexes immutable capability
    descriptors, tracks per-instance health leases, and enforces lifecycle
    transitions (active, quarantined, disabled, revoked, removed). It never
    invokes a capability or selects transport; that belongs to the agent
    gateway.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        provenance_verifier: ProvenanceVerifier | None = None,
        require_provenance: bool = False,
    ) -> None:
        """Initialize an empty catalog.

        Args:
            clock: Injectable monotonic-style clock used for lease and
                expiration arithmetic; tests may supply a fake clock.
            provenance_verifier: Callable used to verify a signed entry when
                provenance verification is required.
            require_provenance: Whether every entry must supply and pass
                provenance verification, regardless of ``trust_policy_ref``.
        """
        self._clock = clock
        self._provenance_verifier = provenance_verifier
        self._require_provenance = require_provenance
        self._lock = threading.RLock()
        self._agents: dict[str, _AgentState] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return the monotonic catalog revision."""
        with self._lock:
            return self._revision

    def refresh(self, provider: CatalogProvider) -> CatalogSnapshot:
        """Admit or update every entry a provider currently offers.

        Missing entries are not removed by refresh; only lease expiration or
        an explicit lifecycle change excludes a previously admitted instance.

        Args:
            provider: The catalog provider to enumerate entries from.

        Returns:
            The resulting eligible catalog snapshot.

        Raises:
            CatalogProviderUnavailableError: If the provider cannot be read.
        """
        try:
            entries = provider.list_entries()
        except CatalogError:
            raise
        except Exception as error:  # noqa: BLE001 - normalize to a typed catalog failure
            raise CatalogProviderUnavailableError(
                f"Catalog provider {provider!r} is unavailable: {error}"
            ) from error
        for entry in entries:
            self.register_instance(entry)
        return self.snapshot()

    def register_instance(self, entry: CatalogEntry) -> CatalogAgentRecord:
        """Validate and admit a single instance entry.

        Args:
            entry: The catalog entry to validate and admit.

        Returns:
            The resulting public agent record.

        Raises:
            CatalogValidationError: If the Agent Card, identity, compatibility,
                or provenance policy rejects this entry.
        """
        raw_card = dict(thaw_json(entry.agent_card))
        try:
            parse_agent_card(raw_card)
        except A2AProtocolError as error:
            raise CatalogValidationError(
                f"Invalid Agent Card for '{entry.agent_id}': {error}"
            ) from error

        card_digest = hashlib.sha256(canonical_json(entry.agent_card).encode()).hexdigest()
        capabilities = _capabilities_from_card(raw_card)
        now = self._clock()

        with self._lock:
            existing = self._agents.get(entry.agent_id)
            self._validate_admission_locked(entry=entry, card=raw_card, existing=existing)

            if existing is not None and existing.lifecycle is not CatalogLifecycleState.ACTIVE:
                lifecycle: CatalogLifecycleState = existing.lifecycle
            else:
                lifecycle = CatalogLifecycleState.ACTIVE

            instances = dict(existing.instances) if existing is not None else {}
            instances[entry.instance_id] = _InstanceState(
                instance_id=entry.instance_id,
                deployment_type=entry.deployment_type,
                agent_card_url=entry.agent_card_url,
                transports=entry.transports,
                healthy=True,
                lease_expires_at=now + entry.lease_seconds,
                last_heartbeat_at=now,
            )
            generation = (existing.generation if existing is not None else 0) + 1
            state = _AgentState(
                agent_id=entry.agent_id,
                owner=entry.owner,
                provenance=entry.provenance,
                trust_policy_ref=entry.trust_policy_ref,
                supported_versions=entry.supported_versions,
                card_name=str(raw_card.get("name", "")),
                card_digest=card_digest,
                capabilities=capabilities,
                lifecycle=lifecycle,
                generation=generation,
                instances=instances,
            )
            self._agents[entry.agent_id] = state
            self._revision += 1
            emit_event(
                CATALOG_AGENT_ADMITTED,
                outcome="success",
                agent_id=entry.agent_id,
                snapshot_revision=self._revision,
            )
            return state.to_record()

    def _validate_admission_locked(
        self,
        *,
        entry: CatalogEntry,
        card: Mapping[str, Any],
        existing: _AgentState | None,
    ) -> None:
        requires_provenance = self._require_provenance or entry.trust_policy_ref is not None
        if requires_provenance:
            if not entry.signature or self._provenance_verifier is None:
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' requires signed provenance verification"
                )
            if not self._provenance_verifier(entry):
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' failed provenance verification"
                )
        if existing is None:
            return
        card_name = str(card.get("name", ""))
        if existing.card_name and card_name != existing.card_name:
            raise CatalogValidationError(
                f"Agent '{entry.agent_id}' Agent Card identity changed from "
                f"'{existing.card_name}' to '{card_name}'; reject rather than "
                "silently replace a trusted registration"
            )
        new_by_id = {
            capability.capability_id: capability for capability in _capabilities_from_card(card)
        }
        for capability in existing.capabilities:
            updated = new_by_id.get(capability.capability_id)
            if updated is None:
                continue
            if (
                updated.version == capability.version
                and updated.input_schema != capability.input_schema
            ):
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' capability '{capability.capability_id}' "
                    "changed its input schema without a version change"
                )

    def renew_lease(
        self,
        agent_id: str,
        instance_id: str,
        *,
        lease_seconds: float | None = None,
    ) -> AgentInstanceRecord:
        """Renew an instance's health lease (heartbeat).

        Args:
            agent_id: The logical agent that owns this instance.
            instance_id: The instance whose lease should be renewed.
            lease_seconds: Optional new lease duration; defaults to reusing
                the instance's previous lease duration.

        Returns:
            The renewed instance record.

        Raises:
            UnknownCatalogAgentError: If the logical agent is not registered.
            UnknownCatalogInstanceError: If the instance is not registered.
        """
        now = self._clock()
        with self._lock:
            state = self._require_agent_locked(agent_id)
            instance = state.instances.get(instance_id)
            if instance is None:
                raise UnknownCatalogInstanceError(
                    f"Instance '{instance_id}' is not registered for agent '{agent_id}'"
                )
            duration = (
                lease_seconds
                if lease_seconds is not None
                else max(instance.lease_expires_at - instance.last_heartbeat_at, 1.0)
            )
            renewed = _InstanceState(
                instance_id=instance.instance_id,
                deployment_type=instance.deployment_type,
                agent_card_url=instance.agent_card_url,
                transports=instance.transports,
                healthy=True,
                lease_expires_at=now + duration,
                last_heartbeat_at=now,
            )
            instances = dict(state.instances)
            instances[instance_id] = renewed
            self._agents[agent_id] = _replace_instances(state, instances)
            self._revision += 1
            emit_event(
                CATALOG_INSTANCE_LEASE_RENEWED,
                outcome="success",
                agent_id=agent_id,
                instance_id=instance_id,
            )
            return renewed.to_record()

    def expire_instances(self) -> tuple[tuple[str, str], ...]:
        """Mark every instance past its lease as unhealthy.

        Crashed instances age out individually; healthy instances of the
        same logical agent are never removed by this operation.

        Returns:
            A tuple of ``(agent_id, instance_id)`` pairs that just expired.
        """
        now = self._clock()
        expired: list[tuple[str, str]] = []
        with self._lock:
            for agent_id, state in list(self._agents.items()):
                instances = dict(state.instances)
                changed = False
                for instance_id, instance in list(instances.items()):
                    if instance.healthy and instance.is_expired(now):
                        instances[instance_id] = _InstanceState(
                            instance_id=instance.instance_id,
                            deployment_type=instance.deployment_type,
                            agent_card_url=instance.agent_card_url,
                            transports=instance.transports,
                            healthy=False,
                            lease_expires_at=instance.lease_expires_at,
                            last_heartbeat_at=instance.last_heartbeat_at,
                        )
                        expired.append((agent_id, instance_id))
                        changed = True
                if changed:
                    self._agents[agent_id] = _replace_instances(state, instances)
                    self._revision += 1
            for agent_id, instance_id in expired:
                emit_event(
                    CATALOG_INSTANCE_EXPIRED,
                    outcome="success",
                    agent_id=agent_id,
                    instance_id=instance_id,
                )
        return tuple(expired)

    def set_lifecycle(self, agent_id: str, lifecycle: CatalogLifecycleState) -> None:
        """Set a lifecycle state (quarantine, disable, revoke, remove, or reactivate).

        Args:
            agent_id: The logical agent to transition.
            lifecycle: The lifecycle state to apply.

        Raises:
            UnknownCatalogAgentError: If the logical agent is not registered.
        """
        with self._lock:
            state = self._require_agent_locked(agent_id)
            self._agents[agent_id] = _AgentState(
                agent_id=state.agent_id,
                owner=state.owner,
                provenance=state.provenance,
                trust_policy_ref=state.trust_policy_ref,
                supported_versions=state.supported_versions,
                card_name=state.card_name,
                card_digest=state.card_digest,
                capabilities=state.capabilities,
                lifecycle=lifecycle,
                generation=state.generation,
                instances=state.instances,
            )
            self._revision += 1
            emit_event(
                CATALOG_AGENT_LIFECYCLE_CHANGED,
                outcome="success",
                agent_id=agent_id,
                lifecycle_state=lifecycle.value,
                snapshot_revision=self._revision,
            )

    def quarantine(self, agent_id: str) -> None:
        """Quarantine a logical agent, excluding it from discovery."""
        self.set_lifecycle(agent_id, CatalogLifecycleState.QUARANTINED)

    def disable(self, agent_id: str) -> None:
        """Disable a logical agent, excluding it from discovery."""
        self.set_lifecycle(agent_id, CatalogLifecycleState.DISABLED)

    def revoke(self, agent_id: str) -> None:
        """Revoke a logical agent's trust, excluding it from discovery."""
        self.set_lifecycle(agent_id, CatalogLifecycleState.REVOKED)

    def reactivate(self, agent_id: str) -> None:
        """Restore a quarantined or disabled logical agent to active."""
        self.set_lifecycle(agent_id, CatalogLifecycleState.ACTIVE)

    def remove(self, agent_id: str) -> None:
        """Remove a logical agent's registration entirely.

        Args:
            agent_id: The logical agent to remove.

        Raises:
            UnknownCatalogAgentError: If the logical agent is not registered.
        """
        with self._lock:
            self._require_agent_locked(agent_id)
            del self._agents[agent_id]
            self._revision += 1
            emit_event(
                CATALOG_AGENT_LIFECYCLE_CHANGED,
                outcome="success",
                agent_id=agent_id,
                lifecycle_state=CatalogLifecycleState.REMOVED.value,
                snapshot_revision=self._revision,
            )

    def get(self, agent_id: str) -> CatalogAgentRecord | None:
        """Return the current record for a logical agent, if registered."""
        with self._lock:
            state = self._agents.get(agent_id)
            return state.to_record() if state is not None else None

    def snapshot(self) -> CatalogSnapshot:
        """Return one coherent snapshot of every eligible registration.

        Untrusted, incompatible, revoked, quarantined, disabled, and
        unhealthy or lease-expired agents and instances are excluded.
        """
        now = self._clock()
        agents: list[CatalogAgentRecord] = []
        with self._lock:
            revision = self._revision
            for _agent_id, state in sorted(self._agents.items()):
                if state.lifecycle is not CatalogLifecycleState.ACTIVE:
                    continue
                healthy_instances = tuple(
                    instance.to_record()
                    for _, instance in sorted(state.instances.items())
                    if instance.healthy and now < instance.lease_expires_at
                )
                if not healthy_instances:
                    continue
                agents.append(
                    CatalogAgentRecord(
                        agent_id=state.agent_id,
                        owner=state.owner,
                        provenance=state.provenance,
                        trust_policy_ref=state.trust_policy_ref,
                        supported_versions=state.supported_versions,
                        card_digest=state.card_digest,
                        capabilities=state.capabilities,
                        lifecycle=state.lifecycle,
                        generation=state.generation,
                        instances=healthy_instances,
                    )
                )
        emit_event(
            CATALOG_DISCOVERY,
            level=10,
            outcome="success",
            agent_count=len(agents),
            snapshot_revision=revision,
        )
        return CatalogSnapshot(revision=revision, agents=tuple(agents))

    def capability_providers(self, capability_id: str) -> tuple[CatalogAgentRecord, ...]:
        """Return every eligible agent that currently offers a capability."""
        return tuple(
            agent
            for agent in self.snapshot().agents
            if any(capability.capability_id == capability_id for capability in agent.capabilities)
        )

    def _require_agent_locked(self, agent_id: str) -> _AgentState:
        state = self._agents.get(agent_id)
        if state is None:
            raise UnknownCatalogAgentError(f"Agent '{agent_id}' is not registered")
        return state


def _replace_instances(state: _AgentState, instances: Mapping[str, _InstanceState]) -> _AgentState:
    return _AgentState(
        agent_id=state.agent_id,
        owner=state.owner,
        provenance=state.provenance,
        trust_policy_ref=state.trust_policy_ref,
        supported_versions=state.supported_versions,
        card_name=state.card_name,
        card_digest=state.card_digest,
        capabilities=state.capabilities,
        lifecycle=state.lifecycle,
        generation=state.generation,
        instances=instances,
    )


def _capabilities_from_card(card: Mapping[str, Any]) -> tuple[CatalogCapabilityDescriptor, ...]:
    version = str(card.get("version", ""))
    parameter_map = capability_parameter_map(card)
    descriptors: list[CatalogCapabilityDescriptor] = []
    for skill in card.get("skills", []):
        capability_id = skill.get("id")
        input_modes = tuple(skill.get("inputModes", ()))
        descriptors.append(
            CatalogCapabilityDescriptor(
                capability_id=capability_id,
                description=skill.get("description"),
                tags=frozenset(skill.get("tags", ())),
                input_schema=parameter_map.get(capability_id, {}),
                output_schema=None,
                version=version,
                modality=input_modes[0] if input_modes else "text/plain",
                required_scopes=_required_scopes(skill),
            )
        )
    return tuple(descriptors)


def _required_scopes(skill: Mapping[str, Any]) -> tuple[str, ...]:
    scopes: set[str] = set()
    for requirement in skill.get("securityRequirements", ()) or ():
        if not isinstance(requirement, Mapping):
            continue
        schemes = requirement.get("schemes", {})
        if not isinstance(schemes, Mapping):
            continue
        for scheme in schemes.values():
            if isinstance(scheme, Mapping):
                scopes.update(scheme.get("list", ()))
    return tuple(sorted(scopes))
