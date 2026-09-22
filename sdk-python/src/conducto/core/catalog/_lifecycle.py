"""Single-lock owner of catalog registrations, lease health, and lifecycle.

Provider loading and card preparation occur outside the lock; admission against
the current registration, mutations, and immutable snapshots share one lock.
The catalog never executes a capability or selects its transport.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ..logging import (
    CATALOG_AGENT_ADMITTED,
    CATALOG_AGENT_LIFECYCLE_CHANGED,
    CATALOG_DISCOVERY,
    CATALOG_INSTANCE_EXPIRED,
    CATALOG_INSTANCE_LEASE_RENEWED,
    emit_event,
)
from ._admission import AdmissionPolicy, ProvenanceVerifier, prepare_entry
from ._managed import ManagedInstances
from ._managed_models import CatalogInstanceState, CatalogManagedCommand, CatalogManagedResult
from ._models import (
    AgentInstanceRecord,
    CatalogAgentRecord,
    CatalogCapabilityDescriptor,
    CatalogEntry,
    CatalogLifecycleState,
    CatalogSnapshot,
    CatalogValidationError,
    DeploymentType,
    UnknownCatalogAgentError,
    UnknownCatalogInstanceError,
)
from ._providers import CatalogProvider, load_entries


@dataclass(frozen=True, slots=True)
class _InstanceState:
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

    def to_record(self) -> AgentInstanceRecord:
        return AgentInstanceRecord(
            instance_id=self.instance_id,
            deployment_type=self.deployment_type,
            agent_card_url=self.agent_card_url,
            transports=self.transports,
            healthy=self.healthy,
            lease_expires_at=self.lease_expires_at,
            last_heartbeat_at=self.last_heartbeat_at,
            environment=self.environment,
            deployment_id=self.deployment_id,
            provenance=self.provenance,
            subject_id=self.subject_id,
            issuer=self.issuer,
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
    logical_digest: str
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
        managed_capacity: int = 10_000,
    ) -> None:
        """Initialize an empty catalog.

        Args:
            clock: Injectable monotonic-style clock used for lease and
                expiration arithmetic; tests may supply a fake clock.
            provenance_verifier: Callable used to verify a signed entry when
                provenance verification is required.
            require_provenance: Whether every entry must supply and pass
                provenance verification, regardless of ``trust_policy_ref``.
            managed_capacity: Maximum managed identities and successful operation
                receipts retained without eviction. One additional terminal receipt
                per identity reserves revocation/removal capacity. Exhaustion denies
                further lease grants without preventing terminal operations.
        """
        self._clock = clock
        self._admission = AdmissionPolicy(provenance_verifier, require_provenance)
        self._lock = threading.RLock()
        self._agents: dict[str, _AgentState] = {}
        self._revision = 0
        self._managed = ManagedInstances(
            capacity=managed_capacity,
            clock=self._clock,
            agents=self._agents,
            admit=self.register_instance,
            changed=self._mark_managed_changed,
        )

    def _mark_managed_changed(self) -> None:
        self._revision += 1

    def manage_instance(self, command: CatalogManagedCommand) -> CatalogManagedResult:
        """Atomically admit or mutate one authenticated managed instance.

        Authentication and authorization belong to the caller. All catalog,
        generation, replay, lease, and identity checks occur under this lock.
        Revoke is instance-specific and never revokes its logical siblings.

        Raises:
            CatalogValidationError: If Agent Card admission rejects the entry.
        """
        with self._lock:
            return self._managed.execute(command)

    def expire_managed_instances(self) -> tuple[tuple[str, str, CatalogManagedResult], ...]:
        """Expire managed leases and consume original-principal attribution receipts.

        Each expiry is returned once, including expirations already observed by a
        command, receipt lookup, or legacy sweep. Delivery failures after this
        handoff must be retried by the caller.
        """
        with self._lock:
            return self._managed.expire()

    def lookup_managed_request(self, command: CatalogManagedCommand) -> CatalogManagedResult | None:
        """Resolve a managed receipt before fetching a registration's Agent Card.

        The caller must first authenticate and authorize the current request.
        Returns ``None`` when its principal-scoped key has no receipt, otherwise
        the original result or a current conflict, generation, or terminal error.
        Expired leases are fenced under the catalog lock; no lease is created or
        extended. A cache miss is not authorization to admit: ``manage_instance``
        must still perform all checks after card retrieval.
        """
        with self._lock:
            return self._managed.lookup(command)

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
        for entry in load_entries(provider):
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
                or provenance policy rejects this entry, or if the identity is
                managed. Direct attempts to replace managed instances permanently
                invalidate their authority; use ``manage_instance`` instead.
        """
        prepared = prepare_entry(entry)
        now = self._clock()

        with self._lock:
            self._managed.guard_external_entry(entry, prepared)
            existing = self._agents.get(entry.agent_id)
            self._admission.validate(
                entry=entry,
                prepared=prepared,
                existing_card_name=existing.card_name if existing is not None else None,
                existing_capabilities=existing.capabilities if existing is not None else (),
            )

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
                card_name=prepared.card_name,
                card_digest=prepared.card_digest,
                logical_digest=prepared.logical_digest,
                capabilities=prepared.capabilities,
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
            CatalogValidationError: If the duration is not finite and positive,
                or if the instance is managed. A direct managed heartbeat fences
                its authority; use ``manage_instance`` instead.
        """
        now = self._clock()
        with self._lock:
            self._managed.reject_external_instance(agent_id, instance_id)
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
            if not math.isfinite(duration) or duration <= 0:
                raise CatalogValidationError("lease_seconds must be finite and positive")
            renewed = replace(
                instance,
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
            expired.extend(self._managed.sweep())
            for agent_id, state in list(self._agents.items()):
                instances = dict(state.instances)
                changed = False
                for instance_id, instance in list(instances.items()):
                    if instance.healthy and instance.is_expired(now):
                        instances[instance_id] = replace(
                            instance,
                            healthy=False,
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

        Any non-active transition permanently fences managed instance authority.
        Reactivation restores legacy eligibility, not previously managed leases.

        Args:
            agent_id: The logical agent to transition.
            lifecycle: The lifecycle state to apply.

        Raises:
            UnknownCatalogAgentError: If the logical agent is not registered.
        """
        with self._lock:
            if lifecycle is not CatalogLifecycleState.ACTIVE:
                terminal = (
                    CatalogInstanceState.REMOVED
                    if lifecycle is CatalogLifecycleState.REMOVED
                    else CatalogInstanceState.REVOKED
                )
                self._managed.fence(agent_id, terminal=terminal)
            state = self._require_agent_locked(agent_id)
            self._agents[agent_id] = _AgentState(
                agent_id=state.agent_id,
                owner=state.owner,
                provenance=state.provenance,
                trust_policy_ref=state.trust_policy_ref,
                supported_versions=state.supported_versions,
                card_name=state.card_name,
                card_digest=state.card_digest,
                logical_digest=state.logical_digest,
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
            self._managed.fence(agent_id, terminal=CatalogInstanceState.REMOVED)
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
        logical_digest=state.logical_digest,
        capabilities=state.capabilities,
        lifecycle=state.lifecycle,
        generation=state.generation,
        instances=instances,
    )
