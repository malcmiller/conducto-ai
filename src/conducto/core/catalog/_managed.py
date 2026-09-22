"""Bounded managed identity ledger; every operation requires the catalog lock."""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ._admission import PreparedEntry, prepare_entry
from ._managed_models import (
    CatalogInstanceState as State,
)
from ._managed_models import (
    CatalogManagedCode as Code,
)
from ._managed_models import (
    CatalogManagedCommand as Command,
)
from ._managed_models import (
    CatalogManagedResult as Result,
)
from ._models import CatalogAgentRecord, CatalogEntry, CatalogLifecycleState, CatalogValidationError

if TYPE_CHECKING:
    from ._lifecycle import _AgentState, _InstanceState


@dataclass(slots=True)
class _Managed:
    owner: str
    environment: str
    subject_id: str
    issuer: str
    handle: str = field(repr=False)
    generation: int
    state: State
    instance: _InstanceState

    def result(self, code: Code = Code.OK, *, handle: bool = False) -> Result:
        return Result(
            code=code,
            generation=self.generation,
            state=self.state,
            lease_expires_at=self.instance.lease_expires_at,
            lease_handle=self.handle if handle else "",
            subject_id=self.subject_id,
            issuer=self.issuer,
        )


@dataclass(frozen=True, slots=True)
class _Receipt:
    fingerprint: str
    identity: tuple[str, str]
    result: Result


class ManagedInstances:
    """Catalog-owned identity tombstones and receipts, never independently locked.

    Tombstones and ordinary successful receipts each have a fixed capacity.
    One extra terminal receipt per identity reserves removal/revocation capacity.
    Nothing is evicted: rejecting further grants is safer than forgetting fences.
    Pending expiry evidence has at most one receipt per retained identity.
    """

    def __init__(
        self,
        *,
        capacity: int,
        clock: Callable[[], float],
        agents: dict[str, _AgentState],
        admit: Callable[[CatalogEntry], CatalogAgentRecord],
        changed: Callable[[], None],
    ) -> None:
        """Bind exclusively to the owning catalog's locked state and mutation hooks."""
        if capacity <= 0:
            raise ValueError("managed_capacity must be positive")
        self.clock = clock
        self.agents = agents
        self.admit = admit
        self.changed = changed
        self.capacity = capacity
        self.instances: dict[tuple[str, str], _Managed] = {}
        self.receipts: dict[tuple[str, str, str], _Receipt] = {}
        self.pending_expiries: dict[tuple[str, str], Result] = {}

    def execute(self, command: Command) -> Result:
        """Check identity, generation, and receipts before applying one mutation."""
        replay = self.lookup(command)
        if replay is not None:
            return replay
        identity = (command.agent_id, command.instance_id)
        key = (command.issuer, command.subject_id, command.idempotency_key)
        managed = self.instances.get(identity)
        if managed is not None:
            denied = self._check_managed(command, identity, managed)
            if denied is not None:
                return denied
        if command.operation == "register":
            if managed is not None:
                return managed.result(Code.IDENTITY_CONFLICT)
            if command.expected_generation != 0:
                return Result(Code.STALE_GENERATION)
        else:
            if managed is None:
                return Result(Code.NOT_FOUND)
            if command.operation != "status" and command.expected_generation != managed.generation:
                return managed.result(Code.STALE_GENERATION)
        if command.operation in ("register", "renew") and (
            not math.isfinite(command.lease_seconds) or command.lease_seconds <= 0
        ):
            return Result(Code.INVALID_LEASE)
        if len(self.receipts) >= self.capacity and command.operation not in (
            "deregister",
            "revoke",
        ):
            return Result(Code.CAPACITY_EXCEEDED)
        if command.operation == "register":
            result = self._register(command, identity)
        else:
            assert managed is not None
            result = self._mutate(command, identity, managed)
        if result.code is Code.OK:
            self.receipts[key] = _Receipt(command.fingerprint, identity, result)
        return result

    def lookup(self, command: Command) -> Result | None:
        """Resolve existing receipts without admitting or extending an instance."""
        key = (command.issuer, command.subject_id, command.idempotency_key)
        receipt = self.receipts.get(key)
        if receipt is None:
            return None
        identity = (command.agent_id, command.instance_id)
        managed = self.instances.get(identity)
        if managed is not None:
            authorization_error = self._authorize(command, managed)
            if authorization_error is not None:
                return Result(authorization_error)
        elif command.operation != "register":
            return Result(Code.NOT_FOUND)
        if receipt.fingerprint != command.fingerprint or receipt.identity != identity:
            return Result(Code.IDEMPOTENCY_CONFLICT)
        if managed is None:
            return Result(Code.REPLAY_REJECTED)
        denied = self._check_managed(command, identity, managed)
        if denied is not None:
            if (
                denied.code is Code.INACTIVE
                and command.operation in ("deregister", "revoke")
                and receipt.result.generation == managed.generation
                and receipt.result.state is managed.state
            ):
                return receipt.result
            return denied
        if receipt.result.generation != managed.generation:
            return managed.result(Code.STALE_GENERATION)
        return receipt.result

    def _check_managed(
        self, command: Command, identity: tuple[str, str], managed: _Managed
    ) -> Result | None:
        denied = self._authorize(command, managed)
        if denied is not None:
            return Result(denied)
        self._expire_one(identity, managed, self.clock())
        if managed.state is State.EXPIRED:
            return managed.result(Code.EXPIRED)
        if managed.state in (State.REMOVED, State.REVOKED):
            return managed.result(Code.INACTIVE)
        return None

    @staticmethod
    def _authorize(command: Command, managed: _Managed) -> Code | None:
        if (command.owner, command.environment) != (managed.owner, managed.environment):
            return Code.IDENTITY_CONFLICT
        if command.operation == "revoke":
            return None
        if (command.issuer, command.subject_id) != (managed.issuer, managed.subject_id):
            return Code.IDENTITY_CONFLICT
        if command.operation != "register" and (
            not command.lease_handle.isascii()
            or not secrets.compare_digest(command.lease_handle, managed.handle)
        ):
            return Code.INVALID_HANDLE
        return None

    def _register(self, command: Command, identity: tuple[str, str]) -> Result:
        if len(self.instances) >= self.capacity:
            return Result(Code.CAPACITY_EXCEEDED)
        entry = command.entry
        if entry is None or (entry.agent_id, entry.instance_id, entry.owner) != (
            command.agent_id,
            command.instance_id,
            command.owner,
        ):
            return Result(Code.IDENTITY_CONFLICT)
        state = self.agents.get(command.agent_id)
        prepared = prepare_entry(entry)
        if state is not None:
            if state.lifecycle is not CatalogLifecycleState.ACTIVE:
                return Result(Code.INACTIVE)
            if state.owner != command.owner or command.instance_id in state.instances:
                return Result(Code.IDENTITY_CONFLICT)
            now = self.clock()
            if any(now < instance.lease_expires_at for instance in state.instances.values()) and (
                state.logical_digest != prepared.logical_digest
                or state.supported_versions != entry.supported_versions
            ):
                return Result(Code.IDENTITY_CONFLICT)
        self.admit(replace(entry, lease_seconds=command.lease_seconds))
        state = self.agents[command.agent_id]
        instance = replace(
            state.instances[command.instance_id],
            environment=command.environment,
            deployment_id=command.deployment_id,
            provenance=command.provenance,
            subject_id=command.subject_id,
            issuer=command.issuer,
        )
        instances = dict(state.instances)
        instances[command.instance_id] = instance
        self.agents[command.agent_id] = replace(state, instances=instances)
        managed = _Managed(
            command.owner,
            command.environment,
            command.subject_id,
            command.issuer,
            secrets.token_urlsafe(32),
            1,
            State.ACTIVE,
            instance,
        )
        self.instances[identity] = managed
        return managed.result(handle=True)

    def _mutate(self, command: Command, identity: tuple[str, str], managed: _Managed) -> Result:
        if command.operation == "status":
            return managed.result()
        if command.operation == "renew":
            if managed.state is not State.ACTIVE:
                return managed.result(Code.INACTIVE)
            now = self.clock()
            instance = replace(
                managed.instance,
                lease_expires_at=max(
                    managed.instance.lease_expires_at, now + command.lease_seconds
                ),
                last_heartbeat_at=now,
            )
            self._publish(identity, managed, State.ACTIVE, instance)
        elif command.operation == "drain":
            if managed.state is not State.DRAINING:
                self._publish(
                    identity, managed, State.DRAINING, replace(managed.instance, healthy=False)
                )
        elif command.operation in ("deregister", "revoke"):
            terminal = State.REMOVED if command.operation == "deregister" else State.REVOKED
            self._publish(identity, managed, terminal, replace(managed.instance, healthy=False))
        else:
            return managed.result(Code.REPLAY_REJECTED)
        return managed.result()

    def _publish(
        self, identity: tuple[str, str], managed: _Managed, state: State, instance: _InstanceState
    ) -> None:
        managed.state = state
        managed.instance = instance
        managed.generation += 1
        agent_id, instance_id = identity
        agent = self.agents.get(agent_id)
        if agent is not None:
            instances = dict(agent.instances)
            if state is State.REMOVED:
                instances.pop(instance_id, None)
            else:
                instances[instance_id] = instance
            self.agents[agent_id] = replace(agent, instances=instances)
        self.changed()

    def _expire_one(self, identity: tuple[str, str], managed: _Managed, now: float) -> bool:
        if managed.state not in (State.ACTIVE, State.DRAINING):
            return False
        if now < managed.instance.lease_expires_at:
            return False
        self._publish(identity, managed, State.EXPIRED, replace(managed.instance, healthy=False))
        self.pending_expiries[identity] = managed.result(Code.EXPIRED)
        return True

    def sweep(self) -> tuple[tuple[str, str], ...]:
        """Expire eligible leases without consuming pending attribution receipts."""
        now = self.clock()
        return tuple(
            identity
            for identity, managed in sorted(self.instances.items())
            if self._expire_one(identity, managed, now)
        )

    def expire(self) -> tuple[tuple[str, str, Result], ...]:
        """Consume attributable expiry receipts, including previously lazy expirations."""
        self.sweep()
        expired = tuple(
            (identity[0], identity[1], result)
            for identity, result in sorted(self.pending_expiries.items())
        )
        self.pending_expiries.clear()
        return expired

    def fence(
        self,
        agent_id: str,
        instance_id: str | None = None,
        *,
        terminal: State = State.REVOKED,
    ) -> None:
        """Permanently invalidate authority affected by a direct catalog mutation."""
        for identity, managed in self.instances.items():
            if identity[0] != agent_id or (instance_id is not None and identity[1] != instance_id):
                continue
            if managed.state in (State.ACTIVE, State.DRAINING):
                self._publish(identity, managed, terminal, replace(managed.instance, healthy=False))

    def reject_external_instance(self, agent_id: str, instance_id: str) -> None:
        """Reject legacy overwrite/heartbeat of managed identities and fence authority."""
        if (agent_id, instance_id) in self.instances:
            self.fence(agent_id, instance_id)
            raise CatalogValidationError("Managed instances require manage_instance")

    def guard_external_entry(self, entry: CatalogEntry, prepared: PreparedEntry) -> None:
        """Prevent legacy registration from overwriting managed sibling metadata."""
        self.reject_external_instance(entry.agent_id, entry.instance_id)
        state = self.agents.get(entry.agent_id)
        if state is None:
            return
        if any(
            identity[0] == entry.agent_id and managed.state in (State.ACTIVE, State.DRAINING)
            for identity, managed in self.instances.items()
        ) and (
            state.owner != entry.owner
            or state.logical_digest != prepared.logical_digest
            or state.supported_versions != entry.supported_versions
        ):
            raise CatalogValidationError("Entry conflicts with managed sibling identity")
