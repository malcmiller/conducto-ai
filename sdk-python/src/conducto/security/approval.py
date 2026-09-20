"""Approval challenge lifecycle and in-memory persistence contracts."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from .context import AuthorizationContext
from .errors import ApprovalExpiredError, InvalidApprovalStateError


class ApprovalState(StrEnum):
    """Pinned lifecycle states for approval-bound execution."""

    REQUIRED = "input-required"
    APPROVED = "working"
    DENIED = "rejected"
    EXPIRED = "failed"
    CANCELED = "canceled"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ApprovalChallenge:
    """Safe approval request payload without protected arguments or credentials."""

    approval_id: str
    agent_id: str
    capability_id: str
    task_id: str
    correlation_id: str
    reason_code: str
    required_role: str
    created_at: datetime
    expires_at: datetime
    display: Mapping[str, str] = field(default_factory=dict)
    required_roles: tuple[str, ...] = ()
    principal_subject_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "display", MappingProxyType(dict(self.display)))


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Immutable decision associated with one challenge."""

    approval_id: str
    approved: bool
    decided_at: datetime
    decided_by: str
    reason_code: str = ""
    role: str | None = None


class Clock(Protocol):
    """Clock abstraction used for deterministic lifecycle tests."""

    def now(self) -> datetime: ...


class IdentifierGenerator(Protocol):
    """Identifier source abstraction used for deterministic tests."""

    def __call__(self) -> str: ...


class ApprovalStore(Protocol):
    """Application-owned persistence boundary for approval challenges."""

    def create(self, challenge: ApprovalChallenge) -> ApprovalChallenge: ...
    def get(self, approval_id: str) -> ApprovalChallenge: ...
    def decide(self, decision: ApprovalDecision) -> ApprovalChallenge: ...
    def cancel(self, approval_id: str) -> ApprovalChallenge: ...
    def complete(self, approval_id: str) -> ApprovalChallenge: ...
    def state(self, approval_id: str) -> ApprovalState: ...


class SystemClock:
    """UTC wall clock implementation."""

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)


class InMemoryApprovalStore:
    """Thread-safe deterministic reference store for one process."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._items: dict[str, ApprovalChallenge] = {}
        self._states: dict[str, ApprovalState] = {}
        self._approved_roles: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def create(self, challenge: ApprovalChallenge) -> ApprovalChallenge:
        with self._lock:
            if challenge.approval_id in self._items:
                raise InvalidApprovalStateError("approval id already exists")
            self._items[challenge.approval_id] = challenge
            self._states[challenge.approval_id] = ApprovalState.REQUIRED
            self._approved_roles[challenge.approval_id] = set()
            return challenge

    def get(self, approval_id: str) -> ApprovalChallenge:
        with self._lock:
            challenge = self._lookup(approval_id)
            if (
                self._states[approval_id] == ApprovalState.REQUIRED
                and self._clock.now() >= challenge.expires_at
            ):
                self._states[approval_id] = ApprovalState.EXPIRED
                raise ApprovalExpiredError("approval challenge expired")
            return challenge

    def decide(self, decision: ApprovalDecision) -> ApprovalChallenge:
        with self._lock:
            challenge = self._lookup(decision.approval_id)
            if (
                self._states[decision.approval_id] == ApprovalState.REQUIRED
                and self._clock.now() >= challenge.expires_at
            ):
                self._states[decision.approval_id] = ApprovalState.EXPIRED
                raise ApprovalExpiredError("approval challenge expired")
            if self._states[decision.approval_id] != ApprovalState.REQUIRED:
                raise InvalidApprovalStateError("approval is no longer pending")
            if not decision.approved:
                self._states[decision.approval_id] = ApprovalState.DENIED
            else:
                roles = challenge.required_roles or (challenge.required_role,)
                role = decision.role or (roles[0] if len(roles) == 1 else None)
                if role not in roles:
                    raise InvalidApprovalStateError("decision role is not required")
                assert role is not None
                self._approved_roles[decision.approval_id].add(role)
                if set(roles).issubset(self._approved_roles[decision.approval_id]):
                    self._states[decision.approval_id] = ApprovalState.APPROVED
            return challenge

    def cancel(self, approval_id: str) -> ApprovalChallenge:
        with self._lock:
            challenge = self._lookup(approval_id)
            if (
                self._states[approval_id] == ApprovalState.REQUIRED
                and self._clock.now() >= challenge.expires_at
            ):
                self._states[approval_id] = ApprovalState.EXPIRED
                raise ApprovalExpiredError("approval challenge expired")
            if self._states[approval_id] != ApprovalState.REQUIRED:
                raise InvalidApprovalStateError("approval is no longer pending")
            self._states[approval_id] = ApprovalState.CANCELED
            return challenge

    def state(self, approval_id: str) -> ApprovalState:
        with self._lock:
            self._lookup(approval_id)
            return self._states[approval_id]

    def complete(self, approval_id: str) -> ApprovalChallenge:
        """Atomically consume an approved challenge exactly once."""
        with self._lock:
            challenge = self._lookup(approval_id)
            if self._states[approval_id] != ApprovalState.APPROVED:
                raise InvalidApprovalStateError("approval is not approved or was already consumed")
            self._states[approval_id] = ApprovalState.COMPLETED
            return challenge

    def _lookup(self, approval_id: str) -> ApprovalChallenge:
        try:
            return self._items[approval_id]
        except KeyError as error:
            raise InvalidApprovalStateError("unknown approval id") from error


def default_challenge(
    context: AuthorizationContext,
    *,
    agent_id: str,
    capability_id: str,
    role: str,
    reason_code: str = "approval_required",
    ttl_seconds: float = 300,
    identifiers: IdentifierGenerator = lambda: str(uuid.uuid4()),
    clock: Clock | None = None,
    required_roles: tuple[str, ...] = (),
    principal_subject_id: str | None = None,
) -> ApprovalChallenge:
    """Build a safe challenge payload for an invocation."""
    now = (clock or SystemClock()).now()
    return ApprovalChallenge(
        identifiers(),
        agent_id,
        capability_id,
        context.task_id,
        context.correlation_id,
        reason_code,
        role,
        now,
        now + timedelta(seconds=ttl_seconds),
        required_roles=required_roles,
        principal_subject_id=principal_subject_id or context.principal.subject_id,
    )
