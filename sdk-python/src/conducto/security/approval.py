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
    """Safe approval request payload without protected arguments or credentials.

    Attributes:
        approval_id: Stable identifier for the approval lifecycle.
        agent_id: Agent bound to the protected invocation.
        capability_id: Capability bound to the protected invocation.
        task_id: Task bound to the protected invocation.
        correlation_id: Correlation identifier bound to the invocation.
        reason_code: Stable application-facing reason for the challenge.
        required_role: Display role or aggregate role description.
        created_at: Challenge creation time.
        expires_at: Challenge expiration time.
        display: Safe metadata suitable for human-facing presentation.
        required_roles: Roles that must independently approve the challenge.
        principal_subject_id: Principal bound to the challenge.
    """

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
    policy_version: str = "1"

    def __post_init__(self) -> None:
        """Freeze display metadata so the challenge remains immutable."""
        object.__setattr__(self, "display", MappingProxyType(dict(self.display)))


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Immutable decision associated with one challenge.

    Attributes:
        approval_id: Identifier of the challenge being decided.
        approved: Whether the decision grants approval.
        decided_at: Time at which the decision was made.
        decided_by: Stable identifier of the approving subject.
        reason_code: Optional stable reason for the decision.
        role: Required role represented by this decision, if applicable.
    """

    approval_id: str
    approved: bool
    decided_at: datetime
    decided_by: str
    reason_code: str = ""
    role: str | None = None


class Clock(Protocol):
    """Clock abstraction used for deterministic lifecycle tests."""

    def now(self) -> datetime:
        """Return the current UTC-aware time."""


class IdentifierGenerator(Protocol):
    """Identifier source abstraction used for deterministic tests."""

    def __call__(self) -> str:
        """Return a new stable identifier."""


class ApprovalStore(Protocol):
    """Application-owned persistence boundary for approval challenges."""

    def create(self, challenge: ApprovalChallenge) -> ApprovalChallenge:
        """Persist a new pending challenge."""

    def get(self, approval_id: str) -> ApprovalChallenge:
        """Return a challenge by identifier."""

    def decide(self, decision: ApprovalDecision) -> ApprovalChallenge:
        """Apply a decision and return its challenge."""

    def cancel(self, approval_id: str) -> ApprovalChallenge:
        """Cancel a pending challenge."""

    def complete(self, approval_id: str) -> ApprovalChallenge:
        """Atomically consume an approved challenge."""

    def state(self, approval_id: str) -> ApprovalState:
        """Return the current lifecycle state."""


class SystemClock:
    """UTC wall clock implementation."""

    @staticmethod
    def now() -> datetime:
        """Return the current UTC-aware wall-clock time."""
        return datetime.now(UTC)


class InMemoryApprovalStore:
    """Thread-safe deterministic reference store for one process."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        """Initialize an empty store.

        Args:
            clock: Clock used for expiration checks. Defaults to UTC wall time.
        """
        self._clock = clock or SystemClock()
        self._items: dict[str, ApprovalChallenge] = {}
        self._states: dict[str, ApprovalState] = {}
        self._approved_roles: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def create(self, challenge: ApprovalChallenge) -> ApprovalChallenge:
        """Persist a new pending challenge.

        Args:
            challenge: Challenge to persist.

        Returns:
            The persisted challenge.

        Raises:
            InvalidApprovalStateError: If the approval identifier already exists.
        """
        with self._lock:
            if challenge.approval_id in self._items:
                raise InvalidApprovalStateError("approval id already exists")
            self._items[challenge.approval_id] = challenge
            self._states[challenge.approval_id] = ApprovalState.REQUIRED
            self._approved_roles[challenge.approval_id] = set()
            return challenge

    def get(self, approval_id: str) -> ApprovalChallenge:
        """Return a challenge and update expired pending state.

        Args:
            approval_id: Identifier of the challenge to retrieve.

        Returns:
            The matching challenge.

        Raises:
            ApprovalExpiredError: If the pending challenge has expired.
            InvalidApprovalStateError: If the identifier is unknown.
        """
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
        """Apply one approval or denial decision.

        Args:
            decision: Decision to apply.

        Returns:
            The challenge associated with the decision.

        Raises:
            ApprovalExpiredError: If the challenge has expired.
            InvalidApprovalStateError: If the transition or role is invalid.
        """
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
        """Cancel a pending challenge.

        Args:
            approval_id: Identifier of the challenge to cancel.

        Returns:
            The canceled challenge.

        Raises:
            ApprovalExpiredError: If the challenge has expired.
            InvalidApprovalStateError: If the challenge is not pending.
        """
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
        """Return to the current state for a challenge.

        Args:
            approval_id: Identifier of the challenge.

        Returns:
            The challenge lifecycle state.

        Raises:
            InvalidApprovalStateError: If the identifier is unknown.
        """
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
    """Build a safe challenge payload for an invocation.

    Args:
        context: Immutable authorization context for the invocation.
        agent_id: Agent bound to the challenge.
        capability_id: Capability bound to the challenge.
        role: Role or aggregate role description displayed to approvers.
        reason_code: Stable reason code for the challenge.
        ttl_seconds: Number of seconds before the challenge expires.
        identifiers: Identifier generator used for the challenge ID.
        clock: Clock used for the creation timestamp.
        required_roles: Roles that must be independently approved.
        principal_subject_id: Principal subject to bind, if overridden.

    Returns:
        An immutable approval challenge.
    """
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
