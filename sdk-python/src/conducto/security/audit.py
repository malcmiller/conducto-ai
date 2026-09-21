"""Vendor-neutral, durable security audit contracts.

Audit events are security evidence, not diagnostic logs.  Applications own
durable storage and export adapters; this module deliberately has no logging,
telemetry, or vendor dependency.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from .errors import SecurityError

AUDIT_SCHEMA_VERSION = "1"
_FORBIDDEN = frozenset(
    {
        "token",
        "authorization",
        "credential",
        "password",
        "secret",
        "signature",
        "private_key",
        "prompt",
        "argument",
        "args",
        "result",
        "payload",
        "traceback",
        "exception",
    }
)


class AuditEventName(StrEnum):
    """Stable event taxonomy. Add names; never repurpose existing names."""

    AUTHORIZATION_ALLOWED = "security.authorization.allowed"
    AUTHORIZATION_DENIED = "security.authorization.denied"
    APPROVAL_REQUESTED = "security.approval.requested"
    APPROVAL_APPROVED = "security.approval.approved"
    APPROVAL_DENIED = "security.approval.denied"
    APPROVAL_EXPIRED = "security.approval.expired"
    APPROVAL_CANCELED = "security.approval.canceled"
    SIGNATURE_VERIFIED = "security.signature.verified"
    SIGNATURE_REJECTED = "security.signature.rejected"
    REPLAY_REJECTED = "security.replay.rejected"
    EXECUTION_ACCEPTED = "security.execution.accepted"
    EXECUTION_STARTED = "security.execution.started"
    EXECUTION_COMPLETED = "security.execution.completed"
    EXECUTION_FAILED = "security.execution.failed"
    DELIVERY_FAILED = "security.audit_delivery.failed"
    TOKEN_EXCHANGE_SUCCEEDED = "security.token_exchange.succeeded"
    TOKEN_EXCHANGE_FAILED = "security.token_exchange.failed"
    TOKEN_VALIDATED = "security.token.validated"
    TOKEN_VALIDATION_REJECTED = "security.token.rejected"
    MTLS_AUTHENTICATED = "security.mtls.authenticated"
    MTLS_REJECTED = "security.mtls.rejected"
    DELEGATION_ATTENUATED = "security.delegation.attenuated"
    DELEGATION_REJECTED = "security.delegation.rejected"


class AuditDecision(StrEnum):
    """Stable authorization or approval decision values."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    NOT_APPLICABLE = "not_applicable"


class AuditOutcome(StrEnum):
    """Stable operation outcomes."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    REJECTED = "rejected"


class AuditSeverity(StrEnum):
    """Severity is evidence classification, not a logging level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AuditCategory(StrEnum):
    """Stable event categories used by delivery policy."""

    AUTHORIZATION = "authorization"
    APPROVAL = "approval"
    CRYPTOGRAPHY = "cryptography"
    EXECUTION = "execution"
    DELIVERY = "delivery"


class AuditDeliveryMode(StrEnum):
    """Explicit, application-selected delivery behavior."""

    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Immutable versioned audit envelope containing only safe identifiers.

    Schema evolution is additive for optional fields within a version. Breaking
    changes require a new ``schema_version``; sinks must reject unknown versions.
    ``sequence`` provides causal ordering only within one task.
    """

    event_name: AuditEventName
    category: AuditCategory
    decision: AuditDecision
    outcome: AuditOutcome
    reason_code: str
    subject_id: str = ""
    issuer: str = ""
    audience: str | tuple[str, ...] = ""
    task_id: str = ""
    challenge_id: str = ""
    agent_id: str = ""
    capability_id: str = ""
    policy_id: str = ""
    policy_version: str = ""
    correlation_id: str = ""
    resource: str = ""
    trace_id: str = ""
    span_id: str = ""
    severity: AuditSeverity = AuditSeverity.INFO
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    sequence: int = 0
    extensions: Mapping[str, str | int | float | bool] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize UTC timestamps and reject unsafe extension data."""
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported audit event schema version")
        if not self.event_id or not self.reason_code:
            raise ValueError("audit event_id and reason_code are required")
        if self.occurred_at.tzinfo is None:
            raise ValueError("audit event timestamp must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        object.__setattr__(self, "extensions", MappingProxyType(_safe_extensions(self.extensions)))

    @property
    def schema_version(self) -> str:
        """Pinned envelope schema version."""
        return AUDIT_SCHEMA_VERSION

    @property
    def idempotency_key(self) -> str:
        """Stable delivery key; sinks may safely deduplicate this event."""
        return f"{self.schema_version}:{self.event_id}"


def _safe_extensions(values: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        normalized = str(key).lower()
        if any(word in normalized for word in _FORBIDDEN):
            raise ValueError(f"unsafe audit extension field: {key}")
        if not isinstance(value, str | int | float | bool) or (
            isinstance(value, str) and len(value) > 256
        ):
            raise ValueError(f"unsafe audit extension value: {key}")
        if isinstance(value, str) and any(word in value.lower() for word in _FORBIDDEN):
            raise ValueError(f"unsafe audit extension value: {key}")
        safe[str(key)] = value
    return safe


@dataclass(frozen=True, slots=True)
class AuditDeliveryReceipt:
    """Explicit sink acceptance acknowledgement."""

    event_id: str
    idempotency_key: str
    accepted_at: datetime
    duplicate: bool = False
    buffered: bool = False


@dataclass(frozen=True, slots=True)
class AuditDeliveryFailure:
    """Safe sink failure; never contains exception details or event payloads."""

    event_id: str
    reason_code: str
    retryable: bool = False


class AuditSink(Protocol):
    """Application-owned asynchronous acceptance boundary."""

    async def emit(self, event: AuditEvent) -> AuditDeliveryReceipt | AuditDeliveryFailure:
        """Accept one event or return an explicit failure."""

    async def flush(self, timeout: float | None = None) -> None:
        """Boundedly drain accepted buffered events during application shutdown."""


class AuditDeliveryError(SecurityError):
    """Required audit evidence was not accepted before protected execution."""

    reason_code = "audit_delivery_failed"

    def __init__(self, failure: AuditDeliveryFailure) -> None:
        super().__init__(failure.reason_code)
        self.failure = failure


@dataclass(frozen=True, slots=True)
class AuditDeliveryPolicy:
    """Bounded delivery policy selected explicitly by the application."""

    mode: AuditDeliveryMode = AuditDeliveryMode.FAIL_CLOSED
    timeout_seconds: float = 1.0
    max_buffered_events: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_buffered_events < 0:
            raise ValueError("invalid audit delivery policy bounds")


class InMemoryAuditSink:
    """Thread-safe-in-practice async reference sink with bounded capacity."""

    def __init__(self, *, max_events: int = 1_000) -> None:
        if max_events < 0:
            raise ValueError("max_events must be non-negative")
        self.max_events = max_events
        self.events: list[AuditEvent] = []
        self._seen: set[str] = set()
        self.dropped_events = 0

    async def emit(self, event: AuditEvent) -> AuditDeliveryReceipt | AuditDeliveryFailure:
        key = event.idempotency_key
        if key in self._seen:
            return AuditDeliveryReceipt(event.event_id, key, datetime.now(UTC), duplicate=True)
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return AuditDeliveryFailure(event.event_id, "audit_backpressure", retryable=True)
        self._seen.add(key)
        self.events.append(event)
        return AuditDeliveryReceipt(event.event_id, key, datetime.now(UTC))

    async def flush(self, timeout: float | None = None) -> None:
        del timeout


class FailingAuditSink:
    """Deterministic failure fixture for conformance and fail-closed tests."""

    def __init__(
        self, *, reason_code: str = "audit_sink_unavailable", retryable: bool = True
    ) -> None:
        self.reason_code = reason_code
        self.retryable = retryable
        self.attempts: list[AuditEvent] = []

    async def emit(self, event: AuditEvent) -> AuditDeliveryFailure:
        self.attempts.append(event)
        return AuditDeliveryFailure(event.event_id, self.reason_code, self.retryable)

    async def flush(self, timeout: float | None = None) -> None:
        del timeout


@dataclass(slots=True)
class _TaskDeliveryState:
    """Transient per-task serialization state, evicted after terminal events."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sequence: int = 0
    active: int = 0
    terminal: bool = False


class AuditEmitter:
    """Serializes per-task delivery and enforces explicit delivery policy."""

    def __init__(self, sink: AuditSink, *, policy: AuditDeliveryPolicy | None = None) -> None:
        self.sink = sink
        self.policy = policy or AuditDeliveryPolicy()
        self._tasks: dict[str, _TaskDeliveryState] = {}
        self._buffer: list[AuditEvent] = []
        self.dropped_events = 0

    async def emit(
        self, event: AuditEvent, *, required: bool = False
    ) -> AuditDeliveryReceipt | None:
        """Deliver evidence with bounded timeout and explicit failure behavior."""
        task_key = event.task_id or event.correlation_id or event.event_id
        state = self._tasks.setdefault(task_key, _TaskDeliveryState())
        state.active += 1
        try:
            async with state.lock:
                state.sequence += 1
                event = dataclass_replace(event, sequence=state.sequence)
                receipt = await self._deliver(event, required=required)
                if self._is_terminal(event):
                    state.terminal = True
                return receipt
        finally:
            state.active -= 1
            if state.terminal and state.active == 0 and self._tasks.get(task_key) is state:
                del self._tasks[task_key]

    async def _deliver(self, event: AuditEvent, *, required: bool) -> AuditDeliveryReceipt | None:
        """Deliver one sequenced event according to bounded policy."""
        try:
            result = await asyncio.wait_for(
                self.sink.emit(event), timeout=self.policy.timeout_seconds
            )
        except TimeoutError:
            result = AuditDeliveryFailure(event.event_id, "audit_sink_timeout", retryable=True)
        if isinstance(result, AuditDeliveryReceipt):
            return result
        self._emergency_signal(result)
        if result.retryable and len(self._buffer) < self.policy.max_buffered_events:
            self._buffer.append(event)
            return AuditDeliveryReceipt(
                event.event_id, event.idempotency_key, datetime.now(UTC), buffered=True
            )
        if result.retryable:
            self.dropped_events += 1
        # Fail-open is intentionally unavailable for required security
        # evidence; it only permits explicit non-required lifecycle data.
        if required:
            raise AuditDeliveryError(result)
        return None

    @staticmethod
    def _is_terminal(event: AuditEvent) -> bool:
        """Return whether no further event is expected in this task lifecycle."""
        return event.event_name in {
            AuditEventName.AUTHORIZATION_DENIED,
            AuditEventName.APPROVAL_DENIED,
            AuditEventName.APPROVAL_EXPIRED,
            AuditEventName.APPROVAL_CANCELED,
            AuditEventName.SIGNATURE_REJECTED,
            AuditEventName.REPLAY_REJECTED,
            AuditEventName.EXECUTION_COMPLETED,
            AuditEventName.EXECUTION_FAILED,
        }

    async def flush(self, timeout: float | None = None) -> None:
        """Attempt each buffered event once; retry remains bounded."""
        deadline = timeout if timeout is not None else self.policy.timeout_seconds
        pending, self._buffer = self._buffer, []
        for event in pending:
            try:
                result = await asyncio.wait_for(self.sink.emit(event), timeout=deadline)
            except TimeoutError:
                result = AuditDeliveryFailure(event.event_id, "audit_sink_timeout", retryable=True)
            if isinstance(result, AuditDeliveryFailure):
                self.dropped_events += 1
                self._emergency_signal(result)
        await self.sink.flush(timeout=deadline)

    @staticmethod
    def _emergency_signal(failure: AuditDeliveryFailure) -> None:
        logging.getLogger("conducto.security.audit").critical(
            "security audit delivery failed: event_id=%s reason=%s",
            failure.event_id,
            failure.reason_code,
        )


def dataclass_replace(event: AuditEvent, **changes: Any) -> AuditEvent:
    """Avoid exposing a mutable event state while assigning a delivery sequence."""
    from dataclasses import replace

    return replace(event, **changes)
