"""Conformance tests for the vendor-neutral security audit contract."""

import asyncio

import pytest

from conducto.security import (
    AuditCategory,
    AuditDecision,
    AuditDeliveryError,
    AuditDeliveryFailure,
    AuditDeliveryMode,
    AuditDeliveryPolicy,
    AuditDeliveryReceipt,
    AuditEmitter,
    AuditEvent,
    AuditEventName,
    AuditOutcome,
    AuthorizationContext,
    FailingAuditSink,
    InMemoryAuditSink,
    Principal,
    SecurityPipeline,
    require_scope,
)


def context() -> AuthorizationContext:
    """Build safe, attributable authorization facts."""
    return AuthorizationContext(
        Principal("subject-1", "issuer-1", "audience-1", scopes=frozenset({"execute"})),
        task_id="task-1",
        correlation_id="correlation-1",
    )


def test_in_memory_sink_receipts_duplicates_and_backpressure() -> None:
    """Reference sink makes duplicate and bounded-delivery behavior deterministic."""
    sink = InMemoryAuditSink(max_events=1)
    event = AuditEvent(
        AuditEventName.AUTHORIZATION_ALLOWED,
        AuditCategory.AUTHORIZATION,
        AuditDecision.ALLOW,
        AuditOutcome.SUCCESS,
        "authorized",
    )
    receipt = asyncio.run(sink.emit(event))
    assert isinstance(receipt, AuditDeliveryReceipt)
    assert receipt.event_id == event.event_id
    duplicate = asyncio.run(sink.emit(event))
    assert isinstance(duplicate, AuditDeliveryReceipt)
    assert duplicate.duplicate
    second = AuditEvent(
        AuditEventName.EXECUTION_ACCEPTED,
        AuditCategory.EXECUTION,
        AuditDecision.ALLOW,
        AuditOutcome.SUCCESS,
        "audit_accepted",
    )
    failure = asyncio.run(sink.emit(second))
    assert isinstance(failure, AuditDeliveryFailure)
    assert failure.reason_code == "audit_backpressure"
    assert sink.dropped_events == 1


def test_unsafe_extensions_are_rejected() -> None:
    """Payload-like extension names cannot enter an audit envelope."""
    with pytest.raises(ValueError, match="unsafe"):
        AuditEvent(
            AuditEventName.AUTHORIZATION_DENIED,
            AuditCategory.AUTHORIZATION,
            AuditDecision.DENY,
            AuditOutcome.REJECTED,
            "missing_context",
            extensions={"raw_token": "never-record-this"},
        )
    with pytest.raises(ValueError, match="unsafe"):
        AuditEvent(
            AuditEventName.AUTHORIZATION_DENIED,
            AuditCategory.AUTHORIZATION,
            AuditDecision.DENY,
            AuditOutcome.REJECTED,
            "missing_context",
            extensions={"diagnostic": "traceback content must not be delivered"},
        )


def test_terminal_events_release_per_task_delivery_state() -> None:
    """Completed task state is evicted rather than retained indefinitely."""
    emitter = AuditEmitter(InMemoryAuditSink())
    terminal = AuditEvent(
        AuditEventName.EXECUTION_COMPLETED,
        AuditCategory.EXECUTION,
        AuditDecision.ALLOW,
        AuditOutcome.SUCCESS,
        "completed",
        task_id="task-1",
    )
    asyncio.run(emitter.emit(terminal))
    assert emitter._tasks == {}


def test_fail_closed_blocks_authorized_protected_execution_before_callback() -> None:
    """Required authorization evidence must be accepted before execution begins."""

    @require_scope("execute")
    def protected() -> None:
        pass

    pipeline = SecurityPipeline(
        audit_emitter=AuditEmitter(
            FailingAuditSink(),
            policy=AuditDeliveryPolicy(mode=AuditDeliveryMode.FAIL_CLOSED),
        )
    )
    with pytest.raises(AuditDeliveryError):
        asyncio.run(pipeline.check_async(protected, context(), {}, agent_id="a", capability_id="c"))


def test_fail_open_is_explicit_and_deterministic_for_noncritical_event() -> None:
    """Non-required delivery may fail without changing an explicit fail-open result."""
    emitter = AuditEmitter(
        FailingAuditSink(),
        policy=AuditDeliveryPolicy(mode=AuditDeliveryMode.FAIL_OPEN),
    )
    event = AuditEvent(
        AuditEventName.EXECUTION_FAILED,
        AuditCategory.EXECUTION,
        AuditDecision.ALLOW,
        AuditOutcome.FAILURE,
        "capability_exception",
    )
    assert asyncio.run(emitter.emit(event, required=False)) is None
