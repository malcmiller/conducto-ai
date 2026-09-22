"""Conformance tests for inbound A2A execution through the canonical runtime."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from a2a.types.a2a_pb2 import (
    CancelTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
)
from google.protobuf.json_format import MessageToDict

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability, get_run_context
from conducto.a2a import (
    A2AASGI,
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2ARequestContext,
    A2ARuntimeHandler,
    create_a2a_app,
)
from conducto.core.agent_card import stable_skill_id
from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationCancelled,
    InvocationFailure,
    InvocationInternalFailure,
    InvocationStaleBinding,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTimeout,
    InvocationValidationFailure,
)
from conducto.core.provider import ModelConfiguration, ProviderResult
from conducto.core.run_context import DelegationBudget
from conducto.security import (
    ApprovalDecision,
    AuditEmitter,
    AuthorizationContext,
    FailingAuditSink,
    InMemoryAuditSink,
    Principal,
    SecurityPipeline,
    require_approval,
    require_scope,
)
from conducto.testing import FakeModel
from conducto.transport import InMemoryTaskRepository

AGENT_ID = "Inbound Agent"
ECHO_SKILL = stable_skill_id(AGENT_ID, "echo")
GUARDED_SKILL = stable_skill_id(AGENT_ID, "guarded")
APPROVED_SKILL = stable_skill_id(AGENT_ID, "approved")
FAIL_SKILL = stable_skill_id(AGENT_ID, "fail")
WAIT_SKILL = stable_skill_id(AGENT_ID, "wait")
CONTEXT_SKILL = stable_skill_id(AGENT_ID, "context")
BUDGET_SKILL = stable_skill_id(AGENT_ID, "budget")
ENDPOINT_URL = "https://agent.example/a2a"


@a2a_agent(name=AGENT_ID, version="1.0", description="Exercises inbound runtime behavior.")
class _InboundAgent(BaseAgent):
    """Deterministic capabilities used by the inbound runtime tests."""

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        super().__init__()

    @a2a_capability(name="echo", description="Echo a validated integer.")
    def echo(self, value: int) -> dict[str, int]:
        """Return a JSON-serializable value."""
        self.calls += 1
        return {"value": value}

    @require_scope("invoke")
    @a2a_capability(name="guarded", description="Require an invocation scope.")
    def guarded(self) -> str:
        """Return the authenticated principal identifier."""
        self.calls += 1
        context = get_run_context()
        assert context is not None and context.authorization is not None
        return context.authorization.principal.subject_id

    @require_approval("reviewer")
    @a2a_capability(name="approved", description="Require approval.")
    def approved(self, value: str) -> str:
        """Return a value after an approved resume."""
        self.calls += 1
        return value

    @a2a_capability(name="fail", description="Raise a capability failure.")
    def fail(self) -> None:
        """Raise an exception that must not cross the A2A boundary."""
        self.calls += 1
        raise RuntimeError("sensitive capability detail")

    @a2a_capability(name="wait", description="Wait for deterministic cancellation.")
    async def wait(self) -> str:
        """Wait until the test releases or cancels this invocation."""
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return "released"

    @a2a_capability(
        name="context",
        description="Return safe invocation context.",
        model_required=True,
    )
    def context(self) -> dict[str, Any]:
        """Return task-local metadata, authority, and model provenance."""
        self.calls += 1
        context = get_run_context()
        assert context is not None
        assert context.authorization is not None
        return {
            "subject": context.authorization.principal.subject_id,
            "model": str(context.model_reference),
            "metadata": context.to_dict()["metadata"],
            "scopes": sorted(context.authorization.principal.scopes),
            "budget_calls": context.remaining_delegation_budget.calls,
            "timeout": context.timeout,
        }

    @a2a_capability(name="budget", description="Inspect and reserve delegation budget.")
    def budget(
        self,
        reserve_calls: int = 0,
        reserve_tokens: int = 0,
        reserve_cost: float = 0,
    ) -> dict[str, Any]:
        """Return request-owned budget state before and after one reservation."""
        self.calls += 1
        context = get_run_context()
        assert context is not None
        before = context.remaining_delegation_budget
        reserved = context.delegation_budget.reserve(
            calls=reserve_calls,
            tokens=reserve_tokens,
            cost=reserve_cost,
        )
        after = context.remaining_delegation_budget
        return {
            "before": {
                "depth": before.depth,
                "calls": before.calls,
                "tokens": before.tokens,
                "cost": before.cost,
            },
            "reserved": reserved,
            "after": {
                "depth": after.depth,
                "calls": after.calls,
                "tokens": after.tokens,
                "cost": after.cost,
            },
        }


class _IdentityResolver:
    """Build authorization from deterministic request headers."""

    def __init__(
        self,
        *,
        scopes: frozenset[str] = frozenset({"invoke"}),
        allowed_capabilities: frozenset[str] | None = None,
        delegation_budget: DelegationBudget | None = None,
        decision: ApprovalDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scopes = scopes
        self.allowed_capabilities = allowed_capabilities
        self.delegation_budget = delegation_budget
        self.decision = decision
        self.error = error
        self.requests: list[A2AAuthenticationRequest] = []

    async def __call__(self, request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
        """Return one immutable identity bound to the accepted task."""
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        subject = request.headers.get("x-test-principal", "principal")
        return A2AAuthenticatedIdentity(
            AuthorizationContext(
                Principal(
                    subject_id=subject,
                    issuer="https://issuer.example",
                    audience=AGENT_ID,
                    scopes=self.scopes,
                ),
                task_id=request.task_id,
                correlation_id=request.correlation_id,
            ),
            allowed_capabilities=self.allowed_capabilities,
            delegation_budget=self.delegation_budget,
            approval_decision=self.decision,
        )


def _message(
    skill_id: str,
    arguments: Any,
    *,
    message_id: str = "message-1",
    task_id: str = "",
    context_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Build the pinned Conducto invocation envelope in an A2A text part."""
    message = Message(
        message_id=message_id,
        role=Role.ROLE_USER,
        parts=[Part(text=json.dumps({"skillId": skill_id, "arguments": arguments}))],
        task_id=task_id,
        context_id=context_id,
    )
    if metadata:
        message.metadata.update({"x-conducto": metadata})
    return message


def _request(request_id: str = "request-1", *, principal: str = "principal") -> A2ARequestContext:
    """Build immutable protocol context for one direct handler call."""
    return A2ARequestContext(
        request_id=request_id,
        headers={"x-test-principal": principal},
    )


def _handler(
    *,
    runtime: Runtime | None = None,
    resolver: _IdentityResolver | None = None,
    clock: Any = None,
    **handler_options: Any,
) -> tuple[A2ARuntimeHandler, _InboundAgent, _IdentityResolver, Runtime]:
    """Build one isolated runtime handler."""
    runtime = runtime or Runtime()
    agent = _InboundAgent()
    resolver = resolver or _IdentityResolver()
    kwargs = {"clock": clock} if clock is not None else {}
    return (
        A2ARuntimeHandler(
            runtime=runtime,
            agent=agent,
            identity_resolver=resolver,
            **kwargs,
            **handler_options,
        ),
        agent,
        resolver,
        runtime,
    )


async def _invoke(
    handler: A2ARuntimeHandler,
    message: Message,
    *,
    task_id: str = "task-1",
    context_id: str = "context-1",
    request: A2ARequestContext | None = None,
) -> Any:
    """Invoke one message with deterministic IDs."""
    return await handler.handle_message(
        message,
        task_id=task_id,
        context_id=context_id,
        request_context=request or _request(),
    )


def test_valid_remote_invocation_matches_canonical_runtime_success() -> None:
    """Remote success uses runtime validation, serialization, and metadata."""
    handler, agent, resolver, runtime = _handler()
    message = _message(
        ECHO_SKILL,
        {"value": 4},
        metadata={
            "correlationId": "correlation-1",
            "metadata": {"safe": "value"},
        },
    )
    message.reference_task_ids.append("parent-task")

    async def run() -> None:
        remote = await _invoke(handler, message)
        local = await runtime.invoke(
            agent,
            "echo",
            {"value": 4},
            correlation_id="correlation-1",
            authorization=(
                await resolver(
                    A2AAuthenticationRequest(
                        "task-local",
                        "context-local",
                        "message-local",
                        "request-local",
                        "correlation-1",
                        {},
                    )
                )
            ).authorization,
        )
        assert isinstance(remote, InvocationSuccess)
        assert isinstance(local, InvocationSuccess)
        assert remote.value == local.value == {"value": 4}
        assert remote.metadata is not None
        assert remote.metadata.run_id == "task-1"
        assert remote.metadata.attributes["safe"] == "value"
        assert remote.metadata.attributes["a2a"]["request_id"] == "request-1"
        assert remote.metadata.attributes["a2a"]["lineage"] == ("parent-task",)

    asyncio.run(run())


def test_malformed_schema_invalid_and_unknown_requests_are_typed() -> None:
    """Adapter and runtime validation failures remain distinct and non-executing."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        malformed = _message(ECHO_SKILL, {})
        malformed.parts[0].text = "{"
        malformed_result = await _invoke(handler, malformed)
        schema_result = await _invoke(
            handler,
            _message(ECHO_SKILL, {"value": "wrong"}, message_id="message-2"),
            request=_request("request-2"),
        )
        unknown_result = await _invoke(
            handler,
            _message("unknown-skill", {}, message_id="message-3"),
            request=_request("request-3"),
        )
        invalid_timeout = await _invoke(
            handler,
            _message(
                ECHO_SKILL,
                {"value": 1},
                message_id="message-4",
                metadata={"deadline": 200.0, "timeoutSeconds": "invalid"},
            ),
            request=_request("request-4"),
        )
        assert isinstance(malformed_result, InvocationValidationFailure)
        assert isinstance(schema_result, InvocationValidationFailure)
        assert isinstance(unknown_result, InvocationTargetNotFound)
        assert isinstance(invalid_timeout, InvocationValidationFailure)
        assert agent.calls == 0

    asyncio.run(run())


def test_scopes_and_capability_allowlist_cannot_be_amplified_by_metadata() -> None:
    """Only resolver-owned authority can admit the root capability."""
    denied_resolver = _IdentityResolver(
        scopes=frozenset(),
        allowed_capabilities=frozenset({"echo"}),
    )
    handler, agent, _, _ = _handler(resolver=denied_resolver)

    async def run() -> None:
        scope_denied = await _invoke(
            handler,
            _message(
                GUARDED_SKILL,
                {},
                metadata={"allowedCapabilities": ["guarded"]},
            ),
        )
        assert isinstance(scope_denied, InvocationAuthorizationFailure)
        assert scope_denied.reason_code == "capability_not_allowed"
        assert agent.calls == 0

    asyncio.run(run())


def test_approval_required_and_authenticated_approval_resume() -> None:
    """Approved continuation reuses the canonical approval store transition."""
    resolver = _IdentityResolver()
    handler, agent, _, _ = _handler(resolver=resolver)

    async def run() -> None:
        pending = await _invoke(
            handler,
            _message(APPROVED_SKILL, {"value": "approved"}),
        )
        assert isinstance(pending, InvocationApprovalRequired)
        resolver.decision = ApprovalDecision(
            approval_id=pending.challenge.approval_id,
            approved=True,
            decided_at=datetime.now(UTC),
            decided_by="reviewer",
            role="reviewer",
        )
        resumed = await _invoke(
            handler,
            _message(
                APPROVED_SKILL,
                {"value": "approved"},
                message_id="message-2",
            ),
            request=_request("request-2"),
        )
        assert isinstance(resumed, InvocationSuccess)
        assert resumed.value == "approved"
        assert agent.calls == 1

    asyncio.run(run())


def test_approval_resume_rejects_argument_substitution() -> None:
    """An approval cannot authorize arguments other than those challenged."""
    resolver = _IdentityResolver()
    handler, agent, _, _ = _handler(resolver=resolver)

    async def run() -> None:
        pending = await _invoke(
            handler,
            _message(APPROVED_SKILL, {"value": "approved"}),
        )
        assert isinstance(pending, InvocationApprovalRequired)
        resolver.decision = ApprovalDecision(
            approval_id=pending.challenge.approval_id,
            approved=True,
            decided_at=datetime.now(UTC),
            decided_by="reviewer",
            role="reviewer",
        )
        substituted = await _invoke(
            handler,
            _message(
                APPROVED_SKILL,
                {"value": "substituted"},
                message_id="message-2",
            ),
            request=_request("request-2"),
        )
        assert isinstance(substituted, InvocationAuthorizationFailure)
        assert substituted.reason_code == "approval_binding_mismatch"
        assert agent.calls == 0

    asyncio.run(run())


def test_required_audit_failure_prevents_capability_execution() -> None:
    """Fail-closed required audit delivery maps to its typed result."""
    pipeline = SecurityPipeline(
        audit_emitter=AuditEmitter(FailingAuditSink()),
    )
    runtime = Runtime(security_pipeline=pipeline)
    handler, agent, _, _ = _handler(runtime=runtime)

    async def run() -> None:
        result = await _invoke(handler, _message(GUARDED_SKILL, {}))
        assert isinstance(result, InvocationAuditFailure)
        assert agent.calls == 0

    asyncio.run(run())


def test_required_audit_success_records_remote_principal_and_target() -> None:
    """Successful protected remote work emits canonical security evidence."""
    sink = InMemoryAuditSink()
    pipeline = SecurityPipeline(audit_emitter=AuditEmitter(sink))
    runtime = Runtime(security_pipeline=pipeline)
    handler, agent, _, _ = _handler(runtime=runtime)

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(GUARDED_SKILL, {}),
            request=_request(principal="audited-principal"),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value == "audited-principal"
        assert agent.calls == 1

    asyncio.run(run())
    assert sink.events
    assert {event.subject_id for event in sink.events} == {"audited-principal"}
    assert {event.capability_id for event in sink.events} == {"guarded"}


def test_capability_and_unexpected_adapter_failures_are_sanitized_and_distinct() -> None:
    """Capability exceptions and authentication bugs map to separate safe families."""
    handler, _, _, _ = _handler()
    broken_handler, _, _, _ = _handler(
        resolver=_IdentityResolver(error=RuntimeError("sensitive resolver detail"))
    )

    async def run() -> None:
        capability = await _invoke(handler, _message(FAIL_SKILL, {}))
        internal = await _invoke(broken_handler, _message(ECHO_SKILL, {"value": 1}))
        assert isinstance(capability, InvocationFailure)
        assert capability.message == "Capability execution failed"
        assert isinstance(internal, InvocationInternalFailure)
        assert internal.reason_code == "authentication_failed"

    asyncio.run(run())


def test_expired_deadline_never_executes_the_capability() -> None:
    """An already-expired caller deadline returns timeout before execution."""
    handler, agent, _, _ = _handler(clock=lambda: 100.0)

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(ECHO_SKILL, {"value": 1}, metadata={"deadline": 99.0}),
        )
        assert isinstance(result, InvocationTimeout)
        assert agent.calls == 0

    asyncio.run(run())


def test_non_positive_timeout_never_executes_the_capability() -> None:
    """A non-positive transport timeout is an immediate typed timeout."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(ECHO_SKILL, {"value": 1}, metadata={"timeoutSeconds": 0}),
        )
        assert isinstance(result, InvocationTimeout)
        assert agent.calls == 0

    asyncio.run(run())


def test_runtime_timeout_cancels_in_flight_capability() -> None:
    """A positive runtime timeout remains a canonical timeout result."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(WAIT_SKILL, {}, metadata={"timeoutSeconds": 0.01}),
        )
        assert isinstance(result, InvocationTimeout)
        assert agent.calls == 1

    asyncio.run(run())


def test_explicit_cancellation_reaches_the_runtime_cancellation_state() -> None:
    """Task cancellation cooperatively cancels the canonical invocation."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        invocation = asyncio.create_task(_invoke(handler, _message(WAIT_SKILL, {})))
        await agent.started.wait()
        await handler.cancel("task-1")
        result = await invocation
        assert isinstance(result, InvocationCancelled)
        assert agent.calls == 1

    asyncio.run(run())


def test_cancellation_during_authentication_prevents_capability_execution() -> None:
    """Cancellation remains sticky until authentication can enter the runtime."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class _PausedResolver(_IdentityResolver):
        async def __call__(self, request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
            entered.set()
            await release.wait()
            return await super().__call__(request)

    handler, agent, _, _ = _handler(resolver=_PausedResolver())

    async def run() -> None:
        invocation = asyncio.create_task(_invoke(handler, _message(ECHO_SKILL, {"value": 1})))
        await entered.wait()
        await handler.cancel("task-1")
        release.set()
        result = await invocation
        assert isinstance(result, InvocationCancelled)
        assert agent.calls == 0

    asyncio.run(run())


def test_duplicate_and_racing_identifiers_execute_once() -> None:
    """Concurrent replays share one in-flight result and execute once."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        message = _message(WAIT_SKILL, {})
        first = asyncio.create_task(_invoke(handler, message))
        await agent.started.wait()
        second = asyncio.create_task(_invoke(handler, message))
        await asyncio.sleep(0)
        agent.release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert isinstance(first_result, InvocationSuccess)
        assert second_result == first_result
        assert agent.calls == 1

    asyncio.run(run())


def test_completed_replay_identifiers_remain_idempotent() -> None:
    """A completed identifier is never evicted and allowed to execute again."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        first = await _invoke(handler, _message(ECHO_SKILL, {"value": 0}))
        for index in range(2, 1_003):
            result = await _invoke(
                handler,
                _message(ECHO_SKILL, {"value": index}, message_id=f"message-{index}"),
                task_id=f"task-{index}",
                context_id=f"context-{index}",
                request=_request(f"request-{index}"),
            )
            assert isinstance(result, InvocationSuccess)
        replay = await _invoke(handler, _message(ECHO_SKILL, {"value": 0}))
        assert replay == first
        assert agent.calls == 1_002

    asyncio.run(run())


def test_duplicate_identifiers_are_scoped_to_authenticated_principal() -> None:
    """Different principals cannot collide with or observe another replay record."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        first = await _invoke(
            handler,
            _message(ECHO_SKILL, {"value": 1}),
            request=_request(principal="principal-a"),
        )
        second = await _invoke(
            handler,
            _message(ECHO_SKILL, {"value": 2}),
            task_id="task-2",
            context_id="context-2",
            request=_request(principal="principal-b"),
        )
        assert isinstance(first, InvocationSuccess)
        assert isinstance(second, InvocationSuccess)
        assert first.value == {"value": 1}
        assert second.value == {"value": 2}
        assert agent.calls == 2

    asyncio.run(run())


def test_stale_and_forged_bindings_fail_while_healthy_expired_binding_refreshes() -> None:
    """Only unchanged active registrations may refresh an expired binding."""
    now = [1.0]
    runtime = Runtime(gateway_clock=lambda: now[0], gateway_binding_ttl=2.0)
    stale_handler, stale_agent, _, _ = _handler(runtime=runtime)
    runtime.agent_registry.remove(AGENT_ID)

    expired_now = [1.0]
    expiring_runtime = Runtime(
        gateway_clock=lambda: expired_now[0],
        gateway_binding_ttl=2.0,
    )
    expired_handler, expired_agent, _, _ = _handler(runtime=expiring_runtime)
    expired_now[0] = 4.0

    invalid_handler, invalid_agent, _, _ = _handler()
    invalid_binding = invalid_handler.binding_for_skill(ECHO_SKILL)
    assert invalid_binding is not None
    object.__setattr__(invalid_binding.capability, "signature", "0" * 64)

    async def run() -> None:
        stale = await _invoke(stale_handler, _message(ECHO_SKILL, {"value": 1}))
        expired = await _invoke(expired_handler, _message(ECHO_SKILL, {"value": 1}))
        invalid = await _invoke(invalid_handler, _message(ECHO_SKILL, {"value": 1}))
        assert isinstance(stale, InvocationStaleBinding)
        assert isinstance(expired, InvocationSuccess)
        assert expired.value == {"value": 1}
        assert isinstance(invalid, InvocationBindingFailure)
        assert invalid.reason_code == "invalid_binding"
        assert stale_agent.calls == invalid_agent.calls == 0
        assert expired_agent.calls == 1

    asyncio.run(run())


def test_transport_budget_is_capped_by_handler_owned_limits() -> None:
    """Transport metadata cannot raise server-owned delegation limits."""
    runtime = Runtime()
    runtime.provider_registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="provider", model="model"),
    )
    handler, _, _, _ = _handler(
        runtime=runtime,
        max_delegation_depth=2,
        max_delegation_calls=3,
    )

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(
                CONTEXT_SKILL,
                {},
                metadata={
                    "modelReference": "model",
                    "budget": {"maxDepth": 100, "calls": 100},
                },
            ),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value["budget_calls"] == 3

    asyncio.run(run())


def test_requested_budget_attenuates_broad_authenticated_budget() -> None:
    """Every requested budget dimension can reduce authenticated authority."""
    authenticated = DelegationBudget(
        max_depth=7,
        calls=9,
        tokens=100,
        cost=20,
    )
    handler, _, _, _ = _handler(resolver=_IdentityResolver(delegation_budget=authenticated))

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(
                BUDGET_SKILL,
                {},
                metadata={
                    "budget": {
                        "maxDepth": 3,
                        "calls": 4,
                        "tokens": 25,
                        "cost": 5,
                    }
                },
            ),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value["before"] == {
            "depth": 2,
            "calls": 4,
            "tokens": 25,
            "cost": 5.0,
        }

    asyncio.run(run())


def test_authenticated_budget_attenuates_broader_request() -> None:
    """Transport metadata cannot increase stricter authenticated authority."""
    authenticated = DelegationBudget(
        max_depth=2,
        calls=3,
        tokens=10,
        cost=2,
    )
    handler, _, _, _ = _handler(resolver=_IdentityResolver(delegation_budget=authenticated))

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(
                BUDGET_SKILL,
                {},
                metadata={
                    "budget": {
                        "maxDepth": 8,
                        "calls": 20,
                        "tokens": 50,
                        "cost": 10,
                    }
                },
            ),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value["before"] == {
            "depth": 1,
            "calls": 3,
            "tokens": 10,
            "cost": 2.0,
        }

    asyncio.run(run())


def test_optional_budget_bounds_use_whichever_side_is_bounded() -> None:
    """Unbounded token and cost dimensions retain the other side's bound."""
    request_bounded, _, _, _ = _handler(
        resolver=_IdentityResolver(delegation_budget=DelegationBudget(tokens=None, cost=None))
    )
    identity_bounded, _, _, _ = _handler(
        resolver=_IdentityResolver(delegation_budget=DelegationBudget(tokens=12, cost=3))
    )

    async def run() -> None:
        request_result = await _invoke(
            request_bounded,
            _message(
                BUDGET_SKILL,
                {},
                metadata={
                    "budget": {
                        "maxDepth": 4,
                        "calls": 8,
                        "tokens": 6,
                        "cost": 1,
                    }
                },
            ),
        )
        identity_result = await _invoke(
            identity_bounded,
            _message(
                BUDGET_SKILL,
                {},
                message_id="message-2",
                metadata={"budget": {"maxDepth": 4, "calls": 8}},
            ),
            task_id="task-2",
            request=_request("request-2"),
        )
        assert isinstance(request_result, InvocationSuccess)
        assert isinstance(identity_result, InvocationSuccess)
        assert request_result.value["before"]["tokens"] == 6
        assert request_result.value["before"]["cost"] == 1.0
        assert identity_result.value["before"]["tokens"] == 12
        assert identity_result.value["before"]["cost"] == 3.0

    asyncio.run(run())


def test_effective_budget_uses_remaining_state_without_mutating_authenticated_ledger() -> None:
    """Reserved authority stays consumed while request-local use remains isolated."""
    authenticated = DelegationBudget(
        max_depth=6,
        calls=5,
        tokens=100,
        cost=10,
    )
    assert authenticated.reserve(calls=2, tokens=30, cost=3)
    handler, _, _, _ = _handler(resolver=_IdentityResolver(delegation_budget=authenticated))

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(
                BUDGET_SKILL,
                {
                    "reserve_calls": 1,
                    "reserve_tokens": 10,
                    "reserve_cost": 1,
                },
                metadata={
                    "budget": {
                        "maxDepth": 8,
                        "calls": 9,
                        "tokens": 200,
                        "cost": 20,
                    }
                },
            ),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value == {
            "before": {
                "depth": 5,
                "calls": 3,
                "tokens": 70,
                "cost": 7.0,
            },
            "reserved": True,
            "after": {
                "depth": 5,
                "calls": 2,
                "tokens": 60,
                "cost": 6.0,
            },
        }

    asyncio.run(run())
    remaining = authenticated.snapshot(current_depth=0, remaining_time=None)
    assert (remaining.calls, remaining.tokens, remaining.cost) == (3, 70, 7)


def test_concurrent_requests_receive_independent_budget_ledgers() -> None:
    """Concurrent requests cannot share or restore mutable budget state."""
    authenticated = DelegationBudget(calls=2, tokens=20, cost=2)
    handler, _, _, _ = _handler(resolver=_IdentityResolver(delegation_budget=authenticated))

    async def run() -> None:
        first, second = await asyncio.gather(
            _invoke(
                handler,
                _message(
                    BUDGET_SKILL,
                    {
                        "reserve_calls": 1,
                        "reserve_tokens": 10,
                        "reserve_cost": 1,
                    },
                    message_id="message-a",
                ),
                task_id="task-a",
                request=_request("request-a"),
            ),
            _invoke(
                handler,
                _message(
                    BUDGET_SKILL,
                    {
                        "reserve_calls": 1,
                        "reserve_tokens": 10,
                        "reserve_cost": 1,
                    },
                    message_id="message-b",
                ),
                task_id="task-b",
                request=_request("request-b"),
            ),
        )
        assert isinstance(first, InvocationSuccess)
        assert isinstance(second, InvocationSuccess)
        assert first.value["before"] == second.value["before"]
        assert first.value["after"] == second.value["after"]
        assert first.value["after"] == {
            "depth": 7,
            "calls": 1,
            "tokens": 10,
            "cost": 1.0,
        }

    asyncio.run(run())
    remaining = authenticated.snapshot(current_depth=0, remaining_time=None)
    assert (remaining.calls, remaining.tokens, remaining.cost) == (2, 20, 2)


def test_transport_timeout_and_application_metadata_use_handler_limits() -> None:
    """Runtime context receives bounded timeout and trusted application metadata."""
    runtime = Runtime()
    runtime.provider_registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="provider", model="model"),
    )
    handler, _, _, _ = _handler(runtime=runtime, max_timeout=2.0)

    async def run() -> None:
        result = await _invoke(
            handler,
            _message(
                CONTEXT_SKILL,
                {},
                metadata={
                    "modelReference": "model",
                    "timeoutSeconds": 100,
                    "metadata": {"application": "untrusted"},
                },
            ),
            request=A2ARequestContext(
                request_id="request-1",
                headers={"x-test-principal": "principal"},
                safe_metadata={"application": "trusted"},
            ),
        )
        assert isinstance(result, InvocationSuccess)
        assert result.value["timeout"] == 2.0
        assert result.value["metadata"]["application"] == "trusted"

    asyncio.run(run())


def test_concurrent_principals_models_and_contexts_are_isolated() -> None:
    """Concurrent inbound calls retain distinct identity, model, and lineage."""
    runtime = Runtime()
    for reference in ("model-a", "model-b"):
        runtime.provider_registry.register_client(
            reference,
            FakeModel(ProviderResult(structured={}, accepted=True)),
            ModelConfiguration(provider=f"provider-{reference}", model=reference),
        )
    handler, _, _, _ = _handler(runtime=runtime)

    async def run() -> None:
        first, second = await asyncio.gather(
            _invoke(
                handler,
                _message(
                    CONTEXT_SKILL,
                    {},
                    message_id="message-a",
                    metadata={
                        "correlationId": "correlation-a",
                        "modelReference": "model-a",
                        "metadata": {"lane": "a"},
                        "budget": {"maxDepth": 2, "calls": 3},
                    },
                ),
                task_id="task-a",
                context_id="context-a",
                request=_request("request-a", principal="principal-a"),
            ),
            _invoke(
                handler,
                _message(
                    CONTEXT_SKILL,
                    {},
                    message_id="message-b",
                    metadata={
                        "correlationId": "correlation-b",
                        "modelReference": "model-b",
                        "metadata": {"lane": "b"},
                        "budget": {"maxDepth": 4, "calls": 7},
                    },
                ),
                task_id="task-b",
                context_id="context-b",
                request=_request("request-b", principal="principal-b"),
            ),
        )
        assert isinstance(first, InvocationSuccess)
        assert isinstance(second, InvocationSuccess)
        assert first.value["subject"] == "principal-a"
        assert second.value["subject"] == "principal-b"
        assert first.value["model"] == "model-a"
        assert second.value["model"] == "model-b"
        assert first.value["metadata"]["a2a"]["task_id"] == "task-a"
        assert second.value["metadata"]["a2a"]["task_id"] == "task-b"
        assert first.value["budget_calls"] == 3
        assert second.value["budget_calls"] == 7
        assert first.metadata is not None and first.metadata.provider == "provider-model-a"
        assert second.metadata is not None and second.metadata.provider == "provider-model-b"

    asyncio.run(run())


def test_concurrent_requests_isolate_cancellation_state() -> None:
    """Canceling one concurrent task does not cancel a different task."""
    handler, agent, _, _ = _handler()

    async def run() -> None:
        first = asyncio.create_task(
            _invoke(
                handler,
                _message(WAIT_SKILL, {}, message_id="message-a"),
                task_id="task-a",
                request=_request("request-a"),
            )
        )
        second = asyncio.create_task(
            _invoke(
                handler,
                _message(WAIT_SKILL, {}, message_id="message-b"),
                task_id="task-b",
                request=_request("request-b"),
            )
        )
        while agent.calls < 2:
            await asyncio.sleep(0)
        await handler.cancel("task-a")
        agent.release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert isinstance(first_result, InvocationCancelled)
        assert isinstance(second_result, InvocationSuccess)

    asyncio.run(run())


def test_in_process_asgi_send_uses_runtime_handler_and_persists_result() -> None:
    """Story 4.4 dispatch composes with the canonical runtime handler end to end."""
    handler, agent, resolver, _ = _handler()
    repository = InMemoryTaskRepository()
    app = A2AASGI(
        agent=agent,
        endpoint_url=ENDPOINT_URL,
        task_repository=repository,
        request_handler=handler,
    )

    async def run() -> None:
        request = SendMessageRequest(
            message=_message(
                ECHO_SKILL,
                {"value": 7},
                metadata={"correlationId": "correlation-asgi"},
            )
        )
        envelope = {
            "jsonrpc": "2.0",
            "id": "request-asgi",
            "method": "SendMessage",
            "params": MessageToDict(request),
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://agent.example",
        ) as client:
            response = await client.post(
                "/a2a",
                json=envelope,
                headers={
                    "A2A-Version": "1.0",
                    "x-test-principal": "remote-principal",
                },
            )
        task = response.json()["result"]["task"]
        persisted = await repository.get(task["id"])
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert json.loads(task["artifacts"][0]["parts"][0]["text"]) == {"value": 7}
        assert persisted is not None
        assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
        assert resolver.requests[0].request_id == "request-asgi"
        assert resolver.requests[0].headers["x-test-principal"] == "remote-principal"
        assert agent.calls == 1

    asyncio.run(run())


def test_create_a2a_app_composes_runtime_identity_repository_and_agent_card() -> None:
    """The recommended factory builds a usable canonical-runtime ASGI app."""
    runtime = Runtime()
    agent = _InboundAgent()
    resolver = _IdentityResolver()
    repository = InMemoryTaskRepository()
    app = create_a2a_app(
        agent=agent,
        runtime=runtime,
        public_url="https://agent.example/",
        identity_resolver=resolver,
        task_repository=repository,
    )

    async def run() -> None:
        request = SendMessageRequest(message=_message(ECHO_SKILL, {"value": 9}))
        envelope = {
            "jsonrpc": "2.0",
            "id": "factory-request",
            "method": "SendMessage",
            "params": MessageToDict(request),
        }
        with patch.object(runtime, "invoke", wraps=runtime.invoke) as invoke:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://agent.example",
            ) as client:
                card_response = await client.get("/.well-known/agent-card.json")
                invocation_response = await client.post(
                    "/a2a",
                    json=envelope,
                    headers={
                        "A2A-Version": "1.0",
                        "x-test-principal": "factory-principal",
                    },
                )
            assert invoke.await_count == 1
        card = card_response.json()
        task = invocation_response.json()["result"]["task"]
        persisted = await repository.get(task["id"])
        assert card["supportedInterfaces"][0]["url"] == ENDPOINT_URL
        assert app.endpoint_url == ENDPOINT_URL
        assert persisted is not None
        assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
        assert agent.calls == 1
        assert len(resolver.requests) == 1
        assert resolver.requests[0].headers["x-test-principal"] == "factory-principal"

    asyncio.run(run())


def test_create_a2a_app_minimal_factory_builds_asgi_application() -> None:
    """The required factory arguments produce a callable ASGI application."""
    app = create_a2a_app(
        agent=_InboundAgent(),
        runtime=Runtime(),
        public_url="https://agent.example",
        identity_resolver=_IdentityResolver(),
    )

    assert callable(app)
    assert app.endpoint_url == ENDPOINT_URL


@pytest.mark.parametrize(
    "public_url",
    [
        "",
        "agent.example",
        "ftp://agent.example",
        "https://user@agent.example",
        "https://agent.example/base",
        "https://agent.example?query=value",
        "https://agent.example#fragment",
        "https://agent.example:invalid",
        " https://agent.example",
    ],
)
def test_create_a2a_app_rejects_invalid_or_ambiguous_public_urls(
    public_url: str,
) -> None:
    """The recommended factory accepts only an unambiguous HTTP(S) origin."""
    with pytest.raises(ValueError, match="public_url"):
        create_a2a_app(
            agent=_InboundAgent(),
            runtime=Runtime(),
            public_url=public_url,
            identity_resolver=_IdentityResolver(),
        )


def test_asgi_cancel_and_caller_cancellation_leave_tasks_terminal() -> None:
    """Explicit protocol and caller cancellation cannot strand accepted tasks."""

    async def exercise_explicit() -> None:
        handler, agent, _, _ = _handler()
        repository = InMemoryTaskRepository()
        task = Task(id="task-explicit", context_id="context-explicit")
        task.status.state = TaskState.TASK_STATE_SUBMITTED
        await repository.create(task)
        app = A2AASGI(
            agent=agent,
            endpoint_url=ENDPOINT_URL,
            task_repository=repository,
            request_handler=handler,
        )
        send = SendMessageRequest(
            message=_message(
                WAIT_SKILL,
                {},
                task_id=task.id,
                context_id=task.context_id,
            )
        )
        cancel = CancelTaskRequest(id=task.id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://agent.example",
        ) as client:
            sending = asyncio.create_task(
                client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "request-send",
                        "method": "SendMessage",
                        "params": MessageToDict(send),
                    },
                    headers={"A2A-Version": "1.0"},
                )
            )
            await agent.started.wait()
            canceled = await client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "request-cancel",
                    "method": "CancelTask",
                    "params": MessageToDict(cancel),
                },
                headers={"A2A-Version": "1.0"},
            )
            completed = await sending
        assert canceled.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        assert completed.json()["result"]["task"]["status"]["state"] == "TASK_STATE_CANCELED"

    async def exercise_caller() -> None:
        handler, agent, _, _ = _handler()
        repository = InMemoryTaskRepository()
        task = Task(id="task-caller", context_id="context-caller")
        task.status.state = TaskState.TASK_STATE_SUBMITTED
        await repository.create(task)
        app = A2AASGI(
            agent=agent,
            endpoint_url=ENDPOINT_URL,
            task_repository=repository,
            request_handler=handler,
        )
        send = SendMessageRequest(
            message=_message(
                WAIT_SKILL,
                {},
                task_id=task.id,
                context_id=task.context_id,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://agent.example",
        ) as client:
            sending = asyncio.create_task(
                client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "request-caller",
                        "method": "SendMessage",
                        "params": MessageToDict(send),
                    },
                    headers={"A2A-Version": "1.0"},
                )
            )
            await agent.started.wait()
            sending.cancel()
            try:
                await sending
            except asyncio.CancelledError:
                pass
        persisted = await repository.get(task.id)
        assert persisted is not None
        assert persisted.status.state == TaskState.TASK_STATE_CANCELED

    asyncio.run(exercise_explicit())
    asyncio.run(exercise_caller())
