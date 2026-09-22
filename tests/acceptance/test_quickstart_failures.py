"""End-to-end failure scenarios for the documented local-agent flow."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest

from conducto import (
    BaseAgent,
    OrchestratorAgent,
    RunConfig,
    Runtime,
    a2a_agent,
    a2a_capability,
    require_run_context,
)
from conducto.core.invocation_results import (
    InvocationCancelled,
    InvocationFailure,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTimeout,
    InvocationValidationFailure,
    RoutingFailure,
)
from conducto.core.logging import configure_logging
from conducto.core.provider import ModelConfiguration, ProviderResult, Usage
from conducto.core.provider_registry import ProviderRegistry
from conducto.testing import FakeModel

pytestmark = pytest.mark.acceptance


@a2a_agent(
    name="FailureAgent",
    version="1.0.0",
    description="Exercises quick-start failure envelopes.",
)
class FailureAgent(BaseAgent):
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        super().__init__()

    @a2a_capability(name="add", description="Adds one to a value.")
    def add(self, value: int) -> int:
        self.calls += 1
        return value + 1

    @a2a_capability(name="explode", description="Raises a local diagnostic error.")
    def explode(self) -> str:
        self.calls += 1
        raise ValueError("private diagnostic detail")

    @a2a_capability(name="slow", description="Waits for an explicit release.")
    async def slow(self) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return "finished"

    @a2a_capability(name="cancel", description="Cooperatively cancels the invocation.")
    async def cancel(self) -> str:
        self.calls += 1
        require_run_context().cancellation.cancel()
        raise asyncio.CancelledError


def _orchestrator(selection: dict[str, Any] | str) -> tuple[OrchestratorAgent, FailureAgent]:
    registry = ProviderRegistry()
    registry.register_client(
        "failure-router",
        FakeModel(
            ProviderResult(
                structured=selection if isinstance(selection, dict) else None,
                content=selection if isinstance(selection, str) else "",
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
        ),
        ModelConfiguration(provider="fake", model="failure-router"),
    )
    orchestrator = OrchestratorAgent(
        model_reference="failure-router",
        runtime=Runtime(provider_registry=registry),
    )
    agent = FailureAgent()
    orchestrator.register_agent(agent)
    return orchestrator, agent


async def _route_with_logs(
    selection: dict[str, Any] | str,
    *,
    correlation_id: str,
    timeout: float | None = None,
    agent_timeout: float | None = None,
) -> tuple[Any, FailureAgent, list[dict[str, Any]]]:
    stream = io.StringIO()
    configure_logging(format="json", stream=stream)
    orchestrator, agent = _orchestrator(selection)
    try:
        result = await orchestrator.route(
            "Exercise failure path.",
            correlation_id=correlation_id,
            timeout=timeout,
            agent_run_config=RunConfig(timeout=agent_timeout)
            if agent_timeout is not None
            else None,
        )
    finally:
        agent.release.set()
    events = [
        event
        for line in stream.getvalue().splitlines()
        if "correlation_id" in (event := json.loads(line))
    ]
    return result, agent, events


def test_malformed_model_output_returns_typed_routing_failure_without_invocation() -> None:
    async def exercise() -> None:
        result, agent, logs = await _route_with_logs(
            "not structured",
            correlation_id="failure-malformed",
        )

        assert isinstance(result, RoutingFailure)
        assert result.metadata is not None
        assert result.metadata.correlation_id == "failure-malformed"
        assert result.usage == Usage(input_tokens=2, output_tokens=1, total_tokens=3)
        assert agent.calls == 0
        assert {event["correlation_id"] for event in logs} == {"failure-malformed"}

    asyncio.run(exercise())


def test_unknown_agent_or_capability_returns_target_failure_without_invocation() -> None:
    async def exercise() -> None:
        for selection in (
            {"agent_id": "MissingAgent", "capability_id": "add", "arguments": {"value": 1}},
            {"agent_id": "FailureAgent", "capability_id": "missing", "arguments": {}},
        ):
            result, agent, _logs = await _route_with_logs(
                selection,
                correlation_id=f"failure-target-{selection['agent_id']}",
            )
            assert isinstance(result, InvocationTargetNotFound)
            assert result.metadata is not None
            assert result.metadata.model_calls[0].purpose == "routing"
            assert agent.calls == 0

    asyncio.run(exercise())


def test_invalid_generated_arguments_return_validation_failure_without_invocation() -> None:
    async def exercise() -> None:
        result, agent, logs = await _route_with_logs(
            {
                "agent_id": "FailureAgent",
                "capability_id": "add",
                "arguments": {"value": "not an integer"},
            },
            correlation_id="failure-invalid-args",
        )

        assert isinstance(result, InvocationValidationFailure)
        assert result.correlation_id == "failure-invalid-args"
        assert result.metadata is not None
        assert [call.purpose for call in result.metadata.model_calls] == ["routing"]
        assert result.errors[0]["loc"] == ("value",)
        assert agent.calls == 0
        assert any(
            event["event"] == "conducto.capability.arguments_validated.v1"
            and event["outcome"] == "failure"
            for event in logs
        )

    asyncio.run(exercise())


def test_capability_exception_returns_safe_failure_with_local_diagnostic_exception() -> None:
    async def exercise() -> None:
        result, agent, logs = await _route_with_logs(
            {"agent_id": "FailureAgent", "capability_id": "explode", "arguments": {}},
            correlation_id="failure-exception",
        )

        assert isinstance(result, InvocationFailure)
        assert result.correlation_id == "failure-exception"
        assert result.message == "Capability execution failed"
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "private diagnostic detail"
        assert result.metadata is not None
        assert result.metadata.correlation_id == "failure-exception"
        assert agent.calls == 1
        for event in logs:
            rendered = json.dumps(event, sort_keys=True)
            assert event["correlation_id"] == "failure-exception"
            assert "private diagnostic detail" not in rendered
            assert "traceback" not in rendered

    asyncio.run(exercise())


def test_timeout_preserves_correlation_id_and_returns_timeout_envelope() -> None:
    async def exercise() -> None:
        result, agent, logs = await _route_with_logs(
            {
                "agent_id": "FailureAgent",
                "capability_id": "slow",
                "arguments": {},
            },
            correlation_id="failure-timeout",
            # Keep this focused on the selected capability deadline. The route-wide
            # timeout also covers structured routing and A2A profile validation.
            agent_timeout=0.001,
        )

        assert isinstance(result, InvocationTimeout)
        assert result.correlation_id == "failure-timeout"
        assert result.metadata is not None
        assert result.metadata.correlation_id == "failure-timeout"
        assert agent.calls == 1
        assert any(
            event["event"] == "conducto.capability.invocation_timed_out.v1"
            and event["correlation_id"] == "failure-timeout"
            for event in logs
        )

    asyncio.run(exercise())


def test_cooperative_cancellation_preserves_correlation_id() -> None:
    async def exercise() -> None:
        result, agent, logs = await _route_with_logs(
            {"agent_id": "FailureAgent", "capability_id": "cancel", "arguments": {}},
            correlation_id="failure-cancelled",
        )

        assert isinstance(result, InvocationCancelled)
        assert result.correlation_id == "failure-cancelled"
        assert result.metadata is not None
        assert result.metadata.correlation_id == "failure-cancelled"
        assert agent.calls == 1
        assert any(
            event["event"] == "conducto.capability.invocation_cancelled.v1"
            and event["correlation_id"] == "failure-cancelled"
            for event in logs
        )

    asyncio.run(exercise())


def test_external_caller_task_cancellation_is_propagated() -> None:
    async def exercise() -> None:
        orchestrator, agent = _orchestrator(
            {
                "agent_id": "FailureAgent",
                "capability_id": "slow",
                "arguments": {},
            }
        )
        task = asyncio.create_task(
            orchestrator.route("Cancel externally.", correlation_id="failure-external-cancel")
        )
        try:
            await asyncio.wait_for(agent.started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert agent.calls == 1
        finally:
            agent.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_success_control_proves_failure_agent_can_execute() -> None:
    async def exercise() -> None:
        result, agent, _logs = await _route_with_logs(
            {"agent_id": "FailureAgent", "capability_id": "add", "arguments": {"value": 2}},
            correlation_id="failure-control-success",
        )

        assert isinstance(result, InvocationSuccess)
        assert result.value == 3
        assert agent.calls == 1

    asyncio.run(exercise())
