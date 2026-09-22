"""Immutable provider arguments retain their JSON semantics through delegation."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.delegation import DelegationConfig, run_delegation
from conducto.core.delegation._arguments import _arguments_match_schema
from conducto.core.gateway_tools import CapabilityUse, ToolboxPolicy, build_toolbox
from conducto.core.provider import (
    ChatMessage,
    ModelConfiguration,
    ProviderResult,
    ProviderToolCallRequest,
)
from conducto.core.provider_registry import ProviderRegistry
from conducto.core.run_context import use_run_context
from conducto.testing import FakeModel


class _Numbers(BaseModel):
    values: list[int]


class _Total(BaseModel):
    total: int


@pytest.mark.parametrize(
    ("value", "schema", "valid"),
    [
        (1, {"type": "integer"}, True),
        (1.5, {"type": "number"}, True),
        (True, {"type": "boolean"}, True),
        (True, {"type": "integer"}, False),
        (False, {"type": "number"}, False),
        (1.5, {"type": "integer"}, False),
        (1, {"type": "boolean"}, False),
        (float("nan"), {"type": "number"}, False),
        (float("inf"), {"type": "number"}, False),
        (2, {"type": "integer", "minimum": 2, "maximum": 2}, True),
        (1, {"type": "integer", "minimum": 2}, False),
        (3, {"type": "integer", "maximum": 2}, False),
        (2, {"type": "number", "exclusiveMinimum": 2}, False),
        (2, {"type": "number", "exclusiveMaximum": 2}, False),
        (0.3, {"type": "number", "multipleOf": 0.1}, True),
        (0.31, {"type": "number", "multipleOf": 0.1}, False),
        ([1, 2], {"type": "array", "items": {"type": "integer"}, "const": [1, 2]}, True),
        ([1, 2], {"type": "array", "items": {"type": "integer"}, "maxItems": 1}, False),
    ],
)
def test_frozen_scalar_and_array_arguments_preserve_schema_constraints(
    value: Any, schema: dict[str, Any], valid: bool
) -> None:
    call = ProviderToolCallRequest(tool_id="tool", arguments={"value": value})
    document = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert _arguments_match_schema(call.arguments, document) is valid


def test_nested_arrays_reach_typed_capabilities_without_mutating_provider_arguments() -> None:
    received: list[_Numbers] = []

    @a2a_agent(name="NestedWorker", version="1.0.0", description="Sums nested values.")
    class Worker(BaseAgent):
        @a2a_capability(name="sum", description="Sums a nested array.")
        def total(self, numbers: _Numbers) -> int:
            received.append(numbers)
            return sum(numbers.values)

    async def exercise() -> None:
        agents = AgentRegistry()
        agents.register(Worker())
        providers = ProviderRegistry()
        runtime = Runtime(agent_registry=agents, provider_registry=providers)
        policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"sum"})),))
        probe = runtime.create_run_context(agent_id="Parent")
        with use_run_context(probe):
            toolbox = await build_toolbox(probe.gateway, policy)
        assert toolbox.snapshot is not None
        arguments = {"numbers": {"values": [1, 2, 3]}}
        request = ProviderToolCallRequest(
            call_id="nested-call",
            tool_id=toolbox.snapshot.tools[0].tool_id,
            arguments=arguments,
        )
        arguments["numbers"]["values"].append(100)
        model = FakeModel(
            script=(
                ProviderResult(tool_calls=(request,), accepted=True),
                ProviderResult(structured={"total": 6}, accepted=True),
            )
        )
        providers.register_client(
            "model", model, ModelConfiguration(provider="fake", model="model")
        )
        context = runtime.create_run_context(agent_id="Parent", call_override="model")
        with use_run_context(context):
            outcome = await run_delegation(
                context,
                (ChatMessage(role="user", content="Sum the numbers."),),
                config=DelegationConfig(toolbox=policy),
                response_type=_Total,
            )
        assert outcome.ok, outcome.code
        assert outcome.value == _Total(total=6)
        assert received == [_Numbers(values=[1, 2, 3])]
        assert len(model.requests[1].tool_results) == 1
        assert request.model_dump()["arguments"] == {"numbers": {"values": [1, 2, 3]}}

    asyncio.run(exercise())
