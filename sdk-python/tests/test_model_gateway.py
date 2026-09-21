"""Focused contracts for model gateway delegation and provenance."""

import asyncio

import pytest
from pydantic import BaseModel

from conducto import (
    ChatMessage,
    FakeModel,
    ModelConfiguration,
    ProviderRegistry,
    Runtime,
    Usage,
)
from conducto.core.model_gateway import ModelCallResult as GatewayModelCallResult
from conducto.core.provider import (
    GenerationOptions,
    MalformedStructuredOutputError,
    ProviderResult,
    StructuredOutputRequest,
    ToolResultMessage,
    build_model_decision_schema,
    complete_with_retries,
    parse_model_decision,
)
from conducto.core.runtime import ModelCallResult, use_run_context


class Response(BaseModel):
    value: str


def test_model_decision_contract_allows_exactly_one_terminal_or_tool_call() -> None:
    schema = build_model_decision_schema(Response).json_schema
    assert len(schema["oneOf"]) == 2

    terminal = parse_model_decision(
        ProviderResult(structured={"type": "terminal", "response": {"value": "ok"}}),
        response_type=Response,
    )
    assert terminal.type == "terminal"

    tool_call = parse_model_decision(
        ProviderResult(
            structured={
                "type": "tool_call",
                "call_id": "call-1",
                "tool_id": "tool-1",
                "arguments": {"query": "value"},
            }
        ),
        response_type=Response,
    )
    assert tool_call.type == "tool_call"

    with pytest.raises(MalformedStructuredOutputError):
        parse_model_decision(
            ProviderResult(
                structured={
                    "type": "tool_call",
                    "call_id": "call-1",
                    "tool_id": "tool-1",
                }
            ),
            response_type=Response,
        )


def test_fake_model_records_scripted_tool_aware_requests() -> None:
    async def exercise() -> None:
        model = FakeModel(
            script=(
                {
                    "type": "tool_call",
                    "call_id": "call-1",
                    "tool_id": "tool-1",
                    "arguments": {},
                },
                {"type": "terminal", "response": {"value": "done"}},
            )
        )
        request = build_model_decision_schema(Response)
        first = await model.complete(
            (ChatMessage(role="user", content="start"),),
            options=GenerationOptions(model="fake"),
            structured_output=request,
            tools=({"id": "tool-1"},),
        )
        second = await model.complete(
            (ChatMessage(role="tool", content="result"),),
            options=GenerationOptions(model="fake"),
            structured_output=request,
            tools=({"id": "tool-1"},),
            tool_results=(
                ToolResultMessage(
                    call_id="call-1",
                    status="success",
                    result={"call_id": "call-1", "status": "success"},
                ),
            ),
        )

        assert first.structured and first.structured["type"] == "tool_call"
        assert second.structured and second.structured["type"] == "terminal"
        assert model.calls == 2
        assert model.requests[1].tool_results[0].call_id == "call-1"

    asyncio.run(exercise())


def test_effective_deadline_is_forwarded_without_tools() -> None:
    async def exercise() -> None:
        model = FakeModel({"value": "ok"})
        result = await complete_with_retries(
            model,
            (ChatMessage(role="user", content="respond"),),
            options=GenerationOptions(model="fake"),
            structured_output=StructuredOutputRequest(
                name="Response",
                schema=Response.model_json_schema(),
            ),
            effective_deadline=100.0,
            clock=lambda: 1.0,
        )

        assert result.structured == {"value": "ok"}
        assert model.requests[0].effective_deadline == 100.0
        assert model.requests[0].tools == ()
        assert model.requests[0].tool_results == ()

    asyncio.run(exercise())


def test_model_gateway_records_typed_completion_provenance() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        registry.register_client(
            "model",
            FakeModel(
                {"value": "ok"},
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
            ModelConfiguration(provider="fake", model="model"),
        )
        runtime = Runtime(provider_registry=registry)
        context = runtime.create_run_context(
            agent_id="Agent",
            call_override="model",
        )
        with use_run_context(context):
            result = await context.models.require().complete_typed(
                (ChatMessage(role="user", content="respond"),),
                response_type=Response,
            )

        assert result == Response(value="ok")
        metadata = context.invocation_metadata()
        assert [call.purpose for call in metadata.model_calls] == ["capability"]
        assert metadata.usage.total_tokens == 3

    assert ModelCallResult is GatewayModelCallResult
    asyncio.run(exercise())
