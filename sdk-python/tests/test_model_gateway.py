"""Focused contracts for model gateway delegation and provenance."""

import asyncio

import pytest
from pydantic import BaseModel

from conducto import (
    ChatMessage,
    FakeModel,
    ModelConfiguration,
    ProviderCapabilities,
    ProviderRegistry,
    Runtime,
    Usage,
)
from conducto.core.model_gateway import ModelCallResult as GatewayModelCallResult
from conducto.core.provider import (
    GenerationOptions,
    MalformedStructuredOutputError,
    ProviderResult,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    build_model_decision_schema,
    complete_with_retries,
    parse_model_decision,
)
from conducto.core.runtime import ModelCallResult, use_run_context
from conducto.testing import ScriptedProvider, assert_provider_tool_call_conformance


class Response(BaseModel):
    value: str


def _tool_definition(
    *,
    tool_id: str = "tool-1",
    name: str = "lookup_tool_1",
    description: str = "Looks up deterministic fixture data.",
) -> ProviderToolDefinition:
    return ProviderToolDefinition(
        tool_id=tool_id,
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def test_model_decision_contract_separates_terminal_schema_and_native_tool_calls() -> None:
    schema = build_model_decision_schema(Response).json_schema
    assert schema == Response.model_json_schema()
    assert "oneOf" not in schema

    terminal = parse_model_decision(
        ProviderResult(structured={"value": "ok"}),
        response_type=Response,
    )
    assert terminal.type == "terminal"
    assert terminal.response == {"value": "ok"}

    tools = (_tool_definition(),)
    tool_call = parse_model_decision(
        ProviderResult(
            request_id="provider-request",
            tool_calls=(
                ProviderToolCallRequest(tool_name="lookup_tool_1", arguments={"query": "value"}),
            ),
        ),
        response_type=Response,
        tools=tools,
    )
    assert tool_call.type == "tool_call"
    assert tool_call.tool_id == "tool-1"
    assert tool_call.arguments == {"query": "value"}
    assert tool_call.call_id.startswith("tool_")
    assert (
        parse_model_decision(
            ProviderResult(
                request_id="provider-request",
                tool_calls=(
                    ProviderToolCallRequest(
                        tool_name="lookup_tool_1", arguments={"query": "value"}
                    ),
                ),
            ),
            response_type=Response,
            tools=tools,
        ).call_id
        == tool_call.call_id
    )

    with pytest.raises(MalformedStructuredOutputError, match="both terminal structured output"):
        parse_model_decision(
            ProviderResult(
                structured={"value": "ok"},
                tool_calls=(
                    ProviderToolCallRequest(tool_id="tool-1", arguments={"query": "value"}),
                ),
            ),
            response_type=Response,
            tools=tools,
        )

    with pytest.raises(MalformedStructuredOutputError, match="no structured model decision"):
        parse_model_decision(ProviderResult(), response_type=Response)

    with pytest.raises(MalformedStructuredOutputError, match="multiple tool calls"):
        parse_model_decision(
            ProviderResult(
                tool_calls=(
                    ProviderToolCallRequest(tool_id="tool-1", arguments={"query": "value"}),
                    ProviderToolCallRequest(tool_id="tool-1", arguments={"query": "other"}),
                )
            ),
            response_type=Response,
            tools=tools,
        )

    unresolved = parse_model_decision(
        ProviderResult(
            tool_calls=(ProviderToolCallRequest(tool_id="forged-tool", arguments={"query": "x"}),)
        ),
        response_type=Response,
        tools=tools,
    )
    assert unresolved.type == "tool_call"
    assert unresolved.tool_id == "forged-tool"

    ambiguous_tools = (
        _tool_definition(tool_id="tool-1", name="shared_tool"),
        _tool_definition(tool_id="tool-2", name="shared_tool"),
    )
    with pytest.raises(MalformedStructuredOutputError, match="ambiguous tool call"):
        parse_model_decision(
            ProviderResult(
                tool_calls=(
                    ProviderToolCallRequest(tool_name="shared_tool", arguments={"query": "x"}),
                )
            ),
            response_type=Response,
            tools=ambiguous_tools,
        )


def test_fake_model_records_scripted_tool_aware_requests() -> None:
    async def exercise() -> None:
        model = FakeModel(
            script=(
                {
                    "type": "tool_call",
                    "call_id": "call-1",
                    "tool_id": "tool-1",
                    "arguments": {"query": "value"},
                },
                {"type": "terminal", "response": {"value": "done"}},
            )
        )
        request = build_model_decision_schema(Response)
        tools = (_tool_definition(),)
        first = await model.complete(
            (ChatMessage(role="user", content="start"),),
            options=GenerationOptions(model="fake"),
            structured_output=request,
            tools=tools,
        )
        second = await model.complete(
            (ChatMessage(role="tool", content="result"),),
            options=GenerationOptions(model="fake"),
            structured_output=request,
            tools=tools,
            tool_results=(
                ToolResultMessage(
                    call_id="call-1",
                    status="success",
                    result={"call_id": "call-1", "status": "success"},
                ),
            ),
        )

        assert first.structured is None
        assert first.tool_calls[0].tool_id == "tool-1"
        assert second.structured == {"value": "done"}
        assert not second.tool_calls
        assert model.calls == 2
        assert model.requests[0].structured_output.json_schema == Response.model_json_schema()
        assert model.requests[0].tools[0].tool_id == "tool-1"
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


def test_provider_contract_requires_only_terminal_schema_features_for_tool_turns() -> None:
    async def exercise() -> None:
        supported_features = frozenset(
            feature
            for feature in ProviderCapabilities(structured_output=True).schema_features
            if feature.value not in {"oneOf", "anyOf", "allOf"}
        )
        provider = FakeModel(
            {"type": "tool_call", "call_id": "call-1", "tool_id": "tool-1", "arguments": {}}
        )
        provider.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=True,
            schema_features=supported_features,
        )
        result = await complete_with_retries(
            provider,
            (ChatMessage(role="user", content="tool please"),),
            options=GenerationOptions(model="fake"),
            structured_output=build_model_decision_schema(Response),
            tools=(_tool_definition(),),
        )

        assert result.tool_calls[0].tool_id == "tool-1"
        assert provider.calls == 1

        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported features"):
            await complete_with_retries(
                provider,
                (ChatMessage(role="user", content="unsupported"),),
                options=GenerationOptions(model="fake"),
                structured_output=StructuredOutputRequest(
                    name="Unsupported",
                    schema={
                        "oneOf": [
                            {"type": "object", "properties": {"value": {"type": "string"}}},
                            {"type": "object", "properties": {"other": {"type": "string"}}},
                        ]
                    },
                ),
                tools=(_tool_definition(),),
            )

        assert provider.calls == 1

    asyncio.run(exercise())


def test_provider_conformance_helper_exercises_separate_tool_contract() -> None:
    async def exercise() -> None:
        provider = ScriptedProvider(
            (
                ProviderResult(
                    request_id="fixture-tool",
                    tool_calls=(
                        ProviderToolCallRequest(
                            tool_name="lookup_tool_1",
                            arguments={"query": "value"},
                        ),
                    ),
                ),
            )
        )

        await assert_provider_tool_call_conformance(provider)

        assert provider.calls[0].structured_output.json_schema["required"] == ["answer"]
        assert provider.calls[0].tools[0].name == "lookup_tool_1"
        assert provider.calls[0].tool_results == ()

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
