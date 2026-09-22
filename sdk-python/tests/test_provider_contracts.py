"""Focused regression tests for provider contract boundaries and execution."""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import conducto.core.provider as contracts
from conducto.core.provider import (
    AcceptanceState,
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ProviderCallContext,
    ProviderCancellationError,
    ProviderCapabilities,
    ProviderError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderToolCall,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    SchemaFeature,
    StructuredOutputRequest,
    ToolCallModelDecision,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    Usage,
    build_terminal_output_request,
    complete_with_retries,
    parse_model_decision,
    validate_provider_contract,
)
from conducto.testing import FakeModel


class _Answer(BaseModel):
    answer: str


def _request() -> StructuredOutputRequest:
    return StructuredOutputRequest("answer", _Answer.model_json_schema())


def _tool() -> ProviderToolDefinition:
    return ProviderToolDefinition("opaque-1", "lookup", "Look up data.", {"type": "object"})


def _complete(provider: FakeModel, **kwargs: Any) -> ProviderResult:
    return asyncio.run(
        complete_with_retries(
            provider,
            (ChatMessage(role="user", content="private message"),),
            options=GenerationOptions(model="test", retries=2),
            structured_output=_request(),
            **kwargs,
        )
    )


def test_aggregation_exposes_contract_owners_without_testing_helpers() -> None:
    assert contracts.ChatMessage.__module__.endswith(".provider.messages")
    assert contracts.ProviderResult.__module__.endswith(".provider.results")
    assert contracts.ProviderError.__module__.endswith(".provider.errors")
    assert contracts.ModelProvider.__module__.endswith(".provider.protocol")
    assert not hasattr(contracts, "FakeModel")
    assert not hasattr(contracts, "FakeModelRequest")
    assert FakeModel.__module__ == "conducto.testing.fake_model"


def test_terminal_request_builder_describes_only_terminal_output() -> None:
    request = build_terminal_output_request(_Answer)
    assert request.name == "_Answer"
    assert request.json_schema == _Answer.model_json_schema()
    assert request.required is False
    assert not hasattr(contracts, "build_model_decision_schema")


def test_schema_snapshot_is_deeply_immutable_and_returns_independent_wire_copies() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"enum": ["yes", "no"]}},
    }
    request = StructuredOutputRequest(" answer ", schema)
    schema["properties"]["answer"]["enum"].append("modified")
    assert request.name == "answer"
    assert request.json_schema["properties"]["answer"]["enum"] == ["yes", "no"]
    with pytest.raises(TypeError):
        request.schema["properties"]["answer"]["enum"][0] = "modified"
    copy = request.json_schema
    copy["properties"]["answer"]["enum"].append("copy")
    assert request.json_schema["properties"]["answer"]["enum"] == ["yes", "no"]


def test_structured_request_rejects_removed_constructor_alias() -> None:
    arguments: dict[str, Any] = {"name": "answer", "json_schema": {"type": "object"}}
    with pytest.raises(TypeError, match="json_schema"):
        StructuredOutputRequest(**arguments)


@pytest.mark.parametrize("entry", ["text", _Answer(answer="ok")])
def test_fake_rejects_implicit_text_and_model_conversion(entry: Any) -> None:
    with pytest.raises(TypeError, match="ProviderResult"):
        FakeModel(entry)


@pytest.mark.parametrize(
    "entry", [{}, {"type": "terminal", "response": {}}, "text", _Answer(answer="ok")]
)
def test_fake_script_requires_explicit_typed_results(entry: Any) -> None:
    with pytest.raises(TypeError, match="ProviderResult"):
        FakeModel(script=[entry])


def test_fake_accepts_plain_mapping_as_terminal_content() -> None:
    result = _complete(FakeModel({"answer": "ok"}))
    assert result.structured == {"answer": "ok"}
    assert result.accepted is True
    assert not result.tool_calls


@pytest.mark.parametrize("key", ["kind", "type"])
@pytest.mark.parametrize("value", ["terminal", "tool_call"])
def test_fake_never_interprets_synthetic_decision_envelopes(key: str, value: str) -> None:
    payload = {key: value, "response": {"answer": "ok"}, "tool_id": "opaque-1"}
    provider = FakeModel(payload)
    result = asyncio.run(
        provider.complete(
            (),
            options=GenerationOptions(model="test"),
            structured_output=_request(),
        )
    )
    assert result.structured == payload
    assert not result.tool_calls


def test_fake_script_is_ordered_and_exhaustion_is_explicit() -> None:
    result = ProviderResult(structured={"answer": "ok"}, accepted=True)
    provider = FakeModel(script=[ProviderRateLimitError(), result])
    assert _complete(provider) is result
    assert provider.calls == 2
    assert provider.requests[0].message_roles == ("user",)
    assert not hasattr(provider.requests[0], "messages")
    with pytest.raises(ProviderError, match="exhausted"):
        _complete(provider)


def test_accepted_failures_are_never_retried_and_diagnostics_are_redacted() -> None:
    failure = ProviderRateLimitError("credential=private", accepted=True, request_id="safe-id")
    provider = FakeModel(failure)
    with pytest.raises(ProviderRateLimitError) as caught:
        _complete(provider)
    assert provider.calls == 1
    assert caught.value.acceptance is AcceptanceState.ACCEPTED
    assert caught.value.retryable is False
    assert caught.value.diagnostic.to_dict() == {
        "category": "rate_limit",
        "message": "provider failure: rate_limit",
        "request_id": "safe-id",
    }


def test_malformed_output_retains_acceptance_and_usage_without_retry() -> None:
    usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)
    provider = FakeModel(
        ProviderResult(structured={"answer": 1}, accepted=True, usage=usage, request_id="req")
    )
    with pytest.raises(MalformedStructuredOutputError) as caught:
        _complete(provider)
    assert caught.value.acceptance is AcceptanceState.ACCEPTED
    assert caught.value.usage is usage
    assert caught.value.request_id == "req"
    assert provider.calls == 1


def test_tool_definitions_reject_legacy_mappings_and_keep_wire_shape() -> None:
    legacy: Any = [{"id": "opaque-1", "name": "lookup"}]
    with pytest.raises(TypeError, match="ProviderToolDefinition"):
        validate_provider_contract(
            FakeModel(ProviderResult()), structured_output=_request(), tools=legacy
        )
    assert not hasattr(ProviderToolDefinition, "from_mapping")
    assert _tool().to_dict() == {
        "id": "opaque-1",
        "name": "lookup",
        "description": "Look up data.",
        "input_schema": {"type": "object"},
    }


@pytest.mark.parametrize(
    "contract", [ProviderToolCallRequest, ProviderToolCall, ToolCallModelDecision]
)
def test_tool_arguments_are_immutable_snapshots_with_unchanged_json(contract: Any) -> None:
    arguments = {"nested": [{"value": "before"}]}
    kwargs: dict[str, Any] = {"call_id": "call", "tool_id": "opaque-1", "arguments": arguments}
    if contract is ToolCallModelDecision:
        kwargs["type"] = "tool_call"
    call = contract(**kwargs)
    arguments["nested"][0]["value"] = "after"
    assert call.arguments["nested"][0]["value"] == "before"
    with pytest.raises(TypeError):
        call.arguments["new"] = "value"
    with pytest.raises(TypeError):
        call.arguments["nested"][0]["value"] = "changed"
    assert json.loads(call.model_dump_json())["arguments"] == {"nested": [{"value": "before"}]}


def test_synthesized_call_ids_are_stable_for_nested_frozen_arguments() -> None:
    def decide(arguments: dict[str, Any]) -> ToolCallModelDecision:
        result = ProviderResult(
            tool_calls=(ProviderToolCallRequest(tool_name="lookup", arguments=arguments),),
            request_id="request",
        )
        decision = parse_model_decision(result, response_type=_Answer, tools=(_tool(),))
        assert isinstance(decision, ToolCallModelDecision)
        return decision

    first = decide({"b": [{"nested": 1}], "a": 2})
    second = decide({"a": 2, "b": [{"nested": 1}]})
    assert first.call_id == second.call_id
    assert first.tool_id == "opaque-1"


def test_default_tool_call_arguments_are_immutable() -> None:
    request = ProviderToolCallRequest(tool_id="opaque-1")
    arguments: Any = request.arguments
    with pytest.raises(TypeError):
        arguments["value"] = "modified"
    assert request.model_dump()["arguments"] == {}


def test_duplicate_opaque_tool_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_provider_contract(
            FakeModel(ProviderResult()), structured_output=_request(), tools=(_tool(), _tool())
        )


def test_nested_schema_features_cannot_be_hidden_by_explicit_features() -> None:
    request = StructuredOutputRequest(
        "nested",
        {"type": "object", "properties": {"items": {"type": "array", "items": {"enum": [1]}}}},
        features={SchemaFeature.OBJECT},
    )
    provider = FakeModel(ProviderResult())
    provider.capabilities = ProviderCapabilities(
        structured_output=True,
        schema_features=frozenset({SchemaFeature.OBJECT, SchemaFeature.ARRAY}),
    )
    with pytest.raises(UnsupportedProviderCapabilityError, match="enum"):
        validate_provider_contract(provider, structured_output=request)
    assert provider.calls == 0


def test_literal_schema_values_are_not_treated_as_validation_keywords() -> None:
    request = StructuredOutputRequest(
        "literal", {"type": "object", "const": {"business_field": {"other_field": "value"}}}
    )
    validate_provider_contract(FakeModel(ProviderResult()), structured_output=request)


def test_expired_deadline_is_not_an_attempt_and_never_dispatches() -> None:
    provider = FakeModel(ProviderResult(structured={"answer": "ok"}))
    with pytest.raises(ProviderTimeoutError) as caught:
        _complete(provider, deadline=9, clock=lambda: 10)
    assert caught.value.acceptance is AcceptanceState.NOT_ATTEMPTED
    assert provider.calls == 0


def test_context_cancellation_prevents_dispatch() -> None:
    provider = FakeModel(ProviderResult(structured={"answer": "ok"}))
    with pytest.raises(ProviderCancellationError):
        _complete(provider, call_context=ProviderCallContext(cancelled=True))
    assert provider.calls == 0


def test_context_deadline_can_only_tighten_request_deadline() -> None:
    provider = FakeModel(ProviderResult(structured={"answer": "ok"}))
    _complete(
        provider, deadline=30, call_context=ProviderCallContext(deadline=20), clock=lambda: 10
    )
    assert provider.requests[0].effective_deadline == 20


def test_task_cancellation_is_preserved_without_retries() -> None:
    class CancelledProvider(FakeModel):
        async def complete(
            self,
            messages: Sequence[ChatMessage],
            *,
            options: GenerationOptions,
            structured_output: StructuredOutputRequest,
            tools: Sequence[ProviderToolDefinition] = (),
            tool_results: Sequence[ToolResultMessage] = (),
            effective_deadline: float | None = None,
            call_context: ProviderCallContext | None = None,
        ) -> ProviderResult:
            self.calls += 1
            raise asyncio.CancelledError

    provider = CancelledProvider(ProviderResult())
    with pytest.raises(asyncio.CancelledError):
        _complete(provider)
    assert provider.calls == 1


@pytest.mark.parametrize("timeout", [float("inf"), float("nan")])
def test_generation_timeout_must_be_finite(timeout: float) -> None:
    with pytest.raises(ValidationError):
        GenerationOptions(model="test", timeout=timeout)
