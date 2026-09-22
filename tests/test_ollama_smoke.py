"""Opt-in smoke tests for an already running Ollama environment."""

from __future__ import annotations

import asyncio
import os

import pytest
from pydantic import BaseModel

from conducto.adapters import AdapterDependencyError, require_adapter
from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    parse_model_decision,
)
from conducto.providers import OLLAMA_TOOL_CAPABLE_PROFILE, OllamaProvider

pytestmark = pytest.mark.ollama


class _SmokeAnswer(BaseModel):
    """Typed terminal response for Ollama smoke tests."""

    answer: str


def _require_smoke_environment() -> tuple[str, str]:
    """Return endpoint/model settings or skip with an actionable reason."""
    if os.environ.get("CONDUCTO_OLLAMA_SMOKE") != "1":
        pytest.skip("Set CONDUCTO_OLLAMA_SMOKE=1 to run Ollama smoke tests.")
    try:
        require_adapter("ollama")
    except AdapterDependencyError as error:
        pytest.skip(str(error))
    model = os.environ.get("CONDUCTO_OLLAMA_MODEL")
    if not model:
        pytest.skip("Set CONDUCTO_OLLAMA_MODEL to an already pulled local Ollama model.")
    endpoint = os.environ.get("CONDUCTO_OLLAMA_ENDPOINT", "http://localhost:11434")
    return endpoint, model


def test_ollama_terminal_structured_smoke() -> None:
    """Verify native JSON Schema terminal output against a running Ollama server."""

    async def exercise() -> None:
        endpoint, model = _require_smoke_environment()
        provider = OllamaProvider(endpoint=endpoint, model=model)
        await provider.check_readiness(model=model)
        result = await provider.complete(
            (ChatMessage(role="user", content="Reply with answer exactly: ok"),),
            options=GenerationOptions(model=model, temperature=0.0, max_tokens=32, timeout=30),
            structured_output=StructuredOutputRequest(
                name="SmokeAnswer",
                schema=_SmokeAnswer.model_json_schema(),
            ),
        )
        assert _SmokeAnswer.model_validate(result.structured).answer
        await provider.aclose()

    asyncio.run(exercise())


def test_ollama_native_tool_flow_smoke() -> None:
    """Verify native tool definition, call, result, and terminal translation."""

    async def exercise() -> None:
        endpoint, default_model = _require_smoke_environment()
        model = os.environ.get("CONDUCTO_OLLAMA_TOOL_MODEL", default_model)
        provider = OllamaProvider(
            endpoint=endpoint,
            model=model,
            profile=OLLAMA_TOOL_CAPABLE_PROFILE,
        )
        await provider.check_readiness(model=model)
        tools = (
            ProviderToolDefinition(
                tool_id="lookup",
                name="lookup_tool_1",
                description="Returns deterministic smoke-test fixture data.",
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )
        first = await provider.complete(
            (
                ChatMessage(
                    role="user",
                    content=(
                        "Use lookup_tool_1 with query 'conducto', then answer only "
                        "after the tool result."
                    ),
                ),
            ),
            options=GenerationOptions(model=model, temperature=0.0, max_tokens=128, timeout=60),
            structured_output=StructuredOutputRequest(
                name="SmokeAnswer",
                schema=_SmokeAnswer.model_json_schema(),
                required=False,
            ),
            tools=tools,
        )
        decision = parse_model_decision(first, response_type=_SmokeAnswer, tools=tools)
        if decision.type != "tool_call":
            pytest.fail("Configured Ollama smoke profile did not return a native tool call.")
        second = await provider.complete(
            (ChatMessage(role="user", content="Use the tool result to answer."),),
            options=GenerationOptions(model=model, temperature=0.0, max_tokens=128, timeout=60),
            structured_output=StructuredOutputRequest(
                name="SmokeAnswer",
                schema=_SmokeAnswer.model_json_schema(),
            ),
            tools=tools,
            tool_results=(
                ToolResultMessage(
                    call_id=decision.call_id,
                    status="success",
                    result={"answer": "conducto fixture"},
                ),
            ),
        )
        assert _SmokeAnswer.model_validate(second.structured).answer
        await provider.aclose()

    asyncio.run(exercise())
