"""Opt-in smoke tests for already running vLLM and LM Studio environments.

These tests never start, stop, or configure a server, and never download
model weights. Each server family is independently skippable so a
contributor with only one server running is not blocked.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from conducto import (
    ChatMessage,
    GenerationOptions,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    parse_model_decision,
)
from conducto.adapters import AdapterDependencyError, require_adapter
from conducto.providers import (
    LM_STUDIO_TOOL_CAPABLE_PROFILE,
    VLLM_TOOL_CAPABLE_PROFILE,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
)


class _SmokeAnswer(BaseModel):
    """Typed terminal response for OpenAI-compatible smoke tests."""

    answer: str


@dataclass(frozen=True, slots=True)
class _SmokeServer:
    """Environment-driven settings for one opt-in smoke server family."""

    enabled_env: str
    endpoint_env: str
    default_endpoint: str
    model_env: str
    tool_model_env: str
    tool_profile: OpenAICompatibleProfile
    marker: str


_VLLM_SERVER = _SmokeServer(
    enabled_env="CONDUCTO_VLLM_SMOKE",
    endpoint_env="CONDUCTO_VLLM_ENDPOINT",
    default_endpoint="http://localhost:8000/v1",
    model_env="CONDUCTO_VLLM_MODEL",
    tool_model_env="CONDUCTO_VLLM_TOOL_MODEL",
    tool_profile=VLLM_TOOL_CAPABLE_PROFILE,
    marker="vllm",
)
_LM_STUDIO_SERVER = _SmokeServer(
    enabled_env="CONDUCTO_LM_STUDIO_SMOKE",
    endpoint_env="CONDUCTO_LM_STUDIO_ENDPOINT",
    default_endpoint="http://localhost:1234/v1",
    model_env="CONDUCTO_LM_STUDIO_MODEL",
    tool_model_env="CONDUCTO_LM_STUDIO_TOOL_MODEL",
    tool_profile=LM_STUDIO_TOOL_CAPABLE_PROFILE,
    marker="lm_studio",
)


def _require_smoke_environment(server: _SmokeServer) -> tuple[str, str]:
    """Return endpoint/model settings or skip with an actionable reason."""
    if os.environ.get(server.enabled_env) != "1":
        pytest.skip(f"Set {server.enabled_env}=1 to run {server.marker} smoke tests.")
    try:
        require_adapter("openai")
    except AdapterDependencyError as error:
        pytest.skip(str(error))
    model = os.environ.get(server.model_env)
    if not model:
        pytest.skip(f"Set {server.model_env} to an already loaded local {server.marker} model.")
    endpoint = os.environ.get(server.endpoint_env, server.default_endpoint)
    return endpoint, model


async def _exercise_terminal_structured_smoke(server: _SmokeServer) -> None:
    """Verify native JSON Schema terminal output against a running server."""
    endpoint, model = _require_smoke_environment(server)
    provider = OpenAICompatibleProvider(
        endpoint=endpoint,
        model=model,
        profile=f"{server.marker.replace('_', '-')}-terminal-json",
    )
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


async def _exercise_native_tool_flow_smoke(server: _SmokeServer) -> None:
    """Verify native tool definition, call, result, and terminal translation."""
    endpoint, default_model = _require_smoke_environment(server)
    model = os.environ.get(server.tool_model_env, default_model)
    provider = OpenAICompatibleProvider(
        endpoint=endpoint,
        model=model,
        profile=server.tool_profile,
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
        pytest.fail(f"Configured {server.marker} smoke profile did not return a native tool call.")
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


@pytest.mark.vllm
def test_vllm_terminal_structured_smoke() -> None:
    """Verify native JSON Schema terminal output against a running vLLM server."""
    asyncio.run(_exercise_terminal_structured_smoke(_VLLM_SERVER))


@pytest.mark.vllm
def test_vllm_native_tool_flow_smoke() -> None:
    """Verify the native tool round trip against a running vLLM server."""
    asyncio.run(_exercise_native_tool_flow_smoke(_VLLM_SERVER))


@pytest.mark.lm_studio
def test_lm_studio_terminal_structured_smoke() -> None:
    """Verify native JSON Schema terminal output against a running LM Studio server."""
    asyncio.run(_exercise_terminal_structured_smoke(_LM_STUDIO_SERVER))


@pytest.mark.lm_studio
def test_lm_studio_native_tool_flow_smoke() -> None:
    """Verify the native tool round trip against a running LM Studio server."""
    asyncio.run(_exercise_native_tool_flow_smoke(_LM_STUDIO_SERVER))
