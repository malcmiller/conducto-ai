"""Deterministic tests for the optional Ollama provider adapter."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from conducto import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ProviderCallContext,
    ProviderClientConfig,
    ProviderOwnership,
    ProviderRegistry,
    ProviderTimeoutError,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    parse_model_decision,
)
from conducto.core.runtime_errors import ContradictoryProviderConfigurationError
from conducto.providers import (
    OLLAMA_TOOL_CAPABLE_PROFILE,
    OllamaConfigurationError,
    OllamaIncompatibleVersionError,
    OllamaModelNotFoundError,
    OllamaOversizedResponseError,
    OllamaProfile,
    OllamaProvider,
    OllamaProviderFactory,
    OllamaReadinessError,
)
from conducto.providers import ollama as ollama_adapter


class _Answer(BaseModel):
    value: str


class _FakeOllamaClient:
    """Scripted official-client-compatible Ollama fixture."""

    def __init__(
        self,
        responses: tuple[Mapping[str, Any], ...] = (),
        *,
        version: str = "0.6.0",
        show_error: Exception | None = None,
        version_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.responses = list(responses)
        self.version_value = version
        self.show_error = show_error
        self.version_error = version_error
        self.delay = delay
        self.chat_calls: list[dict[str, Any]] = []
        self.closed = False

    async def chat(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return the next scripted chat response."""
        self.chat_calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.responses:
            raise AssertionError("script exhausted")
        return self.responses.pop(0)

    async def list(self) -> Mapping[str, Any]:
        """Return a deterministic model list."""
        return {"models": [{"name": "llama3.1:8b"}]}

    async def show(self, model: str) -> Mapping[str, Any]:
        """Return model metadata or the configured failure."""
        _ = model
        if self.show_error is not None:
            raise self.show_error
        return {"model": "llama3.1:8b"}

    async def version(self) -> Mapping[str, Any]:
        """Return server version metadata or the configured failure."""
        if self.version_error is not None:
            raise self.version_error
        return {"version": self.version_value}

    async def aclose(self) -> None:
        """Record async cleanup."""
        self.closed = True


def _schema() -> StructuredOutputRequest:
    """Return the terminal response schema shared by adapter tests."""
    return StructuredOutputRequest(name="Answer", schema=_Answer.model_json_schema())


def _tool_definition(
    *,
    tool_id: str = "tool-1",
    name: str = "lookup_tool_1",
) -> ProviderToolDefinition:
    """Return a deterministic Conducto tool definition."""
    return ProviderToolDefinition(
        tool_id=tool_id,
        name=name,
        description="Looks up deterministic fixture data.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def test_ollama_provider_sends_native_format_and_validates_terminal_json() -> None:
    async def exercise() -> None:
        client = _FakeOllamaClient(
            (
                {
                    "message": {"role": "assistant", "content": '{"value":"ok"}'},
                    "prompt_eval_count": 3,
                    "eval_count": 2,
                    "done_reason": "stop",
                },
            )
        )
        provider = OllamaProvider(model="llama3.1:8b", client=client)

        result = await provider.complete(
            (ChatMessage(role="user", content="answer"),),
            options=GenerationOptions(model="llama3.1:8b", temperature=0.2, max_tokens=20),
            structured_output=_schema(),
        )

        assert result.structured == {"value": "ok"}
        assert result.usage.input_tokens == 3
        assert result.usage.output_tokens == 2
        assert client.chat_calls[0]["format"] == _Answer.model_json_schema()
        assert client.chat_calls[0]["options"]["temperature"] == 0.2
        assert client.chat_calls[0]["options"]["num_predict"] == 20

    asyncio.run(exercise())


def test_ollama_provider_completes_tool_call_result_terminal_flow_without_one_of() -> None:
    async def exercise() -> None:
        client = _FakeOllamaClient(
            (
                {
                    "id": "first-response",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup_tool_1",
                                    "arguments": {"query": "value"},
                                }
                            }
                        ],
                    },
                },
                {"message": {"role": "assistant", "content": '{"value":"done"}'}},
            )
        )
        provider = OllamaProvider(
            model="llama3.1:8b",
            client=client,
            profile=OLLAMA_TOOL_CAPABLE_PROFILE,
        )
        tools = (_tool_definition(),)

        first = await provider.complete(
            (ChatMessage(role="user", content="lookup"),),
            options=GenerationOptions(model="llama3.1:8b"),
            structured_output=_schema(),
            tools=tools,
        )
        decision = parse_model_decision(first, response_type=_Answer, tools=tools)
        assert decision.type == "tool_call"
        assert decision.tool_id == "tool-1"
        assert decision.arguments == {"query": "value"}
        assert decision.call_id.startswith("ollama_")
        assert "oneOf" not in client.chat_calls[0]["format"]
        assert client.chat_calls[0]["tools"][0]["function"]["name"] == "lookup_tool_1"

        second = await provider.complete(
            (ChatMessage(role="user", content="lookup"),),
            options=GenerationOptions(model="llama3.1:8b"),
            structured_output=_schema(),
            tools=tools,
            tool_results=(
                ToolResultMessage(
                    call_id=decision.call_id,
                    status="success",
                    result={"value": "fixture"},
                ),
            ),
        )

        assert second.structured == {"value": "done"}
        assistant_message = client.chat_calls[1]["messages"][-2]
        result_message = client.chat_calls[1]["messages"][-1]
        assert assistant_message["role"] == "assistant"
        assert assistant_message["tool_calls"][0]["function"]["name"] == "lookup_tool_1"
        assert result_message["role"] == "tool"
        assert "tool_call_id" not in result_message
        assert "name" not in result_message
        assert result_message["tool_name"] == "lookup_tool_1"

        with pytest.raises(MalformedStructuredOutputError, match="unknown call ID"):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                tools=tools,
                tool_results=(
                    ToolResultMessage(
                        call_id="unknown",
                        status="success",
                        result={"value": "fixture"},
                    ),
                ),
            )
        assert len(client.chat_calls) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("response", "match"),
    (
        (
            {
                "message": {
                    "content": '{"value":"ok"}',
                    "tool_calls": [{"function": {"name": "lookup_tool_1", "arguments": {}}}],
                }
            },
            "mixed terminal",
        ),
        (
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "lookup_tool_1", "arguments": {}}},
                        {"function": {"name": "lookup_tool_1", "arguments": {}}},
                    ]
                }
            },
            "multiple tool calls",
        ),
        (
            {"message": {"tool_calls": [{"function": {"name": "missing", "arguments": {}}}]}},
            "unknown tool name",
        ),
        (
            {
                "message": {
                    "tool_calls": [{"function": {"name": "lookup_tool_1", "arguments": "not-json"}}]
                }
            },
            "malformed tool arguments",
        ),
        (
            {"message": {"role": "assistant", "content": '{"wrong":"shape"}'}},
            "required",
        ),
    ),
)
def test_ollama_provider_rejects_malformed_outputs_before_business_logic(
    response: Mapping[str, Any],
    match: str,
) -> None:
    async def exercise() -> None:
        client = _FakeOllamaClient((response,))
        provider = OllamaProvider(
            model="llama3.1:8b",
            client=client,
            profile=OLLAMA_TOOL_CAPABLE_PROFILE,
        )
        tools = (_tool_definition(),)
        with pytest.raises(MalformedStructuredOutputError, match=match):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                tools=tools,
            )

    asyncio.run(exercise())


def test_ollama_provider_rejects_ambiguous_tools_and_unsupported_profiles() -> None:
    async def exercise() -> None:
        client = _FakeOllamaClient(
            (
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "shared_tool", "arguments": {"query": "x"}}}
                        ]
                    }
                },
            )
        )
        provider = OllamaProvider(
            model="llama3.1:8b",
            client=client,
            profile=OLLAMA_TOOL_CAPABLE_PROFILE,
        )
        with pytest.raises(MalformedStructuredOutputError, match="ambiguous"):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                tools=(
                    _tool_definition(tool_id="tool-1", name="shared_tool"),
                    _tool_definition(tool_id="tool-2", name="shared_tool"),
                ),
            )

        terminal_only = OllamaProvider(model="llama3.1:8b", client=_FakeOllamaClient(()))
        with pytest.raises(UnsupportedProviderCapabilityError, match="does not support"):
            await terminal_only.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                tools=(_tool_definition(),),
            )
        assert terminal_only.capabilities.tool_calling is False
        with pytest.raises(OllamaConfigurationError, match="has not been conformance tested"):
            OllamaProvider(
                model="llama3.1:8b",
                client=_FakeOllamaClient(()),
                profile=OllamaProfile("custom", tool_calling=True),
            )

    asyncio.run(exercise())


def test_ollama_provider_rejects_unsupported_schemas_before_dispatch() -> None:
    async def exercise() -> None:
        client = _FakeOllamaClient(())
        provider = OllamaProvider(model="llama3.1:8b", client=client)
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported features"):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=StructuredOutputRequest(
                    name="Unsupported",
                    schema={
                        "oneOf": [
                            {"type": "object", "properties": {"value": {"type": "string"}}},
                            {"type": "object", "properties": {"other": {"type": "string"}}},
                        ]
                    },
                ),
            )
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported keywords"):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=StructuredOutputRequest(
                    name="UnsupportedKeyword",
                    schema={
                        "type": "object",
                        "properties": {"value": {"type": "string", "pattern": "^ok$"}},
                    },
                ),
            )
        assert client.chat_calls == []

        tool_provider = OllamaProvider(
            model="llama3.1:8b",
            client=client,
            profile=OLLAMA_TOOL_CAPABLE_PROFILE,
        )
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported keywords"):
            await tool_provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                tools=(
                    ProviderToolDefinition(
                        tool_id="tool-1",
                        name="lookup_tool_1",
                        description="Looks up deterministic fixture data.",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string", "pattern": "^safe$"}},
                        },
                    ),
                ),
            )
        assert client.chat_calls == []

    asyncio.run(exercise())


def test_ollama_provider_enforces_call_timeout_and_deadline() -> None:
    async def exercise() -> None:
        provider = OllamaProvider(
            model="llama3.1:8b",
            client=_FakeOllamaClient(
                ({"message": {"role": "assistant", "content": '{"value":"ok"}'}},),
                delay=0.05,
            ),
        )
        with pytest.raises(ProviderTimeoutError):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="llama3.1:8b", timeout=0.001),
                structured_output=_schema(),
            )

        expired = OllamaProvider(model="llama3.1:8b", client=_FakeOllamaClient(()))
        with pytest.raises(ProviderTimeoutError):
            await expired.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="llama3.1:8b"),
                structured_output=_schema(),
                call_context=ProviderCallContext(deadline=0.0),
            )

    asyncio.run(exercise())


def test_bounded_http_adapter_rejects_oversized_response_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        status_code = 200

        async def __aenter__(self) -> FakeStream:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def aiter_bytes(self) -> Any:
            yield b'{"message":'
            yield b'{"content":"too large"}}'

    class FakeHttpClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def stream(self, method: str, path: str, *, json: Mapping[str, Any] | None = None) -> Any:
            _ = (method, path, json)
            return FakeStream()

        async def aclose(self) -> None:
            return None

    async def exercise() -> None:
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeHttpClient)
        client = ollama_adapter._BoundedOllamaHttpClient(
            endpoint="http://localhost:11434",
            auth_token=None,
            headers={},
            timeout=None,
            transport_options={},
            tls_options={},
            proxy_options={},
            max_response_bytes=8,
        )
        with pytest.raises(OllamaOversizedResponseError):
            await client.chat(model="llama3.1:8b", messages=[])

    asyncio.run(exercise())


def test_ollama_provider_factory_and_preconstructed_client_paths_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_clients: list[_FakeOllamaClient] = []

    def fake_create_official_client(**kwargs: Any) -> _FakeOllamaClient:
        assert kwargs["endpoint"] == "http://localhost:11434"
        assert kwargs["auth_token"] == "secret-token"
        assert kwargs["transport_options"]["max_connections"] == 4
        assert kwargs["max_response_bytes"] == 1_000_000
        client = _FakeOllamaClient(())
        constructed_clients.append(client)
        return client

    monkeypatch.setattr(ollama_adapter, "_create_official_client", fake_create_official_client)
    monkeypatch.setenv("CONDUCTO_OLLAMA_TOKEN", "secret-token")
    registry = ProviderRegistry()
    registry.register_provider_type("ollama", OllamaProviderFactory())
    registration = registry.register_provider(
        "local",
        provider_type="ollama",
        configuration=ProviderClientConfig(
            endpoint="http://localhost:11434",
            credential_ref="CONDUCTO_OLLAMA_TOKEN",
            transport={"max_connections": 4},
            provider_defaults={"model": "llama3.1:8b", "context_window": 8192},
        ),
        model_configuration=ModelConfiguration(provider="ollama", model="llama3.1:8b"),
    )
    assert registration.ownership is ProviderOwnership.RUNTIME_OWNED
    assert registration.client.capabilities.structured_output is True
    assert constructed_clients

    client = _FakeOllamaClient(())
    registry.register_client(
        "borrowed",
        OllamaProvider(model="llama3.1:8b", client=client),
        ModelConfiguration(provider="ollama", model="llama3.1:8b"),
    )
    assert registry.resolve("borrowed").ownership is ProviderOwnership.CALLER_OWNED

    with pytest.raises(ContradictoryProviderConfigurationError):
        OllamaProvider(
            model="llama3.1:8b",
            client=client,
            endpoint="http://localhost:11434",
        )
    monkeypatch.delenv("CONDUCTO_OLLAMA_TOKEN")
    with pytest.raises(OllamaConfigurationError):
        OllamaProviderFactory().create(
            ProviderClientConfig(
                credential_ref="CONDUCTO_OLLAMA_TOKEN",
                provider_defaults={"model": "llama3.1:8b"},
            )
        )


def test_ollama_readiness_distinguishes_version_model_and_connection_failures() -> None:
    async def exercise() -> None:
        await OllamaProvider(
            model="llama3.1:8b",
            client=_FakeOllamaClient(()),
        ).check_readiness()

        with pytest.raises(OllamaIncompatibleVersionError):
            await OllamaProvider(
                model="llama3.1:8b",
                client=_FakeOllamaClient((), version="0.5.0"),
            ).check_readiness()

        not_found = RuntimeError("model not found")
        with pytest.raises(OllamaModelNotFoundError):
            await OllamaProvider(
                model="llama3.1:8b",
                client=_FakeOllamaClient((), show_error=not_found),
            ).check_readiness()

        refused = RuntimeError("connection refused")
        with pytest.raises(OllamaReadinessError):
            await OllamaProvider(
                model="llama3.1:8b",
                client=_FakeOllamaClient((), version_error=refused),
            ).check_readiness()

    asyncio.run(exercise())


def test_ollama_provider_closes_owned_client_only() -> None:
    async def exercise() -> None:
        borrowed = _FakeOllamaClient(())
        borrowed_provider = OllamaProvider(model="llama3.1:8b", client=borrowed)
        await borrowed_provider.aclose()
        assert borrowed.closed is False

        original = os.environ.get("CONDUCTO_OLLAMA_TOKEN")
        try:
            os.environ["CONDUCTO_OLLAMA_TOKEN"] = "token"
            with pytest.raises(OllamaConfigurationError):
                OllamaProviderFactory().create(
                    ProviderClientConfig(
                        credential_ref="CONDUCTO_OLLAMA_TOKEN",
                        provider_defaults={"model": "llama3.1:8b", "unknown": True},
                    )
                )
        finally:
            if original is None:
                os.environ.pop("CONDUCTO_OLLAMA_TOKEN", None)
            else:
                os.environ["CONDUCTO_OLLAMA_TOKEN"] = original

    asyncio.run(exercise())
