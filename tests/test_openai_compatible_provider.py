"""Deterministic tests for the optional OpenAI-compatible provider adapter."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ProviderAuthenticationError,
    ProviderCallContext,
    ProviderTimeoutError,
    ProviderToolDefinition,
    SchemaFeature,
    StructuredOutputRequest,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    parse_model_decision,
)
from conducto.core.provider_registry import (
    ProviderClientConfig,
    ProviderOwnership,
    ProviderRegistry,
)
from conducto.core.runtime_errors import ContradictoryProviderConfigurationError
from conducto.providers import (
    LM_STUDIO_DEFAULT_PROFILE,
    VLLM_TOOL_CAPABLE_PROFILE,
    OpenAICompatibleConfigurationError,
    OpenAICompatibleIncompatibleVersionError,
    OpenAICompatibleModelNotFoundError,
    OpenAICompatibleOversizedResponseError,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
    OpenAICompatibleProviderFactory,
    OpenAICompatibleReadinessError,
)
from conducto.providers import openai_compatible as openai_compatible_adapter


class _Answer(BaseModel):
    value: str


class _FakeChatCompletions:
    """Scripted ``chat.completions`` namespace."""

    def __init__(self, outer: _FakeOpenAICompatibleClient) -> None:
        """Bind this namespace to its owning fixture."""
        self._outer = outer

    async def create(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return the next scripted completion after any configured gate opens."""
        self._outer.chat_calls.append(kwargs)
        if self._outer.response_gate is not None:
            await self._outer.response_gate.wait()
        if not self._outer.responses:
            raise AssertionError("script exhausted")
        return self._outer.responses.pop(0)


class _FakeChat:
    """Scripted ``chat`` namespace."""

    def __init__(self, outer: _FakeOpenAICompatibleClient) -> None:
        """Bind this namespace to its owning fixture."""
        self.completions: openai_compatible_adapter._ChatCompletionsClient = _FakeChatCompletions(
            outer
        )


class _FakeModels:
    """Scripted ``models`` namespace."""

    def __init__(self, outer: _FakeOpenAICompatibleClient) -> None:
        """Bind this namespace to its owning fixture."""
        self._outer = outer

    async def list(self) -> Mapping[str, Any]:
        """Return a deterministic model list or the configured failure."""
        if self._outer.models_error is not None:
            raise self._outer.models_error
        return self._outer.models_payload


class _FakeOpenAICompatibleClient:
    """Scripted official-client-compatible OpenAI-compatible fixture."""

    def __init__(
        self,
        responses: tuple[Mapping[str, Any], ...] = (),
        *,
        version: str = "0.7.0",
        models_payload: Mapping[str, Any] | None = None,
        models_error: Exception | None = None,
        version_error: Exception | None = None,
        response_gate: asyncio.Event | None = None,
    ) -> None:
        """Create a scripted fixture with metadata and an optional response gate."""
        self.responses = list(responses)
        self.version_value = version
        self.models_error = models_error
        self.version_error = version_error
        self.models_payload = (
            models_payload if models_payload is not None else {"data": [{"id": "test-model"}]}
        )
        self.response_gate = response_gate
        self.chat_calls: list[dict[str, Any]] = []
        self.closed = False
        self.chat: openai_compatible_adapter._ChatClient = _FakeChat(self)
        self.models: openai_compatible_adapter._ModelsClient = _FakeModels(self)

    async def get_version(self, path: str) -> Mapping[str, Any]:
        """Return scripted version metadata or the configured failure."""
        _ = path
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


def test_openai_compatible_provider_sends_native_format_and_validates_terminal_json() -> None:
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient(
            (
                {
                    "id": "resp-1",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": '{"value":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        )
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=client,
        )

        result = await provider.complete(
            (ChatMessage(role="user", content="answer"),),
            options=GenerationOptions(model="test-model", temperature=0.2, max_tokens=20),
            structured_output=_schema(),
        )

        assert result.structured == {"value": "ok"}
        assert result.usage.input_tokens == 3
        assert result.usage.output_tokens == 2
        assert result.usage.total_tokens == 5
        assert result.finish_reason == "stop"
        request = client.chat_calls[0]
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["name"] == "Answer"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["temperature"] == 0.2
        assert request["max_tokens"] == 20
        assert "tools" not in request

    asyncio.run(exercise())


def test_openai_compatible_provider_completes_tool_call_result_terminal_flow_without_one_of() -> (
    None
):
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient(
            (
                {
                    "id": "resp-tool-1",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "lookup_tool_1",
                                            "arguments": '{"query":"value"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
                {
                    "id": "resp-tool-2",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": '{"value":"done"}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile=VLLM_TOOL_CAPABLE_PROFILE,
            client=client,
        )
        tools = (_tool_definition(),)

        first = await provider.complete(
            (ChatMessage(role="user", content="lookup"),),
            options=GenerationOptions(model="test-model"),
            structured_output=_schema(),
            tools=tools,
        )
        decision = parse_model_decision(first, response_type=_Answer, tools=tools)
        assert decision.type == "tool_call"
        assert decision.tool_id == "tool-1"
        assert decision.arguments == {"query": "value"}
        assert decision.call_id == "call-1"
        request = client.chat_calls[0]
        assert "oneOf" not in request["response_format"]["json_schema"]["schema"]
        assert request["tools"][0]["function"]["name"] == "lookup_tool_1"
        assert request["parallel_tool_calls"] is False

        second = await provider.complete(
            (ChatMessage(role="user", content="lookup"),),
            options=GenerationOptions(model="test-model"),
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
        assert assistant_message["tool_calls"][0]["id"] == "call-1"
        assert assistant_message["tool_calls"][0]["function"]["name"] == "lookup_tool_1"
        assert result_message["role"] == "tool"
        assert result_message["tool_call_id"] == "call-1"

        with pytest.raises(MalformedStructuredOutputError, match="unknown call ID"):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="test-model"),
                structured_output=_schema(),
                tools=tools,
                tool_results=(ToolResultMessage(call_id="unknown", status="success", result={}),),
            )
        assert len(client.chat_calls) == 2

    asyncio.run(exercise())


def test_openai_compatible_provider_synthesizes_call_id_when_server_omits_it() -> None:
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient(
            (
                {
                    "id": "resp-no-id",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "lookup_tool_1",
                                            "arguments": '{"query":"value"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        )
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile=VLLM_TOOL_CAPABLE_PROFILE,
            client=client,
        )
        result = await provider.complete(
            (ChatMessage(role="user", content="lookup"),),
            options=GenerationOptions(model="test-model"),
            structured_output=_schema(),
            tools=(_tool_definition(),),
        )
        call_id = result.tool_calls[0].call_id
        assert call_id is not None and call_id.startswith("openai_compatible_")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("response", "match"),
    (
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"value":"ok"}',
                            "tool_calls": [
                                {"function": {"name": "lookup_tool_1", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            },
            "mixed terminal",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "lookup_tool_1", "arguments": "{}"}},
                                {"function": {"name": "lookup_tool_1", "arguments": "{}"}},
                            ]
                        }
                    }
                ]
            },
            "multiple tool calls",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "missing", "arguments": "{}"}}]
                        }
                    }
                ]
            },
            "unknown tool name",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "lookup_tool_1", "arguments": "not-json"}}
                            ]
                        }
                    }
                ]
            },
            "malformed",
        ),
        (
            {"choices": [{"message": {"role": "assistant", "content": '{"wrong":"shape"}'}}]},
            "required",
        ),
    ),
)
def test_openai_compatible_provider_rejects_malformed_outputs_before_business_logic(
    response: Mapping[str, Any],
    match: str,
) -> None:
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient((response,))
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile=VLLM_TOOL_CAPABLE_PROFILE,
            client=client,
        )
        tools = (_tool_definition(),)
        with pytest.raises(MalformedStructuredOutputError, match=match):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="test-model"),
                structured_output=_schema(),
                tools=tools,
            )

    asyncio.run(exercise())


def test_openai_compatible_provider_rejects_ambiguous_tools_and_unsupported_profiles() -> None:
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient(
            (
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "shared_tool",
                                            "arguments": '{"query":"x"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        )
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile=VLLM_TOOL_CAPABLE_PROFILE,
            client=client,
        )
        with pytest.raises(MalformedStructuredOutputError, match="ambiguous"):
            await provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="test-model"),
                structured_output=_schema(),
                tools=(
                    _tool_definition(tool_id="tool-1", name="shared_tool"),
                    _tool_definition(tool_id="tool-2", name="shared_tool"),
                ),
            )

        terminal_only = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=_FakeOpenAICompatibleClient(()),
        )
        with pytest.raises(UnsupportedProviderCapabilityError, match="does not support"):
            await terminal_only.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="test-model"),
                structured_output=_schema(),
                tools=(_tool_definition(),),
            )
        assert terminal_only.capabilities.tool_calling is False
        with pytest.raises(
            OpenAICompatibleConfigurationError, match="has not been conformance tested"
        ):
            OpenAICompatibleProvider(
                model="test-model",
                profile=OpenAICompatibleProfile("custom", server_family="vllm", tool_calling=True),
                client=_FakeOpenAICompatibleClient(()),
            )

    asyncio.run(exercise())


def test_openai_compatible_provider_rejects_unsupported_schemas_before_dispatch() -> None:
    async def exercise() -> None:
        client = _FakeOpenAICompatibleClient(())
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=client,
        )
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported keywords"):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="test-model"),
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
                options=GenerationOptions(model="test-model"),
                structured_output=StructuredOutputRequest(
                    name="UnsupportedKeyword",
                    schema={
                        "type": "object",
                        "properties": {"value": {"type": "string", "pattern": "^ok$"}},
                    },
                ),
            )
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported features"):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="test-model"),
                structured_output=StructuredOutputRequest(
                    name="Answer",
                    schema=_Answer.model_json_schema(),
                    features=frozenset({SchemaFeature.ONE_OF}),
                ),
            )
        assert client.chat_calls == []

        tool_provider = OpenAICompatibleProvider(
            model="test-model",
            profile=VLLM_TOOL_CAPABLE_PROFILE,
            client=client,
        )
        with pytest.raises(UnsupportedProviderCapabilityError, match="unsupported keywords"):
            await tool_provider.complete(
                (ChatMessage(role="user", content="lookup"),),
                options=GenerationOptions(model="test-model"),
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


def test_openai_compatible_provider_enforces_call_timeout_and_deadline() -> None:
    async def exercise() -> None:
        response_gate = asyncio.Event()
        provider = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=_FakeOpenAICompatibleClient(
                ({"choices": [{"message": {"role": "assistant", "content": '{"value":"ok"}'}}]},),
                response_gate=response_gate,
            ),
        )
        with pytest.raises(ProviderTimeoutError):
            await provider.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="test-model", timeout=0.001),
                structured_output=_schema(),
            )

        response_gate.set()
        result = await provider.complete(
            (ChatMessage(role="user", content="answer"),),
            options=GenerationOptions(model="test-model"),
            structured_output=_schema(),
        )
        assert result.structured == {"value": "ok"}

        expired = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=_FakeOpenAICompatibleClient(()),
        )
        with pytest.raises(ProviderTimeoutError):
            await expired.complete(
                (ChatMessage(role="user", content="answer"),),
                options=GenerationOptions(model="test-model"),
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
            yield b'{"choices":'
            yield b'[{"message":{"content":"too large"}}]}'

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
        client = openai_compatible_adapter._BoundedOpenAICompatibleHttpClient(
            endpoint="http://localhost:8000/v1",
            api_key=None,
            organization=None,
            project=None,
            headers={},
            timeout=None,
            transport_options={},
            tls_options={},
            proxy_options={},
            max_response_bytes=8,
        )
        with pytest.raises(OpenAICompatibleOversizedResponseError):
            await client.chat.completions.create(model="test-model", messages=[])

    asyncio.run(exercise())


def test_openai_compatible_provider_factory_and_preconstructed_client_paths_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_clients: list[_FakeOpenAICompatibleClient] = []

    def fake_create_bounded_client(**kwargs: Any) -> _FakeOpenAICompatibleClient:
        assert kwargs["endpoint"] == "http://localhost:8000/v1"
        assert kwargs["api_key"] == "secret-token"
        assert kwargs["transport_options"]["max_connections"] == 4
        assert kwargs["max_response_bytes"] == 1_000_000
        client = _FakeOpenAICompatibleClient(())
        constructed_clients.append(client)
        return client

    monkeypatch.setattr(
        openai_compatible_adapter, "_create_bounded_client", fake_create_bounded_client
    )
    monkeypatch.setenv("CONDUCTO_OPENAI_COMPATIBLE_TOKEN", "secret-token")
    registry = ProviderRegistry()
    registry.register_provider_type("openai_compatible", OpenAICompatibleProviderFactory())
    registration = registry.register_provider(
        "local",
        provider_type="openai_compatible",
        configuration=ProviderClientConfig(
            endpoint="http://localhost:8000/v1",
            credential_ref="CONDUCTO_OPENAI_COMPATIBLE_TOKEN",
            transport={"max_connections": 4},
            provider_defaults={
                "model": "test-model",
                "profile": "vllm-terminal-json",
                "context_window": 8192,
            },
        ),
        model_configuration=ModelConfiguration(provider="openai_compatible", model="test-model"),
    )
    assert registration.ownership is ProviderOwnership.RUNTIME_OWNED
    assert registration.client.capabilities.structured_output is True
    assert constructed_clients

    client = _FakeOpenAICompatibleClient(())
    registry.register_client(
        "borrowed",
        OpenAICompatibleProvider(model="test-model", profile="vllm-terminal-json", client=client),
        ModelConfiguration(provider="openai_compatible", model="test-model"),
    )
    assert registry.resolve("borrowed").ownership is ProviderOwnership.CALLER_OWNED

    with pytest.raises(ContradictoryProviderConfigurationError):
        OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=client,
            endpoint="http://localhost:8000/v1",
        )
    monkeypatch.delenv("CONDUCTO_OPENAI_COMPATIBLE_TOKEN")
    with pytest.raises(OpenAICompatibleConfigurationError):
        OpenAICompatibleProviderFactory().create(
            ProviderClientConfig(
                credential_ref="CONDUCTO_OPENAI_COMPATIBLE_TOKEN",
                provider_defaults={"model": "test-model", "profile": "vllm-terminal-json"},
            )
        )


def test_openai_compatible_readiness_distinguishes_version_model_and_connection_failures() -> None:
    async def exercise() -> None:
        await OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=_FakeOpenAICompatibleClient(()),
        ).check_readiness()

        with pytest.raises(OpenAICompatibleIncompatibleVersionError):
            await OpenAICompatibleProvider(
                model="test-model",
                profile="vllm-terminal-json",
                client=_FakeOpenAICompatibleClient((), version="0.1.0"),
            ).check_readiness()

        not_found = RuntimeError("model not found")
        with pytest.raises(OpenAICompatibleModelNotFoundError):
            await OpenAICompatibleProvider(
                model="test-model",
                profile="vllm-terminal-json",
                client=_FakeOpenAICompatibleClient((), models_error=not_found),
            ).check_readiness()

        refused = RuntimeError("connection refused")
        with pytest.raises(OpenAICompatibleReadinessError):
            await OpenAICompatibleProvider(
                model="test-model",
                profile="vllm-terminal-json",
                client=_FakeOpenAICompatibleClient((), version_error=refused),
            ).check_readiness()

        unauthorized = RuntimeError("unauthorized")
        with pytest.raises(ProviderAuthenticationError):
            await OpenAICompatibleProvider(
                model="test-model",
                profile="vllm-terminal-json",
                client=_FakeOpenAICompatibleClient((), models_error=unauthorized),
            ).check_readiness()

        # LM Studio profiles have no documented version endpoint, so an
        # otherwise-incompatible version value is never checked.
        await OpenAICompatibleProvider(
            model="test-model",
            profile=LM_STUDIO_DEFAULT_PROFILE,
            client=_FakeOpenAICompatibleClient((), version="0.0.1"),
        ).check_readiness()

    asyncio.run(exercise())


def test_openai_compatible_provider_closes_owned_client_only() -> None:
    async def exercise() -> None:
        borrowed = _FakeOpenAICompatibleClient(())
        borrowed_provider = OpenAICompatibleProvider(
            model="test-model",
            profile="vllm-terminal-json",
            client=borrowed,
        )
        await borrowed_provider.aclose()
        assert borrowed.closed is False

        original = os.environ.get("CONDUCTO_OPENAI_COMPATIBLE_TOKEN")
        try:
            os.environ["CONDUCTO_OPENAI_COMPATIBLE_TOKEN"] = "token"
            with pytest.raises(OpenAICompatibleConfigurationError):
                OpenAICompatibleProviderFactory().create(
                    ProviderClientConfig(
                        credential_ref="CONDUCTO_OPENAI_COMPATIBLE_TOKEN",
                        provider_defaults={
                            "model": "test-model",
                            "profile": "vllm-terminal-json",
                            "unknown": True,
                        },
                    )
                )
        finally:
            if original is None:
                os.environ.pop("CONDUCTO_OPENAI_COMPATIBLE_TOKEN", None)
            else:
                os.environ["CONDUCTO_OPENAI_COMPATIBLE_TOKEN"] = original

    asyncio.run(exercise())
