"""Deterministic regressions for shared first-party adapter infrastructure."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ProviderAuthenticationError,
    ProviderCallContext,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderToolDefinition,
    StructuredOutputRequest,
    UnsupportedProviderCapabilityError,
)
from conducto.core.provider_registry import ProviderClientConfig, ProviderRegistry
from conducto.providers import (
    OllamaModelNotFoundError,
    OllamaOversizedResponseError,
    OllamaProvider,
    OllamaProviderFactory,
    OpenAICompatibleModelNotFoundError,
    OpenAICompatibleOversizedResponseError,
    OpenAICompatibleProvider,
    OpenAICompatibleProviderFactory,
    _response,
)
from conducto.providers._tool_cache import ToolCallCache

_TOKEN = "adapter-regression-credential"


def _configured_provider(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_response_bytes: int = 1_000_000,
    tools: bool = False,
) -> OllamaProvider | OpenAICompatibleProvider:
    if kind == "ollama":
        monkeypatch.setattr("conducto.providers.ollama.require_adapter", lambda _: None)
        return OllamaProvider(
            model="test-model",
            endpoint="http://adapter.invalid",
            auth_token=_TOKEN,
            profile="llama3.1-tools" if tools else "terminal-json",
            headers={"authorization": "replaced", "X-Adapter-Test": "yes"},
            max_response_bytes=max_response_bytes,
        )
    monkeypatch.setattr("conducto.providers.openai_compatible.require_adapter", lambda _: None)
    return OpenAICompatibleProvider(
        model="test-model",
        profile="lm-studio-tools" if tools else "lm-studio-terminal-json",
        endpoint="http://adapter.invalid/v1",
        api_key=_TOKEN,
        organization="test-org",
        project="test-project",
        headers={"authorization": "replaced", "X-Adapter-Test": "yes"},
        max_response_bytes=max_response_bytes,
    )


def _install_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    client_type = httpx.AsyncClient

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        return client_type(**kwargs, transport=transport)

    monkeypatch.setattr("httpx.AsyncClient", create_client)


def _assert_headers(request: httpx.Request, kind: str) -> None:
    assert request.headers.get_list("authorization") == [f"Bearer {_TOKEN}"]
    assert request.headers["X-Adapter-Test"] == "yes"
    if kind == "openai":
        assert request.headers["OpenAI-Organization"] == "test-org"
        assert request.headers["OpenAI-Project"] == "test-project"


async def _complete(provider: OllamaProvider | OpenAICompatibleProvider) -> None:
    result = await provider.complete(
        (ChatMessage(role="user", content="answer"),),
        options=GenerationOptions(model="test-model"),
        structured_output=StructuredOutputRequest(
            name="Answer", schema={"type": "object", "properties": {"value": {"type": "string"}}}
        ),
    )
    assert result.structured == {"value": "ok"}


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_owned_clients_send_credentials_for_completion_and_readiness(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_headers(request, kind)
        path = request.url.path
        requests.append(path)
        payload: dict[str, Any]
        if path.endswith("/version"):
            payload = {"version": "0.6.0"}
        elif path.endswith("/show"):
            payload = {"model": "test-model"}
        elif path.endswith("/models"):
            payload = {"data": [{"id": "test-model"}]}
        elif kind == "ollama":
            payload = {"message": {"content": '{"value":"ok"}'}}
        else:
            payload = {"choices": [{"message": {"content": '{"value":"ok"}'}}]}
        return httpx.Response(200, json=payload)

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    caplog.set_level(logging.DEBUG)

    async def exercise() -> None:
        provider = _configured_provider(kind, monkeypatch)
        try:
            await _complete(provider)
            await provider.check_readiness()
            assert _TOKEN not in repr(provider)
            assert _TOKEN not in repr(provider._client)
        finally:
            await provider.aclose()

    asyncio.run(exercise())
    assert requests == (
        ["/api/chat", "/api/version", "/api/show"]
        if kind == "ollama"
        else ["/v1/chat/completions", "/v1/models"]
    )
    assert _TOKEN not in caplog.text


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_factory_credentials_reach_http_but_not_registry_diagnostics(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("CONDUCTO_ADAPTER_TEST_CREDENTIAL", _TOKEN)
    module = "ollama" if kind == "ollama" else "openai_compatible"
    monkeypatch.setattr(f"conducto.providers.{module}.require_adapter", lambda _: None)
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.extend(request.headers.get_list("authorization"))
        message = {"content": '{"value":"ok"}'}
        return httpx.Response(
            200,
            json={"message": message} if kind == "ollama" else {"choices": [{"message": message}]},
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    caplog.set_level(logging.DEBUG)
    registry = ProviderRegistry()
    registry.register_provider_type(
        module, OllamaProviderFactory() if kind == "ollama" else OpenAICompatibleProviderFactory()
    )
    registry.register_provider(
        "configured",
        provider_type=module,
        configuration=ProviderClientConfig(
            endpoint="http://adapter.invalid" if kind == "ollama" else "http://adapter.invalid/v1",
            credential_ref="CONDUCTO_ADAPTER_TEST_CREDENTIAL",
            provider_defaults={
                "model": "test-model",
                "profile": "terminal-json" if kind == "ollama" else "lm-studio-terminal-json",
            },
        ),
        model_configuration=ModelConfiguration(provider=module, model="test-model"),
    )
    assert _TOKEN not in repr(registry.snapshot())

    async def exercise() -> None:
        try:
            provider = registry.resolve("configured").client
            assert isinstance(provider, (OllamaProvider, OpenAICompatibleProvider))
            await _complete(provider)
        finally:
            await registry.aclose()

    asyncio.run(exercise())
    assert observed == [f"Bearer {_TOKEN}"]
    assert _TOKEN not in caplog.text


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_owned_clients_serialize_immutable_nested_tool_schemas(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
                "default": {"text": "literal data, not schema keywords"},
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    tool = ProviderToolDefinition(
        tool_id="lookup-id", name="lookup", description="Look up a query.", input_schema=schema
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_headers(request, kind)
        requests.append(json.loads(request.content))
        message = {"content": '{"value":"ok"}'}
        return httpx.Response(
            200,
            json={"message": message} if kind == "ollama" else {"choices": [{"message": message}]},
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async def exercise() -> None:
        provider = _configured_provider(kind, monkeypatch, tools=True)
        try:
            result = await provider.complete(
                (),
                options=GenerationOptions(model="test-model"),
                structured_output=StructuredOutputRequest(
                    name="Answer",
                    schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
                tools=(tool,),
            )
            assert result.structured == {"value": "ok"}
        finally:
            await provider.aclose()

    asyncio.run(exercise())
    assert requests[0]["tools"][0]["function"]["parameters"] == schema


@pytest.mark.parametrize("kind", ["ollama", "openai"])
@pytest.mark.parametrize("status", [-1, 0, 200, 401, 403, 404, 408, 429, 500])
def test_owned_client_errors_and_logged_tracebacks_never_echo_credentials(
    kind: str, status: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_headers(request, kind)
        if status == -1:
            raise httpx.ReadTimeout(_TOKEN, request=request)
        if status == 0:
            raise httpx.ConnectError(f"connection failed with {_TOKEN}", request=request)
        return httpx.Response(status, content=f"remote body echoes {_TOKEN}".encode())

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    error_types: dict[int, type[ProviderError]] = {
        -1: ProviderTimeoutError,
        0: ProviderEndpointUnavailableError,
        200: MalformedStructuredOutputError,
        401: ProviderAuthenticationError,
        403: ProviderAuthenticationError,
        404: OllamaModelNotFoundError if kind == "ollama" else OpenAICompatibleModelNotFoundError,
        408: ProviderTimeoutError,
        429: ProviderRateLimitError,
        500: ProviderEndpointUnavailableError,
    }

    async def exercise() -> None:
        provider = _configured_provider(kind, monkeypatch)
        try:
            with pytest.raises(error_types[status]) as raised:
                await _complete(provider)
            error = raised.value
            diagnostics = str(error) + repr(error) + "".join(traceback.format_exception(error))
            assert _TOKEN not in diagnostics
            logging.getLogger("conducto.adapter.regression").error(
                "Provider failed", exc_info=(type(error), error, error.__traceback__)
            )
        finally:
            await provider.aclose()

    asyncio.run(exercise())
    assert _TOKEN not in caplog.text


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_nested_schema_policy_stays_provider_specific(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": '{"value":"ok"}'}})

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    schema = StructuredOutputRequest(
        name="Answer",
        schema={
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"type": "string", "const": "ok"}},
        },
    )

    async def exercise() -> None:
        provider = _configured_provider(kind, monkeypatch)
        try:
            if kind == "openai":
                with pytest.raises(UnsupportedProviderCapabilityError):
                    await provider.complete(
                        (), options=GenerationOptions(model="test-model"), structured_output=schema
                    )
                assert not requests
            else:
                result = await provider.complete(
                    (), options=GenerationOptions(model="test-model"), structured_output=schema
                )
                assert result.structured == {"value": "ok"}
                assert len(requests) == 1
        finally:
            await provider.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_streamed_response_bound_stops_consumption_and_closes_stream(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ResponseStream(httpx.AsyncByteStream):
        closed = False
        consumed_tail = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"value":'
            yield b'"too large"}'
            self.consumed_tail = True
            yield b"never read"

        async def aclose(self) -> None:
            self.closed = True

    stream = ResponseStream()
    _install_transport(
        monkeypatch, httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    )

    async def exercise() -> None:
        provider = _configured_provider(kind, monkeypatch, max_response_bytes=10)
        try:
            error_type = (
                OllamaOversizedResponseError
                if kind == "ollama"
                else OpenAICompatibleOversizedResponseError
            )
            with pytest.raises(error_type):
                await _complete(provider)
        finally:
            await provider.aclose()

    asyncio.run(exercise())
    assert stream.closed
    assert not stream.consumed_tail


def test_shared_tool_cache_is_bounded_lru_and_instance_local() -> None:
    async def exercise() -> None:
        cache = ToolCallCache(limit=2)
        other = ToolCallCache(limit=2)
        await cache.remember("one", "first", {"role": "assistant"})
        await cache.remember("two", "second", {"role": "assistant"})
        assert await cache.lookup("one") is not None
        await cache.remember("three", "third", {"role": "assistant"})
        assert await cache.lookup("two") is None
        assert await cache.lookup("one") is not None
        assert await other.lookup("one") is None

    asyncio.run(exercise())


def test_deadline_merge_attenuates_timeout_without_wall_clock_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_response.time, "monotonic", lambda: 100.0)
    assert (
        _response.request_timeout(
            GenerationOptions(model="test-model", timeout=30),
            effective_deadline=110,
            call_context=ProviderCallContext(deadline=105),
        )
        == 5
    )
    with pytest.raises(ProviderTimeoutError) as raised:
        _response.request_timeout(
            GenerationOptions(model="test-model"),
            effective_deadline=100,
            call_context=None,
        )
    assert raised.value.attempted is False
