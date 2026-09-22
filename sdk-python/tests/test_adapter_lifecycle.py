"""Deterministic asynchronous adapter ownership and factory regressions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import Any

import httpx
import pytest

from conducto.core.provider import (
    GenerationOptions,
    ProviderEndpointUnavailableError,
    ProviderError,
    StructuredOutputRequest,
)
from conducto.core.provider_registry import ProviderClientConfig
from conducto.providers import (
    OllamaConfigurationError,
    OllamaProvider,
    OllamaProviderFactory,
    OpenAICompatibleConfigurationError,
    OpenAICompatibleProvider,
    OpenAICompatibleProviderFactory,
)


def _provider(kind: str, *, client: Any = None) -> OllamaProvider | OpenAICompatibleProvider:
    if kind == "ollama":
        return OllamaProvider(model="test-model", client=client)
    return OpenAICompatibleProvider(
        model="test-model", profile="lm-studio-terminal-json", client=client
    )


def _adapter_module(kind: str) -> ModuleType:
    return import_module(
        f"conducto.providers.{'ollama' if kind == 'ollama' else 'openai_compatible'}"
    )


async def _complete(provider: OllamaProvider | OpenAICompatibleProvider) -> None:
    result = await provider.complete(
        (),
        options=GenerationOptions(model="test-model"),
        structured_output=StructuredOutputRequest(name="Answer", schema={"type": "object"}),
    )
    assert result.structured == {}


class _TrackedTransport(httpx.AsyncBaseTransport):
    """Track actual HTTP-client shutdown without network access or sleeps."""

    def __init__(self, kind: str, *, block_close: bool = False) -> None:
        self.kind = kind
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.release_close = asyncio.Event()
        if not block_close:
            self.release_close.set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Return an adapter-shaped terminal response."""
        message = {"content": "{}"}
        payload = (
            {"message": message} if self.kind == "ollama" else {"choices": [{"message": message}]}
        )
        return httpx.Response(200, json=payload)

    async def aclose(self) -> None:
        """Count shutdown attempts and optionally wait for deterministic release."""
        self.close_calls += 1
        self.close_started.set()
        await self.release_close.wait()
        self.close_finished.set()


def _install_http_client(
    kind: str, monkeypatch: pytest.MonkeyPatch, transport: _TrackedTransport
) -> list[httpx.AsyncClient]:
    clients: list[httpx.AsyncClient] = []
    client_type = httpx.AsyncClient

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        client = client_type(**kwargs, transport=transport)
        clients.append(client)
        return client

    monkeypatch.setattr("httpx.AsyncClient", create_client)
    monkeypatch.setitem(vars(_adapter_module(kind)), "require_adapter", lambda _: None)
    return clients


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_owned_adapter_has_only_async_shutdown_and_releases_http_pool(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        transport = _TrackedTransport(kind)
        clients = _install_http_client(kind, monkeypatch, transport)
        provider = _provider(kind)
        assert not hasattr(provider, "close")
        assert not hasattr(provider._client, "close")
        assert not clients[0].is_closed
        await _complete(provider)
        await provider.aclose()
        await provider.aclose()
        assert clients[0].is_closed
        assert transport.close_calls == 1
        with pytest.raises(ProviderEndpointUnavailableError) as raised:
            await _complete(provider)
        assert raised.value.attempted is False

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_borrowed_adapter_shutdown_does_not_close_or_disable_client(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        transport = _TrackedTransport(kind)
        clients = _install_http_client(kind, monkeypatch, transport)
        owner = _provider(kind)
        borrowed = _provider(kind, client=owner._client)
        try:
            await borrowed.aclose()
            await borrowed.aclose()
            assert transport.close_calls == 0
            assert not clients[0].is_closed
            await _complete(borrowed)
        finally:
            await owner.aclose()
        assert transport.close_calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_concurrent_shutdown_waits_for_one_owned_close(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        transport = _TrackedTransport(kind, block_close=True)
        _install_http_client(kind, monkeypatch, transport)
        provider = _provider(kind)
        first = asyncio.create_task(provider.aclose())
        await transport.close_started.wait()
        second_started = asyncio.Event()

        async def close_again() -> None:
            second_started.set()
            await provider.aclose()

        second = asyncio.create_task(close_again())
        try:
            await second_started.wait()
            assert not second.done()
            assert transport.close_calls == 1
        finally:
            transport.release_close.set()
            await asyncio.gather(first, second)
        assert transport.close_calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_cancelled_shutdown_waiter_does_not_abandon_the_http_pool(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        transport = _TrackedTransport(kind, block_close=True)
        _install_http_client(kind, monkeypatch, transport)
        provider = _provider(kind)
        closing = asyncio.create_task(provider.aclose())
        await transport.close_started.wait()
        closing.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await closing
        finally:
            transport.release_close.set()
        await provider.aclose()
        assert transport.close_finished.is_set()
        assert transport.close_calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_http_pool_cleanup_failure_remains_explicit_on_later_shutdown(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = ProviderEndpointUnavailableError("Pool shutdown failed")

    class FailingTransport(_TrackedTransport):
        async def aclose(self) -> None:
            self.close_calls += 1
            raise failure

    async def exercise() -> None:
        transport = FailingTransport(kind)
        _install_http_client(kind, monkeypatch, transport)
        provider = _provider(kind)
        for _ in range(2):
            with pytest.raises(ProviderEndpointUnavailableError) as raised:
                await provider.aclose()
            assert raised.value is failure
        assert transport.close_calls == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
@pytest.mark.parametrize("failure", ["typed", "cancelled"])
def test_failed_owned_shutdown_propagates_and_can_be_retried(
    kind: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        error = ProviderEndpointUnavailableError("Shutdown failed")
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class Client:
            close_calls = 0

            async def aclose(self) -> None:
                self.close_calls += 1
                close_started.set()
                if self.close_calls == 1:
                    if failure == "typed":
                        raise error
                    await release_close.wait()

        client = Client()
        monkeypatch.setitem(
            vars(_adapter_module(kind)), "_create_bounded_client", lambda **_: client
        )
        provider = _provider(kind)
        if failure == "typed":
            with pytest.raises(ProviderEndpointUnavailableError) as raised:
                await provider.aclose()
            assert raised.value is error
        else:
            closing = asyncio.create_task(provider.aclose())
            await close_started.wait()
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
        await provider.aclose()
        await provider.aclose()
        assert client.close_calls == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_owned_client_without_async_shutdown_is_a_typed_configuration_failure(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(vars(_adapter_module(kind)), "_create_bounded_client", lambda **_: object())
    error_type = (
        OllamaConfigurationError if kind == "ollama" else OpenAICompatibleConfigurationError
    )
    with pytest.raises(error_type, match="asynchronous shutdown"):
        _provider(kind)


def _factory_configuration(kind: str, defaults: Mapping[str, Any]) -> ProviderClientConfig:
    values = {"model": "test-model", **defaults}
    if kind == "openai":
        values["profile"] = "lm-studio-terminal-json"
    return ProviderClientConfig(provider_defaults=values)


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_factories_share_generation_default_parsing(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class Client:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    def create_client(**kwargs: Any) -> Client:
        captured.update(kwargs)
        return Client()

    monkeypatch.setitem(vars(_adapter_module(kind)), "_create_bounded_client", create_client)
    factory = OllamaProviderFactory() if kind == "ollama" else OpenAICompatibleProviderFactory()
    provider = factory.create(
        _factory_configuration(
            kind,
            {
                "context_window": 1000,
                "max_output_tokens": 100,
                "seed": -1,
                "timeout": 4,
                "max_response_bytes": 2048,
                "sampling_top_p": 0.8,
            },
        )
    )
    if isinstance(provider, OllamaProvider):
        assert provider._context_window == 1000
    else:
        assert provider.capabilities.context_limit == 1000
    assert provider._max_output_tokens == 100
    assert provider._seed == -1
    assert provider._sampling == {"top_p": 0.8}
    assert captured["timeout"] == 4.0
    assert captured["max_response_bytes"] == 2048
    asyncio.run(provider.aclose())


@pytest.mark.parametrize("kind", ["ollama", "openai"])
@pytest.mark.parametrize(
    "defaults",
    [
        {"context_window": False},
        {"max_output_tokens": 0},
        {"seed": "invalid"},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": 10**400},
        {"max_response_bytes": None},
        {"max_response_bytes": 0},
        {"unknown": True},
    ],
)
def test_factories_preserve_typed_default_validation(kind: str, defaults: dict[str, Any]) -> None:
    factory = OllamaProviderFactory() if kind == "ollama" else OpenAICompatibleProviderFactory()
    error_type: type[ProviderError] = (
        OllamaConfigurationError if kind == "ollama" else OpenAICompatibleConfigurationError
    )
    with pytest.raises(error_type):
        factory.create(_factory_configuration(kind, defaults))
