"""Ollama provider adapter for local Conducto model inference.

The adapter keeps the optional official ``ollama`` client behind the shared
Conducto provider protocol. It never starts or configures an Ollama daemon and
never pulls, deletes, or manages model weights.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from packaging.version import InvalidVersion, Version

from conducto.adapters import AdapterDependencyError, require_adapter
from conducto.core.provider import (
    ChatMessage,
    FinishReason,
    GenerationOptions,
    JsonSchemaDialect,
    MalformedStructuredOutputError,
    MessageContentPart,
    ModelProvider,
    ProviderAuthenticationError,
    ProviderCallContext,
    ProviderCancellationError,
    ProviderCapabilities,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderFailureCategory,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    Usage,
    validate_structured_output,
)
from conducto.core.provider_registry import ProviderClientConfig
from conducto.core.runtime_errors import ContradictoryProviderConfigurationError

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
TESTED_MINIMUM_OLLAMA_SERVER = "0.6.0"
TESTED_MINIMUM_OLLAMA_CLIENT = "0.5"
_MAX_ERROR_BODY_CHARS = 512
_DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
_CALL_CACHE_LIMIT = 1024
_TOOL_CAPABLE_PROFILE_NAMES = frozenset({"llama3.1-tools", "qwen2.5-tools"})
_SUPPORTED_SCHEMA_FEATURES = frozenset(
    {
        "type",
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
        "properties",
        "items",
        "required",
        "additionalProperties",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "$ref",
    }
)
_ALLOWED_SCHEMA_KEYWORDS = _SUPPORTED_SCHEMA_FEATURES | frozenset(
    {"$defs", "$schema", "title", "description", "default", "oneOf", "anyOf", "allOf"}
)


class _OllamaClient(Protocol):
    """Subset of the official Ollama async client used by the adapter."""

    async def chat(self, **kwargs: Any) -> Any:
        """Send one chat request to Ollama."""

    async def list(self) -> Any:
        """List locally available models."""

    async def show(self, model: str) -> Any:
        """Return local metadata for one model."""

    async def version(self) -> Any:
        """Return server version metadata."""


@dataclass(frozen=True, slots=True)
class OllamaProfile:
    """Pinned Ollama server/client/model capability profile.

    Attributes:
        name: Stable Conducto profile identifier.
        minimum_server_version: Minimum tested Ollama daemon version.
        minimum_client_version: Minimum tested official Python client version.
        tool_calling: Whether native tool calls are proven for this profile.
        context_limit: Advertised context window, or zero when unknown.
        output_limit: Optional maximum output token count.
    """

    name: str
    minimum_server_version: str = TESTED_MINIMUM_OLLAMA_SERVER
    minimum_client_version: str = TESTED_MINIMUM_OLLAMA_CLIENT
    tool_calling: bool = False
    context_limit: int = 0
    output_limit: int | None = None


OLLAMA_DEFAULT_PROFILE = OllamaProfile("terminal-json")
"""Default terminal-JSON profile that does not advertise native tools."""

OLLAMA_TOOL_CAPABLE_PROFILE = OllamaProfile(
    "llama3.1-tools",
    tool_calling=True,
)
"""Pinned opt-in profile for tested Ollama native tool-call behavior."""


class OllamaConfigurationError(ProviderError):
    """Ollama adapter configuration is contradictory or unsupported."""

    def __init__(self, message: str = "Ollama provider configuration is invalid") -> None:
        """Create a redacted configuration failure."""
        super().__init__(message, category=ProviderFailureCategory.CONFIGURATION)


class OllamaReadinessError(ProviderEndpointUnavailableError):
    """The Ollama server did not pass bounded readiness checks."""


class OllamaModelNotFoundError(ProviderEndpointUnavailableError):
    """The configured Ollama model is not available locally."""


class OllamaIncompatibleVersionError(ProviderError):
    """The Ollama server or client is outside the tested adapter profile."""

    def __init__(self, message: str = "Ollama version is incompatible") -> None:
        """Create a redacted incompatible-version failure."""
        super().__init__(message, category=ProviderFailureCategory.UNSUPPORTED_CAPABILITY)


class OllamaOversizedResponseError(ProviderError):
    """The Ollama response exceeded the configured bounded body size."""

    def __init__(self, message: str = "Ollama response exceeded the configured size limit") -> None:
        """Create a redacted oversized-response failure."""
        super().__init__(
            message,
            accepted=True,
            category=ProviderFailureCategory.PROTOCOL,
            attempted=True,
        )


@dataclass(frozen=True, slots=True)
class _CachedToolCall:
    """Local mapping used to round-trip provider tool result messages."""

    call_id: str
    name: str
    assistant_message: Mapping[str, Any]


class _OllamaHttpStatusError(RuntimeError):
    """Bounded HTTP error raised by the local Ollama protocol adapter."""

    def __init__(self, status_code: int, body: str) -> None:
        """Create an HTTP status error with bounded, local-only body text."""
        super().__init__(body)
        self.status_code = status_code


class _BoundedOllamaHttpClient:
    """Small bounded HTTP adapter for official-client gaps.

    The official Ollama client does not expose a response-body byte limit, so
    configuration-owned clients use this adapter for the same documented
    Ollama endpoints while keeping provider-specific mechanics behind the
    Conducto protocol.
    """

    def __init__(
        self,
        *,
        endpoint: str | None,
        auth_token: str | None,
        headers: Mapping[str, str],
        timeout: float | None,
        transport_options: Mapping[str, Any],
        tls_options: Mapping[str, Any],
        proxy_options: Mapping[str, Any],
        max_response_bytes: int,
    ) -> None:
        """Create a bounded HTTP client for one Ollama endpoint."""
        try:
            import httpx
        except Exception as error:
            raise OllamaConfigurationError("httpx is required for Ollama transport") from error
        self._max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(
            base_url=endpoint or DEFAULT_OLLAMA_ENDPOINT,
            **_official_client_kwargs(
                auth_token=auth_token,
                headers=headers,
                timeout=timeout,
                transport_options=transport_options,
                tls_options=tls_options,
                proxy_options=proxy_options,
            ),
        )

    async def chat(self, **kwargs: Any) -> Any:
        """Send one bounded chat request to Ollama."""
        return await self._request_json("POST", "/api/chat", json_body=kwargs)

    async def list(self) -> Any:
        """List locally available Ollama models."""
        return await self._request_json("GET", "/api/tags")

    async def show(self, model: str) -> Any:
        """Return local metadata for one model."""
        return await self._request_json("POST", "/api/show", json_body={"model": model})

    async def version(self) -> Any:
        """Return Ollama server version metadata."""
        return await self._request_json("GET", "/api/version")

    def close(self) -> None:
        """Release the underlying HTTP connection pool synchronously."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool asynchronously."""
        await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute one request while bounding the response body before decode."""
        async with self._client.stream(method, path, json=json_body) as response:
            body = await self._read_bounded(response)
            if response.status_code >= 400:
                text = body[:_MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace")
                raise _OllamaHttpStatusError(response.status_code, text)
            try:
                return json.loads(body)
            except json.JSONDecodeError as error:
                raise MalformedStructuredOutputError("Ollama returned malformed JSON") from error

    async def _read_bounded(self, response: Any) -> bytes:
        """Read a streaming HTTP response up to the configured byte bound."""
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                raise OllamaOversizedResponseError()
            chunks.append(chunk)
        return b"".join(chunks)


class OllamaProvider(ModelProvider):
    """Conducto provider backed by an existing Ollama server and model.

    Args:
        model: Default Ollama model name used when generation options do not
            override it.
        client: Preconstructed official-compatible async client. When supplied,
            endpoint, authentication, TLS, proxy, and transport settings must
            not also be supplied.
        endpoint: Ollama endpoint used for configuration-owned clients.
        auth_token: Optional bearer token for reverse proxies that protect
            Ollama. Prefer ``OllamaProviderFactory`` with ``credential_ref`` so
            secrets stay outside registry configuration.
        headers: Non-secret static headers to send with configuration-owned
            clients.
        profile: Tested capability profile. Tool calling is advertised only
            for explicit opt-in tool-capable profiles.
        keep_alive: Ollama keep-alive request value.
        context_window: Optional ``num_ctx`` value.
        max_output_tokens: Optional ``num_predict`` value.
        seed: Optional deterministic seed.
        sampling: Allowlisted provider sampling options.
        timeout: Optional client/request timeout in seconds.
        transport_options: Allowlisted official-client transport options.
        tls_options: Allowlisted TLS options.
        proxy_options: Allowlisted proxy options.
        provider_options: Narrowly allowlisted Ollama request options.

    Notes:
        The adapter does not start, stop, supervise, reconfigure, pull, or
        delete Ollama models. Readiness and model availability checks are
        explicit and bounded.
    """

    def __init__(
        self,
        *,
        model: str,
        client: _OllamaClient | None = None,
        endpoint: str | None = None,
        auth_token: str | None = None,
        headers: Mapping[str, str] | None = None,
        profile: OllamaProfile | str | None = None,
        keep_alive: str | int | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        seed: int | None = None,
        sampling: Mapping[str, int | float] | None = None,
        timeout: float | None = None,
        transport_options: Mapping[str, Any] | None = None,
        tls_options: Mapping[str, Any] | None = None,
        proxy_options: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        """Initialize one isolated Ollama provider client."""
        if not model.strip():
            raise OllamaConfigurationError("Ollama model cannot be blank")
        self._model = model.strip()
        self._profile = _coerce_profile(profile)
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=self._profile.tool_calling,
            context_limit=self._profile.context_limit,
            usage_reporting=True,
            streaming=False,
            cancellation=True,
            output_limit=self._profile.output_limit,
            schema_dialects=frozenset({JsonSchemaDialect.DRAFT_2020_12}),
            schema_features=_conducto_schema_features(),
        )
        self._keep_alive = keep_alive
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._seed = seed
        self._sampling = _validate_options(
            sampling or {},
            {"top_k", "top_p", "repeat_penalty", "presence_penalty", "frequency_penalty"},
            "sampling",
        )
        self._provider_options = _validate_options(
            provider_options or {},
            {"mirostat", "mirostat_eta", "mirostat_tau", "num_gpu", "num_thread"},
            "provider_options",
        )
        self._max_response_bytes = max_response_bytes
        if max_response_bytes <= 0:
            raise OllamaConfigurationError("max_response_bytes must be positive")
        self._owned_client = client is None
        self._closed = False
        self._tool_calls_by_id: OrderedDict[str, _CachedToolCall] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        if client is not None:
            if any(
                value is not None
                for value in (
                    endpoint,
                    auth_token,
                    headers,
                    timeout,
                    transport_options,
                    tls_options,
                    proxy_options,
                )
            ):
                raise ContradictoryProviderConfigurationError(
                    "Preconstructed Ollama client cannot be combined with endpoint, "
                    "transport, TLS, proxy, timeout, or authentication settings"
                )
            self._client = client
        else:
            self._client = _create_official_client(
                endpoint=endpoint,
                auth_token=auth_token,
                headers=headers or {},
                timeout=timeout,
                transport_options=transport_options or {},
                tls_options=tls_options or {},
                proxy_options=proxy_options or {},
                max_response_bytes=max_response_bytes,
            )

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
        """Generate one Ollama chat completion through Conducto contracts."""
        if self._closed:
            raise ProviderEndpointUnavailableError("Ollama provider is closed", attempted=False)
        if call_context is not None and call_context.cancelled:
            raise ProviderCancellationError()
        _assert_schema_supported(structured_output)
        if (tools or tool_results) and not self.capabilities.tool_calling:
            raise UnsupportedProviderCapabilityError(
                "Configured Ollama profile does not support native tool calling"
            )
        for tool in tools:
            _assert_schema_supported(
                StructuredOutputRequest(
                    name=f"{tool.name}_arguments",
                    schema=tool.input_schema,
                )
            )
        model = options.model or self._model
        request = {
            "model": model,
            "messages": await self._to_ollama_messages(messages, tool_results),
            "format": structured_output.json_schema,
            "options": self._build_options(options),
            "stream": False,
        }
        if self._keep_alive is not None:
            request["keep_alive"] = self._keep_alive
        if tools:
            request["tools"] = tuple(_to_ollama_tool(tool) for tool in tools)
        try:
            timeout_for_request = _request_timeout(
                options,
                effective_deadline=effective_deadline,
                call_context=call_context,
            )
            completion = self._client.chat(**request)
            if timeout_for_request is not None:
                response = await asyncio.wait_for(completion, timeout_for_request)
            else:
                response = await completion
        except asyncio.CancelledError as error:
            raise ProviderCancellationError() from error
        except ProviderError:
            raise
        except TimeoutError as error:
            raise ProviderTimeoutError(attempted=True) from error
        except Exception as error:
            raise _map_ollama_error(error) from error
        return await self._normalize_response(response, structured_output, tuple(tools))

    async def check_readiness(
        self,
        *,
        model: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Verify server reachability, tested version, and local model availability.

        Args:
            model: Model name to check. Defaults to this provider's configured model.
            timeout: Bound in seconds for each readiness operation.

        Raises:
            OllamaReadinessError: If the server cannot be queried.
            OllamaModelNotFoundError: If the model is not available locally.
            OllamaIncompatibleVersionError: If the server version is too old for
                this provider profile.
            ProviderAuthenticationError: If the server or proxy rejects credentials.
        """
        model_name = model or self._model
        try:
            version_payload = await asyncio.wait_for(self._client.version(), timeout)
        except TimeoutError as error:
            raise ProviderTimeoutError(attempted=True) from error
        except Exception as error:
            mapped = _map_ollama_error(error)
            if isinstance(mapped, ProviderAuthenticationError):
                raise mapped from error
            raise OllamaReadinessError("Ollama readiness check failed") from error
        server_version = _extract_version(version_payload)
        if server_version is not None and not _version_at_least(
            server_version,
            self._profile.minimum_server_version,
        ):
            raise OllamaIncompatibleVersionError("Ollama server version is unsupported")
        try:
            await asyncio.wait_for(self._client.show(model=model_name), timeout)
        except TimeoutError as error:
            raise ProviderTimeoutError(attempted=True) from error
        except Exception as error:
            mapped = _map_ollama_error(error)
            if isinstance(mapped, ProviderAuthenticationError):
                raise mapped from error
            if _looks_not_found(error):
                raise OllamaModelNotFoundError("Ollama model is not available locally") from error
            raise OllamaReadinessError("Ollama model readiness check failed") from error

    def close(self) -> None:
        """Release the owned official client if this provider constructed it."""
        if not self._owned_client or self._closed:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._closed = True

    async def aclose(self) -> None:
        """Release the owned official client through its async close hook."""
        if not self._owned_client or self._closed:
            return
        async_close = getattr(self._client, "aclose", None)
        if callable(async_close):
            await async_close()
        else:
            self.close()
        self._closed = True

    def _build_options(self, options: GenerationOptions) -> dict[str, Any]:
        """Build Ollama generation options from shared Conducto controls."""
        payload: dict[str, Any] = dict(self._sampling)
        payload.update(self._provider_options)
        payload["temperature"] = options.temperature
        if options.max_tokens is not None:
            payload["num_predict"] = options.max_tokens
        elif self._max_output_tokens is not None:
            payload["num_predict"] = self._max_output_tokens
        if self._context_window is not None:
            payload["num_ctx"] = self._context_window
        if self._seed is not None:
            payload["seed"] = self._seed
        if options.stop:
            payload["stop"] = list(options.stop)
        return payload

    async def _to_ollama_messages(
        self,
        messages: Sequence[ChatMessage],
        tool_results: Sequence[ToolResultMessage],
    ) -> list[dict[str, Any]]:
        """Translate Conducto messages and prior tool results."""
        translated = [
            {"role": message.role, "content": _message_content_to_text(message.content)}
            for message in messages
        ]
        for result in tool_results:
            cached = await self._lookup_tool_call(result.call_id)
            if cached is None:
                raise MalformedStructuredOutputError("Ollama tool result has unknown call ID")
            content = _safe_json_dumps(
                {"status": result.status, "result": result.result},
                "tool result",
            )
            translated.append(dict(cached.assistant_message))
            payload: dict[str, Any] = {
                "role": "tool",
                "content": content,
                "tool_name": cached.name,
            }
            translated.append(payload)
        return translated

    async def _normalize_response(
        self,
        response: Any,
        structured_output: StructuredOutputRequest,
        tools: tuple[ProviderToolDefinition, ...],
    ) -> ProviderResult:
        """Normalize an Ollama response without exposing client types."""
        response_map = _object_to_mapping(response)
        self._enforce_response_bound(response_map)
        message = _object_to_mapping(response_map.get("message", {}))
        content = message.get("content")
        tool_calls = _extract_tool_calls(message)
        request_id = _safe_string(response_map.get("id"))
        usage = _usage_from_response(response_map)
        finish_reason = _finish_reason(response_map)
        if tool_calls and _content_has_value(content):
            raise MalformedStructuredOutputError(
                "Ollama returned mixed terminal content and tool calls",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        if len(tool_calls) > 1:
            raise MalformedStructuredOutputError(
                "Ollama returned multiple tool calls",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        if tool_calls:
            call = await self._normalize_tool_call(tool_calls[0], message, tools, request_id)
            return ProviderResult(
                tool_calls=(call,),
                usage=usage,
                accepted=True,
                request_id=request_id,
                finish_reason=finish_reason,
            )
        if not isinstance(content, str):
            raise MalformedStructuredOutputError(
                "Ollama returned no terminal structured content",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        try:
            structured = json.loads(content)
        except json.JSONDecodeError as error:
            raise MalformedStructuredOutputError(
                "Ollama returned invalid terminal JSON",
                accepted=True,
                usage=usage,
                request_id=request_id,
            ) from error
        validate_structured_output(structured, structured_output)
        return ProviderResult(
            structured=structured,
            usage=usage,
            accepted=True,
            request_id=request_id,
            finish_reason=finish_reason,
        )

    async def _normalize_tool_call(
        self,
        raw_call: Mapping[str, Any],
        message: Mapping[str, Any],
        tools: tuple[ProviderToolDefinition, ...],
        request_id: str | None,
    ) -> ProviderToolCallRequest:
        """Resolve and validate one Ollama native tool call."""
        function = _object_to_mapping(raw_call.get("function", raw_call))
        name = _safe_string(function.get("name"))
        if name is None:
            raise MalformedStructuredOutputError("Ollama returned a tool call without a name")
        matches = [tool for tool in tools if tool.name == name]
        if not matches:
            raise MalformedStructuredOutputError("Ollama returned an unknown tool name")
        if len(matches) != 1:
            raise MalformedStructuredOutputError("Ollama returned an ambiguous tool call")
        arguments = _coerce_arguments(function.get("arguments"))
        tool = matches[0]
        validate_structured_output(
            arguments,
            StructuredOutputRequest(
                name=f"{tool.name}_arguments",
                schema=tool.input_schema,
            ),
        )
        call_id = _safe_string(raw_call.get("id")) or _synthesize_call_id(
            request_id=request_id,
            tool_id=tool.tool_id,
            name=name,
            arguments=arguments,
        )
        await self._remember_tool_call(
            call_id,
            name,
            _assistant_tool_message(message, raw_call, name, arguments),
        )
        return ProviderToolCallRequest(
            call_id=call_id,
            tool_id=tool.tool_id,
            arguments=arguments,
        )

    async def _remember_tool_call(
        self,
        call_id: str,
        name: str,
        assistant_message: Mapping[str, Any],
    ) -> None:
        """Cache a bounded call ID to tool-name mapping for result turns."""
        async with self._cache_lock:
            self._tool_calls_by_id[call_id] = _CachedToolCall(
                call_id,
                name,
                assistant_message,
            )
            self._tool_calls_by_id.move_to_end(call_id)
            while len(self._tool_calls_by_id) > _CALL_CACHE_LIMIT:
                self._tool_calls_by_id.popitem(last=False)

    async def _lookup_tool_call(self, call_id: str) -> _CachedToolCall | None:
        """Return the cached tool call associated with a result ID."""
        async with self._cache_lock:
            cached = self._tool_calls_by_id.get(call_id)
            if cached is not None:
                self._tool_calls_by_id.move_to_end(call_id)
            return cached

    def _enforce_response_bound(self, response: Mapping[str, Any]) -> None:
        """Reject oversized decoded response payloads."""
        size = len(json.dumps(response, ensure_ascii=True, default=str))
        if size > self._max_response_bytes:
            raise OllamaOversizedResponseError()


class OllamaProviderFactory:
    """Trusted factory for configuration-owned Ollama providers."""

    def create(self, configuration: ProviderClientConfig) -> OllamaProvider:
        """Construct an Ollama provider from one registry configuration source.

        Args:
            configuration: Provider registry configuration. ``provider_defaults``
                must include ``model``.

        Returns:
            A runtime-owned Ollama provider client.
        """
        defaults = dict(configuration.provider_defaults)
        model = defaults.pop("model", None)
        if not isinstance(model, str) or not model.strip():
            raise OllamaConfigurationError("Ollama provider_defaults must include model")
        auth_token = _token_from_credential_ref(configuration.credential_ref)
        profile = defaults.pop("profile", None)
        keep_alive = defaults.pop("keep_alive", None)
        context_window = _optional_positive_int(
            defaults.pop("context_window", None),
            "context_window",
        )
        max_output_tokens = _optional_positive_int(
            defaults.pop("max_output_tokens", None),
            "max_output_tokens",
        )
        seed = _optional_int(defaults.pop("seed", None), "seed")
        timeout = _optional_positive_float(defaults.pop("timeout", None), "timeout")
        max_response_bytes = _optional_positive_int(
            defaults.pop("max_response_bytes", _DEFAULT_MAX_RESPONSE_BYTES),
            "max_response_bytes",
        )
        sampling = _extract_prefixed(defaults, "sampling_")
        provider_options = _extract_prefixed(defaults, "option_")
        if defaults:
            raise OllamaConfigurationError(
                f"Unsupported Ollama provider defaults: {sorted(defaults)}"
            )
        assert max_response_bytes is not None
        return OllamaProvider(
            model=model,
            endpoint=configuration.endpoint,
            auth_token=auth_token,
            profile=cast(OllamaProfile | str | None, profile),
            keep_alive=cast(str | int | None, keep_alive),
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            seed=seed,
            sampling=cast(Mapping[str, int | float], sampling),
            timeout=timeout,
            transport_options=configuration.transport,
            tls_options=configuration.tls,
            proxy_options=configuration.proxy,
            provider_options=cast(Mapping[str, int | float], provider_options),
            max_response_bytes=max_response_bytes,
        )


ollama_provider_factory = OllamaProviderFactory()
"""Reusable singleton factory suitable for ``ProviderRegistry`` registration."""


def _create_official_client(
    *,
    endpoint: str | None,
    auth_token: str | None,
    headers: Mapping[str, str],
    timeout: float | None,
    transport_options: Mapping[str, Any],
    tls_options: Mapping[str, Any],
    proxy_options: Mapping[str, Any],
    max_response_bytes: int,
) -> _OllamaClient:
    """Construct a bounded client for Ollama's official HTTP protocol."""
    try:
        require_adapter("ollama")
    except AdapterDependencyError:
        raise
    except Exception as error:
        raise OllamaConfigurationError("Ollama client dependency is unavailable") from error
    return _BoundedOllamaHttpClient(
        endpoint=endpoint,
        auth_token=auth_token,
        headers=headers,
        timeout=timeout,
        transport_options=transport_options,
        tls_options=tls_options,
        proxy_options=proxy_options,
        max_response_bytes=max_response_bytes,
    )


def _official_client_kwargs(
    *,
    auth_token: str | None,
    headers: Mapping[str, str],
    timeout: float | None,
    transport_options: Mapping[str, Any],
    tls_options: Mapping[str, Any],
    proxy_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate allowlisted transport settings for the official client."""
    client_headers = dict(headers)
    if auth_token is not None:
        client_headers["Authorization"] = f"Bearer {auth_token}"
    kwargs: dict[str, Any] = {}
    if client_headers:
        kwargs["headers"] = client_headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    allowed_transport = {
        "follow_redirects",
        "max_connections",
        "max_keepalive_connections",
        "keepalive_expiry",
    }
    transport = _validate_options(transport_options, allowed_transport, "transport")
    if "follow_redirects" in transport:
        kwargs["follow_redirects"] = transport["follow_redirects"]
    limits = _limits_from_transport(transport_options)
    if limits is not None:
        kwargs["limits"] = limits
    for key, value in _validate_options(tls_options, {"verify", "cert"}, "tls").items():
        kwargs[key] = value
    for key, value in _validate_options(proxy_options, {"proxy"}, "proxy").items():
        kwargs[key] = value
    return kwargs


def _limits_from_transport(transport_options: Mapping[str, Any]) -> Any:
    """Build an httpx limits object only when pool bounds are configured."""
    allowed = {"max_connections", "max_keepalive_connections", "keepalive_expiry"}
    values = {key: transport_options[key] for key in allowed if key in transport_options}
    unsupported = set(transport_options) - (allowed | {"follow_redirects"})
    if unsupported:
        raise OllamaConfigurationError(
            f"Unsupported Ollama transport options: {sorted(unsupported)}"
        )
    if not values:
        return None
    try:
        import httpx
    except Exception as error:
        raise OllamaConfigurationError("httpx is required for Ollama transport limits") from error
    return httpx.Limits(**values)


def _validate_options(
    values: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> dict[str, Any]:
    """Return allowlisted options or fail explicitly."""
    unsupported = set(values) - allowed
    if unsupported:
        raise OllamaConfigurationError(f"Unsupported Ollama {label} options: {sorted(unsupported)}")
    return dict(values)


def _extract_prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Pop and return provider defaults with a common prefix stripped."""
    extracted: dict[str, Any] = {}
    for key in tuple(values):
        if key.startswith(prefix):
            extracted[key.removeprefix(prefix)] = values.pop(key)
    return extracted


def _coerce_profile(profile: OllamaProfile | str | None) -> OllamaProfile:
    """Normalize profile declarations."""
    if profile is None:
        return OLLAMA_DEFAULT_PROFILE
    if isinstance(profile, OllamaProfile):
        if profile.tool_calling and profile.name not in _TOOL_CAPABLE_PROFILE_NAMES:
            raise OllamaConfigurationError(
                f"Ollama tool profile '{profile.name}' has not been conformance tested"
            )
        return profile
    if profile in _TOOL_CAPABLE_PROFILE_NAMES:
        return OllamaProfile(profile, tool_calling=True)
    if profile == OLLAMA_DEFAULT_PROFILE.name:
        return OLLAMA_DEFAULT_PROFILE
    raise OllamaConfigurationError(f"Unknown Ollama profile '{profile}'")


def _conducto_schema_features() -> frozenset[Any]:
    """Return Conducto schema features supported by Ollama JSON format."""
    from conducto.core.provider import SchemaFeature

    return frozenset(
        feature for feature in SchemaFeature if feature.value in _SUPPORTED_SCHEMA_FEATURES
    )


def _assert_schema_supported(request: StructuredOutputRequest) -> None:
    """Reject schema features not sent to Ollama's native ``format`` field."""
    if request.dialect is not JsonSchemaDialect.DRAFT_2020_12:
        raise UnsupportedProviderCapabilityError("Ollama profile supports Draft 2020-12 schemas")
    unknown = _unknown_schema_keywords(request.json_schema)
    if unknown:
        raise UnsupportedProviderCapabilityError(
            f"Ollama schema uses unsupported keywords: {sorted(unknown)!r}"
        )
    features = {feature.value for feature in request.features}
    features.update(_schema_features(request.json_schema))
    unsupported = features - _SUPPORTED_SCHEMA_FEATURES
    if unsupported:
        raise UnsupportedProviderCapabilityError(
            f"Ollama schema uses unsupported features: {sorted(unsupported)!r}"
        )


def _schema_features(schema: Mapping[str, Any]) -> set[str]:
    """Collect JSON Schema keywords used by a terminal or tool schema."""
    features: set[str] = set()
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            features.add(value)
        if key in _SUPPORTED_SCHEMA_FEATURES or key in {"oneOf", "anyOf", "allOf"}:
            features.add(key)
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, Mapping):
                    features.update(_schema_features(child))
        elif isinstance(value, Mapping):
            features.update(_schema_features(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    features.update(_schema_features(item))
    return features


def _unknown_schema_keywords(schema: Mapping[str, Any]) -> set[str]:
    """Collect JSON Schema keywords outside the adapter's supported subset."""
    unknown: set[str] = set()
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYWORDS and not key.startswith("x-"):
            unknown.add(str(key))
        if key in {"properties", "$defs"} and isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, Mapping):
                    unknown.update(_unknown_schema_keywords(child))
        elif isinstance(value, Mapping):
            unknown.update(_unknown_schema_keywords(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    unknown.update(_unknown_schema_keywords(item))
    return unknown


def _to_ollama_tool(tool: ProviderToolDefinition) -> dict[str, Any]:
    """Translate one Conducto tool descriptor to Ollama's native shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _request_timeout(
    options: GenerationOptions,
    *,
    effective_deadline: float | None,
    call_context: ProviderCallContext | None,
) -> float | None:
    """Return the per-call timeout after applying all known deadlines."""
    timeout = options.timeout
    deadline = effective_deadline
    if call_context is not None and call_context.deadline is not None:
        deadline = (
            min(deadline, call_context.deadline) if deadline is not None else call_context.deadline
        )
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError(attempted=False)
        timeout = min(timeout, remaining) if timeout is not None else remaining
    return timeout


def _assistant_tool_message(
    message: Mapping[str, Any],
    raw_call: Mapping[str, Any],
    name: str,
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the assistant tool-call message Ollama expects before results."""
    content = message.get("content")
    call_payload: dict[str, Any] = {
        "function": {
            "name": name,
            "arguments": dict(arguments),
        }
    }
    raw_id = _safe_string(raw_call.get("id"))
    if raw_id is not None:
        call_payload["id"] = raw_id
    return {
        "role": "assistant",
        "content": content if isinstance(content, str) else "",
        "tool_calls": (call_payload,),
    }


def _message_content_to_text(content: str | tuple[MessageContentPart, ...]) -> str:
    """Translate supported Conducto message content to Ollama text content."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.type == "text" and part.text is not None:
            parts.append(part.text)
        elif part.type == "json" and part.value is not None:
            parts.append(_safe_json_dumps(part.value, "message content"))
    return "\n".join(parts)


def _safe_json_dumps(value: Any, label: str) -> str:
    """Serialize bounded provider payloads without custom object fallback."""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError as error:
        raise MalformedStructuredOutputError(f"Ollama {label} is not JSON serializable") from error


def _object_to_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce official response objects or plain dicts to mappings."""
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, Any], dumped)
    if hasattr(value, "__dict__"):
        return cast(Mapping[str, Any], vars(value))
    return {}


def _extract_tool_calls(message: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return normalized tool-call dictionaries from an Ollama message."""
    raw = message.get("tool_calls", ())
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MalformedStructuredOutputError("Ollama returned malformed tool calls")
    calls: list[Mapping[str, Any]] = []
    for item in raw:
        mapped = _object_to_mapping(item)
        if not mapped:
            raise MalformedStructuredOutputError("Ollama returned malformed tool call")
        calls.append(mapped)
    return tuple(calls)


def _content_has_value(content: Any) -> bool:
    """Return whether provider message content carries terminal data."""
    return isinstance(content, str) and bool(content.strip())


def _coerce_arguments(value: Any) -> Mapping[str, Any]:
    """Decode Ollama tool-call arguments into an immutable-free mapping."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise MalformedStructuredOutputError(
                "Ollama returned malformed tool arguments"
            ) from error
    if not isinstance(value, Mapping):
        raise MalformedStructuredOutputError("Ollama returned non-object tool arguments")
    return dict(value)


def _synthesize_call_id(
    *,
    request_id: str | None,
    tool_id: str,
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    """Create a stable call ID for Ollama protocols that omit IDs."""
    argument_text = json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    seed = f"{request_id or ''}\0{tool_id}\0{name}\0{argument_text}"
    return f"ollama_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"


def _usage_from_response(response: Mapping[str, Any]) -> Usage:
    """Map Ollama-reported token counters without estimating missing values."""
    input_tokens = _optional_non_negative_int(response.get("prompt_eval_count"))
    output_tokens = _optional_non_negative_int(response.get("eval_count"))
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _finish_reason(response: Mapping[str, Any]) -> str:
    """Return provider-neutral finish metadata."""
    raw = _safe_string(response.get("done_reason")) or _safe_string(response.get("finish_reason"))
    if raw in {"stop", "length", "content_filter", "tool_call"}:
        return raw
    if raw == "load":
        return FinishReason.UNKNOWN.value
    return raw or FinishReason.UNKNOWN.value


def _map_ollama_error(error: Exception) -> ProviderError:
    """Map official-client exceptions to redacted Conducto provider errors."""
    status = getattr(error, "status_code", None)
    message = _bounded_error_text(error)
    if status in {401, 403} or "unauthorized" in message or "forbidden" in message:
        return ProviderAuthenticationError()
    if status == 404 or "not found" in message:
        return OllamaModelNotFoundError("Ollama model is not available locally")
    if status == 408 or "timeout" in message:
        return ProviderTimeoutError(attempted=True)
    if status == 429:
        return ProviderRateLimitError()
    if "connection refused" in message or "connect" in message:
        return ProviderEndpointUnavailableError("Ollama endpoint is unavailable")
    return ProviderEndpointUnavailableError("Ollama request failed")


def _bounded_error_text(error: Exception) -> str:
    """Return a lower-case bounded error diagnostic for classification only."""
    text = str(error).lower()
    return text[:_MAX_ERROR_BODY_CHARS]


def _looks_not_found(error: Exception) -> bool:
    """Return whether an exception appears to represent missing local model state."""
    return isinstance(_map_ollama_error(error), OllamaModelNotFoundError)


def _extract_version(payload: Any) -> str | None:
    """Extract the server version from official-client response shapes."""
    mapped = _object_to_mapping(payload)
    return _safe_string(mapped.get("version"))


def _version_at_least(actual: str, minimum: str) -> bool:
    """Compare provider versions using packaging semantics."""
    try:
        return Version(actual) >= Version(minimum)
    except InvalidVersion:
        return False


def _token_from_credential_ref(credential_ref: str | None) -> str | None:
    """Resolve an opaque credential reference through the process environment."""
    if credential_ref is None:
        return None
    value = os.environ.get(credential_ref)
    if value is None:
        raise OllamaConfigurationError("Ollama credential reference is not available")
    if not value.strip():
        raise OllamaConfigurationError("Ollama credential reference is empty")
    return value


def _redacted_endpoint(endpoint: str) -> str:
    """Return an endpoint string with query secrets removed for docs and diagnostics."""
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _safe_string(value: Any) -> str | None:
    """Return a non-empty string value when available."""
    return value if isinstance(value, str) and value.strip() else None


def _optional_int(value: Any, label: str) -> int | None:
    """Validate optional integer provider settings."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise OllamaConfigurationError(f"{label} must be an integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    """Validate optional positive integer provider settings."""
    result = _optional_int(value, label)
    if result is not None and result <= 0:
        raise OllamaConfigurationError(f"{label} must be positive")
    return result


def _optional_positive_float(value: Any, label: str) -> float | None:
    """Validate optional positive float provider settings."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise OllamaConfigurationError(f"{label} must be positive")
    return float(value)


def _optional_non_negative_int(value: Any) -> int | None:
    """Return a non-negative token counter or ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
