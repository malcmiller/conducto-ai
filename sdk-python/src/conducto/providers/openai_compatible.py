"""OpenAI-compatible provider adapter for tested vLLM and LM Studio servers.

The adapter keeps the OpenAI wire protocol (``/chat/completions`` and
``/models``) behind the shared Conducto provider, structured-output,
lifecycle, ownership, and native-tool contracts. "OpenAI-compatible" does not
imply universal compatibility: only the pinned :class:`OpenAICompatibleProfile`
instances exported from this module have been conformance tested, and the
adapter refuses to advertise capabilities for an unrecognized custom profile
that claims native tool support. The adapter never starts, stops, configures,
or supervises a vLLM or LM Studio process, and never downloads or manages
model weights.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

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

DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"
"""Documented default local vLLM OpenAI-compatible endpoint."""

DEFAULT_LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"
"""Documented default local LM Studio OpenAI-compatible endpoint."""

TESTED_MINIMUM_VLLM_SERVER = "0.6.0"
TESTED_MINIMUM_LM_STUDIO_SERVER = "0.3.0"
TESTED_MINIMUM_OPENAI_CLIENT = "1.0"
_MAX_ERROR_BODY_CHARS = 512
_DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
_CALL_CACHE_LIMIT = 1024
_TOOL_CAPABLE_PROFILE_NAMES = frozenset({"vllm-tools", "lm-studio-tools"})

#: JSON Schema keywords Conducto both sends natively and validates locally for
#: this adapter. This subset is intentionally narrower than other adapters
#: because OpenAI-shaped ``strict`` JSON Schema response formats reject
#: keywords such as ``const``, length/item bounds, and schema combinators.
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
        "$ref",
    }
)
_ALLOWED_SCHEMA_KEYWORDS = _SUPPORTED_SCHEMA_FEATURES | frozenset(
    {"$defs", "$schema", "title", "description", "default"}
)
_SCHEMA_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


class _ChatCompletionsClient(Protocol):
    """Subset of the official client's ``chat.completions`` namespace."""

    async def create(self, **kwargs: Any) -> Any:
        """Send one chat completion request."""


class _ChatClient(Protocol):
    """Subset of the official client's ``chat`` namespace."""

    completions: _ChatCompletionsClient


class _ModelsClient(Protocol):
    """Subset of the official client's ``models`` namespace."""

    async def list(self) -> Any:
        """List models available on the server."""


class _OpenAICompatibleClient(Protocol):
    """Subset of the official OpenAI async client used by the adapter."""

    chat: _ChatClient
    models: _ModelsClient


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProfile:
    """Pinned server/model capability profile for one tested server family.

    Attributes:
        name: Stable Conducto profile identifier.
        server_family: Tested server family, either ``"vllm"`` or ``"lm_studio"``.
        minimum_server_version: Minimum tested server version, when the family
            exposes a documented version check.
        minimum_client_version: Minimum tested official ``openai`` client version.
        tool_calling: Whether native tool calls are proven for this profile.
        strict_schema: Whether the profile sends OpenAI ``strict`` JSON Schema
            response formats.
        context_limit: Advertised context window, or zero when unknown.
        output_limit: Optional maximum output token count.
        version_path: Server-relative path used for an optional bounded version
            readiness check, or ``None`` when the family exposes no documented
            version endpoint.
    """

    name: str
    server_family: str
    minimum_server_version: str | None = None
    minimum_client_version: str = TESTED_MINIMUM_OPENAI_CLIENT
    tool_calling: bool = False
    strict_schema: bool = True
    context_limit: int = 0
    output_limit: int | None = None
    version_path: str | None = None


VLLM_DEFAULT_PROFILE = OpenAICompatibleProfile(
    "vllm-terminal-json",
    server_family="vllm",
    minimum_server_version=TESTED_MINIMUM_VLLM_SERVER,
    version_path="/version",
)
"""Tested vLLM profile that advertises terminal JSON Schema output only."""

VLLM_TOOL_CAPABLE_PROFILE = OpenAICompatibleProfile(
    "vllm-tools",
    server_family="vllm",
    minimum_server_version=TESTED_MINIMUM_VLLM_SERVER,
    tool_calling=True,
    version_path="/version",
)
"""Pinned opt-in profile for tested vLLM native tool-call behavior."""

LM_STUDIO_DEFAULT_PROFILE = OpenAICompatibleProfile(
    "lm-studio-terminal-json",
    server_family="lm_studio",
    minimum_server_version=TESTED_MINIMUM_LM_STUDIO_SERVER,
)
"""Tested LM Studio profile that advertises terminal JSON Schema output only."""

LM_STUDIO_TOOL_CAPABLE_PROFILE = OpenAICompatibleProfile(
    "lm-studio-tools",
    server_family="lm_studio",
    minimum_server_version=TESTED_MINIMUM_LM_STUDIO_SERVER,
    tool_calling=True,
)
"""Pinned opt-in profile for tested LM Studio native tool-call behavior.

Notes:
    LM Studio does not expose a documented version endpoint, so this profile
    leaves ``version_path`` unset. Readiness still verifies the configured
    model is listed by the server.
"""

_KNOWN_PROFILES: dict[str, OpenAICompatibleProfile] = {
    profile.name: profile
    for profile in (
        VLLM_DEFAULT_PROFILE,
        VLLM_TOOL_CAPABLE_PROFILE,
        LM_STUDIO_DEFAULT_PROFILE,
        LM_STUDIO_TOOL_CAPABLE_PROFILE,
    )
}


class OpenAICompatibleConfigurationError(ProviderError):
    """OpenAI-compatible adapter configuration is contradictory or unsupported."""

    def __init__(
        self,
        message: str = "OpenAI-compatible provider configuration is invalid",
    ) -> None:
        """Create a redacted configuration failure."""
        super().__init__(message, category=ProviderFailureCategory.CONFIGURATION)


class OpenAICompatibleReadinessError(ProviderEndpointUnavailableError):
    """The configured server did not pass bounded readiness checks."""


class OpenAICompatibleModelNotFoundError(ProviderEndpointUnavailableError):
    """The configured model is not available on the server."""


class OpenAICompatibleIncompatibleVersionError(ProviderError):
    """The server or client is outside the tested adapter profile."""

    def __init__(self, message: str = "OpenAI-compatible server version is incompatible") -> None:
        """Create a redacted incompatible-version failure."""
        super().__init__(message, category=ProviderFailureCategory.UNSUPPORTED_CAPABILITY)


class OpenAICompatibleOversizedResponseError(ProviderError):
    """The server response exceeded the configured bounded body size."""

    def __init__(
        self,
        message: str = "OpenAI-compatible response exceeded the configured size limit",
    ) -> None:
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


class _OpenAICompatibleHttpStatusError(RuntimeError):
    """Bounded HTTP error raised by the local protocol adapter."""

    def __init__(self, status_code: int, body: str) -> None:
        """Create an HTTP status error with bounded, local-only body text."""
        super().__init__(body)
        self.status_code = status_code


class _BoundedOpenAICompatibleHttpClient:
    """Small bounded HTTP adapter mirroring the official client's shape.

    Configuration-owned clients use this transport for the same documented
    OpenAI-compatible endpoints because the official client does not expose a
    response-body byte limit. Preconstructed clients (including the real
    ``openai.AsyncOpenAI``) satisfy :class:`_OpenAICompatibleClient` directly
    and never pass through this bounded transport.
    """

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        organization: str | None,
        project: str | None,
        headers: Mapping[str, str],
        timeout: float | None,
        transport_options: Mapping[str, Any],
        tls_options: Mapping[str, Any],
        proxy_options: Mapping[str, Any],
        max_response_bytes: int,
    ) -> None:
        """Create a bounded HTTP client for one OpenAI-compatible endpoint."""
        try:
            import httpx
        except Exception as error:
            raise OpenAICompatibleConfigurationError(
                "httpx is required for OpenAI-compatible transport"
            ) from error
        self._max_response_bytes = max_response_bytes
        base_url = (endpoint or DEFAULT_VLLM_ENDPOINT).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            **_official_client_kwargs(
                api_key=api_key,
                organization=organization,
                project=project,
                headers=headers,
                timeout=timeout,
                transport_options=transport_options,
                tls_options=tls_options,
                proxy_options=proxy_options,
            ),
        )
        self.chat: _ChatClient = _ChatNamespace(self)
        self.models: _ModelsClient = _ModelsNamespace(self)

    async def get_version(self, path: str) -> Any:
        """Return a bounded, decoded response from a server version endpoint."""
        return await self._request_json("GET", path)

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
                raise _OpenAICompatibleHttpStatusError(response.status_code, text)
            try:
                return json.loads(body)
            except json.JSONDecodeError as error:
                raise MalformedStructuredOutputError(
                    "OpenAI-compatible server returned malformed JSON"
                ) from error

    async def _read_bounded(self, response: Any) -> bytes:
        """Read a streaming HTTP response up to the configured byte bound."""
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                raise OpenAICompatibleOversizedResponseError()
            chunks.append(chunk)
        return b"".join(chunks)


class _ChatCompletionsNamespace:
    """Bounded ``chat.completions`` namespace for the local HTTP transport."""

    def __init__(self, outer: _BoundedOpenAICompatibleHttpClient) -> None:
        """Bind this namespace to its owning bounded transport."""
        self._outer = outer

    async def create(self, **kwargs: Any) -> Any:
        """Send one bounded chat completion request."""
        return await self._outer._request_json("POST", "/chat/completions", json_body=kwargs)


class _ChatNamespace:
    """Bounded ``chat`` namespace for the local HTTP transport."""

    def __init__(self, outer: _BoundedOpenAICompatibleHttpClient) -> None:
        """Bind this namespace to its owning bounded transport."""
        self.completions: _ChatCompletionsClient = _ChatCompletionsNamespace(outer)


class _ModelsNamespace:
    """Bounded ``models`` namespace for the local HTTP transport."""

    def __init__(self, outer: _BoundedOpenAICompatibleHttpClient) -> None:
        """Bind this namespace to its owning bounded transport."""
        self._outer = outer

    async def list(self) -> Any:
        """List models available on the server."""
        return await self._outer._request_json("GET", "/models")


class OpenAICompatibleProvider(ModelProvider):
    """Conducto provider backed by an existing vLLM or LM Studio server.

    Args:
        model: Default model name used when generation options do not
            override it.
        profile: Tested capability profile. Tool calling is advertised only
            for explicit opt-in tool-capable profiles.
        client: Preconstructed official-client-compatible async client (for
            example, ``openai.AsyncOpenAI``). When supplied, endpoint,
            authentication, headers, TLS, proxy, and transport settings must
            not also be supplied.
        endpoint: Server endpoint used for configuration-owned clients.
        api_key: Optional bearer token. Prefer
            ``OpenAICompatibleProviderFactory`` with ``credential_ref`` so
            secrets stay outside registry configuration.
        organization: Optional ``OpenAI-Organization`` header value.
        project: Optional ``OpenAI-Project`` header value.
        headers: Non-secret static headers to send with configuration-owned
            clients.
        context_window: Optional advisory context window used only for
            advertised capabilities; the OpenAI-shaped chat completions
            request does not send it.
        max_output_tokens: Optional ``max_tokens`` value.
        seed: Optional deterministic seed.
        sampling: Allowlisted provider sampling options (``top_p``,
            ``presence_penalty``, ``frequency_penalty``).
        timeout: Optional client/request timeout in seconds.
        transport_options: Allowlisted official-client transport options.
        tls_options: Allowlisted TLS options.
        proxy_options: Allowlisted proxy options.
        provider_options: Narrowly allowlisted request options (``logprobs``,
            ``top_logprobs``).
        max_response_bytes: Bound applied to configuration-owned clients only.

    Notes:
        The adapter does not start, stop, supervise, reconfigure, or manage
        model weights for vLLM or LM Studio. Readiness and model availability
        checks are explicit and bounded. Native tool turns always send
        ``parallel_tool_calls: False`` because Conducto does not support
        parallel or multiple tool calls in one provider turn.
    """

    def __init__(
        self,
        *,
        model: str,
        profile: OpenAICompatibleProfile | str,
        client: _OpenAICompatibleClient | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        headers: Mapping[str, str] | None = None,
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
        """Initialize one isolated OpenAI-compatible provider client."""
        if not model.strip():
            raise OpenAICompatibleConfigurationError("Model cannot be blank")
        self._model = model.strip()
        self._profile = _coerce_profile(profile)
        self.capabilities = ProviderCapabilities(
            structured_output=True,
            tool_calling=self._profile.tool_calling,
            context_limit=context_window
            if context_window is not None
            else self._profile.context_limit,
            usage_reporting=True,
            streaming=False,
            cancellation=True,
            output_limit=max_output_tokens or self._profile.output_limit,
            schema_dialects=frozenset({JsonSchemaDialect.DRAFT_2020_12}),
            schema_features=_conducto_schema_features(),
        )
        self._max_output_tokens = max_output_tokens
        self._seed = seed
        self._sampling = _validate_options(
            sampling or {},
            {"top_p", "presence_penalty", "frequency_penalty"},
            "sampling",
        )
        self._provider_options = _validate_options(
            provider_options or {},
            {"logprobs", "top_logprobs"},
            "provider_options",
        )
        if max_response_bytes <= 0:
            raise OpenAICompatibleConfigurationError("max_response_bytes must be positive")
        self._max_response_bytes = max_response_bytes
        self._owned_client = client is None
        self._closed = False
        self._tool_calls_by_id: OrderedDict[str, _CachedToolCall] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        if client is not None:
            if any(
                value is not None
                for value in (
                    endpoint,
                    api_key,
                    organization,
                    project,
                    headers,
                    timeout,
                    transport_options,
                    tls_options,
                    proxy_options,
                )
            ):
                raise ContradictoryProviderConfigurationError(
                    "Preconstructed OpenAI-compatible client cannot be combined with "
                    "endpoint, transport, TLS, proxy, timeout, or authentication settings"
                )
            self._client = client
        else:
            self._client = _create_official_client(
                endpoint=endpoint,
                api_key=api_key,
                organization=organization,
                project=project,
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
        """Generate one OpenAI-compatible chat completion through Conducto contracts."""
        if self._closed:
            raise ProviderEndpointUnavailableError(
                "OpenAI-compatible provider is closed", attempted=False
            )
        if call_context is not None and call_context.cancelled:
            raise ProviderCancellationError()
        _assert_schema_supported(structured_output, self._profile)
        if (tools or tool_results) and not self.capabilities.tool_calling:
            raise UnsupportedProviderCapabilityError(
                f"Configured profile '{self._profile.name}' does not support native tool calling"
            )
        for tool in tools:
            _assert_schema_supported(
                StructuredOutputRequest(
                    name=f"{tool.name}_arguments",
                    schema=tool.input_schema,
                ),
                self._profile,
            )
        model = options.model or self._model
        payload: dict[str, Any] = {
            "model": model,
            "messages": await self._to_openai_messages(messages, tool_results),
            "response_format": _response_format(structured_output, self._profile.strict_schema),
            "temperature": options.temperature,
        }
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        elif self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        if self._seed is not None:
            payload["seed"] = self._seed
        if options.stop:
            payload["stop"] = list(options.stop)
        payload.update(self._sampling)
        payload.update(self._provider_options)
        if tools:
            payload["tools"] = tuple(_to_openai_tool(tool) for tool in tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        try:
            timeout_for_request = _request_timeout(
                options,
                effective_deadline=effective_deadline,
                call_context=call_context,
            )
            completion = self._client.chat.completions.create(**payload)
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
            raise _map_error(error) from error
        return await self._normalize_response(response, structured_output, tuple(tools))

    async def check_readiness(
        self,
        *,
        model: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Verify server reachability, model availability, and tested version.

        Args:
            model: Model name to check. Defaults to this provider's configured model.
            timeout: Bound in seconds for each readiness operation.

        Raises:
            OpenAICompatibleReadinessError: If the server cannot be queried.
            OpenAICompatibleModelNotFoundError: If the model is not listed by
                the server.
            OpenAICompatibleIncompatibleVersionError: If a documented server
                version check reports an unsupported version.
            ProviderAuthenticationError: If the server or proxy rejects credentials.
        """
        model_name = model or self._model
        try:
            models_payload = await asyncio.wait_for(self._client.models.list(), timeout)
        except TimeoutError as error:
            raise ProviderTimeoutError(attempted=True) from error
        except Exception as error:
            mapped = _map_error(error)
            if isinstance(
                mapped, (ProviderAuthenticationError, OpenAICompatibleModelNotFoundError)
            ):
                raise mapped from error
            raise OpenAICompatibleReadinessError(
                "OpenAI-compatible readiness check failed"
            ) from error
        if not _model_listed(models_payload, model_name):
            raise OpenAICompatibleModelNotFoundError(
                "OpenAI-compatible model is not available on the server"
            )
        if self._profile.version_path is not None:
            await self._check_version(timeout)

    async def _check_version(self, timeout: float) -> None:
        """Perform an optional bounded server version compatibility check."""
        get_version = getattr(self._client, "get_version", None)
        if not callable(get_version):
            return
        assert self._profile.version_path is not None
        try:
            version_payload = await asyncio.wait_for(
                get_version(self._profile.version_path), timeout
            )
        except TimeoutError as error:
            raise ProviderTimeoutError(attempted=True) from error
        except Exception as error:
            mapped = _map_error(error)
            if isinstance(mapped, ProviderAuthenticationError):
                raise mapped from error
            raise OpenAICompatibleReadinessError(
                "OpenAI-compatible version check failed"
            ) from error
        server_version = _extract_version(version_payload)
        if (
            server_version is not None
            and self._profile.minimum_server_version is not None
            and not _version_at_least(server_version, self._profile.minimum_server_version)
        ):
            raise OpenAICompatibleIncompatibleVersionError(
                f"Server version is unsupported for profile '{self._profile.name}' "
                f"(requires >= {self._profile.minimum_server_version})"
            )

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

    async def _to_openai_messages(
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
                raise MalformedStructuredOutputError(
                    "OpenAI-compatible tool result has unknown call ID"
                )
            translated.append(dict(cached.assistant_message))
            translated.append(
                {
                    "role": "tool",
                    "tool_call_id": cached.call_id,
                    "content": _safe_json_dumps(
                        {"status": result.status, "result": result.result},
                        "tool result",
                    ),
                }
            )
        return translated

    async def _normalize_response(
        self,
        response: Any,
        structured_output: StructuredOutputRequest,
        tools: tuple[ProviderToolDefinition, ...],
    ) -> ProviderResult:
        """Normalize an OpenAI-shaped response without exposing client types."""
        response_map = _object_to_mapping(response)
        self._enforce_response_bound(response_map)
        choices = response_map.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise MalformedStructuredOutputError("OpenAI-compatible server returned no choices")
        message = _object_to_mapping(choices[0]).get("message")
        message = _object_to_mapping(message)
        content = message.get("content")
        tool_calls = _extract_tool_calls(message)
        request_id = _safe_string(response_map.get("id"))
        usage = _usage_from_response(response_map)
        finish_reason = _finish_reason(_object_to_mapping(choices[0]))
        if tool_calls and _content_has_value(content):
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned mixed terminal content and tool calls",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        if len(tool_calls) > 1:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned multiple tool calls",
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
                "OpenAI-compatible server returned no terminal structured content",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        try:
            structured = json.loads(content)
        except json.JSONDecodeError as error:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned invalid terminal JSON",
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
        """Resolve and validate one native OpenAI-shaped tool call."""
        function = _object_to_mapping(raw_call.get("function", {}))
        name = _safe_string(function.get("name"))
        if name is None:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned a tool call without a name"
            )
        matches = [tool for tool in tools if tool.name == name]
        if not matches:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned an unknown tool name"
            )
        if len(matches) != 1:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned an ambiguous tool call"
            )
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
        content = message.get("content")
        await self._remember_tool_call(
            call_id,
            name,
            _assistant_tool_message(
                content if isinstance(content, str) else None,
                call_id,
                name,
                arguments,
            ),
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
            raise OpenAICompatibleOversizedResponseError()


class OpenAICompatibleProviderFactory:
    """Trusted factory for configuration-owned OpenAI-compatible providers."""

    def create(self, configuration: ProviderClientConfig) -> OpenAICompatibleProvider:
        """Construct an OpenAI-compatible provider from one registry configuration source.

        Args:
            configuration: Provider registry configuration. ``provider_defaults``
                must include ``model`` and ``profile``.

        Returns:
            A runtime-owned OpenAI-compatible provider client.
        """
        defaults = dict(configuration.provider_defaults)
        model = defaults.pop("model", None)
        if not isinstance(model, str) or not model.strip():
            raise OpenAICompatibleConfigurationError("provider_defaults must include model")
        profile = defaults.pop("profile", None)
        if profile is None:
            raise OpenAICompatibleConfigurationError("provider_defaults must include profile")
        api_key = _secret_from_credential_ref(configuration.credential_ref)
        organization = defaults.pop("organization", None)
        project = defaults.pop("project", None)
        context_window = _optional_positive_int(
            defaults.pop("context_window", None), "context_window"
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
            raise OpenAICompatibleConfigurationError(
                f"Unsupported OpenAI-compatible provider defaults: {sorted(defaults)}"
            )
        assert max_response_bytes is not None
        return OpenAICompatibleProvider(
            model=model,
            profile=cast(OpenAICompatibleProfile | str, profile),
            endpoint=configuration.endpoint,
            api_key=api_key,
            organization=cast(str | None, organization),
            project=cast(str | None, project),
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            seed=seed,
            sampling=cast(Mapping[str, int | float], sampling),
            timeout=timeout,
            transport_options=configuration.transport,
            tls_options=configuration.tls,
            proxy_options=configuration.proxy,
            provider_options=cast(Mapping[str, Any], provider_options),
            max_response_bytes=max_response_bytes,
        )


openai_compatible_provider_factory = OpenAICompatibleProviderFactory()
"""Reusable singleton factory suitable for ``ProviderRegistry`` registration."""


def _create_official_client(
    *,
    endpoint: str | None,
    api_key: str | None,
    organization: str | None,
    project: str | None,
    headers: Mapping[str, str],
    timeout: float | None,
    transport_options: Mapping[str, Any],
    tls_options: Mapping[str, Any],
    proxy_options: Mapping[str, Any],
    max_response_bytes: int,
) -> _OpenAICompatibleClient:
    """Construct a bounded client for the documented OpenAI-compatible protocol."""
    try:
        require_adapter("openai")
    except AdapterDependencyError:
        raise
    except Exception as error:
        raise OpenAICompatibleConfigurationError(
            "OpenAI-compatible client dependency is unavailable"
        ) from error
    return _BoundedOpenAICompatibleHttpClient(
        endpoint=endpoint,
        api_key=api_key,
        organization=organization,
        project=project,
        headers=headers,
        timeout=timeout,
        transport_options=transport_options,
        tls_options=tls_options,
        proxy_options=proxy_options,
        max_response_bytes=max_response_bytes,
    )


def _official_client_kwargs(
    *,
    api_key: str | None,
    organization: str | None,
    project: str | None,
    headers: Mapping[str, str],
    timeout: float | None,
    transport_options: Mapping[str, Any],
    tls_options: Mapping[str, Any],
    proxy_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate allowlisted transport settings for the bounded client."""
    client_headers = dict(headers)
    if api_key is not None:
        client_headers["Authorization"] = f"Bearer {api_key}"
    if organization is not None:
        client_headers["OpenAI-Organization"] = organization
    if project is not None:
        client_headers["OpenAI-Project"] = project
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
        raise OpenAICompatibleConfigurationError(
            f"Unsupported OpenAI-compatible transport options: {sorted(unsupported)}"
        )
    if not values:
        return None
    try:
        import httpx
    except Exception as error:
        raise OpenAICompatibleConfigurationError(
            "httpx is required for OpenAI-compatible transport limits"
        ) from error
    return httpx.Limits(**values)


def _validate_options(
    values: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> dict[str, Any]:
    """Return allowlisted options or fail explicitly."""
    unsupported = set(values) - allowed
    if unsupported:
        raise OpenAICompatibleConfigurationError(
            f"Unsupported OpenAI-compatible {label} options: {sorted(unsupported)}"
        )
    return dict(values)


def _extract_prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Pop and return provider defaults with a common prefix stripped."""
    extracted: dict[str, Any] = {}
    for key in tuple(values):
        if key.startswith(prefix):
            extracted[key.removeprefix(prefix)] = values.pop(key)
    return extracted


def _coerce_profile(profile: OpenAICompatibleProfile | str) -> OpenAICompatibleProfile:
    """Normalize profile declarations to a conformance-tested profile."""
    if isinstance(profile, OpenAICompatibleProfile):
        if profile.tool_calling and profile.name not in _TOOL_CAPABLE_PROFILE_NAMES:
            raise OpenAICompatibleConfigurationError(
                f"OpenAI-compatible tool profile '{profile.name}' has not been conformance tested"
            )
        return profile
    known = _KNOWN_PROFILES.get(profile)
    if known is None:
        raise OpenAICompatibleConfigurationError(f"Unknown OpenAI-compatible profile '{profile}'")
    return known


def _conducto_schema_features() -> frozenset[Any]:
    """Return Conducto schema features supported by this adapter's schema format."""
    from conducto.core.provider import SchemaFeature

    return frozenset(
        feature for feature in SchemaFeature if feature.value in _SUPPORTED_SCHEMA_FEATURES
    )


def _assert_schema_supported(
    request: StructuredOutputRequest,
    profile: OpenAICompatibleProfile,
) -> None:
    """Reject schema features not sent to the server's native response format."""
    if request.dialect is not JsonSchemaDialect.DRAFT_2020_12:
        raise UnsupportedProviderCapabilityError(
            f"Profile '{profile.name}' supports Draft 2020-12 schemas"
        )
    unknown = _unknown_schema_keywords(request.json_schema)
    if unknown:
        raise UnsupportedProviderCapabilityError(
            f"Schema uses unsupported keywords for profile '{profile.name}': {sorted(unknown)!r}"
        )
    features = {feature.value for feature in request.features}
    features.update(_schema_features(request.json_schema))
    unsupported = features - _SUPPORTED_SCHEMA_FEATURES
    if unsupported:
        raise UnsupportedProviderCapabilityError(
            f"Schema uses unsupported features for profile '{profile.name}': "
            f"{sorted(unsupported)!r}"
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


def _to_openai_tool(tool: ProviderToolDefinition) -> dict[str, Any]:
    """Translate one Conducto tool descriptor to the OpenAI-shaped native tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _response_format(
    structured_output: StructuredOutputRequest,
    strict: bool,
) -> dict[str, Any]:
    """Build the OpenAI-shaped ``response_format`` for one terminal schema."""
    json_schema: dict[str, Any] = {
        "name": _safe_schema_name(structured_output.name),
        "schema": dict(structured_output.json_schema),
    }
    if strict:
        json_schema["strict"] = True
    return {"type": "json_schema", "json_schema": json_schema}


def _safe_schema_name(name: str) -> str:
    """Sanitize a Conducto schema name to the server-accepted identifier subset."""
    sanitized = _SCHEMA_NAME_PATTERN.sub("_", name.strip()) or "response"
    return sanitized[:64]


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
    content: str | None,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the assistant tool-call message OpenAI-shaped protocols expect."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": (
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _safe_json_dumps(arguments, "tool call arguments"),
                },
            },
        ),
    }


def _message_content_to_text(content: str | tuple[MessageContentPart, ...]) -> str:
    """Translate supported Conducto message content to plain text content."""
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
        raise MalformedStructuredOutputError(
            f"OpenAI-compatible {label} is not JSON serializable"
        ) from error


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
    """Return normalized tool-call dictionaries from an OpenAI-shaped message."""
    raw = message.get("tool_calls", ())
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MalformedStructuredOutputError(
            "OpenAI-compatible server returned malformed tool calls"
        )
    calls: list[Mapping[str, Any]] = []
    for item in raw:
        mapped = _object_to_mapping(item)
        if not mapped:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned malformed tool call"
            )
        calls.append(mapped)
    return tuple(calls)


def _content_has_value(content: Any) -> bool:
    """Return whether provider message content carries terminal data."""
    return isinstance(content, str) and bool(content.strip())


def _coerce_arguments(value: Any) -> Mapping[str, Any]:
    """Decode OpenAI-shaped tool-call arguments into an immutable-free mapping."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise MalformedStructuredOutputError(
                "OpenAI-compatible server returned malformed tool arguments"
            ) from error
    if not isinstance(value, Mapping):
        raise MalformedStructuredOutputError(
            "OpenAI-compatible server returned non-object tool arguments"
        )
    return dict(value)


def _synthesize_call_id(
    *,
    request_id: str | None,
    tool_id: str,
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    """Create a stable call ID for protocols that omit a native tool-call ID."""
    argument_text = json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    seed = f"{request_id or ''}\0{tool_id}\0{name}\0{argument_text}"
    return f"openai_compatible_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"


def _usage_from_response(response: Mapping[str, Any]) -> Usage:
    """Map server-reported token counters without estimating missing values."""
    usage = _object_to_mapping(response.get("usage", {}))
    input_tokens = _optional_non_negative_int(usage.get("prompt_tokens"))
    output_tokens = _optional_non_negative_int(usage.get("completion_tokens"))
    total_tokens = _optional_non_negative_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _finish_reason(choice: Mapping[str, Any]) -> str:
    """Return provider-neutral finish metadata."""
    raw = _safe_string(choice.get("finish_reason"))
    mapping = {
        "stop": FinishReason.STOP.value,
        "length": FinishReason.LENGTH.value,
        "content_filter": FinishReason.CONTENT_FILTER.value,
        "tool_calls": FinishReason.TOOL_CALL.value,
    }
    if raw in mapping:
        return mapping[raw]
    return FinishReason.UNKNOWN.value


def _model_listed(payload: Any, model_name: str) -> bool:
    """Return whether a model ID appears in an OpenAI-shaped models list response."""
    mapped = _object_to_mapping(payload)
    data = mapped.get("data", ())
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return False
    for item in data:
        item_id = _safe_string(_object_to_mapping(item).get("id"))
        if item_id == model_name:
            return True
    return False


def _map_error(error: Exception) -> ProviderError:
    """Map official-client or bounded-transport exceptions to redacted provider errors."""
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    message = _bounded_error_text(error)
    if status in {401, 403} or "unauthorized" in message or "forbidden" in message:
        return ProviderAuthenticationError()
    if status == 404 or "not found" in message:
        return OpenAICompatibleModelNotFoundError("OpenAI-compatible model is not available")
    if status == 408 or "timeout" in message:
        return ProviderTimeoutError(attempted=True)
    if status == 429 or "rate limit" in message or "quota" in message:
        return ProviderRateLimitError()
    if "connection refused" in message or "connect" in message:
        return ProviderEndpointUnavailableError("OpenAI-compatible endpoint is unavailable")
    return ProviderEndpointUnavailableError("OpenAI-compatible request failed")


def _bounded_error_text(error: Exception) -> str:
    """Return a lower-case bounded error diagnostic for classification only."""
    text = str(error).lower()
    return text[:_MAX_ERROR_BODY_CHARS]


def _extract_version(payload: Any) -> str | None:
    """Extract a server version from official- or bounded-client response shapes."""
    mapped = _object_to_mapping(payload)
    return _safe_string(mapped.get("version"))


def _version_at_least(actual: str, minimum: str) -> bool:
    """Compare provider versions using packaging semantics."""
    try:
        return Version(actual) >= Version(minimum)
    except InvalidVersion:
        return False


def _secret_from_credential_ref(credential_ref: str | None) -> str | None:
    """Resolve an opaque credential reference through the process environment."""
    if credential_ref is None:
        return None
    value = os.environ.get(credential_ref)
    if value is None:
        raise OpenAICompatibleConfigurationError("Credential reference is not available")
    if not value.strip():
        raise OpenAICompatibleConfigurationError("Credential reference is empty")
    return value


def _safe_string(value: Any) -> str | None:
    """Return a non-empty string value when available."""
    return value if isinstance(value, str) and value.strip() else None


def _optional_int(value: Any, label: str) -> int | None:
    """Validate optional integer provider settings."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise OpenAICompatibleConfigurationError(f"{label} must be an integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    """Validate optional positive integer provider settings."""
    result = _optional_int(value, label)
    if result is not None and result <= 0:
        raise OpenAICompatibleConfigurationError(f"{label} must be positive")
    return result


def _optional_positive_float(value: Any, label: str) -> float | None:
    """Validate optional positive float provider settings."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise OpenAICompatibleConfigurationError(f"{label} must be positive")
    return float(value)


def _optional_non_negative_int(value: Any) -> int | None:
    """Return a non-negative token counter or ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
