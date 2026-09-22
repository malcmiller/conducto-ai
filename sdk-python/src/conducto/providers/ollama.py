"""Ollama provider adapter for local Conducto model inference.

The adapter keeps the optional official ``ollama`` client behind the shared
Conducto provider protocol. It never starts or configures an Ollama daemon and
never pulls, deletes, or manages model weights.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from conducto.adapters import AdapterDependencyError, require_adapter
from conducto.core.provider import (
    ChatMessage,
    FinishReason,
    GenerationOptions,
    JsonSchemaDialect,
    MalformedStructuredOutputError,
    ModelProvider,
    ProviderAuthenticationError,
    ProviderCallContext,
    ProviderCancellationError,
    ProviderCapabilities,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderFailureCategory,
    ProviderResult,
    ProviderTimeoutError,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    UnsupportedProviderCapabilityError,
    Usage,
)
from conducto.core.provider_registry import ProviderClientConfig
from conducto.core.runtime_errors import ContradictoryProviderConfigurationError
from conducto.providers._config import AdapterConfig
from conducto.providers._http import (
    DEFAULT_MAX_RESPONSE_BYTES as _DEFAULT_MAX_RESPONSE_BYTES,
)
from conducto.providers._http import (
    BoundedHttpClient,
    ErrorPolicy,
    enforce_response_bound,
)
from conducto.providers._lifecycle import AsyncClientLifecycle
from conducto.providers._response import (
    ResponseCodec,
)
from conducto.providers._response import (
    extract_version as _extract_version,
)
from conducto.providers._response import (
    native_tool as _to_ollama_tool,
)
from conducto.providers._response import (
    object_to_mapping as _object_to_mapping,
)
from conducto.providers._response import (
    optional_non_negative_int as _optional_non_negative_int,
)
from conducto.providers._response import (
    request_timeout as _request_timeout,
)
from conducto.providers._response import (
    safe_string as _safe_string,
)
from conducto.providers._response import (
    version_at_least as _version_at_least,
)
from conducto.providers._schema import (
    conducto_schema_features,
    schema_features,
    unknown_schema_keywords,
)
from conducto.providers._tool_cache import ToolCallCache

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
TESTED_MINIMUM_OLLAMA_SERVER = "0.6.0"
TESTED_MINIMUM_OLLAMA_CLIENT = "0.5"
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


_config = AdapterConfig("Ollama", OllamaConfigurationError)
_validate_options = _config.validate_options
_codec = ResponseCodec("Ollama", "Ollama", "ollama")
_safe_json_dumps = _codec.json_dumps
_message_content_to_text = _codec.message_content_to_text
_synthesize_call_id = _codec.synthesize_call_id
_map_ollama_error = ErrorPolicy(
    "Ollama", OllamaModelNotFoundError, "Ollama model is not available locally"
).map


class _BoundedOllamaHttpClient(BoundedHttpClient):
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
        super().__init__(
            endpoint=endpoint or DEFAULT_OLLAMA_ENDPOINT,
            kwargs=_config.http_kwargs(
                token=auth_token,
                headers=headers,
                timeout=timeout,
                transport_options=transport_options,
                tls_options=tls_options,
                proxy_options=proxy_options,
            ),
            max_response_bytes=max_response_bytes,
            configuration_error=OllamaConfigurationError,
            oversized_error=OllamaOversizedResponseError,
            provider_name="Ollama",
            malformed_message="Ollama returned malformed JSON",
        )

    async def chat(self, **kwargs: Any) -> Any:
        """Send one bounded chat request to Ollama."""
        return await self.request_json("POST", "/api/chat", json_body=kwargs)

    async def list(self) -> Any:
        """List locally available Ollama models."""
        return await self.request_json("GET", "/api/tags")

    async def show(self, model: str) -> Any:
        """Return local metadata for one model."""
        return await self.request_json("POST", "/api/show", json_body={"model": model})

    async def version(self) -> Any:
        """Return Ollama server version metadata."""
        return await self.request_json("GET", "/api/version")


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
        Release configuration-owned resources with ``await provider.aclose()``.
        Borrowed clients remain caller-owned; synchronous shutdown is not supported.
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
            schema_features=conducto_schema_features(_SUPPORTED_SCHEMA_FEATURES),
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
        self._tool_calls = ToolCallCache()
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
            self._client = _create_bounded_client(
                endpoint=endpoint,
                auth_token=auth_token,
                headers=headers or {},
                timeout=timeout,
                transport_options=transport_options or {},
                tls_options=tls_options or {},
                proxy_options=proxy_options or {},
                max_response_bytes=max_response_bytes,
            )
        self._lifecycle = AsyncClientLifecycle(
            self._client if client is None else None,
            configuration_error=OllamaConfigurationError,
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
        if self._lifecycle.closed:
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
        request: dict[str, Any] = {
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

    async def aclose(self) -> None:
        """Await owned-client shutdown once; leave borrowed clients untouched.

        Shutdown failures and cancellation propagate without marking the
        provider closed. Retry with ``await aclose()`` after a failed attempt.
        """
        await self._lifecycle.aclose()

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
            cached = await self._tool_calls.lookup(result.call_id)
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
        enforce_response_bound(response_map, self._max_response_bytes, OllamaOversizedResponseError)
        message = _object_to_mapping(response_map.get("message", {}))
        request_id = _safe_string(response_map.get("id"))
        return await _codec.normalize_result(
            message,
            structured_output=structured_output,
            usage=_usage_from_response(response_map),
            request_id=request_id,
            finish_reason=_finish_reason(response_map),
            normalize_tool=lambda raw: self._normalize_tool_call(raw, message, tools, request_id),
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
        tool, name, arguments = _codec.resolve_tool(function, tools)
        call_id = _safe_string(raw_call.get("id")) or _synthesize_call_id(
            request_id=request_id,
            tool_id=tool.tool_id,
            name=name,
            arguments=arguments,
        )
        await self._tool_calls.remember(
            call_id,
            name,
            _assistant_tool_message(message, raw_call, name, arguments),
        )
        return ProviderToolCallRequest(
            call_id=call_id,
            tool_id=tool.tool_id,
            arguments=arguments,
        )


class OllamaProviderFactory:
    """Trusted factory for configuration-owned Ollama providers."""

    @staticmethod
    def create(configuration: ProviderClientConfig) -> OllamaProvider:
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
        auth_token = _config.credential(
            configuration.credential_ref, label="Ollama credential reference"
        )
        profile = defaults.pop("profile", None)
        keep_alive = defaults.pop("keep_alive", None)
        generation = _config.generation_defaults(defaults)
        if defaults:
            raise OllamaConfigurationError(
                f"Unsupported Ollama provider defaults: {sorted(defaults)}"
            )
        return OllamaProvider(
            model=model,
            endpoint=configuration.endpoint,
            auth_token=auth_token,
            profile=cast(OllamaProfile | str | None, profile),
            keep_alive=cast(str | int | None, keep_alive),
            transport_options=configuration.transport,
            tls_options=configuration.tls,
            proxy_options=configuration.proxy,
            **generation,
        )


ollama_provider_factory = OllamaProviderFactory()
"""Reusable singleton factory suitable for ``ProviderRegistry`` registration."""


def _create_bounded_client(
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


def _assert_schema_supported(request: StructuredOutputRequest) -> None:
    """Reject schema features aren't sent to Ollama's native ``format`` field."""
    if request.dialect is not JsonSchemaDialect.DRAFT_2020_12:
        raise UnsupportedProviderCapabilityError("Ollama profile supports Draft 2020-12 schemas")
    unknown = unknown_schema_keywords(request.json_schema, _ALLOWED_SCHEMA_KEYWORDS)
    if unknown:
        raise UnsupportedProviderCapabilityError(
            f"Ollama schema uses unsupported keywords: {sorted(unknown)!r}"
        )
    features = {feature.value for feature in request.features}
    features.update(schema_features(request.json_schema, _SUPPORTED_SCHEMA_FEATURES))
    unsupported = features - _SUPPORTED_SCHEMA_FEATURES
    if unsupported:
        raise UnsupportedProviderCapabilityError(
            f"Ollama schema uses unsupported features: {sorted(unsupported)!r}"
        )


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


def _looks_not_found(error: Exception) -> bool:
    """Return whether an exception appears to represent missing local model state."""
    return isinstance(_map_ollama_error(error), OllamaModelNotFoundError)
