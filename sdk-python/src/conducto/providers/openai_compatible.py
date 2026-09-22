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
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

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
    native_tool as _to_openai_tool,
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

DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1"
"""Documented default local vLLM OpenAI-compatible endpoint."""

DEFAULT_LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"
"""Documented default local LM Studio OpenAI-compatible endpoint."""

TESTED_MINIMUM_VLLM_SERVER = "0.6.0"
TESTED_MINIMUM_LM_STUDIO_SERVER = "0.3.0"
TESTED_MINIMUM_OPENAI_CLIENT = "1.0"
_TOOL_CAPABLE_PROFILE_NAMES = frozenset({"vllm-tools", "lm-studio-tools"})

# JSON Schema keywords Conducto both sends natively and validates locally for
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


@runtime_checkable
class _VersionedClient(Protocol):
    """Optional version-endpoint hook implemented by version-aware transports."""

    async def get_version(self, path: str) -> Any:
        """Retrieve the server version from the profile's documented endpoint."""
        ...


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


_config = AdapterConfig("OpenAI-compatible", OpenAICompatibleConfigurationError)
_validate_options = _config.validate_options
_codec = ResponseCodec("OpenAI-compatible", "OpenAI-compatible server", "openai_compatible")
_safe_json_dumps = _codec.json_dumps
_message_content_to_text = _codec.message_content_to_text
_synthesize_call_id = _codec.synthesize_call_id
_map_error = ErrorPolicy(
    "OpenAI-compatible",
    OpenAICompatibleModelNotFoundError,
    "OpenAI-compatible model is not available",
    response_status=True,
    text_rate_limit=True,
).map


class _BoundedOpenAICompatibleHttpClient(BoundedHttpClient):
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
        base_url = (endpoint or DEFAULT_VLLM_ENDPOINT).rstrip("/")
        super().__init__(
            endpoint=base_url,
            kwargs=_http_client_kwargs(
                api_key=api_key,
                organization=organization,
                project=project,
                headers=headers,
                timeout=timeout,
                transport_options=transport_options,
                tls_options=tls_options,
                proxy_options=proxy_options,
            ),
            max_response_bytes=max_response_bytes,
            configuration_error=OpenAICompatibleConfigurationError,
            oversized_error=OpenAICompatibleOversizedResponseError,
            provider_name="OpenAI-compatible",
            malformed_message="OpenAI-compatible server returned malformed JSON",
        )
        self.chat: _ChatClient = _ChatNamespace(self)
        self.models: _ModelsClient = _ModelsNamespace(self)

    async def get_version(self, path: str) -> Any:
        """Return a bounded, decoded response from a server version endpoint."""
        return await self.request_json("GET", path)


class _ChatCompletionsNamespace:
    """Bounded ``chat.completions`` namespace for the local HTTP transport."""

    def __init__(self, outer: _BoundedOpenAICompatibleHttpClient) -> None:
        """Bind this namespace to its owning bounded transport."""
        self._outer = outer

    async def create(self, **kwargs: Any) -> Any:
        """Send one bounded chat completion request."""
        return await self._outer.request_json("POST", "/chat/completions", json_body=kwargs)


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
        return await self._outer.request_json("GET", "/models")


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
        Release configuration-owned resources with ``await provider.aclose()``.
        Borrowed clients remain caller-owned; synchronous shutdown is not supported.
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
            schema_features=conducto_schema_features(_SUPPORTED_SCHEMA_FEATURES),
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
        self._tool_calls = ToolCallCache()
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
            self._client = _create_bounded_client(
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
        self._lifecycle = AsyncClientLifecycle(
            self._client if client is None else None,
            configuration_error=OpenAICompatibleConfigurationError,
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
        if self._lifecycle.closed:
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
        client = self._client
        if not isinstance(client, _VersionedClient):
            return
        get_version: Callable[[str], Awaitable[Any]] = client.get_version
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

    async def aclose(self) -> None:
        """Await owned-client shutdown once; leave borrowed clients untouched.

        Shutdown failures and cancellation propagate without marking the
        provider closed. Retry with ``await aclose()`` after a failed attempt.
        """
        await self._lifecycle.aclose()

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
            cached = await self._tool_calls.lookup(result.call_id)
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
        enforce_response_bound(
            response_map, self._max_response_bytes, OpenAICompatibleOversizedResponseError
        )
        choices = response_map.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise MalformedStructuredOutputError("OpenAI-compatible server returned no choices")
        message = _object_to_mapping(_object_to_mapping(choices[0]).get("message"))
        request_id = _safe_string(response_map.get("id"))
        return await _codec.normalize_result(
            message,
            structured_output=structured_output,
            usage=_usage_from_response(response_map),
            request_id=request_id,
            finish_reason=_finish_reason(_object_to_mapping(choices[0])),
            normalize_tool=lambda raw: self._normalize_tool_call(raw, message, tools, request_id),
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
        tool, name, arguments = _codec.resolve_tool(function, tools)
        call_id = _safe_string(raw_call.get("id")) or _synthesize_call_id(
            request_id=request_id,
            tool_id=tool.tool_id,
            name=name,
            arguments=arguments,
        )
        content = message.get("content")
        await self._tool_calls.remember(
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


class OpenAICompatibleProviderFactory:
    """Trusted factory for configuration-owned OpenAI-compatible providers."""

    @staticmethod
    def create(configuration: ProviderClientConfig) -> OpenAICompatibleProvider:
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
        api_key = _config.credential(configuration.credential_ref, label="Credential reference")
        organization = defaults.pop("organization", None)
        project = defaults.pop("project", None)
        generation = _config.generation_defaults(defaults)
        if defaults:
            raise OpenAICompatibleConfigurationError(
                f"Unsupported OpenAI-compatible provider defaults: {sorted(defaults)}"
            )
        return OpenAICompatibleProvider(
            model=model,
            profile=cast(OpenAICompatibleProfile | str, profile),
            endpoint=configuration.endpoint,
            api_key=api_key,
            organization=cast(str | None, organization),
            project=cast(str | None, project),
            transport_options=configuration.transport,
            tls_options=configuration.tls,
            proxy_options=configuration.proxy,
            **generation,
        )


openai_compatible_provider_factory = OpenAICompatibleProviderFactory()
"""Reusable singleton factory suitable for ``ProviderRegistry`` registration."""


def _create_bounded_client(
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


def _http_client_kwargs(
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
    if organization is not None:
        client_headers["OpenAI-Organization"] = organization
    if project is not None:
        client_headers["OpenAI-Project"] = project
    return _config.http_kwargs(
        token=api_key,
        headers=client_headers,
        timeout=timeout,
        transport_options=transport_options,
        tls_options=tls_options,
        proxy_options=proxy_options,
    )


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


def _assert_schema_supported(
    request: StructuredOutputRequest,
    profile: OpenAICompatibleProfile,
) -> None:
    """Reject schema features not sent to the server's native response format."""
    if request.dialect is not JsonSchemaDialect.DRAFT_2020_12:
        raise UnsupportedProviderCapabilityError(
            f"Profile '{profile.name}' supports Draft 2020-12 schemas"
        )
    unknown = unknown_schema_keywords(request.json_schema, _ALLOWED_SCHEMA_KEYWORDS)
    if unknown:
        raise UnsupportedProviderCapabilityError(
            f"Schema uses unsupported keywords for profile '{profile.name}': {sorted(unknown)!r}"
        )
    features = {feature.value for feature in request.features}
    features.update(schema_features(request.json_schema, _SUPPORTED_SCHEMA_FEATURES))
    unsupported = features - _SUPPORTED_SCHEMA_FEATURES
    if unsupported:
        raise UnsupportedProviderCapabilityError(
            f"Schema uses unsupported features for profile '{profile.name}': "
            f"{sorted(unsupported)!r}"
        )


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
    if raw is not None and raw in mapping:
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
