"""Private bounded HTTP transport and safe adapter error classification."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from conducto.core.provider import (
    MalformedStructuredOutputError,
    ProviderAuthenticationError,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

MAX_ERROR_BODY_CHARS = 512
DEFAULT_MAX_RESPONSE_BYTES = 1_000_000


def _classification_hint(text: str) -> str:
    hints = ("unauthorized", "forbidden", "not found", "timeout", "rate limit", "quota", "connect")
    return " ".join(hint for hint in hints if hint in text.lower()[:MAX_ERROR_BODY_CHARS])


def enforce_response_bound(
    response: Mapping[str, Any], limit: int, error_type: Callable[[], ProviderError]
) -> None:
    """Bound decoded SDK payloads using the adapters' existing ASCII JSON metric."""
    if len(json.dumps(response, ensure_ascii=True, default=str)) > limit:
        raise error_type()


class HttpStatusError(RuntimeError):
    """HTTP status with a bounded classification hint, never a remote body."""

    def __init__(self, status_code: int, body: bytes) -> None:
        """Keep only fixed diagnostic tokens from potentially secret response text."""
        text = body[:MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace").lower()
        self.status_code = status_code
        self.classification_hint = _classification_hint(text)
        super().__init__(f"Provider HTTP request failed with status {status_code}")


class HttpTransportError(RuntimeError):
    """Transport diagnostic stripped of request headers and remote text."""

    def __init__(self, error: Exception) -> None:
        """Preserve classification tokens but never retain the original exception."""
        self.classification_hint = _classification_hint(str(error))
        super().__init__("Provider HTTP transport failed")


class BoundedHttpClient:
    """Shared streamed transport with adapter-specific configuration and failures.

    Credentials are held only by the underlying HTTP client's headers. Remote
    error bodies are reduced to fixed classification hints before propagation.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        kwargs: Mapping[str, Any],
        max_response_bytes: int,
        configuration_error: type[ProviderError],
        oversized_error: Callable[[], ProviderError],
        provider_name: str,
        malformed_message: str,
    ) -> None:
        """Create the connection pool without importing httpx at module load."""
        try:
            import httpx
        except ImportError as error:
            raise configuration_error(f"httpx is required for {provider_name} transport") from error
        self._client = httpx.AsyncClient(base_url=endpoint, **kwargs)
        self._max_response_bytes = max_response_bytes
        self._oversized_error = oversized_error
        self._malformed_message = malformed_message
        self._close_task: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        """Keep one pool-close attempt alive if a shutdown waiter is cancelled."""
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._client.aclose())
            self._close_task = task
            task.add_done_callback(_observe_close)
        await asyncio.shield(task)

    async def request_json(
        self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None
    ) -> Any:
        """Execute a bounded request without unsafe transport diagnostics."""
        import httpx

        try:
            return await self._send_json(method, path, json_body=json_body)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(attempted=True) from None
        except httpx.HTTPError as error:
            raise HttpTransportError(error) from None

    async def _send_json(
        self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None
    ) -> Any:
        """Bound the response before decoding and keep remote bodies out of errors."""
        async with self._client.stream(method, path, json=json_body) as response:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self._max_response_bytes:
                    raise self._oversized_error()
                chunks.append(chunk)
            body = b"".join(chunks)
            if response.status_code >= 400:
                raise HttpStatusError(response.status_code, body)
            try:
                return json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Decoder errors may contain fragments of an echoed credential.
                raise MalformedStructuredOutputError(self._malformed_message) from None


def _observe_close(task: asyncio.Task[None]) -> None:
    # A cancelled waiter cannot retrieve failures; every later aclose still awaits this task.
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True, slots=True)
class ErrorPolicy:
    """Retain adapter-specific status and text classification precedence."""

    name: str
    not_found_error: type[ProviderError]
    not_found_message: str
    response_status: bool = False
    text_rate_limit: bool = False

    def map(self, error: Exception) -> ProviderError:
        """Normalize an exception without copying untrusted diagnostics."""
        status = getattr(error, "status_code", None)
        if status is None and self.response_status:
            status = getattr(getattr(error, "response", None), "status_code", None)
        message = (
            error.classification_hint
            if isinstance(error, (HttpStatusError, HttpTransportError))
            else str(error).lower()[:MAX_ERROR_BODY_CHARS]
        )
        if status in {401, 403} or "unauthorized" in message or "forbidden" in message:
            return ProviderAuthenticationError()
        if status == 404 or "not found" in message:
            return self.not_found_error(self.not_found_message)
        if status == 408 or "timeout" in message:
            return ProviderTimeoutError(attempted=True)
        if status == 429 or (
            self.text_rate_limit and ("rate limit" in message or "quota" in message)
        ):
            return ProviderRateLimitError()
        if "connect" in message:
            return ProviderEndpointUnavailableError(f"{self.name} endpoint is unavailable")
        return ProviderEndpointUnavailableError(f"{self.name} request failed")
