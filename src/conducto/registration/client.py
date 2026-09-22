"""Optional typed HTTP client for the deployment registration control plane."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from conducto.core.telemetry import inject_trace_context

from .models import (
    RegisterRequest,
    RegistrationCode,
    RegistrationRequest,
    RegistrationResult,
    RenewRequest,
    request_document,
)

_MAX_RESPONSE_BYTES = 16384
_RETRYABLE = frozenset(
    {
        RegistrationCode.SERVICE_UNAVAILABLE,
        RegistrationCode.CARD_UNAVAILABLE,
        RegistrationCode.CATALOG_UNAVAILABLE,
        RegistrationCode.AUDIT_UNAVAILABLE,
        RegistrationCode.CAPACITY_EXCEEDED,
    }
)


class _InvalidResponse(Exception):
    """Mark an untrusted response without retaining its confidential contents."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise _InvalidResponse
    return dict(pairs)


def _endpoint_origin(endpoint: str, allow_insecure_loopback: bool) -> str:
    try:
        parts = urlsplit(endpoint)
        port = parts.port
        hostname = parts.hostname
    except ValueError:
        raise ValueError("registration endpoint must be a valid origin") from None
    if (
        not hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or "?" in endpoint
        or "#" in endpoint
        or "\\" in endpoint
        or any(ord(character) <= 32 or ord(character) >= 127 for character in endpoint)
        or port == 0
    ):
        raise ValueError("registration endpoint must be a credential-free origin")
    if parts.scheme != "https":
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        if parts.scheme != "http" or not allow_insecure_loopback or not loopback:
            raise ValueError("registration endpoint requires HTTPS")
    return endpoint.rstrip("/")


def _valid_success(request: RegistrationRequest, result: RegistrationResult) -> bool:
    if not result.ok:
        return result.lease_handle is None
    if result.generation <= 0 or result.state is None:
        return False
    if result.state == "active" and (
        result.lease_expires_at is None or result.lease_expires_at <= request.issued_at
    ):
        return False
    if isinstance(request, RegisterRequest):
        return (
            result.state == "active"
            and result.lease_handle is not None
            and 0 < len(result.lease_handle.get_secret_value()) <= 256
            and result.generation > request.expected_generation
        )
    if isinstance(request, RenewRequest):
        return (
            result.state == "active"
            and result.generation > request.expected_generation
            and (
                result.lease_handle is None
                or result.lease_handle.get_secret_value() == request.lease_handle.get_secret_value()
            )
        )
    if request.operation == "drain":
        # Draining an already-draining instance is an intentional generation-preserving no-op.
        return result.state == "draining" and result.generation >= request.expected_generation
    expected_state = {"deregister": "removed", "revoke": "revoked"}
    if request.operation in expected_state:
        return (
            result.state == expected_state[request.operation]
            and result.generation > request.expected_generation
        )
    return True


class RegistrationClient:
    """Send immutable registration requests using an application-owned HTTP client.

    Args:
        endpoint: Exact HTTPS origin, with no credentials, query, fragment, or path.
        http_client: Client owned and closed by the application, never this adapter.
        token_provider: Async provider returning a bare bearer token for each attempt.
        max_attempts: One to three attempts, preserving the exact original request.
        request_timeout: Total seconds allowed per attempt, including token acquisition.
        allow_insecure_loopback: Explicit HTTP opt-in for loopback development only.

    Notes:
        Redirects are never followed. Transport failures and typed unavailable outcomes
        alone are retried. Timeout leaves mutation outcome uncertain; callers must not
        change the request or idempotency key when resolving that uncertainty.
        Terminal mutations require an advanced generation; an already-draining
        instance may acknowledge drain without advancing its generation.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        http_client: httpx.AsyncClient,
        token_provider: Callable[[], Awaitable[str]],
        max_attempts: int = 1,
        request_timeout: float = 10,
        allow_insecure_loopback: bool = False,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer from one to three")
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be an integer from one to three")
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        self._endpoint = _endpoint_origin(endpoint, allow_insecure_loopback)
        self._http_client = http_client
        self._token_provider = token_provider
        self._max_attempts = max_attempts
        self._request_timeout = request_timeout

    async def send(self, request: RegistrationRequest) -> RegistrationResult:
        """Send a typed operation, returning safe failures without remote error text.

        Cancellation propagates. Credentials, grant handles, and raw remote validation
        details are never attached to returned failures or emitted to telemetry.
        """
        payload = json.dumps(request_document(request), separators=(",", ":")).encode("utf-8")
        unavailable = RegistrationResult(
            code=RegistrationCode.SERVICE_UNAVAILABLE, correlation_id=request.correlation_id
        )
        result = unavailable
        for _ in range(self._max_attempts):
            retryable = True
            try:
                async with asyncio.timeout(self._request_timeout):
                    token = await self._token_provider()
                    if not token or any(
                        ord(character) <= 32 or ord(character) >= 127 for character in token
                    ):
                        return RegistrationResult(
                            code=RegistrationCode.UNAUTHENTICATED,
                            correlation_id=request.correlation_id,
                        )
                    headers = inject_trace_context(
                        {
                            "authorization": f"Bearer {token}",
                            "content-type": "application/json",
                            "accept": "application/json",
                            "accept-encoding": "identity",
                            "x-correlation-id": request.correlation_id,
                        }
                    )
                    async with self._http_client.stream(
                        "POST",
                        f"{self._endpoint}/registration/v1/{request.operation}",
                        content=payload,
                        headers=headers,
                        auth=None,
                        timeout=self._request_timeout,
                        follow_redirects=False,
                    ) as response:
                        if response.is_redirect:
                            return unavailable
                        if (
                            response.headers.get("content-encoding", "identity").lower()
                            != "identity"
                        ):
                            return unavailable
                        if (
                            response.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                            != "application/json"
                        ):
                            return unavailable
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                                return unavailable
                            body.extend(chunk)
                        try:
                            document = json.loads(
                                body.decode("utf-8"), object_pairs_hook=_unique_object
                            )
                        except ValueError:
                            # JSON's integer-length limit is also an invalid wire response.
                            raise _InvalidResponse from None
                        result = RegistrationResult.model_validate(document)
                        if (
                            result.correlation_id != request.correlation_id
                            or response.headers.get("x-correlation-id", request.correlation_id)
                            != request.correlation_id
                            or (not response.is_success and result.ok)
                            or not _valid_success(request, result)
                        ):
                            return unavailable
                        retryable = result.code in _RETRYABLE
            except (httpx.TransportError, TimeoutError):
                result = unavailable
            except (
                _InvalidResponse,
                ValidationError,
                UnicodeError,
                RecursionError,
                httpx.DecodingError,
            ):
                return unavailable
            if not retryable:
                return result
        return result
