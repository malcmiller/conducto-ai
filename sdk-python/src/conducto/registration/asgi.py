"""Dependency-free ASGI boundary for authenticated deployment registration."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any, Protocol

from pydantic import ValidationError

from .models import (
    REQUEST_ADAPTER,
    RegistrationCode,
    RegistrationRequest,
    RegistrationResult,
    result_document,
)

_OPERATIONS = frozenset({"register", "renew", "drain", "deregister", "status", "revoke"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_TRACEPARENT = re.compile(r"(?!ff)[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}")
_STATE_KEY = re.compile(
    r"[a-z][a-z0-9_\-*/]{0,255}|[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z][a-z0-9_\-*/]{0,13}"
)
_SINGLE_HEADERS = frozenset(
    {
        "authorization",
        "x-correlation-id",
        "traceparent",
        "tracestate",
        "content-type",
        "content-length",
    }
)
_UNAVAILABLE = frozenset(
    {
        RegistrationCode.CARD_UNAVAILABLE,
        RegistrationCode.CATALOG_UNAVAILABLE,
        RegistrationCode.SERVICE_UNAVAILABLE,
        RegistrationCode.AUDIT_UNAVAILABLE,
        RegistrationCode.CAPACITY_EXCEEDED,
    }
)
_CONFLICT = frozenset(
    {
        RegistrationCode.IDENTITY_CONFLICT,
        RegistrationCode.IDEMPOTENCY_CONFLICT,
        RegistrationCode.STALE_GENERATION,
        RegistrationCode.REPLAY_REJECTED,
        RegistrationCode.INACTIVE,
        RegistrationCode.EXPIRED,
    }
)


class _InvalidRequest(Exception):
    """Mark an invalid envelope without retaining confidential input."""


class _RequestTooLarge(Exception):
    """Mark a body which exceeded its configured byte budget."""


class _RegistrationHandler(Protocol):
    async def handle(
        self,
        request: RegistrationRequest,
        *,
        authorization_header: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> RegistrationResult: ...


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _InvalidRequest
        document[key] = value
    return document


def _valid_trace_headers(headers: Mapping[str, str]) -> bool:
    parent = headers.get("traceparent")
    if parent is not None:
        base, suffix = parent[:55], parent[55:]
        if (
            _TRACEPARENT.fullmatch(base) is None
            or base[3:35] == "0" * 32
            or base[36:52] == "0" * 16
            or (base[:2] == "00" and suffix)
            or (suffix and (not suffix.startswith("-") or len(suffix) == 1))
        ):
            return False
    state = headers.get("tracestate")
    if state is not None:
        if parent is None or len(state) > 512:
            return False
        members = state.split(",")
        if len(members) > 32:
            return False
        keys: set[str] = set()
        for member in members:
            key, separator, value = member.strip(" \t").partition("=")
            if (
                not separator
                or _STATE_KEY.fullmatch(key) is None
                or key in keys
                or not value
                or len(value) > 256
                or value.endswith(" ")
                or any(
                    ord(character) < 32 or ord(character) > 126 or character == "="
                    for character in value
                )
            ):
                return False
            keys.add(key)
    return True


def _request_headers(scope: Mapping[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("ascii").lower()
        if name not in _SINGLE_HEADERS:
            continue
        if name in headers or len(raw_value) > 8192:
            raise _InvalidRequest
        value = raw_value.decode("ascii")
        if any(ord(character) < 32 or ord(character) > 126 for character in value):
            raise _InvalidRequest
        headers[name] = value
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json" or not _valid_trace_headers(headers):
        raise _InvalidRequest
    return headers


def _status_code(result: RegistrationResult) -> int:
    if result.ok:
        return 200
    if result.code is RegistrationCode.UNAUTHENTICATED:
        return 401
    if result.code in {
        RegistrationCode.UNAUTHORIZED,
        RegistrationCode.INVALID_HANDLE,
        RegistrationCode.ENDPOINT_MISMATCH,
    }:
        return 403
    if result.code in _UNAVAILABLE:
        return 503
    if result.code in _CONFLICT:
        return 409
    return 400


class RegistrationASGI:
    """Expose ``POST /registration/v1/{operation}`` without owning resources.

    Args:
        service: Application-owned service which authenticates every request.
        max_request_bytes: Maximum cumulative body size, independent of headers.
        request_timeout: Total seconds allowed for receiving and handling a request.

    Notes:
        Timeout means the mutation outcome is unknown, not that it was rolled back.
        Clients must preserve the original request and idempotency key when retrying.
        Lifespan messages never close the service or application-owned clients.
    """

    def __init__(
        self,
        service: _RegistrationHandler,
        *,
        max_request_bytes: int = 16384,
        request_timeout: float = 10,
    ) -> None:
        if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int):
            raise ValueError("max_request_bytes must be a positive integer")
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be a positive integer")
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        self._service = service
        self._max_request_bytes = max_request_bytes
        self._request_timeout = request_timeout

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Serve one ASGI scope, preserving cancellation and typed public failures."""
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            raise ValueError("registration only supports HTTP and lifespan scopes")

        correlation = ""
        result = RegistrationResult(code=RegistrationCode.INVALID_REQUEST)
        status = 400
        try:
            async with asyncio.timeout(self._request_timeout):
                path = scope.get("path", "")
                prefix = "/registration/v1/"
                operation = path[len(prefix) :] if path.startswith(prefix) else ""
                if operation not in _OPERATIONS:
                    status = 404
                    raise _InvalidRequest
                if scope.get("method") != "POST":
                    status = 405
                    raise _InvalidRequest
                headers = _request_headers(scope)
                content_length = headers.get("content-length")
                if content_length is not None:
                    if not content_length.isascii() or not content_length.isdecimal():
                        raise _InvalidRequest
                    if len(content_length) > 10 or int(content_length) > self._max_request_bytes:
                        raise _RequestTooLarge
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    if message["type"] != "http.request":
                        raise _InvalidRequest
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > self._max_request_bytes:
                        raise _RequestTooLarge
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
                if content_length is not None and int(content_length) != len(body):
                    raise _InvalidRequest
                try:
                    document = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
                except ValueError:
                    # JSON's integer-length limit raises ValueError, not JSONDecodeError.
                    raise _InvalidRequest from None
                request = REQUEST_ADAPTER.validate_python(document)
                if request.operation != operation:
                    raise _InvalidRequest
                supplied_correlation = headers.get("x-correlation-id")
                if (
                    supplied_correlation is not None
                    and supplied_correlation != request.correlation_id
                ):
                    raise _InvalidRequest
                correlation = request.correlation_id
                result = await self._service.handle(
                    request,
                    authorization_header=headers.get("authorization"),
                    trace_headers={
                        name: value
                        for name, value in headers.items()
                        if name in {"traceparent", "tracestate"}
                    },
                )
                status = _status_code(result)
        except (
            _InvalidRequest,
            ValidationError,
            UnicodeError,
            RecursionError,
        ):
            result = RegistrationResult(code=RegistrationCode.INVALID_REQUEST)
        except _RequestTooLarge:
            status = 413
        except TimeoutError:
            result = RegistrationResult(
                code=RegistrationCode.SERVICE_UNAVAILABLE, correlation_id=correlation
            )
            status = 504

        document = result_document(result) if result.ok else result.model_dump(mode="json")
        response_headers = [
            (b"content-type", b"application/json"),
            (b"cache-control", b"no-store"),
        ]
        if (
            correlation
            and result.correlation_id == correlation
            and _IDENTIFIER.fullmatch(correlation)
        ):
            response_headers.append((b"x-correlation-id", correlation.encode("ascii")))
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": json.dumps(document).encode("utf-8")})
