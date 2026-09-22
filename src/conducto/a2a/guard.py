"""ASGI hardening boundary wrapping one Conducto A2A protocol application.

This module owns the HTTP boundary that runs before and around the Story 4.4
protocol adapter: scope validation, bounded request and response bodies,
non-queuing concurrency admission, probe endpoints, and the distinction between
client disconnect, caller cancellation, server deadline, drain, and overload. It
never dispatches JSON-RPC, parses A2A payloads, or invokes capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, Final

from starlette.types import Receive, Scope, Send

from conducto.core.logging import emit_event

from .errors import A2ARequestRejectedError
from .hardening import (
    A2AHostSecurityConfig,
    A2AMTLSIdentityExtractor,
    A2ATransportFacts,
    evaluate_request,
    sanitize_scope,
)
from .lifecycle import A2AConcurrencyLimiter, A2AHostLifecycle

_JSON_CONTENT_TYPE: Final = b"application/json"
_NO_STORE: Final = b"no-store"
_SAFE_RESPONSE_BYTE_CEILING: Final = 512
"""Fixed internal bound for probe/rejection responses, independent of the
user-configurable ``max_response_body_bytes``. These responses carry only a
small, fixed vocabulary of internal reason codes; this ceiling is a defensive
guarantee that they are never emitted unbounded, even under a very small
configured ``max_response_body_bytes`` used to bound arbitrary protocol-adapter
output."""


class _RequestBodyLimitExceeded(Exception):
    """Raised internally when an inbound body exceeds its configured bound."""


class _ResponseLimitExceeded(Exception):
    """Raised internally when a response exceeds its configured bound."""


class _RequestChannel:
    """Bounded inbound message pump that also observes client disconnects.

    Args:
        receive: Server-supplied ASGI receive callable.
        max_body_bytes: Maximum accepted aggregate request body bytes.
        max_chunks: Maximum accepted request body chunks.

    Notes:
        Buffered bytes are bounded by ``max_body_bytes`` and buffered messages by
        ``max_chunks``, so a slow or hostile client cannot grow server memory.
    """

    __slots__ = (
        "_chunks",
        "_max_body_bytes",
        "_max_chunks",
        "_pump",
        "_queue",
        "_receive",
        "_seen",
        "disconnected",
        "limit_exceeded",
    )

    def __init__(self, receive: Receive, *, max_body_bytes: int, max_chunks: int) -> None:
        self._receive = receive
        self._max_body_bytes = max_body_bytes
        self._max_chunks = max_chunks
        self._queue: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._seen = 0
        self._chunks = 0
        self.disconnected = asyncio.Event()
        self.limit_exceeded = asyncio.Event()

    def start(self) -> None:
        """Begin pumping inbound messages in a bounded background task."""
        self._pump = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        """Stop the pump without waiting on further client traffic."""
        pump = self._pump
        if pump is None:
            return
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump

    async def receive(self) -> MutableMapping[str, Any]:
        """Return the next inbound message accepted by the bounded pump."""
        return await self._queue.get()

    async def _run(self) -> None:
        """Read inbound messages, enforcing byte and chunk bounds as they arrive."""
        while True:
            message = await self._receive()
            if message.get("type") == "http.disconnect":
                self.disconnected.set()
                await self._queue.put(message)
                return
            body = message.get("body", b"")
            self._seen += len(body)
            self._chunks += 1
            if self._seen > self._max_body_bytes or self._chunks > self._max_chunks:
                self.limit_exceeded.set()
                return
            await self._queue.put(message)


class _ResponseBuffer:
    """Bounded response accumulator that emits one complete, safe response.

    Args:
        send: Server-supplied ASGI send callable.
        max_bytes: Maximum emitted response body bytes.
    """

    __slots__ = ("_chunks", "_max_bytes", "_send", "_start", "_total", "emitted")

    def __init__(self, send: Send, *, max_bytes: int) -> None:
        self._send = send
        self._max_bytes = max_bytes
        self._start: dict[str, Any] | None = None
        self._chunks: list[bytes] = []
        self._total = 0
        self.emitted = False

    @property
    def has_response(self) -> bool:
        """Return whether a complete response has been buffered."""
        return self._start is not None

    async def send(self, message: MutableMapping[str, Any]) -> None:
        """Accumulate one outbound ASGI message within the configured bound.

        Raises:
            _ResponseLimitExceeded: If the buffered response exceeds its bound.
        """
        kind = message.get("type")
        if kind == "http.response.start":
            self._start = dict(message)
            return
        if kind != "http.response.body":
            return
        body = bytes(message.get("body", b""))
        self._total += len(body)
        if self._total > self._max_bytes:
            self._chunks.clear()
            raise _ResponseLimitExceeded
        self._chunks.append(body)

    async def flush(self) -> None:
        """Emit the buffered response exactly once."""
        if self.emitted or self._start is None:
            return
        self.emitted = True
        await self._send(self._start)
        await self._send(
            {"type": "http.response.body", "body": b"".join(self._chunks), "more_body": False}
        )


class A2ARequestGuard:
    """Hardening boundary executed before and around the A2A protocol adapter.

    Args:
        app: Wrapped protocol application built by the Story 4.4 adapter.
        config: Immutable hardening policy for this host.
        lifecycle: Operational lifecycle owning readiness, drain, and close.
        limiter: Non-queuing admission gate for accepted requests.
        mtls_identity_extractor: Optional server-owned verified-identity seam.
    """

    __slots__ = ("_app", "_config", "_extractor", "_lifecycle", "_limiter")

    def __init__(
        self,
        *,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        config: A2AHostSecurityConfig,
        lifecycle: A2AHostLifecycle,
        limiter: A2AConcurrencyLimiter,
        mtls_identity_extractor: A2AMTLSIdentityExtractor | None = None,
    ) -> None:
        self._app = app
        self._config = config
        self._lifecycle = lifecycle
        self._limiter = limiter
        self._extractor = mtls_identity_extractor

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate, admit, and bound one inbound HTTP request."""
        try:
            facts = evaluate_request(
                scope, config=self._config, mtls_identity_extractor=self._extractor
            )
        except A2ARequestRejectedError as rejection:
            await _reject(send, rejection.status_code, rejection.reason)
            return
        if await self._serve_probe(facts, send):
            return
        if not self._lifecycle.is_ready():
            await _reject(send, 503, "service_unavailable")
            return
        if not self._limiter.try_acquire(facts.caller_key):
            await _reject(send, 429, "concurrency_limit_reached")
            return
        try:
            await self._serve(sanitize_scope(scope, facts), receive, send)
        finally:
            self._limiter.release(facts.caller_key)

    async def _serve_probe(self, facts: A2ATransportFacts, send: Send) -> bool:
        """Serve a liveness or readiness probe without exposing dependency detail."""
        if facts.path == self._config.liveness_path:
            alive = self._lifecycle.is_alive()
            await _respond(send, 200 if alive else 503, {"status": "alive" if alive else "closed"})
            return True
        if facts.path == self._config.readiness_path:
            ready = self._lifecycle.is_ready()
            await _respond(send, 200 if ready else 503, {"status": "ready" if ready else "unready"})
            return True
        return False

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the protocol adapter under bounded body, response, and deadline policy."""
        channel = _RequestChannel(
            receive,
            max_body_bytes=self._config.max_request_body_bytes,
            max_chunks=self._config.max_request_body_chunks,
        )
        channel.start()
        buffer = _ResponseBuffer(send, max_bytes=self._config.max_response_body_bytes)
        execution: asyncio.Task[None] = asyncio.create_task(
            _invoke(self._app, scope, channel.receive, buffer.send)
        )
        self._lifecycle.track(execution)
        disconnect = asyncio.create_task(channel.disconnected.wait())
        overflow = asyncio.create_task(channel.limit_exceeded.wait())
        try:
            try:
                done, _ = await asyncio.wait(
                    {execution, disconnect, overflow},
                    timeout=self._config.request_deadline_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                await self._cancel(execution, reason="caller_cancelled")
                raise
            if execution in done:
                await self._complete(execution, buffer, send)
                return
            if channel.limit_exceeded.is_set():
                await self._cancel(execution, reason="request_body_too_large")
                await _reject(send, 413, "request_body_too_large")
                return
            if channel.disconnected.is_set():
                await self._cancel(execution, reason="client_disconnected")
                return
            await self._cancel(execution, reason="request_deadline_exceeded")
            await _reject(send, 504, "request_deadline_exceeded")
        finally:
            disconnect.cancel()
            overflow.cancel()
            if execution.done():
                self._lifecycle.untrack(execution)
            else:
                execution.add_done_callback(self._reap)
            await channel.aclose()

    async def _complete(
        self, execution: asyncio.Task[None], buffer: _ResponseBuffer, send: Send
    ) -> None:
        """Emit the buffered response or an explicit safe failure."""
        if execution.cancelled():
            await _reject(send, 503, "request_cancelled")
            return
        error = execution.exception()
        if isinstance(error, _ResponseLimitExceeded):
            emit_event(
                "a2a.request.rejected", outcome="failure", error_category="response_too_large"
            )
            await _reject(send, 500, "response_too_large")
            return
        if error is not None:
            emit_event(
                "a2a.request.failed", outcome="failure", error_category="protocol_adapter_failed"
            )
            await _reject(send, 500, "internal_error")
            return
        await buffer.flush()

    async def _cancel(self, execution: asyncio.Task[None], *, reason: str) -> None:
        """Cancel the protocol adapter and bound the cooperative cancellation wait.

        Notes:
            When ``execution`` does not reach a terminal state within the bounded
            wait, it is left running and lifecycle-tracked; the caller registers
            :meth:`_reap` so the execution is never abandoned outside drain and
            shutdown accounting, and it still terminalizes through the protocol
            adapter's own cancellation-handling path once it finishes.
        """
        emit_event(
            "a2a.request.terminated",
            outcome="cancelled" if reason != "request_deadline_exceeded" else "timeout",
            error_category=reason,
        )
        if execution.done():
            return
        execution.cancel()
        await asyncio.wait({execution}, timeout=self._config.cancellation_deadline_seconds)

    def _reap(self, execution: asyncio.Task[None]) -> None:
        """Untrack one execution that outlived its bounded cancellation wait.

        Args:
            execution: Completed protocol-adapter task registered as a done
                callback after a bounded cancellation wait expired.

        Notes:
            The execution's outcome is retrieved so no unhandled exception or
            cancellation is ever left unconsumed by the event loop.
        """
        if not execution.cancelled():
            error = execution.exception()
            if error is not None:
                emit_event(
                    "a2a.request.reaped",
                    outcome="failure",
                    error_category="orphaned_execution_failed",
                )
        self._lifecycle.untrack(execution)


async def _invoke(
    app: Callable[[Scope, Receive, Send], Awaitable[None]],
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """Await the wrapped application as a coroutine so it can be cancelled."""
    await app(scope, receive, send)


async def _respond(send: Send, status: int, payload: dict[str, str]) -> None:
    """Send one small JSON response with no dependency or credential detail.

    Notes:
        The emitted body is bounded by a fixed internal ceiling independent of
        the configurable ``max_response_body_bytes``, so a probe or rejection
        response can never be emitted unbounded even under a very small
        configured limit for arbitrary protocol-adapter output. This never
        triggers for the fixed, well-known reason vocabulary this module emits;
        it is a defensive guarantee, not a normal code path.
    """
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    if len(body) > _SAFE_RESPONSE_BYTE_CEILING:
        status = 500
        body = json.dumps({"reason": "response_too_large"}, sort_keys=True).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", _JSON_CONTENT_TYPE),
                (b"content-length", str(len(body)).encode("latin-1")),
                (b"cache-control", _NO_STORE),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _reject(send: Send, status: int, reason: str) -> None:
    """Send one safe rejection carrying only a stable reason code."""
    emit_event("a2a.request.rejected", outcome="failure", error_category=reason)
    await _respond(send, status, {"reason": reason})


__all__ = ["A2ARequestGuard"]
