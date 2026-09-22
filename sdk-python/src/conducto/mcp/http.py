"""Authenticated Streamable HTTP ASGI adapter for Conducto MCP tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from conducto.security.context import AuthorizationContext
from conducto.core.telemetry import (
    SPAN_MCP_SERVER,
    SPAN_MCP_TOOLS_CALL,
    SPAN_MCP_TOOLS_LIST,
    extract_trace_context,
    start_span,
)

from .errors import McpDependencyError, McpExportError, McpToolNotFoundError
from .export import McpToolExporter
from .profile import MCP_EXTRA, conducto_server_version, require_mcp_dependency
from .server import _tool_result

try:
    from mcp import types
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http import MCP_SESSION_ID_HEADER
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.shared.exceptions import MCPError
    from starlette.responses import PlainTextResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise McpDependencyError(
        f"The MCP HTTP adapter requires the official MCP Python SDK; "
        f"install conducto-ai[{MCP_EXTRA}]",
        extra=MCP_EXTRA,
    ) from error

require_mcp_dependency()

DEFAULT_MCP_HTTP_PATH = "/mcp"
DEFAULT_MAX_HEADER_BYTES = 16_384
DEFAULT_MAX_RESPONSE_BYTES = 1_048_576
DEFAULT_MAX_CONCURRENT_REQUESTS = 16
DEFAULT_MAX_REPLAY_IDS = 1_024


@dataclass(frozen=True, slots=True)
class McpHttpRequest:
    """Validated HTTP facts supplied to an application authentication resolver.

    Attributes:
        method: Uppercase HTTP method.
        headers: Immutable lower-case HTTP headers without request bodies.
        client_host: ASGI-provided peer address, if available.
        scheme: ASGI connection scheme.
    """

    method: str
    headers: Mapping[str, str]
    client_host: str | None
    scheme: str


AuthorizationResolver = Callable[
    [McpHttpRequest], AuthorizationContext | Awaitable[AuthorizationContext]
]
"""Application callback that converts already validated identity to authorization."""


@dataclass(frozen=True, slots=True)
class McpHttpLimits:
    """Bounded Streamable HTTP transport limits.

    Attributes:
        max_request_body_bytes: Maximum JSON-RPC request body size.
        max_header_bytes: Maximum aggregate request header size.
        max_response_bytes: Maximum serialized MCP tool result size.
        max_sessions: Maximum active official MCP sessions.
        max_sessions_per_principal: Maximum active sessions for one principal.
        max_concurrent_requests: Maximum concurrent tool calls per session.
        max_replay_ids: Maximum retained request identities per session.
        session_idle_timeout: Idle session lifetime in seconds.
        call_timeout: Application deadline for one tool call, if configured.
    """

    max_request_body_bytes: int = 1_048_576
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_sessions: int = 256
    max_sessions_per_principal: int = 8
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    max_replay_ids: int = DEFAULT_MAX_REPLAY_IDS
    session_idle_timeout: float = 300.0
    call_timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate finite positive resource bounds."""
        integer_limits = (
            self.max_request_body_bytes,
            self.max_header_bytes,
            self.max_response_bytes,
            self.max_sessions,
            self.max_sessions_per_principal,
            self.max_concurrent_requests,
            self.max_replay_ids,
        )
        if any(limit <= 0 for limit in integer_limits):
            raise McpExportError("MCP HTTP limits must be positive")
        if not math.isfinite(self.session_idle_timeout) or self.session_idle_timeout <= 0:
            raise McpExportError("session_idle_timeout must be a positive finite number")
        if self.call_timeout is not None and (
            not math.isfinite(self.call_timeout) or self.call_timeout <= 0
        ):
            raise McpExportError("call_timeout must be a positive finite number")


class McpHttpServer:
    """Serve canonical Conducto MCP tools through authenticated Streamable HTTP.

    The application supplies authentication and owns ASGI server lifecycle,
    TLS termination, reverse-proxy configuration, and process supervision.
    This adapter binds each official MCP session to the resolved immutable
    Conducto authorization facts and delegates all tool execution to the
    existing exporter and ``Runtime.invoke()`` path.
    """

    def __init__(
        self,
        *,
        exporter: McpToolExporter,
        authorization_resolver: AuthorizationResolver,
        path: str = DEFAULT_MCP_HTTP_PATH,
        allowed_hosts: tuple[str, ...],
        allowed_origins: tuple[str, ...] = (),
        trusted_proxy_hosts: frozenset[str] = frozenset(),
        allowed_schemes: frozenset[str] = frozenset({"https"}),
        limits: McpHttpLimits | None = None,
        server_name: str = "conducto",
        server_version: str = "",
    ) -> None:
        """Configure one authenticated Streamable HTTP adapter.

        Args:
            exporter: Immutable canonical MCP exporter.
            authorization_resolver: Application resolver for validated identity.
            path: Exact absolute ASGI path for the MCP endpoint.
            allowed_hosts: Exact hosts accepted by the SDK DNS-rebinding guard.
            allowed_origins: Exact browser origins accepted by that guard.
            trusted_proxy_hosts: Peers allowed to send forwarded headers.
            allowed_schemes: ASGI schemes accepted at this endpoint.
            limits: Bounded transport, session, replay, and invocation limits.
            server_name: Name reported during MCP initialization.
            server_version: Version reported during MCP initialization.

        Raises:
            McpExportError: If required authentication or transport policy is
                absent or malformed.
        """
        if not callable(authorization_resolver):
            raise McpExportError("authorization_resolver is required")
        if not path.startswith("/") or path == "/" or "?" in path or "#" in path:
            raise McpExportError("path must be a non-root absolute path")
        if not allowed_hosts or any(not host or host == "*" for host in allowed_hosts):
            raise McpExportError("allowed_hosts must contain explicit non-wildcard hosts")
        if any(not origin or origin == "*" for origin in allowed_origins):
            raise McpExportError("allowed_origins cannot contain wildcard origins")
        if not allowed_schemes or not allowed_schemes.issubset({"http", "https"}):
            raise McpExportError("allowed_schemes must contain HTTP schemes")
        limits = limits or McpHttpLimits()
        self._exporter = exporter
        self._authorization_resolver = authorization_resolver
        self._path = path
        self._trusted_proxy_hosts = trusted_proxy_hosts
        self._allowed_schemes = allowed_schemes
        self._limits = limits
        self._sessions: dict[str, AuthorizationContext] = {}
        self._principal_sessions: dict[tuple[str, str], set[str]] = {}
        self._replay_ids: dict[str, set[str | int]] = {}
        self._in_flight: dict[str, int] = {}
        self._active_calls: set[asyncio.Task[Any]] = set()
        self._draining = False
        self._closed = False
        self._server: Server[None] = Server(
            server_name,
            version=server_version or conducto_server_version(),
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )
        self._manager = StreamableHTTPSessionManager(
            app=self._server,
            json_response=True,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(allowed_hosts),
                allowed_origins=list(allowed_origins),
            ),
            max_request_body_size=limits.max_request_body_bytes,
            session_idle_timeout=limits.session_idle_timeout,
            max_sessions=limits.max_sessions,
        )

    @property
    def is_ready(self) -> bool:
        """Report whether new authenticated MCP sessions and calls are accepted."""
        return not self._draining and not self._closed

    @property
    def asgi_app(self) -> ASGIApp:
        """Return the ASGI application owned by this adapter."""
        return self

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve ASGI lifespan and HTTP events without constructing a listener."""
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            async with self._manager.run():
                await self._lifespan(scope, receive, send)
            return
        if scope_type != "http":
            return
        if scope.get("path") != self._path:
            await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
            return
        if self._closed or self._draining:
            await PlainTextResponse("Server unavailable", status_code=503)(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        raw_header_names = tuple(key.decode("latin-1").lower() for key, _ in scope["headers"])
        if (
            sum(len(key) + len(value) for key, value in headers.items())
            > self._limits.max_header_bytes
        ):
            await PlainTextResponse("Request headers too large", status_code=431)(
                scope, receive, send
            )
            return
        client = scope.get("client")
        client_host = client[0] if client else None
        if scope.get("scheme") not in self._allowed_schemes:
            await PlainTextResponse("Invalid request scheme", status_code=400)(scope, receive, send)
            return
        if (
            any(key.startswith("x-forwarded-") or key == "forwarded" for key in headers)
            and client_host not in self._trusted_proxy_hosts
        ):
            await PlainTextResponse("Untrusted forwarded headers", status_code=400)(
                scope, receive, send
            )
            return
        extracted = extract_trace_context(headers, raw_header_names=raw_header_names)
        with start_span(
            SPAN_MCP_SERVER,
            kind="server",
            remote_context=extracted.context,
            attributes={
                "conducto.protocol": "mcp",
                "conducto.transport": "streamable_http",
                "conducto.invalid_remote_context": extracted.invalid_remote_context,
                "http.request.method": str(scope.get("method", "")).upper(),
                "url.scheme": str(scope.get("scheme", "")),
            },
        ) as span:
            request = McpHttpRequest(
                method=str(scope.get("method", "")).upper(),
                headers=headers,
                client_host=client_host,
                scheme=str(scope.get("scheme", "")),
            )
            try:
                authorization = self._authorization_resolver(request)
                if inspect.isawaitable(authorization):
                    authorization = await authorization
                if not isinstance(authorization, AuthorizationContext):
                    raise TypeError
            except (TypeError, ValueError):
                span.set_outcome("denied", reason="authentication_failed")
                await PlainTextResponse("Authentication failed", status_code=401)(
                    scope, receive, send
                )
                return
            session_id = headers.get(MCP_SESSION_ID_HEADER)
            if session_id is not None:
                if not self._same_session_identity(self._sessions.get(session_id), authorization):
                    span.set_outcome("session_not_found", reason="session_not_found")
                    await PlainTextResponse("Session not found", status_code=404)(
                        scope, receive, send
                    )
                    return
                await self._manager.handle_request(scope, receive, send)
                span.set_outcome("success")
                return
            await self._opening_request(scope, receive, send, authorization)
            span.set_outcome("success")

    async def drain(self) -> None:
        """Reject new work and cancel accepted calls after their bounded grace."""
        self._draining = True

    async def aclose(self) -> None:
        """Close idempotently and cancel active adapter-owned tool calls."""
        if self._closed:
            return
        self._draining = True
        self._closed = True
        self._sessions.clear()
        self._principal_sessions.clear()
        self._replay_ids.clear()
        self._in_flight.clear()
        current_task = asyncio.current_task()
        active_calls = tuple(task for task in self._active_calls if task is not current_task)
        for task in active_calls:
            task.cancel()
        if active_calls:
            await asyncio.gather(*active_calls, return_exceptions=True)

    async def _lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward the ASGI lifespan protocol while the SDK manager is running."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _opening_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        authorization: AuthorizationContext,
    ) -> None:
        """Bind a successfully initialized official session to authorization."""
        principal_key = (authorization.principal.issuer, authorization.principal.subject_id)
        sessions = self._principal_sessions.setdefault(principal_key, set())
        if len(sessions) >= self._limits.max_sessions_per_principal:
            await PlainTextResponse("Session limit reached", status_code=429)(scope, receive, send)
            return
        session_id = ""

        async def capture(message: Any) -> None:
            nonlocal session_id
            if message["type"] == "http.response.start":
                for key, value in message.get("headers", ()):
                    if key.lower() == MCP_SESSION_ID_HEADER.encode():
                        session_id = value.decode("latin-1")
                        break
            await send(message)

        await self._manager.handle_request(scope, receive, capture)
        if session_id:
            self._sessions[session_id] = authorization
            sessions.add(session_id)

    async def _on_list_tools(
        self,
        context: ServerRequestContext[None, Any],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        """List exporter tools authorized for the bound HTTP session."""
        authorization = self._authorization_for_context(context)
        with start_span(
            SPAN_MCP_TOOLS_LIST,
            attributes={
                "conducto.protocol": "mcp",
                "conducto.transport": "streamable_http",
                "conducto.task.id": authorization.task_id,
                "conducto.correlation_id": authorization.correlation_id,
            },
        ) as span:
            result = types.ListToolsResult(
                tools=[
                    types.Tool(
                        name=definition.name,
                        title=definition.title,
                        description=definition.description,
                        input_schema=definition.input_schema_dict(),
                        output_schema=definition.output_schema_dict(),
                    )
                    for definition in self._exporter.list_tools(authorization.principal)
                ]
            )
            span.set_outcome("success")
            return result

    async def _on_call_tool(
        self,
        context: ServerRequestContext[None, Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        """Dispatch one authorized call through the canonical MCP exporter."""
        if not self.is_ready:
            raise MCPError(types.INVALID_REQUEST, "Server unavailable")
        authorization = self._authorization_for_context(context)
        session_id = self._session_id(context)
        request_id = context.request_id
        if request_id is None:
            raise MCPError(types.INVALID_REQUEST, "Request identity is required")
        replay_ids = self._replay_ids.setdefault(session_id, set())
        if request_id in replay_ids or len(replay_ids) >= self._limits.max_replay_ids:
            raise MCPError(types.INVALID_REQUEST, "Duplicate request")
        if self._in_flight.get(session_id, 0) >= self._limits.max_concurrent_requests:
            raise MCPError(types.INVALID_REQUEST, "Too many concurrent requests")
        replay_ids.add(request_id)
        self._in_flight[session_id] = self._in_flight.get(session_id, 0) + 1
        task_id = f"mcp-http-{uuid4().hex}"
        call_authorization = replace(
            authorization,
            task_id=task_id,
            correlation_id=uuid4().hex,
        )
        task = asyncio.current_task()
        if task is not None:
            self._active_calls.add(task)
        try:
            with start_span(
                SPAN_MCP_TOOLS_CALL,
                attributes={
                    "conducto.protocol": "mcp",
                    "conducto.transport": "streamable_http",
                    "conducto.task.id": task_id,
                    "conducto.correlation_id": call_authorization.correlation_id,
                    "conducto.capability.id": params.name,
                },
            ) as span:
                try:
                    outcome = await self._exporter.call_tool(
                        params.name,
                        params.arguments or {},
                        authorization=call_authorization,
                        task_id=task_id,
                        timeout=self._limits.call_timeout,
                    )
                except McpToolNotFoundError as error:
                    span.set_outcome("not_found", reason="tool_not_found")
                    raise MCPError(types.INVALID_PARAMS, str(error)) from None
                span.set_outcome("success" if not outcome.is_error else "failure")
        finally:
            self._in_flight[session_id] -= 1
            if task is not None:
                self._active_calls.discard(task)
        result = _tool_result(outcome)
        encoded = json.dumps(
            result.model_dump(mode="json", by_alias=True, exclude_none=True),
            separators=(",", ":"),
        ).encode()
        if len(encoded) > self._limits.max_response_bytes:
            raise MCPError(types.INTERNAL_ERROR, "Response exceeds size limit")
        return result

    def _authorization_for_context(
        self, context: ServerRequestContext[None, Any]
    ) -> AuthorizationContext:
        """Return the immutable authorization bound to one initialized session."""
        authorization = self._sessions.get(self._session_id(context))
        if authorization is None:
            raise MCPError(types.INVALID_REQUEST, "Session is not initialized")
        return authorization

    @staticmethod
    def _session_id(context: ServerRequestContext[None, Any]) -> str:
        """Read the SDK-provided session header from a request context."""
        request = context.request
        session_id = None if request is None else request.headers.get(MCP_SESSION_ID_HEADER)
        if not session_id:
            raise MCPError(types.INVALID_REQUEST, "Session is not initialized")
        return str(session_id)

    @staticmethod
    def _same_session_identity(
        bound: AuthorizationContext | None,
        candidate: AuthorizationContext,
    ) -> bool:
        """Compare only immutable identity and policy facts across requests."""
        return bound is not None and (
            bound.principal == candidate.principal
            and bound.policy_metadata == candidate.policy_metadata
        )


__all__ = [
    "AuthorizationResolver",
    "DEFAULT_MCP_HTTP_PATH",
    "McpHttpLimits",
    "McpHttpRequest",
    "McpHttpServer",
]
