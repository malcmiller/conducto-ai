"""Stdio MCP server adapter around the pinned official MCP Python SDK."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from typing import Any

from conducto.core.telemetry import (
    SPAN_MCP_SERVER,
    SPAN_MCP_TOOLS_CALL,
    SPAN_MCP_TOOLS_LIST,
    start_span,
)
from conducto.security.context import Principal

from .errors import McpDependencyError, McpExportError, McpServerStateError, McpToolNotFoundError
from .export import McpToolExporter
from .mapping import McpToolOutcome
from .profile import MCP_EXTRA, conducto_server_version, require_mcp_dependency

try:
    from mcp import types
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.shared.exceptions import MCPError
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise McpDependencyError(
        f"The MCP export adapter requires the official MCP Python SDK; "
        f"install conducto-ai[{MCP_EXTRA}]",
        extra=MCP_EXTRA,
    ) from error

require_mcp_dependency()

CORRELATION_META_KEY = "ai.conducto/correlationId"
"""Result metadata key carrying the Conducto correlation identifier."""

REASON_META_KEY = "ai.conducto/reasonCode"
"""Result metadata key carrying the mapped Conducto reason code."""

PrincipalResolver = Callable[[], Principal | None]


class McpStdioServer:
    """Serve policy-admitted Conducto capabilities over MCP stdio.

    The adapter owns export policy, session identity, runtime dispatch, and
    result mapping. The official SDK owns protocol framing, initialization,
    version negotiation, cancellation notifications, and stdio mechanics.

    Notes:
        The embedding application owns process launch, standard-stream
        plumbing, principal selection, and process supervision. The exported
        tool list is immutable for the lifetime of one server instance.
    """

    def __init__(
        self,
        *,
        exporter: McpToolExporter,
        principal: Principal | None,
        principal_resolver: PrincipalResolver | None = None,
        server_name: str = "conducto",
        server_version: str = "",
        call_timeout: float | None = None,
        drain_grace_period: float = 5.0,
        cancel_on_drain: bool = True,
    ) -> None:
        """Configure one immutable stdio server instance.

        Args:
            exporter: Exporter holding the immutable admitted tool list.
            principal: Immutable session principal, or ``None`` for an
                anonymous session that fails closed on protected capabilities.
            principal_resolver: Callable resolving the session principal when
                the application cannot supply it at construction.
            server_name: Server name advertised during initialization.
            server_version: Server version advertised during initialization.
                Defaults to the installed Conducto distribution version.
            call_timeout: Application deadline applied to every tool call.
            drain_grace_period: Bounded seconds accepted work may continue
                running while the server drains.
            cancel_on_drain: Whether work still running after the grace period
                is cancelled.

        Raises:
            McpExportError: If both a principal and a resolver are supplied or
                if a bound is not positive.
        """
        if principal is not None and principal_resolver is not None:
            raise McpExportError("Configure either a principal or a principal resolver, not both")
        if call_timeout is not None and call_timeout <= 0:
            raise McpExportError("call_timeout must be a positive number of seconds")
        if drain_grace_period < 0:
            raise McpExportError("drain_grace_period cannot be negative")
        self._exporter = exporter
        self._principal = principal
        self._principal_resolver = principal_resolver
        self._call_timeout = call_timeout
        self._drain_grace_period = drain_grace_period
        self._cancel_on_drain = cancel_on_drain
        self._draining = False
        self._closed = False
        self._session_counter = 0
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._serve_task: asyncio.Task[Any] | None = None
        self._server: Server[None] = Server(
            server_name,
            version=server_version or conducto_server_version(),
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    @property
    def mcp_server(self) -> Server[None]:
        """Return the official SDK server owned by this adapter."""
        return self._server

    @property
    def principal(self) -> Principal | None:
        """Return the resolved immutable session principal."""
        if self._principal is None and self._principal_resolver is not None:
            self._principal = self._principal_resolver()
        return self._principal

    @property
    def is_ready(self) -> bool:
        """Report whether the server currently accepts new tool calls."""
        return not (self._draining or self._closed)

    async def serve(self, read_stream: Any, write_stream: Any) -> None:
        """Serve one MCP session over caller-owned streams.

        Args:
            read_stream: Inbound message stream owned by the application.
            write_stream: Outbound message stream owned by the application.

        Raises:
            McpServerStateError: If the server is already closed or serving.
        """
        if self._closed:
            raise McpServerStateError("This MCP server instance is closed")
        if self._serve_task is not None:
            raise McpServerStateError("This MCP server instance is already serving")
        self._serve_task = asyncio.current_task()
        try:
            with start_span(
                SPAN_MCP_SERVER,
                kind="server",
                attributes={
                    "conducto.protocol": "mcp",
                    "conducto.transport": "stdio",
                },
            ) as span:
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
                span.set_outcome("success")
        finally:
            self._serve_task = None

    async def serve_stdio(self) -> None:
        """Serve one MCP session over the process standard streams."""
        async with stdio_server() as (read_stream, write_stream):
            await self.serve(read_stream, write_stream)

    async def drain(self) -> None:
        """Reject new calls and bound the lifetime of accepted work.

        Notes:
            Draining is idempotent. Accepted work runs for at most the
            configured grace period; remaining work is cancelled when the
            configured policy allows it.
        """
        self._draining = True
        deadline = asyncio.get_running_loop().time() + self._drain_grace_period
        while self._in_flight and asyncio.get_running_loop().time() < deadline:
            pending = tuple(self._in_flight)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.wait(pending, timeout=remaining)
        if self._in_flight and self._cancel_on_drain:
            for task in tuple(self._in_flight):
                task.cancel()

    async def aclose(self) -> None:
        """Drain, release owned SDK resources, and close idempotently."""
        if self._closed:
            return
        await self.drain()
        self._closed = True
        serve_task = self._serve_task
        if serve_task is not None and serve_task is not asyncio.current_task():
            serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task

    async def _on_list_tools(
        self,
        context: ServerRequestContext[None, Any],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        """Return the tools admitted by policy for the configured principal."""
        with start_span(
            SPAN_MCP_TOOLS_LIST,
            attributes={
                "conducto.protocol": "mcp",
                "conducto.transport": "stdio",
            },
        ) as span:
            tools = [
                types.Tool(
                    name=definition.name,
                    title=definition.title,
                    description=definition.description,
                    input_schema=definition.input_schema_dict(),
                    output_schema=definition.output_schema_dict(),
                )
                for definition in self._exporter.list_tools(self.principal)
            ]
            span.set_outcome("success")
            return types.ListToolsResult(tools=tools)

    async def _on_call_tool(
        self,
        context: ServerRequestContext[None, Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        """Dispatch one tool call through the runtime and map its outcome."""
        if not self.is_ready:
            raise MCPError(types.INVALID_REQUEST, "This MCP server is draining")
        arguments: Mapping[str, Any] = params.arguments or {}
        task = asyncio.current_task()
        if task is not None:
            self._in_flight.add(task)
        try:
            with start_span(
                SPAN_MCP_TOOLS_CALL,
                attributes={
                    "conducto.protocol": "mcp",
                    "conducto.transport": "stdio",
                    "conducto.capability.id": params.name,
                },
            ) as span:
                try:
                    outcome = await self._exporter.call_tool(
                        params.name,
                        arguments,
                        principal=self.principal,
                        task_id=self._next_task_id(),
                        timeout=self._call_timeout,
                    )
                except McpToolNotFoundError as error:
                    span.set_outcome("not_found", reason="tool_not_found")
                    raise MCPError(types.INVALID_PARAMS, str(error)) from None
                span.set_outcome("success" if not outcome.is_error else "failure")
        finally:
            if task is not None:
                self._in_flight.discard(task)
        return _tool_result(outcome)

    def _next_task_id(self) -> str:
        """Return a monotonic task identifier for audit and approval lineage."""
        self._session_counter += 1
        return f"mcp-stdio-{self._session_counter}"


def _tool_result(outcome: McpToolOutcome) -> types.CallToolResult:
    """Convert a mapped Conducto outcome into an MCP tool result."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=outcome.message)],
        structured_content=(
            None if outcome.structured_content is None else dict(outcome.structured_content)
        ),
        is_error=outcome.is_error,
        _meta={
            CORRELATION_META_KEY: outcome.correlation_id,
            REASON_META_KEY: outcome.reason_code,
        },
    )


__all__ = ["CORRELATION_META_KEY", "REASON_META_KEY", "McpStdioServer", "PrincipalResolver"]
