"""End-to-end MCP stdio coverage using the official SDK in-memory transport."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.a2a import invocation_result_to_task
from conducto.core.invocation_results import InvocationSuccess
from conducto.mcp import McpExportPolicy, McpExportRule, McpToolExporter
from conducto.mcp.server import CORRELATION_META_KEY, REASON_META_KEY, McpStdioServer
from conducto.security import Principal

pytestmark = pytest.mark.acceptance

AGENT_ID = "LedgerAgent"
CALLS: list[dict[str, Any]] = []
CANCELLED: list[str] = []


@a2a_agent(name=AGENT_ID, version="1.0.0", description="Records ledger entries.")
class LedgerAgent(BaseAgent):
    """Agent whose single capability is shared by A2A and MCP callers."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_started = asyncio.Event()
        self.wait_cancelled = asyncio.Event()

    @a2a_capability(name="record", description="Records one ledger entry.")
    def record(self, entry: str) -> dict[str, str]:
        """Record an entry and return its stored state."""
        CALLS.append({"entry": entry})
        return {"entry": entry, "state": "recorded"}

    @a2a_capability(name="wait", description="Waits until the caller cancels.")
    async def wait(self, label: str) -> str:
        """Await cancellation so cancellation propagation can be observed."""
        self.wait_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            CANCELLED.append(label)
            self.wait_cancelled.set()
            raise
        return label  # pragma: no cover - the capability never completes


def _principal() -> Principal:
    """Return the immutable stdio principal used by the tests."""
    return Principal(
        subject_id="operator-1",
        issuer="https://issuer.invalid",
        audience="conducto",
    )


def _server(**kwargs: Any) -> tuple[McpStdioServer, Runtime, LedgerAgent]:
    """Build a runtime, registry, exporter, and stdio server for one test."""
    registry = AgentRegistry()
    agent = LedgerAgent()
    registry.register(agent)
    runtime = Runtime(agent_registry=registry)
    exporter = McpToolExporter(
        runtime=runtime,
        policy=McpExportPolicy(
            rules=(
                McpExportRule(AGENT_ID, "record"),
                McpExportRule(AGENT_ID, "wait"),
            )
        ),
        registry=registry,
    )
    return (
        McpStdioServer(exporter=exporter, principal=_principal(), **kwargs),
        runtime,
        agent,
    )


@asynccontextmanager
async def _session(server: McpStdioServer) -> AsyncIterator[ClientSession]:
    """Connect an official SDK client to a served session over in-memory pipes."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        serving = asyncio.ensure_future(server.serve(*server_streams))
        try:
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
        finally:
            await server.aclose()
            serving.cancel()
            try:
                await serving
            except asyncio.CancelledError:
                pass


def test_official_client_initializes_lists_and_calls_exported_tools() -> None:
    server, _, _ = _server()
    CALLS.clear()

    async def run() -> tuple[Any, Any, Any]:
        async with _session(server) as session:
            listed = await session.list_tools()
            success = await session.call_tool("ledgeragent__record", {"entry": "invoice-1"})
            invalid = await session.call_tool("ledgeragent__record", {"entry": 7})
            return listed, success, invalid

    listed, success, invalid = asyncio.run(run())

    assert [tool.name for tool in listed.tools] == ["ledgeragent__record", "ledgeragent__wait"]
    assert listed.tools[0].output_schema is not None
    assert success.is_error is False
    assert success.structured_content == {"result": {"entry": "invoice-1", "state": "recorded"}}
    assert success.meta is not None and success.meta[REASON_META_KEY] == "ok"
    assert invalid.is_error is True
    assert invalid.meta is not None and invalid.meta[REASON_META_KEY] == "invalid_arguments"
    assert CALLS == [{"entry": "invoice-1"}]


def test_unknown_tool_names_are_protocol_errors() -> None:
    server, _, _ = _server()

    async def run() -> MCPError | None:
        async with _session(server) as session:
            try:
                await session.call_tool("missing_tool", {})
            except MCPError as error:
                return error
        return None

    error = asyncio.run(run())

    assert error is not None
    assert "Unknown MCP tool" in str(error)


def test_client_cancellation_propagates_into_the_invocation() -> None:
    server, _, agent = _server()
    CANCELLED.clear()

    async def run() -> None:
        async with _session(server) as session:
            call = asyncio.ensure_future(
                session.call_tool("ledgeragent__wait", {"label": "cancel-me"})
            )
            await asyncio.wait_for(agent.wait_started.wait(), timeout=1)
            call.cancel()
            try:
                await call
            except asyncio.CancelledError:
                pass
            await asyncio.wait_for(agent.wait_cancelled.wait(), timeout=1)

    asyncio.run(run())

    assert CANCELLED == ["cancel-me"]


def test_draining_rejects_new_calls_and_close_is_idempotent() -> None:
    server, _, _ = _server(drain_grace_period=0.0)

    async def run() -> None:
        async with _session(server) as session:
            await session.call_tool("ledgeragent__record", {"entry": "invoice-2"})
            assert server.is_ready is True
            await server.drain()
            assert server.is_ready is False
            with pytest.raises(MCPError, match="draining"):
                await session.call_tool("ledgeragent__record", {"entry": "invoice-3"})
        await server.aclose()

    asyncio.run(run())


def test_mcp_and_a2a_calls_share_one_capability_and_result_contract() -> None:
    server, runtime, agent = _server()
    CALLS.clear()

    async def run() -> tuple[Any, Any]:
        direct = await runtime.invoke(agent, "record", {"entry": "invoice-9"})
        async with _session(server) as session:
            projected = await session.call_tool("ledgeragent__record", {"entry": "invoice-9"})
        return direct, projected

    direct, projected = asyncio.run(run())
    task = invocation_result_to_task(direct, task_id="task-1", context_id="context-1")

    assert isinstance(direct, InvocationSuccess)
    assert projected.structured_content == {"result": direct.value}
    assert json.loads(task.artifacts[0].parts[0].text) == direct.value
    assert CALLS == [{"entry": "invoice-9"}, {"entry": "invoice-9"}]
    assert projected.meta is not None
    assert projected.meta[CORRELATION_META_KEY] != direct.correlation_id
