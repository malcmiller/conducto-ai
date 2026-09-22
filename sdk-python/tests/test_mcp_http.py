"""Authenticated Streamable HTTP MCP adapter coverage."""

from __future__ import annotations

from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.mcp import (
    McpExportError,
    McpExportPolicy,
    McpExportRule,
    McpHttpServer,
    McpToolExporter,
)
from conducto.security import AuthorizationContext, Principal, require_scope

CALLS: list[int] = []


@a2a_agent(name="HttpLedger", version="1.0", description="HTTP MCP test agent.")
class HttpLedger(BaseAgent):
    """Agent that records canonical MCP calls for transport tests."""

    @a2a_capability(name="record", description="Record one value.")
    def record(self, value: int) -> dict[str, int]:
        """Record and return one integer value."""
        CALLS.append(value)
        return {"value": value}

    @a2a_capability(name="protected", description="Return a protected value.")
    @require_scope("ledger.write")
    def protected(self) -> dict[str, str]:
        """Return the protected result."""
        return {"state": "protected"}


def _app() -> McpHttpServer:
    """Build an HTTP adapter with resolver-provided generated identities."""

    def resolve(request: Any) -> AuthorizationContext:
        subject = request.headers.get("x-subject")
        if subject is None:
            raise ValueError("missing identity")
        scopes = (
            frozenset({"ledger.write"})
            if request.headers.get("x-scope") == "write"
            else frozenset()
        )
        return AuthorizationContext(
            principal=Principal(
                subject_id=subject,
                issuer="https://issuer.invalid",
                audience="conducto",
                scopes=scopes,
            ),
            task_id="resolver-task",
            correlation_id="resolver-correlation",
            policy_metadata={"environment": "test"},
        )

    agent = HttpLedger()
    exporter = McpToolExporter(
        runtime=Runtime(),
        policy=McpExportPolicy(
            rules=(
                McpExportRule("HttpLedger", "record"),
                McpExportRule("HttpLedger", "protected"),
            )
        ),
        agents=(agent,),
    )
    return McpHttpServer(
        exporter=exporter,
        authorization_resolver=resolve,
        allowed_hosts=("testserver",),
        allowed_schemes=frozenset({"http"}),
    )


def _initialize(client: TestClient, subject: str, *, scope: str = "") -> str:
    """Initialize one official-protocol session and return its opaque identifier."""
    response = client.post(
        "/mcp",
        headers={"x-subject": subject, "x-scope": scope},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id is not None
    return session_id


def _request_headers(session_id: str, subject: str, *, scope: str = "") -> dict[str, str]:
    """Build authenticated request headers for an established session."""
    return {
        "mcp-session-id": session_id,
        "mcp-protocol-version": "2025-11-25",
        "x-subject": subject,
        "x-scope": scope,
    }


def test_http_sessions_filter_tools_and_cannot_be_exchanged() -> None:
    """Session identity filters discovery and rejects cross-principal reuse."""
    with TestClient(_app()) as client:
        reader_session = _initialize(client, "reader")
        writer_session = _initialize(client, "writer", scope="write")
        reader_tools = client.post(
            "/mcp",
            headers=_request_headers(reader_session, "reader"),
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        writer_tools = client.post(
            "/mcp",
            headers=_request_headers(writer_session, "writer", scope="write"),
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        exchanged = client.post(
            "/mcp",
            headers=_request_headers(reader_session, "writer", scope="write"),
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )

    assert [tool["name"] for tool in reader_tools.json()["result"]["tools"]] == [
        "httpledger__record"
    ]
    assert [tool["name"] for tool in writer_tools.json()["result"]["tools"]] == [
        "httpledger__protected",
        "httpledger__record",
    ]
    assert exchanged.status_code == 404


def test_http_calls_use_exporter_and_reject_replay_before_business_logic() -> None:
    """A canonical call executes once when its JSON-RPC request identity is replayed."""
    CALLS.clear()
    request = {
        "jsonrpc": "2.0",
        "id": "record-1",
        "method": "tools/call",
        "params": {"name": "httpledger__record", "arguments": {"value": 4}},
    }
    with TestClient(_app()) as client:
        session_id = _initialize(client, "operator")
        headers = _request_headers(session_id, "operator")
        first = client.post("/mcp", headers=headers, json=request)
        replay = client.post("/mcp", headers=headers, json=request)

    assert first.status_code == 200
    assert first.json()["result"]["structuredContent"] == {"result": {"value": 4}}
    assert replay.status_code == 200
    assert replay.json()["error"]["message"] == "Duplicate request"
    assert CALLS == [4]


def test_http_authentication_and_transport_failures_do_not_invoke_tools() -> None:
    """Unauthenticated and unsafe requests fail before canonical invocation."""
    CALLS.clear()
    with TestClient(_app()) as client:
        missing_identity = client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        invalid_host = client.post(
            "/mcp",
            headers={"host": "attacker.invalid", "x-subject": "operator"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        forwarded = client.post(
            "/mcp",
            headers={"x-subject": "operator", "x-forwarded-host": "attacker.invalid"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )

    assert missing_identity.status_code == 401
    assert invalid_host.status_code == 421
    assert forwarded.status_code == 400
    assert CALLS == []


def test_http_requires_an_authorization_resolver() -> None:
    """Construction rejects any attempt to configure an anonymous adapter."""
    with pytest.raises(McpExportError):
        McpHttpServer(
            exporter=cast(Any, object()),
            authorization_resolver=cast(Any, None),
            allowed_hosts=("testserver",),
        )
