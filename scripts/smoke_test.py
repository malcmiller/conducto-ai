"""Installed-wheel smoke test.

This script is intentionally standalone (no pytest, no dev dependencies) and
is executed against a *built and installed wheel* in a clean virtual
environment. It proves that the published artifact is importable and that
the documented quick-start flow works without any of the source tree's
development tooling on ``sys.path``.

Usage (from an environment with only the built wheel installed):

    python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import io
import json
import runpy
import sys
from importlib import metadata
from pathlib import Path


def _smoke_a2a_server() -> str:
    """Exercise the optional A2A server extra through the public ASGI API."""
    extras = metadata.metadata("conducto-ai").get_all("Provides-Extra") or []
    if "a2a-server" not in extras:
        return "A2A server smoke skipped: install the 'a2a-server' extra to exercise it."

    import fastapi
    import httpx
    import starlette
    import uvicorn

    assert fastapi.__version__ and starlette.__version__ and uvicorn.__version__

    from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
    from conducto.a2a import (
        A2AAuthenticatedIdentity,
        A2AAuthenticationRequest,
        A2AHostSecurityConfig,
        create_a2a_app,
    )
    from conducto.security import AuthorizationContext, Principal

    @a2a_agent(
        name="SmokeA2AServerAgent",
        version="1.0.0",
        description="Installed-wheel A2A server smoke test agent.",
    )
    class SmokeA2AServerAgent(BaseAgent):
        """Expose one deterministic capability through the standard host factory."""

        @a2a_capability(name="echo", description="Echo a deterministic value.")
        def echo(self, value: str) -> dict[str, str]:
            """Return the supplied value."""
            return {"value": value}

    async def resolve_identity(request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
        """Return deterministic authority for the bounded local smoke request."""
        return A2AAuthenticatedIdentity(
            AuthorizationContext(
                principal=Principal(
                    subject_id="smoke",
                    issuer="smoke",
                    audience="SmokeA2AServerAgent",
                    scopes=frozenset(),
                ),
                task_id=request.task_id,
                correlation_id=request.correlation_id,
            )
        )

    async def exercise() -> None:
        app = create_a2a_app(
            agent=SmokeA2AServerAgent(),
            runtime=Runtime(),
            public_url="http://127.0.0.1:8999",
            identity_resolver=resolve_identity,
            security_config=A2AHostSecurityConfig(readiness_path="/readyz"),
        )
        await app.startup()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8999",
            ) as client:
                ready = await client.get("/readyz")
                card = await client.get("/.well-known/agent-card.json")
            assert ready.json() == {"status": "ready"}
            assert card.json()["name"] == "SmokeA2AServerAgent"
        finally:
            await app.aclose()

    asyncio.run(exercise())
    return "A2A server smoke passed: optional extra metadata and ASGI host are available."


def _smoke_mcp_stdio() -> str:
    """Drive the MCP example server with an official SDK stdio client."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        return "MCP stdio smoke skipped: install the 'mcp' extra to exercise it."

    example = Path(__file__).resolve().parents[1] / "examples" / "mcp_stdio_server.py"
    parameters = StdioServerParameters(command=sys.executable, args=[str(example)])

    async def exercise() -> None:
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert [tool.name for tool in listed.tools] == ["weatheragent__temperature"]
                success = await session.call_tool("weatheragent__temperature", {"city": "Seattle"})
                assert success.is_error is False
                assert success.structured_content == {"result": {"celsius": 21, "city": "Seattle"}}
                invalid = await session.call_tool("weatheragent__temperature", {"city": 7})
                assert invalid.is_error is True

    asyncio.run(exercise())
    return "MCP stdio smoke passed: official SDK client listed and called one exported tool."


def main() -> int:
    import conducto
    from conducto import BaseAgent, OrchestratorAgent, Runtime, a2a_agent, a2a_capability
    from conducto.core.agent_card import A2A_AGENT_CARD_SPEC_VERSION
    from conducto.core.invocation_results import InvocationSuccess
    from conducto.core.logging import configure_logging
    from conducto.core.provider import ModelConfiguration, ProviderResult
    from conducto.core.provider_registry import ProviderRegistry
    from conducto.registration import RegistrationCode, RegistrationResult
    from conducto.registration.asgi import RegistrationASGI
    from conducto.testing import FakeModel

    source_root = Path(__file__).resolve().parents[1] / "src"
    imported_from = Path(conducto.__file__ or "").resolve()
    if imported_from.is_relative_to(source_root):
        raise AssertionError(f"conducto imported from source checkout: {imported_from}")
    assert RegistrationASGI is not None
    assert not RegistrationResult(code=RegistrationCode.SERVICE_UNAVAILABLE).ready

    @a2a_agent(
        name="SmokeInvoiceAgent",
        version="1.0.0",
        description="Installed-wheel invoice smoke test agent.",
    )
    class SmokeInvoiceAgent(BaseAgent):
        @a2a_capability(name="classify", description="Classifies an invoice.")
        def classify(self, vendor_id: str, amount: float) -> dict[str, object]:
            return {"vendor_id": vendor_id, "amount": amount, "approved": amount < 1000}

    @a2a_agent(
        name="SmokeIncidentAgent",
        version="1.0.0",
        description="Installed-wheel incident smoke test agent.",
    )
    class SmokeIncidentAgent(BaseAgent):
        @a2a_capability(name="summarize", description="Summarizes an incident.")
        def summarize(self, service: str, severity: int) -> dict[str, object]:
            return {"service": service, "severity": severity, "priority": severity >= 4}

    agents = (SmokeInvoiceAgent(), SmokeIncidentAgent())
    cards = [
        agents[0].get_agent_card("https://smoke.conducto.test/invoice"),
        agents[1].get_agent_card("https://smoke.conducto.test/incident"),
    ]
    assert [card["supportedInterfaces"][0]["protocolVersion"] for card in cards] == [
        A2A_AGENT_CARD_SPEC_VERSION
    ] * 2
    assert [card["name"] for card in cards] == ["SmokeInvoiceAgent", "SmokeIncidentAgent"]

    async def invoke() -> None:
        stream = io.StringIO()
        configure_logging(format="json", stream=stream)
        registry = ProviderRegistry()
        registry.register_client(
            "smoke-model",
            FakeModel(
                ProviderResult(
                    structured={
                        "agent_id": "SmokeIncidentAgent",
                        "capability_id": "summarize",
                        "arguments": {"service": "checkout", "severity": 5},
                    }
                ),
            ),
            ModelConfiguration(provider="fake", model="smoke-model"),
        )
        orchestrator = OrchestratorAgent(
            model_reference="smoke-model",
            runtime=Runtime(provider_registry=registry),
        )
        for agent in agents:
            orchestrator.register_agent(agent)
        result = await orchestrator.route("summarize checkout", correlation_id="smoke")
        assert isinstance(result, InvocationSuccess)
        assert result.value == {"priority": True, "service": "checkout", "severity": 5}
        assert result.metadata is not None
        assert result.metadata.model_calls[0].model_reference == "smoke-model"
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert {(event["event"], event.get("correlation_id")) for event in events} >= {
            ("conducto.model.selected.v1", "smoke"),
            ("conducto.capability.invocation_completed.v1", "smoke"),
        }

    asyncio.run(invoke())

    print(_smoke_a2a_server())
    example = Path(__file__).resolve().parents[1] / "examples" / "agent_chaining.py"
    chaining = runpy.run_path(str(example))
    asyncio.run(chaining["main"]())

    print(_smoke_mcp_stdio())
    print("Smoke test passed: installed conducto-ai wheel routed local and chained flows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
