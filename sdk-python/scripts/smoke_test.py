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

import ast
import asyncio
import io
import json
import runpy
import sys
from pathlib import Path


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
    from conducto import (
        A2A_AGENT_CARD_SPEC_VERSION,
        BaseAgent,
        FakeModel,
        InvocationSuccess,
        ModelConfiguration,
        OrchestratorAgent,
        a2a_agent,
        a2a_capability,
        configure_logging,
    )

    source_root = Path(__file__).resolve().parents[1] / "src"
    imported_from = Path(conducto.__file__ or "").resolve()
    if imported_from.is_relative_to(source_root):
        raise AssertionError(f"conducto imported from source checkout: {imported_from}")

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
        orchestrator = OrchestratorAgent(
            model_provider=FakeModel(
                {
                    "agent_id": "SmokeIncidentAgent",
                    "capability_id": "summarize",
                    "arguments": {"service": "checkout", "severity": 5},
                },
            ),
            model_config=ModelConfiguration(provider="fake", model="smoke-model"),
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

    example = Path(__file__).resolve().parents[1] / "examples" / "agent_chaining.py"
    tree = ast.parse(example.read_text(encoding="utf-8"), filename=str(example))
    for node in ast.walk(tree):
        module = (
            node.module
            if isinstance(node, ast.ImportFrom)
            else next((name.name for name in node.names if name.name == "conducto.core"), None)
            if isinstance(node, ast.Import)
            else None
        )
        if module == "conducto.core" or (
            isinstance(module, str) and module.startswith("conducto.core.")
        ):
            raise AssertionError("agent chaining example imports a non-public conducto.core module")
    chaining = runpy.run_path(str(example))
    asyncio.run(chaining["main"]())

    print(_smoke_mcp_stdio())
    print("Smoke test passed: installed conducto-ai wheel routed local and chained flows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
