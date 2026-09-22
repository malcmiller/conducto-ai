"""Minimal MCP stdio export for one canonical Conducto capability.

Run this module to serve the allowlisted capability over standard streams::

    uv run python examples/mcp_stdio_server.py

The embedding application owns process launch, standard-stream plumbing,
principal selection, and process supervision.
"""

import asyncio

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.mcp import McpExportPolicy, McpExportRule, McpStdioServer, McpToolExporter
from conducto.security import Principal


@a2a_agent(name="WeatherAgent", version="1.0.0", description="Provides local weather.")
class WeatherAgent(BaseAgent):
    @a2a_capability(name="temperature", description="Returns a local temperature.")
    def temperature(self, city: str) -> dict[str, object]:
        return {"city": city, "celsius": 21}


def build_server() -> McpStdioServer:
    """Build an MCP stdio server exporting one allowlisted capability."""
    registry = AgentRegistry()
    registry.register(WeatherAgent())
    exporter = McpToolExporter(
        runtime=Runtime(agent_registry=registry),
        policy=McpExportPolicy(rules=(McpExportRule("WeatherAgent", "temperature"),)),
        registry=registry,
    )
    return McpStdioServer(
        exporter=exporter,
        principal=Principal(
            subject_id="local-operator",
            issuer="https://local.invalid",
            audience="conducto",
        ),
        call_timeout=30.0,
    )


async def main() -> None:
    """Serve one MCP stdio session and close the server deterministically."""
    server = build_server()
    try:
        await server.serve_stdio()
    finally:
        await server.aclose()


if __name__ == "__main__":
    asyncio.run(main())
