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
import sys


def main() -> int:
    from conducto import (
        A2A_AGENT_CARD_SPEC_VERSION,
        BaseAgent,
        InvocationSuccess,
        OrchestratorAgent,
        a2a_agent,
        a2a_capability,
    )

    @a2a_agent(name="SmokeAgent", version="1.0.0", description="Installed-wheel smoke test agent.")
    class SmokeAgent(BaseAgent):
        @a2a_capability(name="ping", description="Replies with pong.")
        def ping(self) -> str:
            return "pong"

    agent = SmokeAgent()
    card = agent.get_agent_card("https://smoke.conducto.test/a2a")
    assert card["protocolVersion"] == A2A_AGENT_CARD_SPEC_VERSION
    assert card["name"] == "SmokeAgent"

    async def invoke() -> None:
        orchestrator = OrchestratorAgent()
        orchestrator.register_agent(agent)
        result = await orchestrator.invoke("SmokeAgent", "ping", {}, correlation_id="smoke")
        assert isinstance(result, InvocationSuccess)
        assert result.value == "pong"

    asyncio.run(invoke())

    print("Smoke test passed: conducto-ai wheel is importable and functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
