"""Milestone 1 quick-start acceptance suite.

This module exercises the documented quick-start flow end-to-end exactly as a
new SDK consumer would: define an agent with the public decorators, publish an
A2A Agent Card, register the agent with a local ``OrchestratorAgent``, and
invoke a capability through the orchestrator.

It is intentionally kept independent of the unit test suite so it can be
run as its own CI job/marker and act as the stable, story-level acceptance
gate for Milestone 1 ("Core Protocol & Reflection Discovery"), separate from
fine-grained unit coverage of individual modules.
"""

import asyncio
import json

import pytest

from conducto import (
    BaseAgent,
    InvocationSuccess,
    OrchestratorAgent,
    a2a_agent,
    a2a_capability,
)

pytestmark = pytest.mark.acceptance


@a2a_agent(
    name="AuditAgent",
    version="1.0",
    description="Evaluates transaction risk against compliance rules.",
)
class AuditAgent(BaseAgent):
    @a2a_capability(description="Evaluates transaction risk against compliance rules.")
    def evaluate_transaction_risk(self, vendor_id: str, amount: float) -> dict:
        _ = amount
        return {
            "vendor_id": vendor_id,
            "status": "APPROVED",
            "risk_score": 0.05,
        }


def test_quickstart_defines_agent_and_publishes_a2a_agent_card() -> None:
    """A decorated agent reflects its capability into a standards-conformant card."""
    agent = AuditAgent()

    card = agent.get_agent_card("https://agent.example/a2a")

    assert card["name"] == "AuditAgent"
    assert card["version"] == "1.0"
    assert card["protocolVersion"] == "0.3.0"
    skill = next(skill for skill in card["skills"] if skill["name"] == "evaluate_transaction_risk")
    parameters = card["x-conducto"]["parameters"][skill["id"]]
    assert set(parameters["required"]) == {"vendor_id", "amount"}

    # The canonical JSON form must round-trip and stay stable for consumers
    # that persist or diff it (e.g. `/.well-known/agent-card.json`).
    serialized = agent.get_agent_card_json("https://agent.example/a2a")
    assert json.loads(serialized) == card


def test_quickstart_orchestrator_discovers_and_invokes_the_agent() -> None:
    """The local OrchestratorAgent can discover and invoke the quick-start agent."""

    async def exercise() -> None:
        orchestrator = OrchestratorAgent()
        orchestrator.register_agent(AuditAgent())

        assert "AuditAgent" in {entry["name"] for entry in orchestrator.get_routing_metadata()}

        result = await orchestrator.invoke(
            "AuditAgent",
            "evaluate_transaction_risk",
            {"vendor_id": "vendor-42", "amount": 1250.0},
            correlation_id="quickstart-acceptance",
        )

        assert isinstance(result, InvocationSuccess)
        assert result.value == {
            "vendor_id": "vendor-42",
            "status": "APPROVED",
            "risk_score": 0.05,
        }

    asyncio.run(exercise())
