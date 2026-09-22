r"""Deterministic two-agent Conducto quick start.

Run from the repository root after installing the built wheel:

    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from conducto import BaseAgent, OrchestratorAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.invocation_results import InvocationApprovalRequired, InvocationSuccess
from conducto.core.logging import configure_logging
from conducto.core.provider import ModelConfiguration, ProviderResult, Usage
from conducto.core.provider_registry import ProviderRegistry
from conducto.security import (
    ApprovalDecision,
    AuthorizationContext,
    Principal,
    require_approval,
    require_scope,
)
from conducto.testing import FakeModel

CARD_BASE_URL = "https://local.conducto.invalid/a2a"
QUICKSTART_CORRELATION_ID = "quickstart-local-001"


def _requires_finance_approval(
    _context: AuthorizationContext, arguments: Mapping[str, Any]
) -> bool:
    """Require finance approval only for high-value invoices."""
    amount = arguments["amount"]
    return isinstance(amount, (int, float)) and amount >= 5000


@a2a_agent(
    name="InvoiceAgent",
    version="1.0.0",
    description="Classifies invoices for deterministic local approval workflows.",
)
class InvoiceAgent(BaseAgent):
    @a2a_capability(
        name="classify_invoice",
        description="Classifies an invoice amount for approval routing.",
    )
    @require_scope("invoices:read")
    @require_approval("finance", condition=_requires_finance_approval)
    def classify_invoice(self, vendor_id: str, amount: float) -> dict[str, object]:
        band = "review" if amount >= 1000 else "auto-approve"
        return {
            "approved": band == "auto-approve",
            "amount": amount,
            "decision": band,
            "vendor_id": vendor_id,
        }


@a2a_agent(
    name="IncidentAgent",
    version="1.0.0",
    description="Summarizes service incidents for local operations handoff.",
)
class IncidentAgent(BaseAgent):
    @a2a_capability(
        name="summarize_incident",
        description="Summarizes an incident severity for an affected service.",
    )
    def summarize_incident(self, service: str, severity: int) -> dict[str, object]:
        priority = "page" if severity >= 4 else "ticket"
        return {
            "priority": priority,
            "service": service,
            "severity": severity,
            "summary": f"{service} severity {severity} requires {priority}",
        }


def default_selection() -> dict[str, Any]:
    """Return the deterministic routing selection used by the CLI example."""
    return {
        "agent_id": "InvoiceAgent",
        "capability_id": "classify_invoice",
        "arguments": {"vendor_id": "vendor-42", "amount": 6000.0},
    }


def create_agents() -> tuple[InvoiceAgent, IncidentAgent]:
    """Create the two reflected agents used in the quick start."""
    return InvoiceAgent(), IncidentAgent()


def create_agent_cards() -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish deterministic Agent Cards for both quick-start agents."""
    invoice, incident = create_agents()
    return (
        invoice.get_agent_card(f"{CARD_BASE_URL}/invoice"),
        incident.get_agent_card(f"{CARD_BASE_URL}/incident"),
    )


def build_orchestrator(selection: dict[str, Any]) -> OrchestratorAgent:
    """Create a local orchestrator whose model deterministically returns ``selection``."""
    registry = ProviderRegistry()
    registry.register_client(
        "quickstart-router",
        FakeModel(
            ProviderResult(
                structured=selection,
                usage=Usage(input_tokens=9, output_tokens=4, total_tokens=13),
            ),
        ),
        ModelConfiguration(provider="fake", model="quickstart-router"),
    )
    orchestrator = OrchestratorAgent(
        model_reference="quickstart-router",
        runtime=Runtime(provider_registry=registry),
    )
    for agent in create_agents():
        orchestrator.register_agent(agent)
    return orchestrator


async def route_once(
    request: str,
    selection: dict[str, Any],
    *,
    correlation_id: str,
) -> tuple[InvocationSuccess, list[dict[str, Any]]]:
    """Route one request and return the success envelope plus structured logs."""
    log_stream = io.StringIO()
    configure_logging(format="json", stream=log_stream)

    orchestrator = build_orchestrator(selection)
    authorization = AuthorizationContext(
        Principal(
            "quickstart-user",
            "quickstart-issuer",
            "quickstart-audience",
            scopes=frozenset({"invoices:read"}),
        ),
        task_id="quickstart-task",
        correlation_id=correlation_id,
    )
    result = await orchestrator.route(
        request,
        correlation_id=correlation_id,
        authorization=authorization,
    )
    if isinstance(result, InvocationApprovalRequired):
        challenge = result.challenge
        print(f"approval required: role={challenge.required_role}; granting local demo approval")
        # Production applications should collect this decision from an authorized approver.
        resumed = await orchestrator.resume_approval(
            challenge.agent_id,
            challenge.capability_id,
            selection["arguments"],
            ApprovalDecision(
                approval_id=challenge.approval_id,
                approved=True,
                decided_at=datetime.now(UTC),
                decided_by="quickstart-finance-reviewer",
                role="finance",
            ),
            authorization=authorization,
        )
        if isinstance(resumed, InvocationSuccess) and result.metadata and resumed.metadata:
            metadata = resumed.metadata.with_prior_model_calls(result.metadata.model_calls)
            resumed = dataclasses.replace(resumed, usage=metadata.usage, metadata=metadata)
        result = resumed

    if not isinstance(result, InvocationSuccess):
        raise RuntimeError(f"Quickstart route failed: {result!r}")

    events = [
        event
        for line in log_stream.getvalue().splitlines()
        if "correlation_id" in (event := json.loads(line))
    ]
    return result, events


async def run_quickstart() -> InvocationSuccess:
    """Execute the documented local-agent flow end to end."""
    result, events = await route_once(
        "Please classify invoice vendor-42 for 6000 dollars.",
        default_selection(),
        correlation_id=QUICKSTART_CORRELATION_ID,
    )
    expected_events = {
        "conducto.model.selected.v1",
        "conducto.capability.arguments_validated.v1",
        "conducto.capability.invocation_started.v1",
        "conducto.capability.invocation_completed.v1",
    }
    observed = {event["event"] for event in events}
    if not expected_events <= observed:
        missing = ", ".join(sorted(expected_events - observed))
        raise RuntimeError(f"Quickstart did not emit expected structured events: {missing}")
    if any(event.get("correlation_id") != QUICKSTART_CORRELATION_ID for event in events):
        raise RuntimeError("Quickstart logs did not preserve the correlation ID")
    return result


def main() -> int:
    result = asyncio.run(run_quickstart())
    selection = default_selection()
    print(
        "quickstart result: "
        f"agent={selection['agent_id']} "
        f"capability={selection['capability_id']} "
        f"value={json.dumps(result.value, sort_keys=True)} "
        f"correlation_id={result.correlation_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
