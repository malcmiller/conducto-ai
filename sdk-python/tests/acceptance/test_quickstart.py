"""Acceptance coverage for the documented two-agent quick-start flow."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from examples.quickstart import (
    CARD_BASE_URL,
    build_orchestrator,
    create_agent_cards,
    default_selection,
    route_once,
)

from conducto import (
    A2A_AGENT_CARD_SPEC_VERSION,
    InvocationSuccess,
    OrchestratorAgent,
)

pytestmark = pytest.mark.acceptance


def _assert_supported_a2a_card(card: Mapping[str, Any]) -> None:
    assert card["protocolVersion"] == A2A_AGENT_CARD_SPEC_VERSION
    assert isinstance(card["name"], str) and card["name"]
    assert isinstance(card["description"], str) and card["description"]
    assert isinstance(card["url"], str) and card["url"].startswith("https://")
    assert card["preferredTransport"] == "JSONRPC"
    assert isinstance(card["version"], str) and card["version"]
    assert card["capabilities"] == {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    }
    assert card["defaultInputModes"] == ["text"]
    assert card["defaultOutputModes"] == ["text"]
    assert isinstance(card["securitySchemes"], dict)
    assert isinstance(card["security"], list)

    skills = card["skills"]
    assert isinstance(skills, Sequence) and not isinstance(skills, (str, bytes)) and skills
    parameter_schemas = card["x-conducto"]["parameters"]
    assert isinstance(parameter_schemas, Mapping)
    for skill in skills:
        assert isinstance(skill, Mapping)
        assert isinstance(skill["id"], str) and skill["id"].startswith("conducto-")
        assert isinstance(skill["name"], str) and skill["name"]
        assert isinstance(skill["description"], str) and skill["description"]
        assert skill["inputModes"] == ["text"]
        assert skill["outputModes"] == ["text"]
        assert skill["id"] in parameter_schemas
        assert parameter_schemas[skill["id"]]["type"] == "object"


def test_quickstart_agent_cards_validate_and_are_deterministic() -> None:
    first_cards = create_agent_cards()
    second_cards = create_agent_cards()

    assert first_cards == second_cards
    assert [card["name"] for card in first_cards] == ["InvoiceAgent", "IncidentAgent"]
    for card in first_cards:
        _assert_supported_a2a_card(card)
        serialized = json.dumps(card, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        assert serialized.isascii()
        assert json.loads(serialized) == card


def test_quickstart_discovers_two_agents_from_one_orchestrator() -> None:
    orchestrator = build_orchestrator(default_selection())

    assert isinstance(orchestrator, OrchestratorAgent)
    assert orchestrator.get_registered_agent_names() == ("IncidentAgent", "InvoiceAgent")
    assert [entry["name"] for entry in orchestrator.get_routing_metadata()] == [
        "IncidentAgent",
        "InvoiceAgent",
    ]
    assert {
        capability["name"]
        for entry in orchestrator.get_routing_metadata()
        for capability in entry["capabilities"]
    } == {"classify_invoice", "summarize_incident"}


def test_quickstart_routes_either_agent_with_the_same_orchestrator_shape() -> None:
    async def exercise() -> None:
        invoice_selection = default_selection()
        incident_selection: dict[str, Any] = {
            "agent_id": "IncidentAgent",
            "capability_id": "summarize_incident",
            "arguments": {"service": "checkout", "severity": 5},
        }

        invoice_result, invoice_logs = await route_once(
            "Classify the invoice.",
            invoice_selection,
            correlation_id="quickstart-invoice",
        )
        incident_result, incident_logs = await route_once(
            "Summarize the incident.",
            incident_selection,
            correlation_id="quickstart-incident",
        )

        assert invoice_result.value == {
            "amount": 6000.0,
            "approved": False,
            "decision": "review",
            "vendor_id": "vendor-42",
        }
        assert incident_result.value == {
            "priority": "page",
            "service": "checkout",
            "severity": 5,
            "summary": "checkout severity 5 requires page",
        }
        assert {event["correlation_id"] for event in invoice_logs} == {"quickstart-invoice"}
        assert {event["correlation_id"] for event in incident_logs} == {"quickstart-incident"}

    asyncio.run(exercise())


def test_quickstart_success_path_has_typed_immutable_envelope_and_provenance() -> None:
    async def exercise() -> None:
        result, logs = await route_once(
            "Classify the invoice.",
            default_selection(),
            correlation_id="quickstart-provenance",
        )

        assert isinstance(result, InvocationSuccess)
        assert result.correlation_id == "quickstart-provenance"
        assert result.value == {
            "amount": 6000.0,
            "approved": False,
            "decision": "review",
            "vendor_id": "vendor-42",
        }
        assert dataclasses.is_dataclass(result)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(result, "correlation_id", "mutated") # noqa: B010
        assert result.metadata is not None
        assert result.metadata.correlation_id == "quickstart-provenance"
        assert [(call.purpose, call.model_reference) for call in result.metadata.model_calls] == [
            ("routing", "quickstart-router")
        ]
        assert result.metadata.usage == result.usage

        event_names = [event["event"] for event in logs]
        assert "conducto.model.selected.v1" in event_names
        assert "conducto.capability.arguments_validated.v1" in event_names
        assert "conducto.capability.invocation_started.v1" in event_names
        assert "conducto.capability.invocation_completed.v1" in event_names
        for event in logs:
            rendered = json.dumps(event, sort_keys=True)
            assert event["correlation_id"] == "quickstart-provenance"
            assert "Classify the invoice." not in rendered
            assert "vendor-42" not in rendered
            assert "1250" not in rendered
            assert "review" not in rendered
            assert "approved" not in rendered
            assert "model_response" not in rendered
            assert "traceback" not in rendered
            assert "credentials" not in rendered

    asyncio.run(exercise())


def test_quickstart_script_executes_and_prints_concise_result() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path("examples") / "quickstart.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        "approval required: role=finance; granting local demo approval",
        (
            "quickstart result: "
            "agent=InvoiceAgent "
            "capability=classify_invoice "
            'value={"amount": 6000.0, "approved": false, "decision": "review", '
            '"vendor_id": "vendor-42"} '
            "correlation_id=quickstart-local-001"
        ),
    ]


def test_quickstart_cards_use_documented_local_urls() -> None:
    invoice_card, incident_card = create_agent_cards()

    assert invoice_card["url"] == f"{CARD_BASE_URL}/invoice"
    assert incident_card["url"] == f"{CARD_BASE_URL}/incident"
