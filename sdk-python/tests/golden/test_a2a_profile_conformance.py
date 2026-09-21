"""Golden conformance tests for Conducto's pinned A2A 1.0 profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conducto import (
    A2A_JSONRPC_BINDING,
    A2A_NORMATIVE_COMMIT,
    A2A_NORMATIVE_PROTO_SHA256,
    A2A_NORMATIVE_SOURCE,
    A2A_PROTOCOL_RELEASE,
    A2A_PROTOCOL_VERSION,
    A2A_PYTHON_SDK_PACKAGE,
    A2A_PYTHON_SDK_VERSION,
    A2AProtocolError,
    parse_agent_card,
    parse_message,
    parse_task,
    validate_jsonrpc_method,
    validate_task_transition,
)

pytestmark = pytest.mark.golden

FIXTURE = Path(__file__).parent / "fixtures" / "a2a" / "a2a_1_0_conformance.json"


@pytest.fixture(scope="module")
def conformance_fixture() -> dict[str, Any]:
    """Return the language-neutral A2A 1.0 conformance fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_profile_fixture_records_pinned_a2a_provenance(
    conformance_fixture: dict[str, Any],
) -> None:
    profile = conformance_fixture["profile"]

    assert profile["protocolRelease"] == A2A_PROTOCOL_RELEASE == "1.0.0"
    assert profile["protocolVersion"] == A2A_PROTOCOL_VERSION == "1.0"
    assert profile["jsonrpcBinding"] == A2A_JSONRPC_BINDING == "JSONRPC"
    assert profile["normativeSource"] == A2A_NORMATIVE_SOURCE
    assert profile["normativeCommit"] == A2A_NORMATIVE_COMMIT
    assert profile["normativeProtoSha256"] == A2A_NORMATIVE_PROTO_SHA256
    assert profile["pythonSdk"] == {
        "package": A2A_PYTHON_SDK_PACKAGE,
        "version": A2A_PYTHON_SDK_VERSION,
    }
    assert profile["directMessageResponses"] is False


def test_positive_a2a_conformance_fixtures_parse_with_official_sdk(
    conformance_fixture: dict[str, Any],
) -> None:
    positive = conformance_fixture["positive"]

    card = parse_agent_card(positive["agentCard"])
    message = parse_message(positive["messageMaxBoundary"])
    task = parse_task(positive["taskMaxBoundary"])

    assert card.name == "GoldenAgent"
    assert len(message.parts) == conformance_fixture["profile"]["limits"]["messageParts"]
    assert len(task.history) == conformance_fixture["profile"]["limits"]["historyMessages"]
    assert len(task.artifacts[0].parts) == conformance_fixture["profile"]["limits"]["artifactParts"]
    for method in conformance_fixture["profile"]["supportedMethods"]:
        validate_jsonrpc_method(method)


@pytest.mark.parametrize(
    ("fixture_name", "validator", "match"),
    [
        ("legacy03AgentCard", parse_agent_card, "0.3 protocolVersion"),
        ("unknownRequiredExtensionCard", parse_agent_card, "Unsupported required"),
        ("unsupportedMediaMessage", parse_message, "Unsupported media type"),
        ("overLimitMessageParts", parse_message, "Message parts exceed"),
        ("overLimitMetadataMessage", parse_message, "Metadata exceeds"),
        ("overLimitTaskHistory", parse_task, "Task history exceeds"),
        ("overLimitArtifactParts", parse_task, "Artifact parts exceed"),
    ],
)
def test_negative_a2a_conformance_fixtures_fail_explicitly(
    conformance_fixture: dict[str, Any],
    fixture_name: str,
    validator: Any,
    match: str,
) -> None:
    with pytest.raises(A2AProtocolError, match=match):
        validator(conformance_fixture["negative"][fixture_name])


def test_task_transitions_and_deferred_methods_fail_explicitly(
    conformance_fixture: dict[str, Any],
) -> None:
    transition = conformance_fixture["negative"]["terminalTransition"]
    with pytest.raises(A2AProtocolError, match="Terminal A2A tasks are immutable"):
        validate_task_transition(transition["currentState"], transition["nextState"])
    with pytest.raises(A2AProtocolError, match="Unsupported A2A JSON-RPC method"):
        validate_jsonrpc_method(conformance_fixture["negative"]["unsupportedMethod"])
