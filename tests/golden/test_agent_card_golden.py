"""Schema/golden-fixture tests for the A2A Agent Card contract.

These tests pin the exact, byte-for-byte serialization of a generated Agent
Card against a checked-in golden fixture. A diff here signals an unintentional
(and potentially breaking) change to the public wire schema used by A2A
network consumers.
"""

from pathlib import Path

import pytest

from conducto import BaseAgent, a2a_agent, a2a_capability

pytestmark = pytest.mark.golden

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@a2a_agent(
    name="GoldenAgent", version="1.0.0", description="Golden fixture agent for CI schema checks."
)
class GoldenAgent(BaseAgent):
    @a2a_capability(name="echo", description="Echoes the provided message back to the caller.")
    def echo(self, message: str, repeat: int = 1) -> str:
        return message * repeat


def test_agent_card_json_matches_golden_fixture() -> None:
    agent = GoldenAgent()
    actual = agent.get_agent_card_json("https://golden.conducto.test/a2a")
    expected = (FIXTURES_DIR / "golden_agent_card.json").read_text(encoding="utf-8").strip()

    assert actual == expected, (
        "Agent Card JSON schema drifted from the golden fixture. If this "
        "change is intentional, regenerate tests/golden/fixtures/"
        "golden_agent_card.json and review the diff carefully, since it is a "
        "public wire contract."
    )


def test_agent_card_json_is_valid_utf8_ascii_and_deterministic() -> None:
    agent = GoldenAgent()
    url = "https://golden.conducto.test/a2a"

    first = agent.get_agent_card_json(url)
    second = agent.get_agent_card_json(url)

    assert first == second
    assert first.isascii()
