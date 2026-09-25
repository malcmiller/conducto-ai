"""Golden fixtures for derived capability structured-output schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from conducto import BaseAgent, a2a_agent, a2a_capability

pytestmark = pytest.mark.golden

FIXTURES_DIR = Path(__file__).parent / "capability_schemas"


class GoldenAnswer(BaseModel):
    """Golden object response model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str


class GoldenItem(BaseModel):
    """Golden sequence item response model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


@a2a_agent(
    name="GoldenStructuredAgent",
    version="1.0.0",
    description="Pins derived structured-output schemas.",
)
class GoldenStructuredAgent(BaseAgent):
    """Agent whose output schemas are pinned as golden fixtures."""

    @a2a_capability(name="answer", description="Returns a golden answer.")
    def answer(self) -> GoldenAnswer:
        """Return the pinned answer model."""
        return GoldenAnswer(text="ok")

    @a2a_capability(name="items", description="Returns golden items.")
    def items(self) -> list[GoldenItem]:
        """Return the pinned root-level item sequence."""
        return [GoldenItem(name="ok")]


def _canonical(value: object) -> str:
    """Serialize JSON values with the repository's golden-fixture conventions."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def test_pydantic_model_output_schema_matches_golden_fixture() -> None:
    agent = GoldenStructuredAgent()
    contract = agent.capabilities["answer"].output_contract
    assert contract is not None

    actual = _canonical(contract.schema)
    expected = (FIXTURES_DIR / "golden_answer_schema.json").read_text(encoding="utf-8").strip()

    assert actual == expected


def test_root_sequence_output_schema_matches_golden_fixture() -> None:
    agent = GoldenStructuredAgent()
    contract = agent.capabilities["items"].output_contract
    assert contract is not None

    actual = _canonical(contract.schema)
    expected = (FIXTURES_DIR / "golden_item_list_schema.json").read_text(encoding="utf-8").strip()

    assert actual == expected
