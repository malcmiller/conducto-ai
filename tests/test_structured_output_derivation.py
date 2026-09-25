"""Tests for deriving capability structured-output contracts from return types."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.agent_card import capability_output_schema_map, stable_skill_id
from conducto.core.capability_errors import CapabilityFailureCode, MalformedModelOutputError
from conducto.core.invocation_results import InvocationFailure, InvocationSuccess
from conducto.core.provider import ModelConfiguration, ProviderResult
from conducto.core.provider_registry import ProviderOwnership, ProviderRegistry
from conducto.testing import FakeModel


class _Answer(BaseModel):
    """Typed response used by structured-output derivation tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str


class _Item(BaseModel):
    """Typed item used by root-level sequence structured-output tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


_SCHEMA_OVERRIDE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
}


@a2a_agent(
    name="StructuredAgent",
    version="1.0",
    description="Exercises derived structured output.",
    default_model="model",
    model_required=True,
)
class _StructuredAgent(BaseAgent):
    @a2a_capability(name="answer", description="Return a typed answer.")
    async def answer(self, prompt: str) -> _Answer:
        """Return an answer validated from the derived model schema."""
        call = await self.complete(prompt)
        value = call.result.structured
        assert isinstance(value, _Answer)
        return value

    @a2a_capability(name="items", description="Return typed items.")
    async def items(self, prompt: str) -> list[_Item]:
        """Return root-level typed items validated from the derived schema."""
        call = await self.complete(prompt)
        value = call.result.structured
        assert isinstance(value, list)
        assert all(isinstance(item, _Item) for item in value)
        return value

    @a2a_capability(
        name="override",
        description="Return an explicitly schema-constrained response.",
        output_schema=_SCHEMA_OVERRIDE,
    )
    async def override(self, prompt: str) -> Mapping[str, Any]:
        """Return raw structured output constrained by an explicit schema."""
        call = await self.complete(prompt)
        value = call.result.structured
        assert isinstance(value, dict)
        return value


def _runtime(result: ProviderResult) -> tuple[Runtime, FakeModel]:
    """Create a runtime backed by one deterministic fake model response."""
    provider = FakeModel(result)
    registry = ProviderRegistry()
    registry.register_client(
        "model",
        provider,
        ModelConfiguration(provider="fake", model="model"),
        ownership=ProviderOwnership.RUNTIME_OWNED,
    )
    return Runtime(provider_registry=registry), provider


def test_pydantic_return_annotation_derives_schema_and_typed_result() -> None:
    async def exercise() -> tuple[InvocationSuccess, FakeModel, _StructuredAgent]:
        runtime, provider = _runtime(ProviderResult(structured={"text": "done"}, accepted=True))
        agent = _StructuredAgent()
        result = await runtime.invoke(agent, "answer", {"prompt": "draft"})
        assert isinstance(result, InvocationSuccess)
        return result, provider, agent

    result, provider, agent = asyncio.run(exercise())

    assert result.value == {"text": "done"}
    request_schema = provider.requests[0].structured_output.json_schema
    registered_schema = agent.capabilities["answer"].output_contract.schema
    assert request_schema == registered_schema
    assert request_schema == _Answer.model_json_schema()

    card = agent.get_agent_card("https://structured.conducto.test/a2a")
    skill_id = stable_skill_id(agent.agent_metadata.name, "answer")
    assert capability_output_schema_map(card)[skill_id] == request_schema


def test_root_level_sequence_return_round_trips_with_derived_schema() -> None:
    async def exercise() -> tuple[InvocationSuccess, FakeModel]:
        runtime, provider = _runtime(
            ProviderResult(structured=[{"name": "first"}, {"name": "second"}], accepted=True)
        )
        result = await runtime.invoke(_StructuredAgent(), "items", {"prompt": "list"})
        assert isinstance(result, InvocationSuccess)
        return result, provider

    result, provider = asyncio.run(exercise())

    assert result.value == [{"name": "first"}, {"name": "second"}]
    request_schema = provider.requests[0].structured_output.json_schema
    assert request_schema["type"] == "array"
    assert request_schema["items"] == {"$ref": "#/$defs/_Item"}


def test_explicit_output_schema_override_is_published_and_sent_to_provider() -> None:
    async def exercise() -> tuple[InvocationSuccess, FakeModel, _StructuredAgent]:
        runtime, provider = _runtime(
            ProviderResult(structured={"message": "override"}, accepted=True)
        )
        agent = _StructuredAgent()
        result = await runtime.invoke(agent, "override", {"prompt": "raw"})
        assert isinstance(result, InvocationSuccess)
        return result, provider, agent

    result, provider, agent = asyncio.run(exercise())

    assert result.value == {"message": "override"}
    assert provider.requests[0].structured_output.json_schema == _SCHEMA_OVERRIDE
    skill_id = stable_skill_id(agent.agent_metadata.name, "override")
    card = agent.get_agent_card("https://structured.conducto.test/a2a")
    assert capability_output_schema_map(card)[skill_id] == _SCHEMA_OVERRIDE


def test_invalid_provider_output_raises_typed_malformed_model_failure() -> None:
    async def exercise() -> InvocationFailure:
        runtime, _provider = _runtime(ProviderResult(structured={"text": 123}, accepted=True))
        result = await runtime.invoke(_StructuredAgent(), "answer", {"prompt": "bad"})
        assert isinstance(result, InvocationFailure)
        return result

    result = asyncio.run(exercise())

    assert result.classification == CapabilityFailureCode.MALFORMED_OUTPUT.value
    assert isinstance(result.exception, MalformedModelOutputError)
