"""Acceptance failure boundaries for public model/tool delegation."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from conducto import (
    AgentRegistry,
    CapabilityUse,
    CapabilityUseRequirement,
    ChatMessage,
    DelegationConfig,
    DelegationOutcomeCode,
    FakeModel,
    ModelConfiguration,
    ProviderRegistry,
    Runtime,
    ToolboxPolicy,
    run_delegation,
)
from conducto.core.runtime import use_run_context

pytestmark = pytest.mark.acceptance


class _Answer(BaseModel):
    answer: str


def _runtime(model: FakeModel) -> Runtime:
    providers = ProviderRegistry()
    providers.register("test-model", model, ModelConfiguration(provider="fake", model="test-model"))
    return Runtime(provider_registry=providers, agent_registry=AgentRegistry())


def test_malformed_required_capability_and_invalid_terminal_are_typed_failures() -> None:
    async def exercise() -> None:
        malformed_runtime = _runtime(FakeModel("prose only"))
        malformed_context = malformed_runtime.create_run_context(
            agent_id="Parent",
            call_override="test-model",
        )
        with use_run_context(malformed_context):
            malformed = await run_delegation(
                malformed_context,
                (ChatMessage(role="user", content="question"),),
                config=DelegationConfig(),
                response_type=_Answer,
            )
        assert malformed.code is DelegationOutcomeCode.MALFORMED_DECISION

        required_runtime = _runtime(
            FakeModel({"type": "terminal", "response": {"answer": "unused"}})
        )
        required_context = required_runtime.create_run_context(
            agent_id="Parent",
            call_override="test-model",
        )
        required_policy = ToolboxPolicy(
            uses=(
                CapabilityUse(
                    capability_ids=frozenset({"documentation.search"}),
                    requirement=CapabilityUseRequirement.REQUIRED,
                ),
            )
        )
        with use_run_context(required_context):
            required = await run_delegation(
                required_context,
                (ChatMessage(role="user", content="question"),),
                config=DelegationConfig(toolbox=required_policy),
                response_type=_Answer,
            )
        assert required.code is DelegationOutcomeCode.REQUIRED_CAPABILITY_UNAVAILABLE

        terminal_runtime = _runtime(FakeModel({"type": "terminal", "response": {"wrong": "shape"}}))
        terminal_context = terminal_runtime.create_run_context(
            agent_id="Parent",
            call_override="test-model",
        )
        with use_run_context(terminal_context):
            terminal = await run_delegation(
                terminal_context,
                (ChatMessage(role="user", content="question"),),
                config=DelegationConfig(),
                response_type=_Answer,
            )
        assert terminal.code is DelegationOutcomeCode.FINAL_OUTPUT_VALIDATION_FAILURE

    asyncio.run(exercise())
