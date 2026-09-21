"""Acceptance failure boundaries for public model/tool delegation."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from conducto import (
    AgentRegistry,
    BaseAgent,
    CapabilityUse,
    CapabilityUseRequirement,
    ChatMessage,
    DelegationConfig,
    DelegationOutcome,
    DelegationOutcomeCode,
    FakeModel,
    ModelConfiguration,
    ProviderError,
    ProviderRegistry,
    Runtime,
    ToolboxPolicy,
    Usage,
    a2a_agent,
    a2a_capability,
    build_toolbox,
    run_delegation,
)
from conducto.core.runtime import use_run_context

pytestmark = pytest.mark.acceptance


class _Answer(BaseModel):
    answer: str


@a2a_agent(name="FailureDocumentation", version="1.0.0", description="Failure test provider.")
class _Documentation(BaseAgent):
    calls = 0

    @a2a_capability(name="documentation.search", description="Searches deterministic docs.")
    def search(self, query: str) -> dict[str, str]:
        type(self).calls += 1
        return {"excerpt": query}


def _runtime(model: FakeModel, *, registry: AgentRegistry | None = None) -> Runtime:
    providers = ProviderRegistry()
    providers.register_client(
        "test-model",
        model,
        ModelConfiguration(provider="fake", model="test-model"),
    )
    return Runtime(provider_registry=providers, agent_registry=registry or AgentRegistry())


async def _outcome(
    runtime: Runtime,
    config: DelegationConfig,
) -> tuple[DelegationOutcome[_Answer], FakeModel]:
    context = runtime.create_run_context(agent_id="Parent", call_override="test-model")
    with use_run_context(context):
        outcome = await run_delegation(
            context,
            (ChatMessage(role="user", content="question"),),
            config=config,
            response_type=_Answer,
        )
    model = runtime.provider_registry.resolve("test-model").client
    assert isinstance(model, FakeModel)
    return outcome, model


async def _tool_id(runtime: Runtime) -> str:
    context = runtime.create_run_context(agent_id="Parent")
    policy = ToolboxPolicy(
        uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
    )
    with use_run_context(context):
        toolbox = await build_toolbox(context.gateway, policy)
    assert toolbox.snapshot is not None
    return toolbox.snapshot.tools[0].tool_id


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


def test_optional_unknown_replayed_and_invalid_tools_stop_without_duplicate_execution() -> None:
    async def exercise() -> None:
        optional_runtime = _runtime(FakeModel({"type": "terminal", "response": {"answer": "ok"}}))
        optional, optional_model = await _outcome(optional_runtime, DelegationConfig())
        assert optional.code is DelegationOutcomeCode.SUCCESS
        assert optional_model.requests[0].tools == ()

        registry = AgentRegistry()
        registry.register(_Documentation())
        probe = _runtime(
            FakeModel({"type": "terminal", "response": {"answer": "unused"}}), registry=registry
        )
        tool_id = await _tool_id(probe)

        _Documentation.calls = 0
        unknown_runtime = _runtime(
            FakeModel(
                {
                    "type": "tool_call",
                    "call_id": "unknown",
                    "tool_id": "forged-tool",
                    "arguments": {},
                }
            ),
            registry=registry,
        )
        unknown, _ = await _outcome(
            unknown_runtime,
            DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
                )
            ),
        )
        assert unknown.code is DelegationOutcomeCode.UNKNOWN_TOOL_CALL
        assert _Documentation.calls == 0

        replay = {
            "type": "tool_call",
            "call_id": "once",
            "tool_id": tool_id,
            "arguments": {"query": "one"},
        }
        replay_runtime = _runtime(FakeModel(script=(replay, replay)), registry=registry)
        replay_outcome, _ = await _outcome(
            replay_runtime,
            DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
                )
            ),
        )
        assert replay_outcome.code is DelegationOutcomeCode.REPLAYED_TOOL_CALL
        assert _Documentation.calls == 1

        invalid_runtime = _runtime(
            FakeModel(
                {
                    "type": "tool_call",
                    "call_id": "invalid",
                    "tool_id": tool_id,
                    "arguments": {},
                }
            ),
            registry=registry,
        )
        invalid, _ = await _outcome(
            invalid_runtime,
            DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
                )
            ),
        )
        assert invalid.code is DelegationOutcomeCode.INVALID_ARGUMENTS
        assert _Documentation.calls == 1

    asyncio.run(exercise())


def test_provider_and_budget_boundaries_are_typed_failures() -> None:
    async def exercise() -> None:
        provider_runtime = _runtime(FakeModel(script=(ProviderError("unavailable"),)))
        provider, _ = await _outcome(provider_runtime, DelegationConfig())
        assert provider.code is DelegationOutcomeCode.PROVIDER_FAILURE

        registry = AgentRegistry()
        registry.register(_Documentation())
        probe = _runtime(
            FakeModel({"type": "terminal", "response": {"answer": "unused"}}), registry=registry
        )
        tool_id = await _tool_id(probe)
        turn_runtime = _runtime(
            FakeModel(
                script=(
                    {
                        "type": "tool_call",
                        "call_id": "only-turn",
                        "tool_id": tool_id,
                        "arguments": {"query": "question"},
                    },
                    {"type": "terminal", "response": {"answer": "unreachable"}},
                )
            ),
            registry=registry,
        )
        turns, _ = await _outcome(
            turn_runtime,
            DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
                ),
                max_model_turns=1,
            ),
        )
        assert turns.code is DelegationOutcomeCode.TURN_LIMIT_EXHAUSTED

        token_runtime = _runtime(
            FakeModel(
                {"type": "terminal", "response": {"answer": "ok"}}, usage=Usage(total_tokens=2)
            )
        )
        tokens, _ = await _outcome(token_runtime, DelegationConfig(token_budget=1))
        assert tokens.code is DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED

        cost_runtime = _runtime(
            FakeModel(
                {"type": "terminal", "response": {"answer": "ok"}},
                usage=Usage(cost=2.0),
            )
        )
        cost, _ = await _outcome(cost_runtime, DelegationConfig(cost_budget=1.0))
        assert cost.code is DelegationOutcomeCode.COST_BUDGET_EXHAUSTED

    asyncio.run(exercise())
