import asyncio
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.delegation import (
    DelegationConfig,
    DelegationFallbackPolicy,
    DelegationOutcome,
    DelegationOutcomeCode,
    DelegationRequirement,
    ToolResultStatus,
    run_delegation,
)
from conducto.core.gateway_tools import (
    CapabilityUse,
    CapabilityUseRequirement,
    ToolboxPolicy,
    build_toolbox,
)
from conducto.core.provider import (
    ChatMessage,
    ModelConfiguration,
    ProviderCapabilities,
    ProviderError,
    ProviderResult,
    ProviderToolCallRequest,
    Usage,
)
from conducto.core.provider_registry import ProviderRegistry
from conducto.core.run_context import DelegationBudget, use_run_context
from conducto.testing import FakeModel


class Answer(BaseModel):
    value: str


def _terminal(value: str, *, usage: Usage | None = None) -> ProviderResult:
    return ProviderResult(structured={"value": value}, usage=usage or Usage(), accepted=True)


def _call(
    tool_id: str, call_id: str, arguments: Mapping[str, Any], *, usage: Usage | None = None
) -> ProviderResult:
    return ProviderResult(
        tool_calls=(
            ProviderToolCallRequest(tool_id=tool_id, call_id=call_id, arguments=arguments),
        ),
        usage=usage or Usage(),
        accepted=True,
    )


@a2a_agent(name="Worker", version="1.0.0", description="Test worker.")
class Worker(BaseAgent):
    calls = 0

    @a2a_capability(name="echo", description="Echo a value.")
    def echo(self, value: str) -> dict[str, str]:
        type(self).calls += 1
        return {"value": value}

    @a2a_capability(name="fail", description="Fail deterministically.")
    def fail(self) -> str:
        type(self).calls += 1
        raise RuntimeError("sensitive child failure")


@a2a_agent(name="BlockingWorker", version="1.0.0", description="Blocking worker.")
class BlockingWorker(BaseAgent):
    started: asyncio.Event | None = None

    @a2a_capability(name="block", description="Wait until cancelled.")
    async def block(self) -> str:
        started = type(self).started
        assert started is not None
        started.set()
        await asyncio.Event().wait()
        return "unreachable"


def _policy(capability: str = "echo", *, required: bool = False) -> ToolboxPolicy:
    return ToolboxPolicy(
        uses=(
            CapabilityUse(
                capability_ids=frozenset({capability}),
                requirement=(
                    CapabilityUseRequirement.REQUIRED
                    if required
                    else CapabilityUseRequirement.OPTIONAL
                ),
            ),
        )
    )


async def _tool_id(runtime: Runtime, policy: ToolboxPolicy) -> str:
    context = runtime.create_run_context(agent_id="Inspector")
    with use_run_context(context):
        result = await build_toolbox(context.gateway, policy)
    assert result.snapshot is not None
    return result.snapshot.tools[0].tool_id


async def _tool_name(runtime: Runtime, policy: ToolboxPolicy) -> str:
    context = runtime.create_run_context(agent_id="Inspector")
    with use_run_context(context):
        result = await build_toolbox(context.gateway, policy)
    assert result.snapshot is not None
    return result.snapshot.tools[0].name


def _runtime(
    model: FakeModel,
    *,
    registry: AgentRegistry | None = None,
) -> tuple[Runtime, FakeModel]:
    providers = ProviderRegistry()
    providers.register_client(
        "model",
        model,
        ModelConfiguration(provider="fake", model="model"),
    )
    return Runtime(provider_registry=providers, agent_registry=registry), model


async def _run(
    runtime: Runtime,
    config: DelegationConfig,
    *,
    clock: Callable[[], float] | None = None,
) -> DelegationOutcome[Answer]:
    context = runtime.create_run_context(agent_id="Caller", call_override="model")
    with use_run_context(context):
        if clock is not None:
            return await run_delegation(
                context,
                (ChatMessage(role="user", content="complete the task"),),
                config=config,
                response_type=Answer,
                clock=clock,
            )
        return await run_delegation(
            context,
            (ChatMessage(role="user", content="complete the task"),),
            config=config,
            response_type=Answer,
        )


def test_no_tool_terminal_response_does_not_invoke_gateway() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        runtime, model = _runtime(FakeModel(_terminal("done")))
        outcome = await _run(runtime, DelegationConfig())
        assert outcome.code is DelegationOutcomeCode.SUCCESS
        assert outcome.value == Answer(value="done")
        assert model.calls == 1
        assert Worker.calls == 0

    asyncio.run(exercise())


def test_required_capability_unavailable_fails_before_model_call() -> None:
    async def exercise() -> None:
        runtime, model = _runtime(FakeModel(_terminal("unused")))
        outcome = await _run(
            runtime,
            DelegationConfig(toolbox=_policy("missing", required=True)),
        )
        assert outcome.code is DelegationOutcomeCode.REQUIRED_CAPABILITY_UNAVAILABLE
        assert model.calls == 0

    asyncio.run(exercise())


def test_one_tool_result_is_matched_and_terminal_output_is_validated() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        probe = Runtime(agent_registry=agents)
        tool_id = await _tool_id(probe, _policy())
        runtime, model = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "call-1", {"value": "hello"}),
                    _terminal("complete"),
                )
            ),
            registry=agents,
        )
        # Tool IDs are registry-derived and stable across runtime-bound snapshots.
        runtime_tool_id = await _tool_id(runtime, _policy())
        assert runtime_tool_id == tool_id

        outcome = await _run(runtime, DelegationConfig(toolbox=_policy()))

        assert outcome.code is DelegationOutcomeCode.SUCCESS
        assert Worker.calls == 1
        assert model.calls == 2
        result = model.requests[1].tool_results[0]
        assert result.call_id == "call-1"
        assert result.status == "success"
        assert outcome.provenance.tool_calls[0].child_task_id is not None
        assert outcome.metadata.usage == Usage()
        assert [call.purpose for call in outcome.metadata.model_calls] == [
            "delegation_turn",
            "delegation_turn",
        ]

    asyncio.run(exercise())


def test_missing_provider_call_id_is_synthesized_once_and_round_tripped() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        probe = Runtime(agent_registry=agents)
        tool_id = await _tool_id(probe, _policy())
        tool_name = await _tool_name(probe, _policy())
        runtime, model = _runtime(
            FakeModel(
                script=(
                    ProviderResult(
                        tool_calls=(
                            ProviderToolCallRequest(
                                tool_name=tool_name, arguments={"value": "hello"}
                            ),
                        ),
                        accepted=True,
                    ),
                    _terminal("complete"),
                )
            ),
            registry=agents,
        )

        outcome = await _run(runtime, DelegationConfig(toolbox=_policy()))

        assert outcome.code is DelegationOutcomeCode.SUCCESS
        assert Worker.calls == 1
        synthesized = outcome.provenance.tool_calls[0].call_id
        assert synthesized.startswith("tool_")
        assert outcome.provenance.tool_calls[0].tool_id == tool_id
        assert model.requests[1].tool_results[0].call_id == synthesized

    asyncio.run(exercise())


def test_sequential_tool_calls_preserve_all_results_and_ordered_provenance() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        probe = Runtime(agent_registry=agents)
        tool_id = await _tool_id(probe, _policy())
        usage = Usage(input_tokens=2, output_tokens=1, total_tokens=3, cost=0.25)
        runtime, model = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "call-1", {"value": "first"}, usage=usage),
                    _call(tool_id, "call-2", {"value": "second"}, usage=usage),
                    _terminal("complete", usage=usage),
                ),
            ),
            registry=agents,
        )

        outcome = await _run(runtime, DelegationConfig(toolbox=_policy()))

        assert outcome.code is DelegationOutcomeCode.SUCCESS
        assert Worker.calls == 2
        assert model.calls == 3
        assert [result.call_id for result in model.requests[1].tool_results] == ["call-1"]
        assert [result.call_id for result in model.requests[2].tool_results] == [
            "call-1",
            "call-2",
        ]
        assert [record.call_id for record in outcome.provenance.tool_calls] == [
            "call-1",
            "call-2",
        ]
        assert outcome.metadata.usage == Usage(
            input_tokens=6,
            output_tokens=3,
            total_tokens=9,
            cost=0.75,
        )

    asyncio.run(exercise())


def test_unknown_invalid_and_replayed_calls_never_reexecute_business_logic() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())

        unknown_runtime, _ = _runtime(
            FakeModel(_call("foreign-snapshot-tool", "unknown", {})),
            registry=agents,
        )
        unknown = await _run(
            unknown_runtime,
            DelegationConfig(toolbox=_policy()),
        )
        assert unknown.code is DelegationOutcomeCode.MALFORMED_DECISION
        assert Worker.calls == 0

        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy())
        invalid_runtime, _ = _runtime(
            FakeModel(_call(tool_id, "invalid", {})),
            registry=agents,
        )
        invalid = await _run(invalid_runtime, DelegationConfig(toolbox=_policy()))
        assert invalid.code is DelegationOutcomeCode.INVALID_ARGUMENTS

        call = _call(tool_id, "duplicate", {"value": "once"})
        replay_runtime, _ = _runtime(
            FakeModel(script=(call, call)),
            registry=agents,
        )
        replay = await _run(replay_runtime, DelegationConfig(toolbox=_policy()))
        assert replay.code is DelegationOutcomeCode.REPLAYED_TOOL_CALL
        assert len(replay.provenance.tool_results) == 1
        assert Worker.calls == 1

    asyncio.run(exercise())


def test_explicit_eligible_fallback_retains_child_failure() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy("fail"))
        runtime, model = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "failed-child", {}),
                    _terminal("fallback"),
                )
            ),
            registry=agents,
        )
        outcome = await _run(
            runtime,
            DelegationConfig(
                toolbox=_policy("fail"),
                fallback=DelegationFallbackPolicy(frozenset({ToolResultStatus.EXECUTION_FAILURE})),
            ),
        )
        assert outcome.code is DelegationOutcomeCode.FALLBACK_SUCCESS
        assert outcome.value == Answer(value="fallback")
        assert outcome.provenance.tool_results[0].status is ToolResultStatus.EXECUTION_FAILURE
        assert outcome.provenance.tool_calls[0].fallback_allowed
        assert "sensitive child failure" not in repr(model.requests[1].tool_results)

    asyncio.run(exercise())


def test_independent_loop_limits_are_typed_and_deterministic() -> None:
    async def exercise() -> None:
        agents = AgentRegistry()
        agents.register(Worker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy())
        runtime, _ = _runtime(
            FakeModel(_call(tool_id, "limited", {"value": "x"})),
            registry=agents,
        )
        calls = await _run(
            runtime,
            DelegationConfig(toolbox=_policy(), max_tool_calls=0),
        )
        assert calls.code is DelegationOutcomeCode.TOOL_CALL_LIMIT_EXHAUSTED

        depth = await _run(
            runtime,
            DelegationConfig(toolbox=_policy(), max_depth=0),
        )
        assert depth.code is DelegationOutcomeCode.DEPTH_LIMIT_EXHAUSTED

        runtime, _ = _runtime(FakeModel(_terminal("again")), registry=agents)
        turns = await _run(
            runtime,
            DelegationConfig(
                requirement=DelegationRequirement.REQUIRED,
                max_model_turns=1,
            ),
        )
        assert turns.code is DelegationOutcomeCode.REQUIRED_DELEGATION_NOT_PERFORMED

        times: Iterator[float] = iter((0.0, 2.0))
        deadline = await _run(
            runtime,
            DelegationConfig(timeout=1.0),
            clock=lambda: next(times),
        )
        assert deadline.code is DelegationOutcomeCode.DEADLINE_EXHAUSTED

    asyncio.run(exercise())


def test_token_cost_and_result_size_limits_are_enforced() -> None:
    async def exercise() -> None:
        agents = AgentRegistry()
        agents.register(Worker())
        token_runtime, _ = _runtime(
            FakeModel(
                _terminal("unused", usage=Usage(total_tokens=2)),
            ),
            registry=agents,
        )
        tokens = await _run(token_runtime, DelegationConfig(token_budget=1))
        assert tokens.code is DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED

        cost_runtime, _ = _runtime(
            FakeModel(
                _terminal("unused", usage=Usage(cost=2.0)),
            ),
            registry=agents,
        )
        cost = await _run(cost_runtime, DelegationConfig(cost_budget=1.0))
        assert cost.code is DelegationOutcomeCode.COST_BUDGET_EXHAUSTED

        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy())
        size_runtime, _ = _runtime(
            FakeModel(_call(tool_id, "large", {"value": "large"})),
            registry=agents,
        )
        size = await _run(
            size_runtime,
            DelegationConfig(toolbox=_policy(), max_result_bytes=8),
        )
        assert size.code is DelegationOutcomeCode.RESULT_SIZE_EXHAUSTED

    asyncio.run(exercise())


def test_parent_cancellation_is_not_reported_as_success_or_generic_failure() -> None:
    async def exercise() -> None:
        runtime, model = _runtime(FakeModel(_terminal("unused")))
        context = runtime.create_run_context(agent_id="Caller", call_override="model")
        context.cancellation.cancel()
        with use_run_context(context):
            outcome = await run_delegation(
                context,
                (ChatMessage(role="user", content="cancel"),),
                config=DelegationConfig(),
                response_type=Answer,
            )
        assert outcome.code is DelegationOutcomeCode.CANCELLATION
        assert model.calls == 0

    asyncio.run(exercise())


def test_parent_cancellation_stops_active_child_work() -> None:
    async def exercise() -> None:
        BlockingWorker.started = asyncio.Event()
        agents = AgentRegistry()
        agents.register(BlockingWorker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy("block"))
        runtime, model = _runtime(
            FakeModel(_call(tool_id, "blocking", {})),
            registry=agents,
        )
        context = runtime.create_run_context(agent_id="Caller", call_override="model")

        async def invoke() -> DelegationOutcome[Answer]:
            with use_run_context(context):
                return await run_delegation(
                    context,
                    (ChatMessage(role="user", content="block"),),
                    config=DelegationConfig(toolbox=_policy("block")),
                    response_type=Answer,
                )

        task = asyncio.create_task(invoke())
        started = BlockingWorker.started
        assert started is not None
        await started.wait()
        context.cancellation.cancel()
        outcome = await task
        assert outcome.code is DelegationOutcomeCode.CANCELLATION
        assert model.calls == 1

    asyncio.run(exercise())


def test_agent_facade_and_fallback_safety_are_explicit() -> None:
    async def exercise() -> None:
        config = DelegationConfig()
        agent = Worker(delegation_config=config)
        runtime, _ = _runtime(FakeModel(_terminal("facade")))
        context = runtime.create_run_context(agent_id="Worker", call_override="model")
        with use_run_context(context):
            outcome = await agent.run_delegation(
                (ChatMessage(role="user", content="facade"),),
                response_type=Answer,
            )
        assert outcome.value == Answer(value="facade")

        for prohibited in (
            ToolResultStatus.DENIED,
            ToolResultStatus.CANCELLATION,
            ToolResultStatus.BUDGET_REJECTED,
            ToolResultStatus.DEPTH_REJECTED,
            ToolResultStatus.CYCLE_REJECTED,
        ):
            try:
                DelegationFallbackPolicy(frozenset({prohibited}))
            except ValueError:
                pass
            else:
                raise AssertionError(f"{prohibited} unexpectedly allowed fallback")

    asyncio.run(exercise())


def test_malformed_provider_and_incompatible_model_fail_with_typed_outcomes() -> None:
    async def exercise() -> None:
        malformed_runtime, _ = _runtime(
            FakeModel(ProviderResult(content="prose only", accepted=True))
        )
        malformed = await _run(malformed_runtime, DelegationConfig())
        assert malformed.code is DelegationOutcomeCode.MALFORMED_DECISION

        failed_runtime, _ = _runtime(FakeModel(script=(ProviderError("provider failed"),)))
        failed = await _run(failed_runtime, DelegationConfig())
        assert failed.code is DelegationOutcomeCode.PROVIDER_FAILURE

        agents = AgentRegistry()
        agents.register(Worker())
        incompatible_model = FakeModel(_terminal("unused"))
        incompatible_model.capabilities = ProviderCapabilities(structured_output=True)
        incompatible_runtime, _ = _runtime(incompatible_model, registry=agents)
        incompatible = await _run(
            incompatible_runtime,
            DelegationConfig(toolbox=_policy()),
        )
        assert incompatible.code is DelegationOutcomeCode.PROVIDER_FAILURE

    asyncio.run(exercise())


def test_deadline_before_dispatch_does_not_execute_or_reserve_child_budget() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy())
        runtime, model = _runtime(
            FakeModel(_call(tool_id, "deadline", {"value": "never"})),
            registry=agents,
        )
        budget = DelegationBudget(calls=1)
        context = runtime.create_run_context(
            agent_id="Caller",
            call_override="model",
            delegation_budget=budget,
        )
        with use_run_context(context):
            outcome = await run_delegation(
                context,
                (ChatMessage(role="user", content="deadline"),),
                config=DelegationConfig(toolbox=_policy(), timeout=1.0),
                response_type=Answer,
                clock=lambda: 2.0 if model.calls else 0.0,
            )
        assert outcome.code is DelegationOutcomeCode.DEADLINE_EXHAUSTED
        assert budget.snapshot(current_depth=0, remaining_time=None).calls == 1
        assert Worker.calls == 0

    asyncio.run(exercise())


def test_concurrent_loops_isolate_replay_state_and_share_atomic_budget() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy())
        runtime_a, _ = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "same-call-id", {"value": "a"}),
                    _terminal("a"),
                )
            ),
            registry=agents,
        )
        runtime_b, _ = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "same-call-id", {"value": "b"}),
                    _terminal("b"),
                )
            ),
            registry=agents,
        )
        shared = DelegationBudget(calls=1)

        async def invoke(runtime: Runtime) -> DelegationOutcome[Answer]:
            context = runtime.create_run_context(
                agent_id="Caller",
                call_override="model",
                delegation_budget=shared,
            )
            with use_run_context(context):
                return await run_delegation(
                    context,
                    (ChatMessage(role="user", content="concurrent"),),
                    config=DelegationConfig(toolbox=_policy()),
                    response_type=Answer,
                )

        first, second = await asyncio.gather(invoke(runtime_a), invoke(runtime_b))
        codes = {first.code, second.code}
        assert codes == {
            DelegationOutcomeCode.SUCCESS,
            DelegationOutcomeCode.SHARED_BUDGET_EXHAUSTED,
        }
        assert first.provenance.loop_id != second.provenance.loop_id
        assert Worker.calls == 1

    asyncio.run(exercise())


def test_eligible_fallback_allows_only_a_terminal_turn_not_another_child() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())
        tool_id = await _tool_id(Runtime(agent_registry=agents), _policy("fail"))
        runtime, model = _runtime(
            FakeModel(
                script=(
                    _call(tool_id, "first-failure", {}),
                    _call(tool_id, "must-not-execute", {}),
                    _terminal("must-not-succeed"),
                )
            ),
            registry=agents,
        )
        outcome = await _run(
            runtime,
            DelegationConfig(
                toolbox=_policy("fail"),
                fallback=DelegationFallbackPolicy(frozenset({ToolResultStatus.EXECUTION_FAILURE})),
            ),
        )
        assert outcome.code is DelegationOutcomeCode.CHILD_FAILURE
        assert outcome.failure_code == "fallback_must_be_terminal"
        assert Worker.calls == 1
        assert model.calls == 2
        assert [result.to_dict() for result in outcome.provenance.tool_results] == [
            {
                "call_id": "first-failure",
                "status": "execution_failure",
                "reason_code": "execution_failure",
            }
        ]

    asyncio.run(exercise())
