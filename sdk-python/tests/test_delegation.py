import asyncio
from collections.abc import Callable, Iterator

from pydantic import BaseModel

from conducto import (
    AgentRegistry,
    BaseAgent,
    CapabilityUse,
    CapabilityUseRequirement,
    ChatMessage,
    DelegationBudget,
    DelegationConfig,
    DelegationFallbackPolicy,
    DelegationOutcome,
    DelegationOutcomeCode,
    DelegationRequirement,
    FakeModel,
    ModelConfiguration,
    ProviderCapabilities,
    ProviderError,
    ProviderRegistry,
    Runtime,
    ToolboxPolicy,
    ToolResultStatus,
    Usage,
    a2a_agent,
    a2a_capability,
    build_toolbox,
    run_delegation,
)
from conducto.core.runtime import use_run_context


class Answer(BaseModel):
    value: str


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


def _runtime(
    model: FakeModel,
    *,
    registry: AgentRegistry | None = None,
) -> tuple[Runtime, FakeModel]:
    providers = ProviderRegistry()
    providers.register(
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
        runtime, model = _runtime(FakeModel({"type": "terminal", "response": {"value": "done"}}))
        outcome = await _run(runtime, DelegationConfig())
        assert outcome.code is DelegationOutcomeCode.SUCCESS
        assert outcome.value == Answer(value="done")
        assert model.calls == 1
        assert Worker.calls == 0

    asyncio.run(exercise())


def test_required_capability_unavailable_fails_before_model_call() -> None:
    async def exercise() -> None:
        runtime, model = _runtime(FakeModel({"type": "terminal", "response": {"value": "unused"}}))
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
                    {
                        "type": "tool_call",
                        "call_id": "call-1",
                        "tool_id": tool_id,
                        "arguments": {"value": "hello"},
                    },
                    {"type": "terminal", "response": {"value": "complete"}},
                )
            ),
            registry=agents,
        )
        # Tool IDs are registry-derived and stable across runtime-bound snapshots.
        runtime_tool_id = await _tool_id(runtime, _policy())
        model._script = (
            {
                "type": "tool_call",
                "call_id": "call-1",
                "tool_id": runtime_tool_id,
                "arguments": {"value": "hello"},
            },
            {"type": "terminal", "response": {"value": "complete"}},
        )

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


def test_unknown_invalid_and_replayed_calls_never_reexecute_business_logic() -> None:
    async def exercise() -> None:
        Worker.calls = 0
        agents = AgentRegistry()
        agents.register(Worker())

        unknown_runtime, _ = _runtime(
            FakeModel(
                {
                    "type": "tool_call",
                    "call_id": "unknown",
                    "tool_id": "foreign-snapshot-tool",
                    "arguments": {},
                }
            ),
            registry=agents,
        )
        unknown = await _run(
            unknown_runtime,
            DelegationConfig(toolbox=_policy()),
        )
        assert unknown.code is DelegationOutcomeCode.UNKNOWN_TOOL_CALL

        invalid_runtime, invalid_model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        invalid_id = await _tool_id(invalid_runtime, _policy())
        invalid_model.selection = {
            "type": "tool_call",
            "call_id": "invalid",
            "tool_id": invalid_id,
            "arguments": {},
        }
        invalid = await _run(invalid_runtime, DelegationConfig(toolbox=_policy()))
        assert invalid.code is DelegationOutcomeCode.INVALID_ARGUMENTS

        replay_runtime, replay_model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        replay_id = await _tool_id(replay_runtime, _policy())
        call = {
            "type": "tool_call",
            "call_id": "duplicate",
            "tool_id": replay_id,
            "arguments": {"value": "once"},
        }
        replay_model._script = (call, call)
        replay_model.selection = None
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
        runtime, model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id = await _tool_id(runtime, _policy("fail"))
        model._script = (
            {
                "type": "tool_call",
                "call_id": "failed-child",
                "tool_id": tool_id,
                "arguments": {},
            },
            {"type": "terminal", "response": {"value": "fallback"}},
        )
        model.selection = None
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
        runtime, model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id = await _tool_id(runtime, _policy())
        tool_call = {
            "type": "tool_call",
            "call_id": "limited",
            "tool_id": tool_id,
            "arguments": {"value": "x"},
        }

        model.selection = tool_call
        calls = await _run(
            runtime,
            DelegationConfig(toolbox=_policy(), max_tool_calls=0),
        )
        assert calls.code is DelegationOutcomeCode.TOOL_CALL_LIMIT_EXHAUSTED

        model.selection = {**tool_call, "call_id": "depth"}
        depth = await _run(
            runtime,
            DelegationConfig(toolbox=_policy(), max_depth=0),
        )
        assert depth.code is DelegationOutcomeCode.DEPTH_LIMIT_EXHAUSTED

        model.selection = {"type": "terminal", "response": {"value": "again"}}
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
                {"type": "terminal", "response": {"value": "unused"}},
                usage=Usage(total_tokens=2),
            ),
            registry=agents,
        )
        tokens = await _run(token_runtime, DelegationConfig(token_budget=1))
        assert tokens.code is DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED

        cost_runtime, _ = _runtime(
            FakeModel(
                {"type": "terminal", "response": {"value": "unused"}},
                usage=Usage(cost=2.0),
            ),
            registry=agents,
        )
        cost = await _run(cost_runtime, DelegationConfig(cost_budget=1.0))
        assert cost.code is DelegationOutcomeCode.COST_BUDGET_EXHAUSTED

        size_runtime, size_model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id = await _tool_id(size_runtime, _policy())
        size_model.selection = {
            "type": "tool_call",
            "call_id": "large",
            "tool_id": tool_id,
            "arguments": {"value": "large"},
        }
        size = await _run(
            size_runtime,
            DelegationConfig(toolbox=_policy(), max_result_bytes=8),
        )
        assert size.code is DelegationOutcomeCode.RESULT_SIZE_EXHAUSTED

    asyncio.run(exercise())


def test_parent_cancellation_is_not_reported_as_success_or_generic_failure() -> None:
    async def exercise() -> None:
        runtime, model = _runtime(FakeModel({"type": "terminal", "response": {"value": "unused"}}))
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
        runtime, model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id = await _tool_id(runtime, _policy("block"))
        model.selection = {
            "type": "tool_call",
            "call_id": "blocking",
            "tool_id": tool_id,
            "arguments": {},
        }
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
        runtime, _ = _runtime(FakeModel({"type": "terminal", "response": {"value": "facade"}}))
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
        malformed_runtime, _ = _runtime(FakeModel("prose only"))
        malformed = await _run(malformed_runtime, DelegationConfig())
        assert malformed.code is DelegationOutcomeCode.MALFORMED_DECISION

        failed_runtime, _ = _runtime(FakeModel(script=(ProviderError("provider failed"),)))
        failed = await _run(failed_runtime, DelegationConfig())
        assert failed.code is DelegationOutcomeCode.PROVIDER_FAILURE

        agents = AgentRegistry()
        agents.register(Worker())
        incompatible_model = FakeModel({"type": "terminal", "response": {"value": "unused"}})
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
        runtime, model = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id = await _tool_id(runtime, _policy())
        model.selection = {
            "type": "tool_call",
            "call_id": "deadline",
            "tool_id": tool_id,
            "arguments": {"value": "never"},
        }
        budget = DelegationBudget(calls=1)
        context = runtime.create_run_context(
            agent_id="Caller",
            call_override="model",
            delegation_budget=budget,
        )
        times: Iterator[float] = iter((0.0, 0.0, 0.0, 0.0, 2.0))
        with use_run_context(context):
            outcome = await run_delegation(
                context,
                (ChatMessage(role="user", content="deadline"),),
                config=DelegationConfig(toolbox=_policy(), timeout=1.0),
                response_type=Answer,
                clock=lambda: next(times),
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
        runtime_a, model_a = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        runtime_b, model_b = _runtime(
            FakeModel({"type": "terminal", "response": {"value": "placeholder"}}),
            registry=agents,
        )
        tool_id_a = await _tool_id(runtime_a, _policy())
        tool_id_b = await _tool_id(runtime_b, _policy())
        model_a._script = (
            {
                "type": "tool_call",
                "call_id": "same-call-id",
                "tool_id": tool_id_a,
                "arguments": {"value": "a"},
            },
            {"type": "terminal", "response": {"value": "a"}},
        )
        model_a.selection = None
        model_b._script = (
            {
                "type": "tool_call",
                "call_id": "same-call-id",
                "tool_id": tool_id_b,
                "arguments": {"value": "b"},
            },
            {"type": "terminal", "response": {"value": "b"}},
        )
        model_b.selection = None
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
