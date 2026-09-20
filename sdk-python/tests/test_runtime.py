import asyncio
import json
from dataclasses import FrozenInstanceError

import pytest

from conducto import (
    AgentModelConfig,
    BaseAgent,
    ChatMessage,
    FakeModel,
    IncompatibleProviderCapabilitiesError,
    InvocationSuccess,
    MissingModelDefaultError,
    ModelConfiguration,
    ModelOverrideDeniedError,
    ModelReference,
    ModelRequirement,
    ModelResolutionSource,
    OrchestratorAgent,
    ProviderCapabilities,
    ProviderRegistry,
    ProviderUnavailableError,
    RunConfig,
    Runtime,
    RuntimeConfig,
    StructuredOutputRequest,
    UnknownModelReferenceError,
    a2a_agent,
    a2a_capability,
    get_run_context,
)


def _runtime(*references: str, default: str | None = None) -> tuple[Runtime, dict[str, FakeModel]]:
    registry = ProviderRegistry()
    providers: dict[str, FakeModel] = {}
    for reference in references:
        provider = FakeModel({"agent_id": "unused", "capability_id": "unused"})
        providers[reference] = provider
        registry.register(
            reference,
            provider,
            ModelConfiguration(provider=f"provider-{reference}", model=f"model-{reference}"),
        )
    return Runtime(
        provider_registry=registry,
        config=RuntimeConfig(ModelReference(default)) if default else RuntimeConfig(),
    ), providers


def test_model_resolution_precedence_and_immutable_configuration() -> None:
    runtime, _ = _runtime("runtime", "agent", "run", "call", default="runtime")
    agent = AgentModelConfig(
        default_model=ModelReference("agent"),
        requirement=ModelRequirement.REQUIRED,
    )

    runtime_context = runtime.create_run_context(agent_id="Agent", agent_config=AgentModelConfig())
    agent_context = runtime.create_run_context(agent_id="Agent", agent_config=agent)
    run_context = runtime.create_run_context(
        agent_id="Agent",
        agent_config=agent,
        run_config=RunConfig(model=ModelReference("run")),
    )
    call_context = runtime.create_run_context(
        agent_id="Agent",
        agent_config=agent,
        run_config=RunConfig(model=ModelReference("run")),
        call_override="call",
    )

    assert (
        runtime_context.model
        and runtime_context.model.source is ModelResolutionSource.RUNTIME_DEFAULT
    )
    assert agent_context.model and agent_context.model.source is ModelResolutionSource.AGENT_DEFAULT
    assert run_context.model and run_context.model.source is ModelResolutionSource.RUN_OVERRIDE
    assert call_context.model and call_context.model.source is ModelResolutionSource.CALL_OVERRIDE
    with pytest.raises(FrozenInstanceError):
        setattr(run_context, "correlation_id", "changed")  # noqa: B010
    with pytest.raises(FrozenInstanceError):
        setattr(agent, "default_model", ModelReference("call"))  # noqa: B010


def test_concurrent_runs_of_one_agent_isolate_model_overrides() -> None:
    runtime, _ = _runtime("first", "second")

    @a2a_agent(name="ConcurrentAgent", description="Tests isolated model contexts.")
    class ConcurrentAgent(BaseAgent):
        @a2a_capability(name="selected", description="Returns the selected model.")
        async def selected(self) -> str:
            await asyncio.sleep(0)
            context = get_run_context()
            assert context is not None and context.model is not None
            return f"{context.correlation_id}:{context.model.reference}"

    async def exercise() -> None:
        orchestrator = OrchestratorAgent(runtime=runtime)
        orchestrator.register_agent(ConcurrentAgent())
        first, second = await asyncio.gather(
            orchestrator.invoke(
                "ConcurrentAgent",
                "selected",
                {},
                run_config=RunConfig(model=ModelReference("first")),
                correlation_id="one",
            ),
            orchestrator.invoke(
                "ConcurrentAgent",
                "selected",
                {},
                run_config=RunConfig(model=ModelReference("second")),
                correlation_id="two",
            ),
        )
        assert isinstance(first, InvocationSuccess)
        assert isinstance(second, InvocationSuccess)
        assert {first.value, second.value} == {"one:first", "two:second"}
        assert first.metadata and first.metadata.model_reference == "first"
        assert second.metadata and second.metadata.model_reference == "second"

    asyncio.run(exercise())


def test_call_override_does_not_mutate_enclosing_run_context() -> None:
    runtime, providers = _runtime("run", "call")
    context = runtime.create_run_context(
        agent_id="Agent",
        agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
        run_config=RunConfig(model=ModelReference("run")),
    )

    async def exercise() -> None:
        result = await runtime.complete(
            context,
            (ChatMessage(role="user", content="route"),),
            structured_output=StructuredOutputRequest(
                name="selection",
                schema={"type": "object"},
            ),
            model="call",
        )
        assert result.metadata.model_reference == "call"
        assert result.metadata.resolution_source is ModelResolutionSource.CALL_OVERRIDE

    asyncio.run(exercise())
    assert context.model_reference == ModelReference("run")
    assert providers["call"].calls == 1
    assert providers["run"].calls == 0


def test_orchestrator_and_selected_agent_use_different_models() -> None:
    runtime, _ = _runtime("orchestrator", "worker")
    routing_provider = FakeModel({"agent_id": "Worker", "capability_id": "work"})
    runtime.provider_registry.register(
        "orchestrator",
        routing_provider,
        ModelConfiguration(provider="router", model="router-model"),
        replace=True,
    )

    @a2a_agent(
        name="Worker",
        description="Performs model-backed work.",
        default_model="worker",
        model_required=True,
    )
    class Worker(BaseAgent):
        @a2a_capability(name="work", description="Reports its model.")
        def work(self) -> str:
            context = get_run_context()
            assert context is not None and context.model is not None
            return str(context.model.reference)

    async def exercise() -> None:
        orchestrator = OrchestratorAgent(
            runtime=runtime,
            model_reference="orchestrator",
        )
        orchestrator.register_agent(Worker())
        result = await orchestrator.route("work")
        assert isinstance(result, InvocationSuccess)
        assert result.value == "worker"
        assert result.metadata and result.metadata.model_reference == "worker"

    asyncio.run(exercise())
    assert routing_provider.calls == 1


def test_deterministic_capability_runs_without_a_model() -> None:
    @a2a_agent(name="Deterministic", description="Does not need a model.", model_required=True)
    class Deterministic(BaseAgent):
        @a2a_capability(
            name="add",
            description="Adds one without a model.",
            model_required=False,
        )
        def add(self, value: int) -> int:
            context = get_run_context()
            assert context is not None and context.model is None
            return value + 1

    async def exercise() -> None:
        orchestrator = OrchestratorAgent(runtime=Runtime())
        orchestrator.register_agent(Deterministic())
        result = await orchestrator.invoke("Deterministic", "add", {"value": 1})
        assert isinstance(result, InvocationSuccess)
        assert result.value == 2

    asyncio.run(exercise())


def test_resolution_failures_happen_before_capability_or_provider_calls() -> None:
    registry = ProviderRegistry()
    incompatible = FakeModel({})
    incompatible.capabilities = ProviderCapabilities()  # type: ignore[misc]
    registry.register(
        "incompatible",
        incompatible,
        ModelConfiguration(provider="fake", model="incompatible"),
    )
    unavailable = FakeModel({})
    registry.register(
        "unavailable",
        unavailable,
        ModelConfiguration(provider="fake", model="unavailable"),
        available=False,
    )
    runtime = Runtime(
        provider_registry=registry,
        policy=lambda context: context.model_reference.value != "denied",
    )
    registry.register(
        "denied",
        FakeModel({}),
        ModelConfiguration(provider="fake", model="denied"),
    )

    with pytest.raises(UnknownModelReferenceError):
        runtime.create_run_context(
            agent_id="Agent",
            run_config=RunConfig(model=ModelReference("unknown")),
        )
    with pytest.raises(IncompatibleProviderCapabilitiesError):
        runtime.create_run_context(
            agent_id="Agent",
            run_config=RunConfig(model=ModelReference("incompatible")),
            required_capabilities=frozenset({"structured_output"}),
        )
    with pytest.raises(ModelOverrideDeniedError):
        runtime.create_run_context(
            agent_id="Agent",
            run_config=RunConfig(model=ModelReference("denied")),
        )
    with pytest.raises(ProviderUnavailableError):
        runtime.create_run_context(
            agent_id="Agent",
            run_config=RunConfig(model=ModelReference("unavailable")),
        )
    with pytest.raises(MissingModelDefaultError):
        runtime.create_run_context(
            agent_id="Agent",
            agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
        )
    assert incompatible.calls == unavailable.calls == 0


def test_run_context_serialization_excludes_clients_and_provider_configuration() -> None:
    runtime, _ = _runtime("safe")
    context = runtime.create_run_context(
        agent_id="Agent",
        run_config=RunConfig(
            model=ModelReference("safe"),
            metadata={"classification": "internal"},
        ),
    )
    serialized = json.dumps(context.to_dict(), sort_keys=True)

    assert '"model_reference": "safe"' in serialized
    assert "model-safe" not in serialized
    assert "client" not in serialized
    assert "configuration" not in serialized
    with pytest.raises(ValueError, match="Sensitive values"):
        RunConfig(metadata={"credentials": "must-not-enter-context"})


def test_runtime_callable_invocation_supports_classmethods_and_rejects_other_instances() -> None:
    @a2a_agent(name="CallableAgent", description="Tests callable targets.")
    class CallableAgent(BaseAgent):
        def __init__(self, label: str) -> None:
            self.label = label
            super().__init__()

        @classmethod
        @a2a_capability(name="class-call", description="Invokes a class method.")
        def class_call(cls, value: str) -> str:
            return f"{cls.__name__}:{value}"

        @a2a_capability(name="instance-call", description="Invokes an instance method.")
        def instance_call(self) -> str:
            return self.label

    async def exercise() -> None:
        runtime = Runtime()
        first = CallableAgent("first")
        second = CallableAgent("second")

        class_result = await runtime.invoke(first, first.class_call, {"value": "ok"})
        assert isinstance(class_result, InvocationSuccess)
        assert class_result.value == "CallableAgent:ok"

        wrong_instance = await runtime.invoke(first, second.instance_call, {})
        assert not isinstance(wrong_instance, InvocationSuccess)

    asyncio.run(exercise())


def test_model_free_capability_clears_inherited_provider_requirements() -> None:
    @a2a_agent(name="MixedAgent", description="Has deterministic work.")
    class MixedAgent(BaseAgent):
        @a2a_capability(
            name="deterministic",
            description="Runs without a model.",
            model_required=False,
        )
        def deterministic(self) -> str:
            return "done"

    agent = MixedAgent(
        agent_config=AgentModelConfig(
            requirement=ModelRequirement.REQUIRED,
            required_capabilities=frozenset({"structured_output"}),
        )
    )

    async def exercise() -> None:
        result = await Runtime().invoke(agent, "deterministic", {})
        assert isinstance(result, InvocationSuccess)
        assert result.value == "done"

    asyncio.run(exercise())
