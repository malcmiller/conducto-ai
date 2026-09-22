"""Regression contracts for the core module boundaries."""

import asyncio
from typing import Any

import pytest

from conducto import AgentRegistry, BaseAgent, OrchestratorAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.invocation_results import InvocationSuccess, RoutingFailure
from conducto.core.provider import ModelConfiguration, ProviderResult
from conducto.core.provider_registry import ProviderRegistry
from conducto.core.registration import AgentRegistrationError, RegisteredMethod
from conducto.testing import FakeModel


def test_registration_constructs_one_parameter_model_per_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conducto.core.parameter_schema as parameter_schema

    calls = 0
    original_create_model = parameter_schema.create_parameter_model

    def counting_create_model(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_create_model(*args, **kwargs)

    monkeypatch.setattr(parameter_schema, "create_parameter_model", counting_create_model)

    class SingleModelAgent(BaseAgent):
        @a2a_capability(name="echo", description="Echoes a value.")
        def echo(self, value: str) -> str:
            return value

    agent = SingleModelAgent()

    assert calls == 1
    assert agent.capabilities["echo"].parameter_schema == (
        agent.capabilities["echo"].parameter_model.model_json_schema()
    )


def test_route_generates_routing_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    @a2a_agent(name="RouteAgent", version="1.0.0", description="Routes.")
    class RouteAgent(BaseAgent):
        @a2a_capability(name="echo", description="Echoes.")
        def echo(self, value: str) -> str:
            return value

    async def exercise() -> None:
        registry = ProviderRegistry()
        registry.register_client(
            "test",
            FakeModel(
                ProviderResult(
                    structured={
                        "agent_id": "RouteAgent",
                        "capability_id": "echo",
                        "arguments": {"value": "ok"},
                    }
                )
            ),
            ModelConfiguration(provider="fake", model="test"),
        )
        orchestrator = OrchestratorAgent(
            model_reference="test", runtime=Runtime(provider_registry=registry)
        )
        orchestrator.register_agent(RouteAgent())
        calls = 0
        original = orchestrator.get_routing_metadata

        def counting_metadata() -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(orchestrator, "get_routing_metadata", counting_metadata)
        result = await orchestrator.route("Echo ok")

        assert isinstance(result, InvocationSuccess)
        assert calls == 1

    asyncio.run(exercise())


def test_public_exports_reference_owning_modules() -> None:
    assert InvocationSuccess.__module__ == "conducto.core.invocation_results"
    assert issubclass(AgentRegistrationError, ValueError)
    assert RegisteredMethod.__module__ == "conducto.core.registration"
    assert RoutingFailure.__module__ == "conducto.core.invocation_results"


def test_registry_preserves_all_capability_providers_without_first_choice_projection() -> None:
    @a2a_agent(name="Alpha", version="1.0.0", description="First provider.")
    class Alpha(BaseAgent):
        @a2a_capability(name="echo", description="Echoes.")
        def echo(self, value: str) -> str:
            return value

    @a2a_agent(name="Bravo", version="1.0.0", description="Second provider.")
    class Bravo(Alpha):
        pass

    registry = AgentRegistry()
    alpha, bravo = Alpha(), Bravo()
    registry.register(bravo)
    registry.register(alpha)

    assert registry.capability_providers("echo") == (alpha, bravo)
    assert [agent.agent_id for agent in registry.snapshot().agents] == ["Alpha", "Bravo"]
    assert not hasattr(registry, "registered_capabilities")
    assert not hasattr(registry, "capabilities")

    registry.remove(alpha)
    assert registry.capability_providers("echo") == (bravo,)
