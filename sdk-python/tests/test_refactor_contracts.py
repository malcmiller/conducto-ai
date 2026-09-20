"""Regression contracts for the core module boundaries."""

import asyncio
from typing import Any

import pytest

from conducto import (
    BaseAgent,
    FakeModel,
    InvocationSuccess,
    ModelConfiguration,
    OrchestratorAgent,
    a2a_agent,
    a2a_capability,
)
from conducto.core import (
    InvocationSuccess as CoreInvocationSuccess,
)
from conducto.core import (
    RoutingFailure as CoreRoutingFailure,
)
from conducto.core.agent import (
    AgentRegistrationError,
    RegisteredMethod,
)
from conducto.core.invocation import InvocationSuccess as ModuleInvocationSuccess
from conducto.core.invocation_results import RoutingFailure as ResultRoutingFailure
from conducto.core.orchestrator import RoutingFailure
from conducto.core.registration import RegisteredMethod as RegistrationRegisteredMethod


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
        orchestrator = OrchestratorAgent(
            model_provider=FakeModel(
                {
                    "agent_id": "RouteAgent",
                    "capability_id": "echo",
                    "arguments": {"value": "ok"},
                }
            ),
            model_config=ModelConfiguration(provider="fake", model="test"),
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


def test_public_and_direct_module_re_exports_remain_available() -> None:
    assert InvocationSuccess is CoreInvocationSuccess is ModuleInvocationSuccess
    assert issubclass(AgentRegistrationError, ValueError)
    assert RegisteredMethod is RegistrationRegisteredMethod
    assert RoutingFailure is CoreRoutingFailure is ResultRoutingFailure
