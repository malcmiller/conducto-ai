"""Public API acceptance coverage for invocation-scoped local delegation."""

import asyncio

import pytest

from conducto import (
    AgentRegistry,
    BaseAgent,
    Runtime,
    a2a_agent,
    a2a_capability,
    require_run_context,
)
from conducto.core.gateway_models import DiscoveryQuery
from conducto.core.invocation_results import InvocationSuccess

pytestmark = pytest.mark.acceptance


@a2a_agent(name="MathProvider", version="1.0.0", description="Provides local arithmetic.")
class MathProvider(BaseAgent):
    @a2a_capability(name="increment", description="Adds one.", tags=("math",))
    def increment(self, value: int) -> int:
        return value + 1


@a2a_agent(name="CallingAgent", version="1.0.0", description="Delegates through its context.")
class CallingAgent(BaseAgent):
    @a2a_capability(name="run", description="Invokes an allowed local capability.")
    async def run(self, value: int) -> int:
        gateway = require_run_context().gateway
        selected = await gateway.select(
            DiscoveryQuery(
                capability_ids=frozenset({"increment"}),
                tags=frozenset({"math"}),
            )
        )
        assert selected.binding is not None
        result = await gateway.invoke(selected.binding, {"value": value})
        assert isinstance(result, InvocationSuccess)
        assert isinstance(result.value, int)
        return result.value


def test_agent_discovers_and_invokes_without_orchestrator_or_target_ownership() -> None:
    registry = AgentRegistry()
    registry.register(MathProvider())
    runtime = Runtime(agent_registry=registry)

    result = asyncio.run(
        runtime.invoke(
            CallingAgent(),
            "run",
            {"value": 41},
            correlation_id="gateway-acceptance",
            allowed_capabilities=frozenset({"increment"}),
        )
    )

    assert isinstance(result, InvocationSuccess)
    assert result.value == 42
    assert result.metadata is not None
    assert result.metadata.correlation_id == "gateway-acceptance"
