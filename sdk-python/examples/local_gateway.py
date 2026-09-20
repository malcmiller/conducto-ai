"""Minimal model-free local capability discovery and invocation example."""

import asyncio

from conducto import (
    AgentRegistry,
    BaseAgent,
    InvocationSuccess,
    Runtime,
    a2a_agent,
    a2a_capability,
    require_run_context,
)


@a2a_agent(name="WeatherAgent", version="1.0.0", description="Provides local weather.")
class WeatherAgent(BaseAgent):
    @a2a_capability(name="temperature", description="Returns a local temperature.")
    def temperature(self, city: str) -> dict[str, object]:
        return {"city": city, "celsius": 21}


@a2a_agent(name="TravelAgent", version="1.0.0", description="Uses allowed local services.")
class TravelAgent(BaseAgent):
    @a2a_capability(name="plan", description="Builds a local travel summary.")
    async def plan(self, city: str) -> dict[str, object]:
        selection = await require_run_context().gateway.lookup(
            "WeatherAgent",
            "temperature",
        )
        assert selection.binding is not None
        result = await require_run_context().gateway.invoke(
            selection.binding,
            {"city": city},
        )
        assert isinstance(result, InvocationSuccess)
        assert isinstance(result.value, dict)
        return result.value


async def main() -> None:
    """Run one invocation with explicitly bounded gateway authority."""
    registry = AgentRegistry()
    registry.register(WeatherAgent())
    runtime = Runtime(agent_registry=registry)
    result = await runtime.invoke(
        TravelAgent(),
        "plan",
        {"city": "Toronto"},
        allowed_capabilities=frozenset({"WeatherAgent:temperature"}),
    )
    assert isinstance(result, InvocationSuccess)
    print(result.value)


if __name__ == "__main__":
    asyncio.run(main())
