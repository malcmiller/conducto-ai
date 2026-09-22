# Connect two local agents

This example lets a travel agent call a weather agent without networking.

## The flow

```mermaid
flowchart LR
    App --> T[Travel agent]
    T -->|gateway lookup and invoke| W[Weather agent]
    W --> T
```

## Code

```python
import asyncio

from conducto import (
    AgentRegistry,
    BaseAgent,
    Runtime,
    a2a_agent,
    a2a_capability,
    require_run_context,
)
from conducto.core.invocation_results import InvocationSuccess


@a2a_agent(name="WeatherAgent", version="1.0.0", description="Provides weather.")
class WeatherAgent(BaseAgent):
    @a2a_capability(name="temperature", description="Returns one temperature.")
    def temperature(self, city: str) -> dict[str, object]:
        return {"city": city, "celsius": 21}


@a2a_agent(name="TravelAgent", version="1.0.0", description="Plans local trips.")
class TravelAgent(BaseAgent):
    @a2a_capability(name="plan", description="Builds a weather-aware trip plan.")
    async def plan(self, city: str) -> dict[str, object]:
        selection = await require_run_context().gateway.lookup(
            "WeatherAgent",
            "temperature",
        )
        if selection.binding is None:
            raise RuntimeError("Weather capability is unavailable")

        result = await require_run_context().gateway.invoke(
            selection.binding,
            {"city": city},
        )
        if not isinstance(result, InvocationSuccess):
            raise RuntimeError(f"Weather call failed: {type(result).__name__}")

        return {"destination": city, "weather": result.value}


async def main() -> None:
    registry = AgentRegistry()
    registry.register(WeatherAgent())
    runtime = Runtime(agent_registry=registry)

    result = await runtime.invoke(
        TravelAgent(),
        "plan",
        {"city": "Toronto"},
        allowed_capabilities=frozenset({"WeatherAgent:temperature"}),
    )
    if not isinstance(result, InvocationSuccess):
        raise RuntimeError(f"Travel call failed: {type(result).__name__}")

    print(result.value)


asyncio.run(main())
```

## Important details

- The travel agent asks the active runtime gateway for a capability.
- The gateway returns an opaque binding, not a mutable agent object.
- `allowed_capabilities` restricts the child call.
- The child call enters the same runtime pipeline as the parent call.
- The weather agent does not need to know who called it.

A similar executable version lives in `examples/local_gateway.py`.

## When to add a model

The target above is explicit. Add model-driven selection when the application
needs to choose among compatible capabilities from natural-language context.
The model sees bounded tool definitions, not registry objects or credentials.

See [orchestration and delegation](../orchestration-and-delegation.md) for the
three call modes and their limits.

Next: [host an agent for another process](./host-an-agent.md).
