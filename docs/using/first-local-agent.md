# Create one local agent

This example uses no model and no network. It demonstrates the smallest
governed capability invocation.

## Install

```bash
pip install conducto-ai
```

Conducto requires Python 3.12 or newer.

## Define and invoke the agent

```python
import asyncio

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.invocation_results import InvocationSuccess


@a2a_agent(
    name="WeatherAgent",
    version="1.0.0",
    description="Provides local weather.",
)
class WeatherAgent(BaseAgent):
    @a2a_capability(
        name="temperature",
        description="Returns the temperature for one city.",
    )
    def temperature(self, city: str) -> dict[str, object]:
        return {"city": city, "celsius": 21}


async def main() -> None:
    runtime = Runtime()
    result = await runtime.invoke(
        WeatherAgent(),
        "temperature",
        {"city": "Toronto"},
    )

    if not isinstance(result, InvocationSuccess):
        raise RuntimeError(f"Invocation failed: {type(result).__name__}")

    print(result.value)


asyncio.run(main())
```

Expected output:

```text
{'city': 'Toronto', 'celsius': 21}
```

## What Conducto did

1. Reflected the capability's typed argument schema.
2. Confirmed `city` was a string.
3. Ran the method through `Runtime`.
4. Serialized the return value.
5. Returned an `InvocationSuccess` envelope instead of a raw value.

## Why not call `WeatherAgent().temperature(...)`?

Direct Python calls are still possible, but they bypass Conducto's validation,
authorization, approval, auditing, deadline, cancellation, model-resolution,
and result contracts. Application entry points should use the runtime.

## Next

- Add a model with [an Ollama-backed agent](./ollama-agent.md).
- Let one agent use another with
  [a two-agent workflow](./two-agent-workflow.md).
- Understand the internals in the
  [invocation lifecycle](../system-design/invocation-lifecycle.md).
