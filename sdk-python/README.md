# `conducto-ai` Python SDK

The **`conducto-ai`** Python SDK provides the Milestone 1 local-agent runtime:
decorated Python agents, deterministic Agent Card generation, model-mediated
local routing, validated capability invocation, typed result envelopes, runtime
model provenance, and structured correlation-safe logs.

Agent Cards are generated against pinned A2A Agent Card specification
`0.3.0` (`A2A_AGENT_CARD_SPEC_VERSION`). Use
`BaseAgent.get_agent_card("https://agent.example/a2a")` for the card
dictionary or `get_agent_card_json()` for canonical JSON.

## Quick start

From a clean checkout:

```bash
cd sdk-python
uv sync --locked --group dev
uv run pytest -m acceptance
uv build
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python examples/quickstart.py
uv run --no-project --with "$wheel" python scripts/smoke_test.py
```

Expected example output:

```text
quickstart result: agent=InvoiceAgent capability=classify_invoice value={"amount": 1250.0, "approved": false, "decision": "review", "vendor_id": "vendor-42"} correlation_id=quickstart-local-001
```

See the detailed, copy/pasteable guide in
[`docs/quickstart.md`](docs/quickstart.md), including Windows PowerShell
commands, package build steps, smoke testing, and troubleshooting.

## Public API example

```python
from conducto import BaseAgent, a2a_agent, a2a_capability


@a2a_agent(
    name="InvoiceAgent",
    version="1.0.0",
    description="Classifies invoices for deterministic local workflows.",
)
class InvoiceAgent(BaseAgent):
    @a2a_capability(
        name="classify_invoice",
        description="Classifies an invoice amount for approval routing.",
    )
    def classify_invoice(self, vendor_id: str, amount: float) -> dict[str, object]:
        return {
            "vendor_id": vendor_id,
            "amount": amount,
            "approved": amount < 1000,
        }
```

`examples/quickstart.py` shows the complete two-agent flow using only public
imports from `conducto` and a deterministic `FakeModel`.

## Local capability gateway

Agents obtain the model-neutral gateway only from their active
`require_run_context().gateway`. Applications own an `AgentRegistry`, inject it
into `Runtime`, and declare the caller's allowed capability IDs when starting a
run. Discovery returns immutable descriptors and opaque bindings; invocation
revalidates the binding and dispatches through the normal runtime pipeline.

```python
registry = AgentRegistry()
registry.register(WeatherAgent())
runtime = Runtime(agent_registry=registry)

result = await runtime.invoke(
    TravelAgent(),
    "plan",
    {"city": "Toronto"},
    allowed_capabilities=frozenset({"WeatherAgent:temperature"}),
)
```

Inside `TravelAgent.plan`, use `await context.gateway.lookup(...)` or
`discover(DiscoveryQuery(...))`, then pass the returned binding to
`context.gateway.invoke(...)`. See
[`examples/local_gateway.py`](examples/local_gateway.py) for a complete,
model-free example. A later registry registration is visible to subsequent
discovery calls without recreating the caller or runtime.

## Repository layout

```text
sdk-python/
├── docs/
│   ├── logging.md
│   └── quickstart.md
├── examples/
│   └── quickstart.py
├── scripts/
│   └── smoke_test.py
├── src/
│   └── conducto/
│       ├── __init__.py
│       └── core/
│           ├── agent.py
│           ├── agent_card.py
│           ├── decorators.py
│           ├── invocation.py
│           ├── invocation_results.py
│           ├── logging.py
│           ├── model_config.py
│           ├── model_gateway.py
│           ├── model_resolution.py
│           ├── orchestrator.py
│           ├── parameter_schema.py
│           ├── provider.py
│           ├── provider_registry.py
│           ├── registration.py
│           ├── registry.py
│           ├── run_context.py
│           ├── runtime.py
│           ├── runtime_errors.py
│           └── serialization.py
├── tests/
│   ├── acceptance/
│   │   ├── test_quickstart.py
│   │   ├── test_quickstart_failures.py
│   │   └── test_runtime_invocation.py
│   ├── golden/
│   └── test_*.py
├── pyproject.toml
└── uv.lock
```

## Development checks

These commands match `.github/workflows/python-ci.yml`:

```bash
uv sync --locked --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src examples scripts tests/acceptance
uv run pytest -m "not acceptance"
uv run pytest -m acceptance
uv build
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python scripts/smoke_test.py
```

Supported CI targets are Python `3.12` and `3.13` on `ubuntu-latest` and
`windows-latest`. The single required aggregate check is **Python CI
(required)**.

## Logging and model provenance

Conducto emits versioned standard-library logging events but never configures
the root logger during import. Applications can attach handlers to the
`conducto` logger or opt into:

```python
from conducto import configure_logging

configure_logging(format="json")
```

Lifecycle logs carry schema version, correlation ID, agent/capability IDs,
outcome, duration, and model provenance where applicable. Prompts, model
responses, capability arguments/results, credentials, tokens, approval data,
and exception tracebacks are excluded by default. See
[`docs/logging.md`](docs/logging.md) for the full event contract.

Provider clients and credentials belong to a runtime-owned `ProviderRegistry`.
Agents and callers use opaque model references. Resolution is deterministic:
**call override -> run override -> agent default -> runtime default**.
