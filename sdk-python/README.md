# `conducto-ai` Python SDK

The **`conducto-ai`** Python SDK provides the local reference runtime:
decorated Python agents, deterministic Agent Card generation, governed
capability discovery, model-mediated routing and delegation, validated
invocation, typed result envelopes, runtime model provenance, and structured
correlation-safe logs.

Agent Cards are generated against the pinned Conducto A2A `1.0` JSON-RPC
profile (`A2A_AGENT_CARD_SPEC_VERSION`). Use
`BaseAgent.get_agent_card("https://agent.example/a2a")` for the card
dictionary or `get_agent_card_json()` for canonical JSON.

## Documentation

The [Python SDK documentation index](docs/README.md) separates the major
implementation responsibilities:

- [Architecture](docs/architecture.md)
- [Agents and registration](docs/agents-and-registration.md)
- [Gateway and discovery](docs/gateway-and-discovery.md)
- [Orchestration and delegation](docs/orchestration-and-delegation.md)
- [Providers and models](docs/providers-and-models.md)
- [Ollama provider](docs/ollama-provider.md)
- [Runtime and invocation](docs/runtime-and-invocation.md)
- [MCP tool export](docs/mcp-export.md)
- [SDK reference](docs/sdk-reference.md)
- [Security and governance](docs/security-and-governance.md)

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

## Optional MCP tool export

Projecting existing `@a2a_capability` declarations into MCP stdio tools
requires the optional `mcp` extra:

```bash
uv sync --locked --extra mcp --group dev
uv run python examples/mcp_stdio_server.py
```

Importing `conducto` never imports the MCP SDK. See
[MCP tool export](docs/mcp-export.md).

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

## Agent chaining

The three-agent, installed-package reference demonstrates orchestrator routing,
bounded model-selected documentation delegation, typed terminal output, and a
deterministic direct invocation:

```bash
uv build
uv run --no-project --with dist/conducto_ai-*.whl python examples/agent_chaining.py
```

See [`docs/agent-chaining.md`](docs/agent-chaining.md) for configuration,
provider lifecycle snapshots, limits, security boundaries, failure handling,
and PowerShell commands.

## Local capability gateway

Agents get the model-neutral gateway only from their active
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
model-free example. A later registry registration is visible to later
discovery calls without recreating the caller or runtime.

## Model-facing toolbox projection

`conducto.core.gateway_tools` (re-exported from `conducto`) bridges gateway
discovery and a future model/tool execution loop. An agent declares which
capability families it may use – never a concrete provider agent ID -- and
`build_toolbox` projects only already-authorized candidates into a bounded,
immutable, schema-valid snapshot for one model decision boundary.

```python
from conducto import CapabilityUse, CapabilityUseRequirement, ToolboxPolicy, build_toolbox

policy = ToolboxPolicy(
    uses=(
        CapabilityUse(
            capability_ids=frozenset({"documentation.search"}),
            requirement=CapabilityUseRequirement.REQUIRED,
        ),
    )
)
result = await build_toolbox(context.gateway, policy)
if result.ok:
    tools = result.snapshot.as_model_payload()  # Safe to send to a provider.
```

A missing optional capability yields a partial toolbox; a missing required
capability fails with a typed `ToolboxResult` before any model call. A tool ID
returned by a model only resolves through `result.snapshot.resolve(tool_id)`
when it belongs to that exact snapshot.

## Bounded model delegation

Agents opt into the reusable model/tool loop with an immutable
`DelegationConfig`, then explicitly call `run_delegation()` from an active
capability. Each model turn receives one immutable toolbox snapshot and must
return exactly one structured terminal response or one tool call.

```python
from conducto import BaseAgent, CapabilityUse, ChatMessage, DelegationConfig, ToolboxPolicy
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str


class ResearchAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            delegation_config=DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"search"})),)
                ),
                max_model_turns=4,
                max_tool_calls=2,
            )
        )

    async def answer(self, question: str) -> Answer:
        outcome = await self.run_delegation(
            (ChatMessage(role="user", content=question),),
            response_type=Answer,
        )
        if not outcome.ok or outcome.value is None:
            raise RuntimeError(outcome.code.value)
        return outcome.value
```

The loop refreshes discovery only at the next model-decision boundary. An
accepted call always executes against its originating snapshot through
`AgentGateway`, where authority, binding freshness, lifecycle, cycle/depth, and
shared budgets are revalidated atomically. Calls are sequential and never
implicitly retried. Replayed call IDs terminate with a typed failure.
Child failures terminate unless an explicit fallback policy marks their safe
category eligible; a later model response is then reported as a distinct
fallback success with the child failure retained in provenance.

## Repository layout

```text
sdk-python/
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── agents-and-registration.md
│   ├── gateway-and-discovery.md
│   ├── logging.md
│   ├── orchestration-and-delegation.md
│   ├── provider-registration.md
│   ├── providers-and-models.md
│   ├── quickstart.md
│   ├── runtime-and-invocation.md
│   ├── sdk-reference.md
│   └── security-and-governance.md
├── examples/
│   ├── agent_chaining.py
│   ├── local_gateway.py
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
│           ├── delegation.py
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
**call override → run override → agent default → runtime default**.

Applications register an allowlisted provider-type factory once and bind
credential-free model references to configuration-constructed or
preconstructed clients — see
[`docs/provider-registration.md`](docs/provider-registration.md) for the full
registration API, typed failures, and the migration path from the deprecated
`ProviderRegistry.register(...)`.
