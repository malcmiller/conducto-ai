# `conducto-ai` (Python SDK)

The **`conducto-ai`** Python SDK is the foundational client and server framework for **Conducto**. It provides runtime reflection, Pydantic-based schema generation, security guardrails, and A2A (Agent2Agent) protocol transport for Python-based agents.

Agent Cards are generated against the pinned **A2A Agent Card specification
0.3.0** (`A2A_AGENT_CARD_SPEC_VERSION`). Use
`BaseAgent.get_agent_card("https://agent.example/a2a")` to produce the
standards-conformant card dictionary, or `get_agent_card_json()` for canonical
golden fixtures. Reflected parameter schemas are available under the
documented `x-conducto.parameters` extension because they are not part of the
standard `AgentSkill` object.

---

## 🛠️ Requirements & Tooling

* **Python:** 3.12+
* **Package & Project Manager:** [`uv`](https://github.com/astral-sh/uv?utm_source=gemini) (recommended) or `poetry` / `pip`
* **Core Dependencies:**
* `pydantic-ai` for dynamic model execution and structured outputs
* `fastapi` & `uvicorn` for hosting JSON-RPC / Agent Card endpoints
* `httpx` for mTLS client transport
* `authlib` & `cryptography` for OAuth 2.0 OBO token exchange and ECDSA signatures



---

## 📂 Folder Structure

```text
sdk-python/
├── pyproject.toml               # Project metadata and dependencies (uv/pip)
├── README.md                    # Sub-README for the Python SDK
├── src/
│   └── conducto/
│       ├── __init__.py          # Top-level SDK exports (@a2a_capability, BaseAgent, etc.)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py         # BaseAgent implementation & reflection engine
│       │   ├── decorators.py    # @a2a_agent, @a2a_capability, @tool decorators
│       │   └── orchestrator.py  # OrchestratorAgent state machine and discovery engine
│       ├── security/
│       │   ├── __init__.py
│       │   ├── guardrails.py    # @require_approval, @require_scope decorators
│       │   ├── crypto.py        # ECDSA challenge signing & verification handlers
│       │   └── auth.py          # OAuth 2.0 OBO token exchange & JWT claims validation
│       ├── transport/
│       │   ├── __init__.py
│       │   ├── server.py        # FastAPI A2A server & /.well-known/agent-card.json route
│       │   └── client.py        # httpx mTLS JSON-RPC 2.0 transport client
│       └── exporters/
│           ├── __init__.py
│           └── mcp.py           # Dual-protocol MCP server exporter (@mcp_tool)
└── tests/
    ├── unit/
    │   ├── test_reflection.py  # Inspection & agent card schema tests
    │   ├── test_guardrails.py  # HITL pause/resume & crypto challenge tests
    │   └── test_security.py    # Scope & token validation tests
    └── integration/
        └── test_a2a_network.py # End-to-end FastAPI + HTTP/2 JSON-RPC transport tests

```

---

## 🚀 Quick Usage

### Logging

Conducto emits versioned standard-library logging events but never configures
the root logger or adds handlers during import. Applications can attach their
own handler to the `conducto` logger, or opt in to a small development setup:

```python
from conducto import configure_logging

configure_logging(format="json")  # or format="development"
```

Events carry schema version, correlation ID, agent/capability identifiers,
outcome, duration, stable error category, and model provider/provenance where
applicable. Discovery is `DEBUG`; normal lifecycle events are `INFO`; timeouts
are `WARNING`; unexpected capability failures are `ERROR`. Prompts, model
responses, capability arguments/results, credentials, tokens, approval data,
and exception tracebacks are excluded by default. Sensitive payload logging is
available only through both per-event and `configure_logging(
include_sensitive_data=True)` opt-ins; enable it only in controlled
development environments. See [the logging guide](docs/logging.md) for the
event schema, formatter integration, context propagation, and compatibility
guarantees.

### Per-run model configuration

Provider clients and credentials belong to a runtime-owned `ProviderRegistry`.
Agents and callers use opaque `ModelReference` values. Resolution is deterministic:
**call override → run override → agent default → runtime default**.

```python
from conducto import (
    ModelConfiguration,
    ModelReference,
    ProviderRegistry,
    RunConfig,
    Runtime,
    RuntimeConfig,
)

registry = ProviderRegistry()
registry.register(
    "fast",
    provider_client,  # Credentials stay inside this runtime-owned client.
    ModelConfiguration(provider="example", model="fast-model"),
)
runtime = Runtime(
    provider_registry=registry,
    config=RuntimeConfig(default_model=ModelReference("fast")),
)

result = await runtime.invoke(
    worker,
    "work",
    {},
    run_config=RunConfig(model=ModelReference("fast")),
)
```

`RunContext`, `RunConfig`, `RuntimeConfig`, `AgentModelConfig`, and
`InvocationMetadata` are immutable public contracts. `RunContext.to_dict()`
includes only model provenance, timing, cancellation state, and
provider-neutral metadata; it excludes provider clients and configuration.
Model-backed capabilities use the invocation-scoped gateway rather than a raw
provider:

```python
from conducto import ChatMessage, require_run_context

context = require_run_context()
draft = await context.models.require().complete_typed(
    messages=(ChatMessage(role="user", content="Draft the audit report"),),
    response_type=AuditReportDraft,
)
```

`require_run_context()` raises `NoActiveRunContextError` when model-backed code
is called directly instead of through `Runtime.invoke()` or
`OrchestratorAgent.invoke()`. Routed invocation metadata retains ordered
provenance and usage for both the routing model and capability model in
`metadata.model_calls`.
Use `model_required=False` on `@a2a_agent` or `@a2a_capability` for
deterministic work. A runtime policy callback can deny selections using agent,
caller, environment, cost-tier, and data-classification facts before any
provider request or capability invocation.

### Defining an Agent

```python
from conducto import BaseAgent, a2a_agent, a2a_capability, require_approval, require_scope

@a2a_agent(name="AuditAgent", version="1.0")
class AuditAgent(BaseAgent):
    
    @a2a_capability(description="Evaluates transaction risk against compliance rules.")
    @require_scope("audit:read")
    @require_approval(role="compliance_officer")
    async def evaluate_transaction_risk(self, vendor_id: str, amount: float) -> dict:
        # Pydantic AI or custom logic
        return {
            "vendor_id": vendor_id,
            "status": "APPROVED",
            "risk_score": 0.05
        }

```

### Running the A2A Server

```python
import uvicorn
from conducto.transport import create_a2a_app
from my_agents import AuditAgent

agent = AuditAgent()
app = create_a2a_app(agent)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

---

## 🧪 Development & Testing

Every command below is exactly what CI runs (see
`../.github/workflows/python-ci.yml`), so a clean checkout can reproduce any
CI failure locally.

**Supported matrix:** Python `3.12` and `3.13` on `ubuntu-latest` and
`windows-latest`.

1. **Install dependencies (including dev tools), from the lockfile:**
```bash
uv sync --locked --group dev

```

2. **Format check:**
```bash
uv run ruff format --check .

```

3. **Lint:**
```bash
uv run ruff check .

```

4. **Static type-check:**
```bash
uv run mypy src

```

5. **Unit & schema/golden-fixture tests:**
```bash
uv run pytest -m "not acceptance"

```

6. **Milestone 1 quick-start acceptance suite** (the stable, story-level gate
   for `@a2a_agent` / `@a2a_capability` / `get_agent_card()` / `OrchestratorAgent`):
```bash
uv run pytest -m acceptance

```

7. **Build the package:**
```bash
uv build

```

8. **Installed-wheel smoke test** (verifies the *built artifact* imports and
   runs the quick-start flow with no dev dependencies on the path):
```bash
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python scripts/smoke_test.py

```

### Continuous Integration

Pull requests and pushes to `main` run a single required check named
**`Python CI (required)`**. It aggregates: formatting, linting, `mypy`,
unit/schema tests, and package build + installed-wheel smoke tests across
the full OS/Python matrix, plus the Milestone 1 acceptance suite across the
OS matrix (Python 3.12 only). The check still reports (as a fast no-op
success) on documentation-only or unrelated-path changes, so it is safe to
mark as required in branch protection / repository rulesets. See
`../.github/workflows/python-ci.yml` for details, and
`../.github/README.md` for how to configure the required-check ruleset and
how .NET CI will be added alongside it.