# Conducto 🚀

**Conducto** is a secure, polyglot (Python / C#) multi-agent orchestration framework designed to provide a standardized, developer-friendly "Agent2Agent" (A2A) abstraction layer. It brings native Object-Oriented Programming (OOP) paradigms (`@decorators` in Python, `[Attributes]` in C#) to multi-agent networking, governance, discovery, and execution.

## Documentation

The repository documentation lives in the [`docs/`](./docs/README.md) folder and covers the repository overview, architecture, SDK reference, security model, and local development workflow.

---

## 🌟 Key Features

* **Polyglot Agent Interoperability:** Build agents in Python (`conducto-ai`) or C# (`Conducto.NET`) that discover, negotiate, and delegate tasks to each other over a shared protocol.
* **Declarative Governance & Guardrails:** Built-in Human-in-the-Loop (HITL) approval challenges via `@require_approval` / `[RequireApproval]` and fine-grained authorization via `@require_scope` / `[RequireScope]`.
* **Zero-Trust Security:** Multi-layer defense using Mutual TLS (mTLS 1.3), OAuth 2.0 On-Behalf-Of (OBO) token exchange, and cryptographically signed HITL challenges.
* **Dual-Protocol Standards:** Expose capabilities as **A2A** network endpoints (JSON-RPC 2.0 over HTTP/2) or **MCP** (Model Context Protocol) server tools with zero duplicate code.
* **Dynamic Model Ingestion:** Native support for runtime model assignment via `pydantic-ai` and local inference endpoints (Ollama, vLLM, LM Studio).

---

## 🏗️ Architecture Overview

Conducto bridges low-level network protocols with clean developer abstractions:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Orchestrator Agent                                │
 │               (Python conducto-ai / C# Conducto.NET)                   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                      mTLS + OAuth 2.0 OBO Tokens
                      JSON-RPC 2.0 over HTTP/2
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       Target Sub-Agent Host                            │
 │  ┌───────────────────────┐  ┌────────────────────┐  ┌───────────────┐ │
 │  │ FastAPI / ASP.NET     │─>│ BaseAgent Engine   │─>│ Capability /  │ │
 │  │ (Auth & Verification) │  │ (Reflection Engine)│  │ Guardrails    │ │
 │  └───────────────────────┘  └────────────────────┘  └───────────────┘ │
 └────────────────────────────────────────────────────────────────────────┘

```

---

## 🚀 Quick Start

### Python (`conducto-ai`)

```python
from conducto import BaseAgent, a2a_capability, require_approval, require_scope

class FinancialAgent(BaseAgent):
    @a2a_capability(description="Executes vendor payout after compliance clearance.")
    @require_scope("payout:write")
    @require_approval(role="finance_lead")
    async def execute_payout(self, vendor_id: str, amount: float) -> str:
        # Cross-agent execution or business logic
        return f"Successfully processed payout of ${amount} to {vendor_id}"

```

### C# (`Conducto.NET`)

```csharp
using Conducto.Core;
using Conducto.Security;

namespace Conducto.Financial;

[A2AAgent(Name = "FinancialAgent", Version = "1.0")]
public class FinancialAgent : BaseAgent 
{
    [A2ACapability("Executes vendor payout after compliance clearance.")]
    [RequireScope("payout:write")]
    [RequireApproval(Role = "finance_lead")]
    public async Task<string> ExecutePayoutAsync(string vendorId, decimal amount)
    {
        return $"Successfully processed payout of ${amount} to {vendorId}";
    }
}

```

---

## 📋 Development Roadmap & Milestones

Development is tracked across eight milestones. Python establishes the reference
flow, followed by secure network interoperability, operations and deployment,
.NET parity, hybrid orchestration, and optional cross-organization federation.

See [Deployment topologies and federation](./docs/deployment-and-federation.md)
for the local, container, Microsoft Foundry, hybrid, and cross-organization
topologies; ownership boundaries; trust model; and complete delivery path.

The required promotion order is **local on one machine → local containers →
Azure deployment → optional cross-organization federation**. Each stage uses
the same agent capability contract and must pass before cloud-specific hosting
is introduced.

### Milestone 1: Core Protocol & Reflection Discovery

* **Story 1.1 (Python):** Implement `@a2a_agent`, `@a2a_capability`, `@tool` decorators & reflection engine.
* **Story 1.2 (Python):** Build `get_agent_card()` generator mapping to `/.well-known/agent-card.json`.
* **Story 1.3 (Python):** Build local `OrchestratorAgent` for discovery and dynamic prompt context aggregation.
* **Story 1.4 (C#):** Build `[A2AAgent]`, `[A2ACapability]`, `[ConductoTool]` attributes and `System.Reflection` engine.

### Milestone 2: Security & Governance Guardrail Interceptors

* **Story 2.1 (Python):** Implement `@require_approval` and `@require_scope` with `INPUT_REQUIRED` state challenges.
* **Story 2.2 (Python):** Implement ECDSA challenge signature verification.
* **Story 2.3 (C#):** Implement `[RequireApproval]` and `[RequireScope]` ASP.NET Core interceptors.

### Milestone 3: Polyglot Network Transport (A2A Wire Protocol)

* **Story 3.1 (Python):** Implement FastAPI JSON-RPC 2.0 handler and remote endpoint discovery.
* **Story 3.2 (Python):** Implement `httpx` mTLS client transport and OAuth 2.0 OBO token exchange.
* **Story 3.3 (C#):** Implement ASP.NET Core middleware for JSON-RPC dispatching and JWT verification.
* **Story 3.4 (Integration):** Execute end-to-end Python $\rightarrow$ C# $\rightarrow$ Python cross-language integration tests.

### Milestone 4: Dual-Protocol Exporters & Operations

* **Story 4.1 (Python):** Build `@mcp_tool` exporter allowing capability methods to serve as MCP tools.
* **Story 4.2 (Polyglot):** Add OpenTelemetry W3C trace context propagation across JSON-RPC headers.
* **Story 4.3 (C#):** Implement `to_mcp_server()` exporter in `Conducto.NET`.

### Milestone 5: Model Runtimes & Microsoft Foundry Deployment

* First run agents on one machine against local Llama or self-hosted models.
* Next package and verify the same agents in local containers with external
  model and trust configuration.
* Then promote the proven artifacts to remote Azure containers or
  same-organization Foundry hosting without requiring federation.
* Preserve independent per-run model selection and execution context.

### Milestone 6: .NET SDK Parity

* Bring .NET metadata, execution, security, transport, MCP, and contract behavior
  to parity with the Python reference implementation.

### Milestone 7: Hybrid Deployment & Workflow Orchestration

* Organize local, containerized, and Foundry-hosted agents through one
  orchestrator without changing the agent programming model.
* Add a governed remote-agent catalog with registration, refresh, quarantine,
  revocation, and compatibility handling.
* Add policy-aware workflows across mixed deployment types.

### Milestone 8: Cross-Organization Federation

* Layer optional federation over remotely deployed container or Microsoft
  Foundry agents.
* Define global agent identity, signed discovery metadata, and external trust
  onboarding.
* Verify multi-organization Azure discovery, delegation, audit, revocation, and
  operations end to end.

---

## 🤝 How to Contribute

We welcome contributions from both Python and .NET developers!

1. **Pick an Issue:** Check out the [GitHub Issues](https://www.google.com/search?q=https://github.com/your-org/conducto/issues&utm_source=gemini) tagged with `good first issue`, `area:python`, or `area:dotnet`.
2. **Branching Strategy:** Fork the repository and create a feature branch using `feat/milestone-X-description` or `fix/issue-number`.
3. **Local Setup (Python):**
```bash
cd sdk-python
uv sync --locked --group dev
uv run pytest

```


4. **Local Setup (.NET):**
```bash
cd sdk-dotnet
dotnet restore
dotnet test

```


5. **Submit a PR:** Every pull request must pass the single required
   **`Python CI (required)`** check (formatting, linting, `mypy`, unit and
   schema/golden-fixture tests, the Milestone 1 acceptance suite, and package
   build + installed-wheel smoke tests — see `sdk-python/README.md` and
   `.github/README.md`).

---

## 📜 License

Licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE&utm_source=gemini).
