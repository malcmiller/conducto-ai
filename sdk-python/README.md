# `conducto-ai` (Python SDK)

The **`conducto-ai`** Python SDK is the foundational client and server framework for **Conducto**. It provides runtime reflection, Pydantic-based schema generation, security guardrails, and A2A (Agent2Agent) protocol transport for Python-based agents.

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

1. **Install Dependencies:**
```bash
uv sync

```


2. **Run Unit & Integration Tests:**
```bash
uv run pytest

```


3. **Format & Lint:**
```bash
uv run ruff check .
uv run ruff format .

```