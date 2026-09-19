# Conducto documentation

This directory contains the canonical documentation for the Conducto core repository and the Python SDK implementation it currently ships.

## What this repository contains

- `README.md` at the repository root introduces the Conducto vision, architecture, and the Python SDK.
- `sdk-python/` contains the shipped Python package (`conducto-ai`).
- `sdk-python/src/conducto/` is the implementation of the agent registration, routing, validation, and model-provider abstractions.
- `sdk-python/tests/` documents the public behavior expected by the SDK through unit, golden, and acceptance tests.

## Documentation map

- [Repository overview](./repository-overview.md) — repository purpose, structure, and key concepts.
- [Architecture](./architecture.md) — runtime layers, data flow, and the relationship between agents, orchestrators, and providers.
- [SDK reference](./sdk-reference.md) — API surface for `BaseAgent`, decorators, orchestration, and provider contracts.
- [Security and governance](./security-and-governance.md) — validation, A2A Agent Card guarantees, and trust boundaries.
- [Development guide](./development-guide.md) — local setup, formatting, testing, CI commands, and contribution workflow.

## Quick start

```python
from conducto import BaseAgent, a2a_agent, a2a_capability

@a2a_agent(name="AuditAgent", version="1.0", description="Evaluates transaction risk.")
class AuditAgent(BaseAgent):
    @a2a_capability(description="Evaluates transaction risk against compliance rules.")
    def evaluate_transaction_risk(self, vendor_id: str, amount: float) -> dict:
        return {
            "vendor_id": vendor_id,
            "status": "APPROVED",
            "risk_score": 0.05,
        }
```

The project uses structured metadata to reflect Python methods into A2A-compatible Agent Cards and deterministic routing payloads.

## Scope note

The codebase in this repository is centered on the Python SDK and the core protocol primitives. The repository is intentionally designed to be the reference implementation for A2A-style capability discovery, local orchestration, and provider-independent contracts.
