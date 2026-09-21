# Conducto repository documentation

This directory contains repository-wide product, protocol, deployment, and
automation documentation. Implementation-specific Python guides live with the
package in [`sdk-python/docs/`](../sdk-python/docs/README.md).

## What this repository contains

- `README.md` at the repository root introduces the Conducto vision and roadmap.
- `sdk-python/` contains the shipped Python package (`conducto-ai`).
- `sdk-python/docs/` documents the Python implementation and public API.
- `sdk-python/tests/` records expected behavior through unit, golden, and
  acceptance tests.

## Documentation map

- [Repository overview](./repository-overview.md) — repository purpose, structure, and key concepts.
- [Communication paths and standards](./communication-and-standards.md) — A2A sequencing, local orchestration, container transport, and Microsoft Foundry-based model routing.
- [Conducto A2A 1.0 profile](./a2a-1-profile.md) — pinned protocol provenance, feature matrix, compatibility policy, conformance fixtures, and update procedure.
- [Deployment topologies and federation](./deployment-and-federation.md) — local, container, Foundry, hybrid, and cross-organization scenarios plus ownership, trust, catalog, and workflow orchestration.
- [Repository automation and agent guidance](./repository-automation.md) — coding-agent instructions, required validation, CI behavior, branch protection, and workflow maintenance.
- [Python SDK documentation](../sdk-python/docs/README.md) — architecture,
  registration, gateway, orchestration, provider, runtime, security, API, and
  development guides.

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

The Python SDK uses structured metadata to reflect methods into A2A-compatible
Agent Cards and deterministic routing payloads.

## Scope note

The current implementation is Python-first, while repository-level contracts
are designed for later .NET parity and cross-language conformance.
