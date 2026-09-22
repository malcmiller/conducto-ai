# ADR 0002: Keep agents capability-first and infrastructure-free

**Status:** Accepted

## Context

Agent classes need a stable contract across local, remote, container, and
hosted deployments. If agents own registries, credentials, endpoints, or model
clients, moving them changes business logic and complicates testing.

## Decision

Agents declare metadata, capabilities, optional tools, and model requirements.
Applications and runtimes supply registries, gateways, provider bindings,
identity, transports, and lifecycle.

## Consequences

- Agent code remains portable and easy to test.
- Provider references are opaque names rather than clients.
- Remote endpoints are deployment records, not agent identity.
- More composition is explicit at application startup.

## Alternatives considered

- Service-locator or global registry access from agents. Rejected because it
  hides dependencies and shares mutable state.
- Provider clients stored on agents. Rejected because ownership, credentials,
  shutdown, and concurrent model selection become ambiguous.

## Related code and evidence

- `BaseAgent`, `@a2a_agent`, `@a2a_capability`
- `AgentRegistry`, `AgentCatalog`, `ProviderRegistry`
- Public API, registration, provider lifecycle, and concurrent runtime tests
