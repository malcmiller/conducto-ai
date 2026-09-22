# Python SDK documentation

These guides describe the implemented Python SDK. Repository-wide protocol,
deployment, roadmap, and automation documentation remains in
[`../../docs/`](../../docs/README.md).

## Start here

- [Quick start](./quickstart.md) — install, run, build, and smoke-test the SDK.
- [Architecture](./architecture.md) — component boundaries and end-to-end data
  flow.
- [SDK reference](./sdk-reference.md) — supported public imports and API
  contracts.
- [Development guide](./development-guide.md) — local validation and
  contribution workflow.

## Core components

- [Agents and registration](./agents-and-registration.md) — decorators,
  reflection, schemas, Agent Cards, and the local `AgentRegistry`.
- [Gateway and discovery](./gateway-and-discovery.md) — policy-filtered
  discovery across local and cataloged remote agents, immutable snapshots,
  opaque bindings, lifecycle, and invocation.
- [Remote-agent catalog](./remote-agent-catalog.md) — governed catalog
  provider contract, capability indexing, lease-based instance health, and
  quarantine, disablement, revocation, and removal lifecycle transitions.
- [Deployment registration](./deployment-registration.md) — authenticated remote
  admission, opaque leases, readiness, renewal, drain, and shutdown hooks.
- [Orchestration and delegation](./orchestration-and-delegation.md) —
  deterministic routing, model-facing toolboxes, and bounded nested agent
  calls.
- [Providers and models](./providers-and-models.md) — provider contracts,
  model precedence, capability checks, and invocation-scoped model access.
- [Ollama provider](./ollama-provider.md) — local Ollama structured output,
  native tools, profiles, transport configuration, and opt-in smoke tests.
- [Runtime and invocation](./runtime-and-invocation.md) — execution context,
  validation, security, deadlines, cancellation, results, and provenance.

## Focused guides

- [Agent chaining](./agent-chaining.md)
- [MCP tool export](./mcp-export.md)
- [Provider registration](./provider-registration.md)
- [Security and governance](./security-and-governance.md)
- [Logging](./logging.md)

The code is the final source of truth. These documents describe current
behavior and distinguish it from roadmap work.
