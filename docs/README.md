# Conducto documentation

Start with the idea, try it, and then look under the hood. You do not need to
understand the whole framework before creating an agent.

## Choose your path

### I am new to Conducto

Read the five short pages in [Understanding Conducto](./understanding/README.md):

1. [What Conducto is](./understanding/what-is-conducto.md)
2. [Agents, capabilities, and orchestrators](./understanding/agents-capabilities-orchestrators.md)
3. [How agents communicate](./understanding/how-agents-communicate.md)
4. [Security in plain English](./understanding/security-in-plain-english.md)
5. [Authentication, authorization, and scopes](./understanding/authentication-and-scopes.md)
6. [Where agents can run](./understanding/where-agents-run.md)

These pages avoid implementation detail until the underlying idea is clear.

### I want to build something

Follow [Using Conducto](./using/README.md):

- [Create one local agent](./using/first-local-agent.md)
- [Connect an agent to Ollama](./using/ollama-agent.md)
- [Connect two local agents](./using/two-agent-workflow.md)
- [Host an agent for another process](./using/host-an-agent.md)

### I want to understand the system

Continue with [System design](./system-design/README.md):

- [Architecture overview](./system-design/architecture-overview.md)
- [Module map](./system-design/module-map.md)
- [Invocation lifecycle](./system-design/invocation-lifecycle.md)
- [System invariants](./system-design/invariants.md)

### I need to change the code

Use [Maintainer guides](./maintainers/README.md):

- [How to approach a change](./maintainers/how-to-change-conducto.md)
- [Common change recipes](./maintainers/change-recipes.md)
- [Validation and debugging](./maintainers/validation-and-debugging.md)
- [Architecture decisions](./decisions/README.md)

### I need exact contracts

The detailed reference remains authoritative:

- [SDK reference](./sdk-reference.md)
- [A2A 1.0 profile](./a2a-1-profile.md)
- [A2A ASGI server](./a2a-server.md)
- [Runtime and invocation](./runtime-and-invocation.md)
- [Gateway and discovery](./gateway-and-discovery.md)
- [Providers and models](./providers-and-models.md)
- [Security and governance](./security-and-governance.md)
- [Deployment and federation](./deployment-and-federation.md)

## Detailed topic index

### Agents and orchestration

- [Agents and registration](./agents-and-registration.md)
- [Capabilities and tools](./capabilities-and-tools.md)
- [Agent chaining](./agent-chaining.md)
- [Orchestration and delegation](./orchestration-and-delegation.md)
- [Gateway and discovery](./gateway-and-discovery.md)
- [Remote-agent catalog](./remote-agent-catalog.md)

### Protocols and deployment

- [Communication paths and standards](./communication-and-standards.md)
- [A2A 1.0 profile](./a2a-1-profile.md)
- [A2A ASGI server](./a2a-server.md)
- [MCP export](./mcp-export.md)
- [Deployment registration](./deployment-registration.md)
- [Deployment and federation](./deployment-and-federation.md)

### Models and operation

- [Provider registration and lifecycle](./provider-registration.md)
- [Ollama provider](./ollama-provider.md)
- [OpenAI-compatible provider](./openai-compatible-provider.md)
- [Logging and telemetry](./logging.md)

### Repository development

- [Repository overview](./repository-overview.md)
- [Development guide](./development-guide.md)
- [Repository automation](./repository-automation.md)
- [Release guide](./releasing.md)

## Documentation principle

Each path follows the same progression:

> plain-language idea → mental model → runnable example → technical flow →
> implementation map → maintenance rules

Conducto is Python-only. The root `README.md` introduces the product;
`src/conducto/` contains the package; `examples/` contains runnable examples;
and `tests/` pin expected behavior.
