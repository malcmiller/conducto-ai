# Repository overview

Conducto is a capability-first framework for governed multi-agent systems. An
agent publishes typed capabilities, callers discover only the capabilities
they are authorized to use, and every invocation follows the same validation,
security, deadline, cancellation, result, and provenance rules.

## Repository shape

```text
conducto-ai/
├── .github/                  Workflows and path-scoped agent instructions
├── docs/                     Product and Python SDK documentation
├── examples/                 Runnable application examples
├── scripts/                  Package and release verification
├── src/conducto/             Python package
├── tests/                    Unit, acceptance, and golden tests
├── AGENTS.md                 Repository-wide agent guidance
├── pyproject.toml            Package and tool configuration
├── README.md                 Product and SDK overview
├── uv.lock                   Reproducible dependency lock
└── LICENSE
```

The repository is Python-only. The SDK establishes reference behavior and
protocol fixtures before cross-organization federation is added. Federation
follows only after local, remote, container, and Foundry behavior is proven.

## Product concepts

### Agent and capability

An agent publishes identity, version, and typed capabilities. Capability
descriptors provide stable identifiers, schemas, tags, security requirements,
and enough metadata for compatible discovery without exposing implementation
objects.

### Registry and catalog

A registry or catalog owns identity, capability metadata, lifecycle, health,
versions, and coherent snapshots. It does not call models, make
caller-specific policy decisions, or execute capabilities.

### Gateway

The gateway combines catalog facts with caller authority and policy. It owns
filtered discovery, target selection, opaque bindings, transport choice, and
normalized invocation. A model may select only from the bounded tools the
gateway has already authorized.

### Runtime

The runtime owns invocation state: identity, authorization, model resolution,
deadlines, cancellation, budgets, audit, execution, and typed results. Local
execution is the reference path; transports must preserve its behavior rather
than create alternate validation or error semantics.

### Provider

Model providers implement a neutral structured-output contract behind
credential-free references. Vendor clients and credentials remain
application-owned runtime concerns and do not leak into agent contracts.

### Orchestration

Orchestration may be deterministic application logic, model-assisted
top-level routing, bounded model-selected delegation, or a durable workflow.
All modes use the same discovery, authority, invocation, and result
boundaries.

## Documentation boundaries

Documents in [`docs/`](./README.md) define shared protocols, deployment
progression, federation, project automation, Python APIs, and implementation
details.
