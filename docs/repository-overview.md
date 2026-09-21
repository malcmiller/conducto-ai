# Repository overview

Conducto is a capability-first framework for governed multi-agent systems. An
agent publishes typed capabilities, callers discover only the capabilities
they are authorized to use, and every invocation follows the same validation,
security, deadline, cancellation, result, and provenance rules.

## Repository shape

```text
conducto-ai/
├── .github/                  Workflows and path-scoped agent instructions
├── docs/                     Product, protocol, deployment, and automation docs
├── sdk-python/               Python reference SDK and implementation docs
├── AGENTS.md                 Repository-wide agent guidance
├── README.md                 Product overview and roadmap
└── LICENSE
```

The repository is intentionally Python-first. The Python SDK establishes
reference behavior and language-neutral fixtures before .NET parity is added.
Cross-organization federation follows only after local, remote, and
cross-language behavior is proven.

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

Repository-wide documents in [`docs/`](./README.md) define shared protocols,
deployment progression, federation, and project automation. Python APIs and
implementation details are documented beside the package in
[`sdk-python/docs/`](../sdk-python/docs/README.md).
