# Repository overview

Conducto is a secure multi-agent orchestration framework built around a small but explicit abstraction: a Python or .NET agent exposes named capabilities, those capabilities are reflected into a structured Agent Card, and a local orchestrator can route a user request to the correct capability.

## Repository shape

```text
conducto-ai/
├── README.md
├── LICENSE
├── .github/
├── sdk-python/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   │   └── conducto/
│   │       ├── __init__.py
│   │       └── core/
│   │           ├── __init__.py
│   │           ├── agent.py
│   │           ├── decorators.py
│   │           ├── orchestrator.py
│   │           └── provider.py
│   ├── tests/
│   │   ├── acceptance/
│   │   ├── golden/
│   │   └── *.py
│   └── uv.lock
└── docs/
```

## Core concepts

### Agent

An agent is a Python class that subclasses `BaseAgent`. It is the primary unit of discovery and execution.

The class declares:

- a published name and version via `@a2a_agent`
- one or more exported capabilities via `@a2a_capability`
- optional internal helper methods via `@tool`

The agent does not need to be network-aware to be useful. It is a reflection-driven registry object that can be installed into an `OrchestratorAgent` and invoked locally.

### Capability

A capability is a method on an agent that is published as a callable action. Each capability is introspected to create a parameter schema, a stable identifier, and a description used in the A2A Agent Card.

The `BaseAgent` class uses Pydantic to convert Python signatures into structured JSON Schema payloads. This is crucial because the capability metadata is used both for validation and for agent discovery.

### Orchestrator

An `OrchestratorAgent` is a local registry of agents and their capabilities. It:

- stores agents by published name
- tracks capability collisions
- exposes routing metadata
- validates invocation arguments before execution
- runs custom model-based routing when a `ModelProvider` is configured

The orchestrator aims to be deterministic and safe: it serializes return values, prevents invalid data from leaking through the public contract, and exposes typed result envelopes instead of raw exceptions.

### Provider abstraction

The provider layer is deliberately neutral. `ModelProvider` is a protocol, and `ModelConfiguration` configures the runtime model that will rank or select capabilities. The project includes a `FakeModel` implementation for deterministic tests and a structured-output schema that routes the orchestrator's decisions.

## Why the repository exists

This repository solves a specific orchestration problem:

1. an agent must advertise what it can do
2. a caller must be able to discover those capabilities without manual registry wiring
3. a capability invocation must validate arguments, run safely, and return a typed result
4. the output must be portable and structured enough to be used in a larger multi-agent system

The result is a foundation for Agent2Agent interoperability and for future polyglot agent ecosystems.

## Current implementation status

The codebase clearly separates product concepts from implementation details:

- `decorators.py` stores metadata declaratively
- `agent.py` materializes metadata into Agent Cards and JSON schemas
- `orchestrator.py` performs local routing, validation, and timeout-safe invocation
- `provider.py` provides the model contract and deterministic routing helpers

This is intentionally minimal, explicit, and testable rather than hidden behind a large framework layer.
