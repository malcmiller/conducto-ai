# Common change recipes

## Add or change an agent capability

Usually inspect:

- `conducto.core.agent`, decorators, and reflection;
- Agent Card generation;
- argument validation and return serialization;
- local and remote invocation tests.

Preserve:

- one declaration path;
- deterministic schemas;
- runtime-only public execution; and
- Agent Card/profile consistency.

Update golden Agent Card fixtures only for an intentional public contract
change.

## Add an invocation result

Usually update:

1. `conducto.core.invocation_results`;
2. every producer in runtime, gateway, delegation, and security paths;
3. A2A result mapping;
4. MCP result mapping;
5. golden mapping fixtures;
6. SDK and error reference documentation.

Search exhaustively for `InvocationResult` and all `isinstance` mappings.
Unknown outcomes must remain sanitized rather than silently appearing as
success.

## Add a model provider

Implement the provider protocols and normalize behavior into shared contracts.
Do not expose vendor clients to agents.

Cover:

- structured terminal output;
- supported schema features;
- native tool calls when advertised;
- timeout and cancellation;
- bounded response sizes;
- lifecycle and ownership;
- readiness;
- sanitized failures;
- optional dependency behavior; and
- independently installed package extra.

See [provider registration](../provider-registration.md).

## Add an A2A operation

Start with the pinned [A2A profile](../a2a-1-profile.md). Do not implement a
method in the server without also advertising it accurately.

Review:

- official SDK support;
- protocol constants and validation;
- server dispatch;
- client operation;
- task transitions;
- runtime/result mapping;
- limits and security;
- golden fixtures; and
- compatibility notes.

## Add a transport adapter

The adapter may parse, authenticate, map, and serialize. It may not directly
execute capability methods or create alternate guardrails.

Require:

- a typed boundary into `Runtime`;
- explicit lifecycle and resource ownership;
- bounded input and output;
- cancellation mapping;
- sanitized typed failures;
- optional imports; and
- in-process deterministic tests.

## Add a security guardrail

Place transport-independent policy in `conducto.security` or the canonical
runtime pipeline. Transport code should only obtain and normalize trusted
facts.

Test:

- allow and deny;
- missing and malformed facts;
- required audit failure;
- nested authority attenuation;
- approval resume binding;
- concurrent isolation; and
- redaction.

## Add an optional package extra

Update:

- `pyproject.toml`;
- `uv.lock`;
- lazy import and actionable missing-dependency error;
- release workflow extra matrix;
- package metadata assertion;
- independent import and smoke test;
- installation documentation.

Core imports must continue to work without the extra.
