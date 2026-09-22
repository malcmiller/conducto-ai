# ADR 0001: Use one canonical capability runtime

**Status:** Accepted

## Context

Capabilities can be invoked locally, through a gateway, over A2A, or through
MCP. Separate execution implementations would drift in validation, security,
timeouts, cancellation, auditing, serialization, and failures.

## Decision

Every public capability invocation enters the canonical `Runtime` pipeline.
Transport and protocol adapters translate requests and results but never call
capability methods directly.

## Consequences

- Local behavior is the reference for every transport.
- Adapters require typed seams into the runtime.
- Runtime changes must be tested across local and projected protocols.
- Protocol-specific failures map from `InvocationResult` rather than raw
  exceptions.

## Alternatives considered

- Let each adapter invoke reflected methods directly. Rejected because it
  creates parallel guardrail and serialization paths.
- Put protocol logic inside `Runtime`. Rejected because the runtime must remain
  transport-independent.

## Related code and evidence

- `conducto.core.runtime*`
- `conducto.a2a.A2ARuntimeHandler`
- `conducto.mcp.McpToolExporter`
- Runtime, gateway, A2A runtime, MCP, and golden mapping tests
