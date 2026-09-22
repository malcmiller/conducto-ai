# Python SDK architecture

The Python SDK separates declaration, discovery, model access, and execution
so each concern can evolve without turning `Runtime` or `OrchestratorAgent`
into a monolith.

## Component map

```mermaid
flowchart TD
    A[Agent class and decorators] --> R[Method reflection and schemas]
    R --> C[Agent Card and capability descriptors]
    C --> AR[AgentRegistry]
    AR --> G[LocalAgentGateway]
    O[OrchestratorAgent] --> AR
    O --> RT[Runtime]
    G --> RT
    RT --> I[Invocation pipeline]
    RT --> MR[ModelResolver]
    MR --> PR[ProviderRegistry]
    RT --> MG[ModelGateway]
    MG --> P[ModelProvider]
    D[Delegation loop] --> G
    D --> MG
    I --> S[Security and audit]
    I --> X[Typed invocation result]
```

## Responsibility boundaries

| Component | Owns | Does not own |
|---|---|---|
| `BaseAgent` and decorators | Agent metadata, capability declarations, agent defaults | Global registration, provider clients, transport |
| `registration.py` | Reflection of decorated methods and registration metadata | Runtime agent discovery |
| `AgentRegistry` | Local agent instances, capability indexes, lifecycle, health, immutable snapshots | Policy decisions, model calls, capability execution |
| `AgentGateway` | Caller-aware discovery, deterministic selection, opaque bindings, invocation revalidation | Mutable registration state, provider construction |
| `OrchestratorAgent` | Application-facing local registry facade and optional model-based top-level routing | Nested delegation state or provider lifecycle |
| `Runtime` | Run context, model resolution, gateway construction, security, invocation, provenance | Agent metadata declaration |
| `ProviderRegistry` | Provider factories, clients, ownership metadata, model-reference bindings | Selecting a model for a particular call |
| `ModelResolver` | Call/run/agent/runtime precedence, policy, provider capability checks | Provider execution |
| `ModelGateway` | Invocation-scoped provider calls, deadlines, cancellation, typed output, usage | Long-lived provider ownership |
| `run_delegation` | Bounded model/tool loop and terminal outcome | Discovery authority or direct registry access |
| `conducto.mcp` | MCP export policy, deterministic tool naming, schema projection, result mapping, stdio lifecycle | Capability declaration, validation, authorization, MCP framing |
| `conducto.core.telemetry` | Optional Conducto span names, safe attributes, W3C trace-context helpers, and test tracing helper | Application tracer-provider/exporter lifecycle or auto-instrumentation |
| `conducto.core.otel_logs` | Optional bridge attaching an application-owned `LoggerProvider` to `conducto` events, bounded/redacted attribute mapping, and test in-memory helper | Global provider/handler/exporter installation, security audit delivery |

The similarly named registries solve different problems:
`AgentRegistry` indexes callable agents and capabilities, while
`ProviderRegistry` binds credential-free model references to model clients.

## Primary flows

### Direct capability invocation

1. An application registers an agent or already holds its instance.
2. `Runtime.invoke()` creates or attenuates a `RunContext`.
3. The invocation pipeline resolves the decorated capability.
4. Security checks run before business logic.
5. Pydantic validates arguments.
6. The capability executes under deadline and cancellation controls.
7. The return value is serialized and wrapped in a typed
   `InvocationResult`.

### Gateway invocation

1. `AgentRegistry.snapshot()` publishes immutable descriptors.
2. `LocalAgentGateway.discover()` filters candidates by capability, tags,
   version, schema, lifecycle, health, caller authority, and policy.
3. Discovery returns an opaque, runtime-bound `CapabilityBinding`.
4. `LocalAgentGateway.invoke()` revalidates the binding, lifecycle, schema,
   authority, path, and shared budget.
5. Dispatch enters the same `Runtime.invoke()` path as a direct call.

### Model-assisted orchestration

1. `ModelResolver` resolves a credential-free model reference.
2. `ModelGateway` sends a structured request through a registered
   `ModelProvider`.
3. `OrchestratorAgent.route()` validates the returned top-level target, or
   `run_delegation()` validates one terminal response or one tool call.
4. Any selected capability still executes through the gateway/runtime
   boundaries rather than directly from model output.

### MCP tool export

1. An application configures `McpToolExporter` with a `Runtime`, agents or a
   registry, and a default-deny `McpExportPolicy`.
2. The exporter projects allowlisted capability descriptors into immutable MCP
   tool definitions at construction time, rejecting collisions, bound
   violations, and unsupported schemas.
3. `McpStdioServer` publishes those definitions through the official MCP Python
   SDK for a configured stdio principal.
4. Each `tools/call` enters the same `Runtime.invoke()` path as a direct or
   gateway call, and the typed result is mapped to an MCP tool result.

See [MCP tool export](./mcp-export.md).

### Optional tracing

OpenTelemetry support is an optional package extra. Importing `conducto` does
not configure global tracing, exporters, samplers, propagators, or framework
auto-instrumentation. Applications own those choices; Conducto creates
best-effort explicit spans only at SDK-owned boundaries and keeps invocation,
security, audit, deadline, retry, and cancellation behavior unchanged when
tracing is absent. W3C `traceparent`/`tracestate` context is propagated at
supported remote boundaries, while baggage is not forwarded by default.

### Optional OpenTelemetry Logs export

`OpenTelemetryLogBridge` (`conducto.core.otel_logs`) attaches an
application-supplied OpenTelemetry `LoggerProvider` to the `conducto` logger,
bridging existing versioned events (never a second event taxonomy) to OTLP
log records with trace/span correlation from the active context. The
application owns the provider, exporter, resource, and shutdown; the bridge
owns only the handler it creates, applies bounded/redacted attribute mapping
before any record reaches a processor or exporter, and never blocks
invocation on exporter failures. It is entirely independent from the Story
2.3 security audit sink: accepting a log record here never satisfies
mandatory audit delivery. See [Python logging](./logging.md).

## State and concurrency

- `Runtime` and both registries are application-owned objects; importing the
  package creates no clients or global runtime.
- `RunContext` is invocation-scoped and propagated with `contextvars`, so
  concurrent asyncio tasks do not share active state.
- Registry snapshots and toolbox snapshots are immutable decision-boundary
  views.
- A registration mutation affects later discovery. It does not rewrite an
  already published snapshot or bypass binding revalidation.
- Deadlines, cancellation, authority, and delegation budgets are inherited or
  attenuated for child calls, never expanded.

## Determinism and failure semantics

The SDK sorts published metadata, canonicalizes serialized values, uses stable
capability identifiers, and keeps golden fixtures for public wire contracts.
Expected failures are typed: validation, authorization, approval, target,
binding, lifecycle, schema, budget, timeout, cancellation, provider, routing,
and delegation outcomes remain distinguishable.

For the details behind each layer, continue with
[agents and registration](./agents-and-registration.md),
[gateway and discovery](./gateway-and-discovery.md),
[orchestration and delegation](./orchestration-and-delegation.md),
[providers and models](./providers-and-models.md), and
[runtime and invocation](./runtime-and-invocation.md).
