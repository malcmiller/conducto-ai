# SDK reference

This reference describes the pre-v1 Python APIs implemented in `sdk-python/src/conducto`.

## Package root exports

The package root is a small application facade: `BaseAgent`, `a2a_agent`,
`a2a_capability`, `tool`, `AgentRegistry`, `OrchestratorAgent`, `Runtime`,
`RunContext`, `AgentModelConfig`, `ModelReference`, `ModelRequirement`,
`RunConfig`, `RuntimeConfig`, `get_run_context`, and `require_run_context`.

Specialized contracts have explicit owning packages. `conducto.core` is a
namespace, not a second umbrella export; `core.runtime` exports `Runtime`,
not unrelated model, context, provider, or error types.

| Import path | Public responsibility |
|---|---|
| `conducto.core.provider` | Provider protocol, messages, schema requests, tool calls, results, failures |
| `conducto.core.provider_registry` | Factories, client configuration, model registration, ownership, cleanup reports, snapshots |
| `conducto.core.model_config` | Model identities and agent/run/runtime configuration |
| `conducto.core.model_gateway` | Invocation-scoped typed model access and results |
| `conducto.core.model_resolution` | Resolution and credential-free resolved identities |
| `conducto.core.run_context` | Context, budgets, cancellation, policy, provenance |
| `conducto.core.runtime_errors` | Typed configuration, resolution, and lifecycle failures |
| `conducto.core.invocation_results` | Invocation outcomes and routing failures |
| `conducto.core.gateway` | Policy-filtered gateway contract and local implementation |
| `conducto.core.gateway_models` | Discovery queries, descriptors, opaque bindings, outcomes |
| `conducto.core.gateway_tools` | Bounded model-facing toolbox projection |
| `conducto.core.catalog` | Governed remote catalog, providers, lifecycle and immutable records |
| `conducto.core.delegation` | Bounded model/tool loop, configuration, outcomes, fallback |
| `conducto.core.a2a_profile` | Pinned A2A constants and protocol validation |
| `conducto.core.agent_card` | Agent Card specification version |
| `conducto.core.logging`, `telemetry`, `otel_logs` | Safe diagnostics and optional telemetry integration |
| `conducto.providers` | Optional provider adapters and factories |
| `conducto.testing` | Deterministic `FakeModel` and provider conformance helpers |

These are intentional pre-v1 import paths, not aliases for historical module
locations. Wire fixtures remain versioned independently of Python import paths.

## Pre-v1 API cleanup

Python source compatibility is deliberately not preserved for the earlier
prototype API:

| Removed API or usage | Preferred API |
|---|---|
| Large `conducto` or `conducto.core` umbrella imports | Application facade plus owning domain imports above |
| Importing model/context/provider types through `core.runtime` | Import from the owning model, context, or provider package |
| `ProviderRegistry.register(...)` | `register_client(...)` or factory-backed `register_provider(...)` |
| Raw `model_provider` / `model_config` on agents or routing calls | Register clients on the runtime and select a `ModelReference` |
| `OrchestratorAgent.invoke_capability(...)` | `invoke(...)` |
| `OrchestratorAgent.agents` / `get_routing_context()` | `registered_agents` / `get_routing_prompt_context()` |
| `replace_agent(agent)` / `discover_agents()` | `register_agent(agent, replace=True)` / `registered_agents` |
| `routing_metadata` / `routing_prompt_context` properties | `get_routing_metadata()` / `get_routing_prompt_context()` |
| `authorization_context=` invocation argument | `authorization=` |
| Synchronous `close()` on first-party async provider adapters | `await provider.aclose()` or runtime-managed shutdown |
| `FakeModel` imported as a production provider contract | `conducto.testing.FakeModel` |

Tool definitions use `ProviderToolDefinition`, not untyped legacy mappings.
Scripted tool decisions use `ProviderResult(tool_calls=(ProviderToolCallRequest(...),))`;
terminal scripted responses use `ProviderResult(structured=...)`, not untyped
mappings or synthetic `type`/`response` wrappers. Usage and acceptance state
are explicit fields of each scripted result. A single `FakeModel({...})`
remains a literal terminal-content convenience with `accepted=True`; `type`
and `kind` fields are ordinary content, never instructions to unwrap a result.
Use `build_terminal_output_request()` to construct a delegation terminal
schema; the misleading `build_model_decision_schema()` name is removed.
This removes Python coercion paths without changing native tool channels,
Agent Cards, invocation envelopes, or versioned golden wire fixtures.

MCP export types are published from `conducto.mcp` instead of the package root
so that importing `conducto` never requires the optional `mcp` extra:
`McpExportPolicy`, `McpExportRule`, `McpCapabilityQuery`, `McpToolExporter`,
`McpToolDefinition`, `McpToolOutcome`, `McpStdioServer`, and the typed
`McpExportError` hierarchy. See [MCP tool export](./mcp-export.md).

## `@a2a_agent`

Declared on a class to mark it as an A2A-capable agent.

### Signature

```python
@a2a_agent(
    cls=None,
    *,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    default_model: str | None = None,
    model_required: bool = False,
    tags: tuple[str, ...] = (),
)
```

### Behavior

- uses the class name as the default agent name
- validates that `version`, `name`, and `description` are not blank
- stores `AgentMetadata` on the class
- defaults `description` to the class docstring

### Example

```python
from conducto import BaseAgent, a2a_agent

@a2a_agent(name="InventoryAgent", version="1.2.0", description="Handles inventory queries.")
class InventoryAgent(BaseAgent):
    pass
```

## `@a2a_capability`

Declared on a method to expose it as an A2A capability.

### Purpose

- reflects method metadata
- exposes a stable capability name
- captures the published description
- registers the method as part of the agent's public contract

### Example

```python
from conducto import BaseAgent, a2a_capability

class InventoryAgent(BaseAgent):
    @a2a_capability(name="lookup", description="Looks up inventory for a SKU.")
    def lookup_inventory(self, sku: str) -> dict:
        return {"sku": sku, "count": 12}
```

## `@tool`

Declared on a method to mark it as an internal Conducto tool, separate from the public A2A capability layer.

### Purpose

- keep helper methods out of the public capability set
- preserve the original method semantics while adding internal registry metadata
- allow the same method to carry both capability and tool metadata when needed

## `BaseAgent`

`BaseAgent` is the main class consumers inherit from.

### Responsibilities

- resolve agent metadata declared with `@a2a_agent`
- discover decorated methods
- create Pydantic parameter models for each capability
- generate A2A Agent Card JSON
- expose capability and tool registries as deterministic mappings

### Important members

- `agent_metadata` — resolved `AgentMetadata`
- `registered_methods` — tuple of all reflected methods
- `capabilities` — mapping of capability name to `RegisteredMethod`
- `tools` — mapping of helper-tool name to `RegisteredMethod`
- `get_agent_card(url, ...)`
- `get_agent_card_json(url, ...)`

### Constraint

The default card generation requires:

- a valid absolute HTTP or HTTPS URL
- a non-empty `name` and `version`
- a non-empty `description` on the agent class or explicit metadata
- JSON-RPC A2A protocol version `1.0`
- `text/plain` input and output modes

Conducto rejects A2A 0.3 Agent Cards, unknown required extensions, unsupported
JSON-RPC methods, unsupported media types, and terminal task-state transitions
explicitly. The pinned wire profile and update procedure are documented in the repository's
[A2A 1.0 profile](../../docs/a2a-1-profile.md).

## `OrchestratorAgent`

The orchestrator is the application-facing facade for a local agent registry,
direct invocation, and optional model-assisted top-level routing.

### Primary operations

- `register_agent(agent, replace=False)`
- `remove_agent(agent_or_name)`
- `clear_agents()`
- `get_agent_by_name(name)`
- `invoke(agent_id, capability_id, arguments, ...)`
- `route(...)`
- `get_routing_metadata()`
- `get_routing_prompt_context()`

### Invocation result types

The orchestrator returns strongly typed result envelopes:

- `InvocationSuccess`
- `InvocationValidationFailure`
- `InvocationTargetNotFound`
- `InvocationTimeout`
- `InvocationCancelled`
- `InvocationFailure`
- authorization, approval, and audit failures
- binding, lifecycle, schema, budget, and delegation failures

These ensure callers can handle protocol-level outcomes without relying on thrown exceptions for ordinary workflow states.

## Provider contracts

The relevant provider-building types are:

### `ChatMessage`

```python
class ChatMessage(BaseModel):
    role: str
    content: str
```

### `GenerationOptions`

Configures the provider call:

- `model`
- `temperature`
- `max_tokens`
- `timeout`
- `retries`

### `StructuredOutputRequest`

Captures a named schema used for structured provider responses.

### `ProviderResult`

The provider response object includes:

- `content`
- `structured` for terminal schema-constrained output
- `tool_calls` for provider-native tool decisions
- `usage`
- `accepted`
- `request_id`

### `ProviderToolDefinition` and `ProviderToolCall`

`ProviderToolDefinition` describes one tool from the exact toolbox snapshot
supplied to a provider turn. `ProviderToolCall` is the normalized single tool
decision resolved back to that snapshot, with an opaque `call_id`, resolved
`tool_id`, and validated argument mapping.

### `ModelProvider`

A protocol representing the required provider interface:

```python
class ModelProvider(Protocol):
    capabilities: ProviderCapabilities

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        effective_deadline: float | None = None,
    ) -> ProviderResult: ...
```

### `RoutingSelection`

The orchestrator expects a structured selection with:

```python
class RoutingSelection(BaseModel):
    agent_id: str
    capability_id: str
    arguments: dict[str, Any] = {}
```

## Return serialization rules

Capability return values are converted to JSON-compatible values through
`serialization.serialize_result()`. Supported values include:

- `None`, strings, booleans, ints
- finite floats
- `Enum`
- Pydantic models
- dataclasses
- mappings with string keys
- lists, tuples, sets, and other sequences

Unsupported values raise `UnsupportedReturnValueError` and become an `InvocationFailure` result rather than crashing the orchestrator.

## Component guides

The reference lists public types. The following guides explain how the
components cooperate and where ownership boundaries sit:

- [Agents and registration](./agents-and-registration.md)
- [Gateway and discovery](./gateway-and-discovery.md)
- [Remote-agent catalog](./remote-agent-catalog.md)
- [Orchestration and delegation](./orchestration-and-delegation.md)
- [Providers and models](./providers-and-models.md)
- [Runtime and invocation](./runtime-and-invocation.md)
- [MCP tool export](./mcp-export.md)
