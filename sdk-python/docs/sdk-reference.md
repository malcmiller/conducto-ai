# SDK reference

This reference describes the public Python SDK APIs currently shipped by `sdk-python/src/conducto`.

## Package root exports

The package root exports the stable public API from `conducto`:

- `BaseAgent`
- A2A protocol constants and validators such as `A2A_PROTOCOL_VERSION`,
  `A2A_PYTHON_SDK_VERSION`, `parse_agent_card`, `parse_message`, and
  `parse_task`
- `OrchestratorAgent`
- `Runtime`, `RunContext`, and runtime configuration types
- `AgentRegistry`, `AgentGateway`, discovery queries, descriptors, and typed
  gateway outcomes
- `a2a_agent`
- `a2a_capability`
- `tool`
- `InvocationSuccess`, `InvocationFailure`, and related result types
- delegation and toolbox contracts such as `DelegationConfig`,
  `CapabilityUse`, and `ToolboxPolicy`
- provider registration, model-provider, and routing contracts

For consumers, the package root is the supported import surface.

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
- `replace_agent(agent)`
- `remove_agent(agent_or_name)`
- `clear_agents()`
- `get_agent_by_name(name)`
- `invoke(agent_id, capability_id, arguments, ...)`
- `invoke_capability(...)`
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
- `structured`
- `usage`
- `accepted`
- `request_id`

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
- [Orchestration and delegation](./orchestration-and-delegation.md)
- [Providers and models](./providers-and-models.md)
- [Runtime and invocation](./runtime-and-invocation.md)
