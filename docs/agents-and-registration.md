# Agents and registration

Conducto has two registration stages:

1. **Class reflection** turns decorated Python methods into immutable metadata
   and validation models on one `BaseAgent` instance.
2. **Runtime registration** publishes agent and capability descriptors through
   an application-owned `AgentRegistry`.

Keeping these stages separate lets an agent be constructed and tested without
global discovery infrastructure.

## Declaring an agent

`@a2a_agent` declares published identity and defaults. `@a2a_capability`
declares a callable public capability, while `@tool` marks an internal export.

```python
from conducto import BaseAgent, a2a_agent, a2a_capability


@a2a_agent(
    name="InventoryAgent",
    version="1.0.0",
    description="Looks up inventory.",
)
class InventoryAgent(BaseAgent):
    @a2a_capability(
        name="inventory.lookup",
        description="Returns stock for one SKU.",
        tags=("inventory", "read"),
    )
    def lookup(self, sku: str) -> dict[str, object]:
        return {"sku": sku, "quantity": 12}
```

Decorators store metadata; they do not register a global singleton or alter
transport state.

## Reflection

During `BaseAgent.__init__()`, `registration.py`:

1. resolves inherited class and method metadata
2. walks decorated attributes in deterministic order
3. rejects invalid or duplicate export definitions
4. binds each method to a `RegisteredMethod`
5. asks `parameter_schema.py` to build the Pydantic argument model and JSON
   Schema
6. separates public capabilities from internal tools

`BaseAgent.capabilities`, `tools`, and `registered_methods` return copies or
immutable tuples rather than mutable internal dictionaries.

## Agent Cards

`BaseAgent.get_agent_card()` converts reflected metadata into the pinned A2A
Agent Card profile. `agent_card.py` owns stable skill IDs, URL and transport
validation, modes, security schemes, requirements, and canonical JSON
serialization.

The card is a publication format, not the live execution registry. The
language-neutral profile and compatibility policy are documented in the
repository [A2A 1.0 profile](./a2a-1-profile.md).

## The local `AgentRegistry`

Applications register agent instances explicitly:

```python
from conducto import AgentRegistry

registry = AgentRegistry()
registry.register(InventoryAgent())
snapshot = registry.snapshot()
```

The registry owns:

- agent identity and replacement generations
- one-to-many capability indexes
- active, draining, disabled, and removed lifecycle state
- health state
- monotonic revisions
- immutable `RegistrySnapshot` publication

Registration validates the agent's card before publishing it. Publication is
atomic under the registry lock, so discovery never sees a partially indexed
agent.

## Replacement and lifecycle

`register(agent, replace=True)` publishes a new generation. Existing bindings
retain their original generation and will be rejected as stale when invoked;
later discovery sees the replacement.

Use `set_lifecycle()` to drain or disable a target and `set_health()` to
publish health changes. Removal creates a tombstone distinction for lifecycle
checks. These mutations affect future discovery and are rechecked when a
binding is invoked.

`OrchestratorAgent.register_agent()` uses the same registry but rejects
capability collisions for its explicit single-provider routing behavior.
Direct `AgentRegistry.register()` permits multiple providers for one
capability by default so the gateway can resolve or report ambiguity.

## What not to put in registration

Agent registration must not:

- construct model clients or resolve credentials
- call a model
- select a caller-specific target
- execute capability business logic
- expose mutable agents, callables, secrets, or endpoints in public snapshots

Provider factories and model references belong to
[provider registration](./provider-registration.md). Caller-specific
selection belongs to the [gateway](./gateway-and-discovery.md).
