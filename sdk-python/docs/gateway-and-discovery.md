# Gateway and discovery

`AgentGateway` is the invocation-scoped boundary between an agent and the
capabilities it is allowed to discover. `LocalAgentGateway` implements that
contract over an `AgentRegistry` and the shared `Runtime`.

The gateway is not a registry alias. The registry publishes facts; the gateway
applies caller context and policy to those facts.

## Obtaining a gateway

Applications inject the registry into the runtime and define the caller's
maximum capability set:

```python
from conducto import AgentRegistry, Runtime

registry = AgentRegistry()
registry.register(weather_agent)
runtime = Runtime(agent_registry=registry)

result = await runtime.invoke(
    travel_agent,
    "plan",
    {"city": "Toronto"},
    allowed_capabilities=frozenset({"weather.temperature"}),
)
```

Inside an active capability, call `require_run_context().gateway`. Agents do
not construct gateways or reach into the registry directly.

## Discovery

`discover(DiscoveryQuery(...))` evaluates one immutable registry snapshot.
Queries can constrain:

- exact agent identity
- one or more capability IDs
- tags
- agent version
- compatible input and output JSON Schemas
- whether approval-required capabilities may be returned
- result count

The gateway then filters lifecycle, health, allowed capabilities, required
authorization scopes, application gateway policy, remaining budget, and
cycle/depth constraints. Results are deterministically ordered and bounded by
count and serialized size.

Use `lookup(agent_id, capability_id)` for an exact target. Use `select(query)`
when exactly one eligible result is required. Selection returns explicit
`selected`, `no_match`, `denied`, `ambiguous`, or `failed` outcomes rather than
guessing among providers.

## Bindings

Discovery does not return a callable. It returns a descriptor plus an opaque
`CapabilityBinding` tied to:

- runtime identity
- agent and capability identity
- registry revision
- registration generation
- schema digest
- issuance and expiration time
- an integrity signature

Applications and models must treat the binding as opaque. It cannot be moved
to another runtime, edited, reconstructed from identifiers, or retained
indefinitely.

## Invocation revalidation

`gateway.invoke(binding, arguments)` checks the binding before reserving a
shared delegation call:

1. integrity, runtime identity, and expiry
2. registration generation and schema digest
3. current lifecycle and health
4. current authorization and gateway policy
5. deadline, cycle, and depth state
6. remaining calls, tokens, and cost

Only then does it dispatch through `Runtime.invoke()`. Registry changes can
therefore invalidate a previously discovered target safely, and no alternate
execution path bypasses argument validation, security, audit, or typed result
construction.

## Model-facing tools

`discover_tools()` projects eligible capabilities into bounded,
provider-neutral tool descriptors. `build_toolbox()` adds author-declared
`ToolboxPolicy` rules for required and optional capability families.

Each toolbox snapshot:

- exposes safe descriptions and JSON Schemas, not callable objects
- gives tools collision-resistant opaque IDs
- resolves IDs only within the originating snapshot
- reports missing required capabilities before a model call
- can return a partial toolbox when only optional capabilities are absent

The model can nominate a tool ID and arguments, but the SDK resolves that ID
through the snapshot and invokes its binding through the gateway. Model output
never becomes authority.

## Failures and limits

Gateway failures distinguish no match, denial, ambiguity, unsupported schema,
policy callback failure, invalid/stale binding, unavailable target, cycle,
depth, and result limits. JSON Schema compatibility intentionally supports a
controlled subset; unsupported features fail explicitly rather than being
treated as compatible.

Configure binding lifetime, preferred agents, result count, serialized size,
and application policy on `Runtime`. Keep policy callbacks deterministic,
side-effect free, and fast.
