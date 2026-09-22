# Gateway and discovery

`AgentGateway` is the invocation-scoped boundary between an agent and the
capabilities it is allowed to discover. `LocalAgentGateway` implements that
contract over an `AgentRegistry` and the shared `Runtime`.
`HybridAgentGateway` preserves the same public discovery, selection, binding,
tool, and typed-invocation contracts while composing local registry facts with
governed remote facts from `AgentCatalog`.

The gateway is not a registry alias. The registry publishes facts; the gateway
applies caller context and policy to those facts.

## Module ownership

Import the gateway protocol, policy, and local implementation from
`conducto.core.gateway`. Discovery records remain in
`conducto.core.gateway_models`; toolbox declarations and projection remain in
`conducto.core.gateway_tools`.

Inside the gateway package, schema compatibility, safe tool projection,
discovery filtering, and binding issuance/revalidation are separate private
collaborators. The public gateway coordinates them against the same registry
snapshot and runtime context. Moving these helpers does not weaken
authorization or add a shortcut around runtime invocation.

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

When a runtime is configured with both `agent_catalog=` and
`gateway_transport=`, `RunContext.gateway` returns `HybridAgentGateway`
automatically. The calling code does not branch on whether a selected
capability is local, containerized, remote A2A, or Foundry-hosted.

```python
from conducto import AgentRegistry, Runtime
from conducto.core.catalog import AgentCatalog
from conducto.core.gateway import GatewaySelectionMode, GatewaySelectionPolicy

runtime = Runtime(
    agent_registry=registry,
    agent_catalog=AgentCatalog(),
    gateway_transport=transport_adapter,
    gateway_selection_policy=GatewaySelectionPolicy(
        mode=GatewaySelectionMode.LOCAL_PREFERRED,
        allow_pre_acceptance_failover=False,
    ),
)
```

## Discovery

`discover(DiscoveryQuery(...))` evaluates one immutable local/remote discovery
snapshot pair and merges the eligible candidates into one bounded result.
Queries can constrain:

- exact agent identity
- one or more capability IDs
- tags
- agent version
- compatible input and output JSON Schemas
- whether approval-required capabilities may be returned
- result count

The gateway filters lifecycle, health, allowed capabilities, required
authorization scopes, application gateway policy, remaining budget, and
cycle/depth constraints before any relevance or preference ranking. Hybrid
discovery also excludes quarantined, revoked, stale, transport-incompatible,
deployment-incompatible, and no-longer-leased remote instances before tools are
projected.

Results are deterministically ordered and bounded by count and serialized size.

Use `lookup(agent_id, capability_id)` for an exact target. Use `select(query)`
when exactly one eligible result is required. Selection returns explicit
`selected`, `no_match`, `denied`, `ambiguous`, or `failed` outcomes rather than
guessing among providers.

For mixed local/remote discovery, `GatewaySelectionPolicy` adds deterministic
selection hooks after authorization filtering:

- `ambiguous` — surface explicit ambiguity
- `local_preferred` — choose local before remote
- `remote_preferred` — choose remote before local
- `round_robin` — stable runtime-owned rotation for equally eligible matches
- `sticky_task` / `sticky_session` — stable hashing for task- or
  correlation-scoped affinity

These hooks also govern remote instance ordering. A transport adapter may retry
another healthy instance only when `allow_pre_acceptance_failover=True` and the
adapter can prove the failed instance did not accept the request.

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

Hybrid bindings still expose only the public `CapabilityBinding`. Runtime-owned
binding state keeps the immutable local registration snapshot or remote catalog
snapshot private. Mutable remote instance endpoints are re-resolved only at
dispatch time, so catalog changes affect later discovery without silently
rewriting an already-issued binding.

## Invocation revalidation

`gateway.invoke(binding, arguments)` checks the binding before reserving a
shared delegation call:

1. integrity, runtime identity, and expiry
2. registration generation and schema digest
3. current lifecycle and health
4. current authorization and gateway policy
5. deadline, cycle, and depth state
6. remaining calls, tokens, and cost

Only then does it dispatch through `Runtime.invoke()`. Local dispatch and remote
A2A dispatch share the same validation, approval, authorization, timeout,
cancellation, budget, and typed-result contract.

Remote invocation keeps the binding stable while re-checking the latest catalog
generation, schema digest, lifecycle, and eligible instances. A successful
dispatch preserves correlation IDs, parent/child run hierarchy, authorization
identity, attenuated capability scope, cancellation, deadlines, model
provenance, and shared delegation budgets across the boundary.

Registry or catalog changes can therefore invalidate a previously discovered
target safely, but an in-flight accepted task is never silently redirected to a
different instance.

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
policy callback failure, invalid/stale binding, unavailable target, no eligible
instance, cycle, depth, budget, and result limits. Hybrid transport failures
identify the selected runtime boundary and never become success-shaped
capability results. JSON Schema compatibility intentionally supports a
controlled subset; unsupported features fail explicitly rather than being
treated as compatible.

Configure binding lifetime, preferred agents, result count, serialized size,
and application policy on `Runtime`. Keep policy callbacks deterministic,
side effect free, and fast.

The optional external `agentgateway` network data plane remains an adapter seam,
not a replacement for Conducto gateway semantics. TLS termination, JWT/OAuth
enforcement, routing, rate limits, or OpenTelemetry collection may sit behind a
future `gateway_transport` implementation, but semantic capability filtering,
opaque bindings, authority attenuation, and typed invocation outcomes stay in
the Conducto gateway/runtime boundary.
