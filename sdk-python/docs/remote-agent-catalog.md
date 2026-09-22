# Remote-agent catalog

`AgentCatalog` is the governed control plane for independently operated
agents. It owns stable logical and instance identity, ownership, deployment
type, Agent Card provenance, trust-policy references, capability indexing,
lease-based instance health, and lifecycle state. It never selects transport
or invokes a capability; `HybridAgentGateway` consumes catalog snapshots the
same way `LocalAgentGateway` consumes `AgentRegistry` snapshots while the
runtime-owned transport adapter owns remote dispatch.

Deployment automation uses the authenticated
[registration control plane](deployment-registration.md), not direct
`CatalogEntry` submission. It retrieves cards through transport policy and uses
catalog-owned atomic managed operations for idempotency, instance generations,
identity-bound leases, drain, removal, and revocation.

## Package ownership

Import catalog contracts from `conducto.core.catalog`. The package separates
immutable public records and typed failures, admission-source providers,
Agent Card/provenance validation, and the service that coordinates lifecycle,
health, leases, and coherent snapshots. Provider loading produces candidate
entries; it cannot bypass admission or publish partially validated state.
Lease and lifecycle transitions remain serialized with registry mutation.

The catalog package has no dependency on concrete model providers and never
constructs a transport client or executes capabilities. A catalog snapshot is
metadata, not authorization to invoke a target.

## Provider contract

Any admission source implements `CatalogProvider`:

```python
class CatalogProvider(Protocol):
    def list_entries(self) -> Sequence[CatalogEntry]: ...
```

Two reference providers are included:

- `InMemoryCatalogProvider` — an explicit, replaceable entry list, useful for
  tests and programmatic composition.
- `StaticFileCatalogProvider` — loads a JSON array of entries from a static
  configuration file.

A future hosted registry (for example a Microsoft Entra Agent Registry
adapter) implements the same protocol without changing `AgentCatalog` or
gateway code. See [gateway and discovery](./gateway-and-discovery.md) for how
the runtime combines catalog facts with local registry facts.

## Admission

Each `CatalogEntry` carries the logical agent identity, instance identity,
owner, deployment type, Agent Card URL and payload, optional provenance and
trust-policy reference, supported versions, transports, and a requested lease
duration:

```python
from conducto.core.catalog import (
    AgentCatalog,
    CatalogEntry,
    DeploymentType,
    InMemoryCatalogProvider,
)

entry = CatalogEntry(
    agent_id="org-a.finance.payout",
    instance_id="payout-7f3c",
    owner="org-a",
    deployment_type=DeploymentType.REMOTE_CONTAINER,
    agent_card_url="https://payout.org-a.example/.well-known/agent-card.json",
    agent_card=agent_card_payload,
    lease_seconds=60,
)

catalog = AgentCatalog()
catalog.register_instance(entry)
```

`register_instance` validates the Agent Card with the same A2A 1.0 profile
used for local registration, indexes immutable
`CatalogCapabilityDescriptor` entries, and rejects the entry with
`CatalogValidationError` rather than silently replacing a trusted
registration when:

- the Agent Card's declared name changes for an already-admitted logical
  agent identity, or
- a capability keeps the same published version but changes its input
  schema, or
- the entry requires signed provenance (because `trust_policy_ref` is set or
  the catalog was constructed with `require_provenance=True`) and either no
  signature is supplied or the configured `provenance_verifier` rejects it.

`refresh(provider)` admits every entry a provider currently offers and
returns the resulting snapshot; it never removes an instance that a
provider's current list omits; only lease expiration or an explicit lifecycle
change does that.

Catalog descriptors retain the admitted Agent Card skill identifier for audit
and transport correlation, while the gateway projects the stable Conducto
capability name back into the caller-facing `CapabilityDescriptor` contract.

## Instance health and lease expiration

Multiple instances of the same logical agent are supported without treating
endpoint URLs as identity. Each instance carries its own lease:

```python
catalog.renew_lease("org-a.finance.payout", "payout-7f3c", lease_seconds=60)
expired = catalog.expire_instances()  # [(agent_id, instance_id), ...]
```

`snapshot()` always filters by the catalog's injected clock, so a
lease-expired instance is excluded from discovery immediately, even before
`expire_instances()` runs. A crashed instance ages out on its own; healthy
instances of the same logical agent are never removed as a side effect.

The catalog accepts an injectable `clock: Callable[[], float]` so lease
renewal and expiration are deterministic under a fake clock in tests.

## Lifecycle transitions

```python
catalog.quarantine(agent_id)
catalog.disable(agent_id)
catalog.revoke(agent_id)
catalog.reactivate(agent_id)   # restore a quarantined or disabled agent
catalog.remove(agent_id)       # delete the registration entirely
```

`snapshot()` and `capability_providers()` only return agents in the `active`
lifecycle state that currently have at least one healthy, non-expired
instance. Quarantined, disabled, revoked, and removed agents are excluded
without deleting their audit history until `remove()` is called explicitly.

## Audit events

Catalog admission, lifecycle transitions, lease renewal, instance expiration,
and discovery emit attributable `conducto.catalog.*` events through the same
structured logging facade used elsewhere in the SDK. Agent Card payloads and
signatures are never included in these events.
