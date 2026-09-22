# Deployment registration control plane

`conducto.registration` admits independently deployed remote instances to an
already-running `AgentCatalog`. This is **not an A2A capability**: registration
uses dedicated authenticated JSON operations; A2A remains the invocation data
plane. Existing catalog consumers see new capabilities in subsequent snapshots
without a restart. Cached snapshots remain immutable and do not update in place.

Install `conducto-ai[registration]` for HTTPX discovery and the Python client.
Importing `conducto` does not import these adapters. `RegistrationASGI` implements
ASGI directly; an ASGI server is application-owned, not a core dependency.

## Server composition and trust

Create `RegistrationService` with the existing catalog, a `TokenValidator` (for
example `JWTBearerTokenValidator` and an application-owned key resolver), a
dedicated `TrustPolicy`, `RegistrationGrant` values, a Story 4 `DiscoveryPolicy`,
and a required `AuditEmitter`. Wrap it in
`conducto.registration.asgi.RegistrationASGI`.

```python
from conducto.registration import RegistrationGrant, RegistrationService
from conducto.registration.asgi import RegistrationASGI
from conducto.transport import DiscoveryPolicy

grant = RegistrationGrant(
    issuer="https://identity.example",
    subject_id="invoice-deployer",
    owner="payments",
    environment="production",
    agent_id="payments.invoice",
    card_name="InvoiceAgent",
    agent_card_url="https://invoice.example/.well-known/agent-card.json",
    endpoint_url="https://invoice.example/a2a",
    max_lease_seconds=120,
)
service = RegistrationService(
    catalog=catalog, validator=validator, trust_policy=registration_trust,
    grants=(grant,), discovery_policy=DiscoveryPolicy(), audit=audit,
)
app = RegistrationASGI(service)
```

The host supplies the named objects above. Use a registration-specific audience;
an A2A invocation token is not automatically a deployment token. Each operation
also requires its `registration:<operation>` scope. Grants bind the verified
issuer and subject to the exact owner, environment, logical agent, card name,
card URL, endpoint, deployment types, and maximum lease. No wildcard ownership
or implicit administrator rights exist. Give administrators a separate grant
with only `operations=frozenset({"revoke"})` and the corresponding token scope.

Use HTTPS and an authenticated workload/deployment identity in production.
The TLS server owns certificate validation; this reference service validates
OAuth bearer tokens and rejects certificate-bearing `TrustPolicy` objects rather
than silently skipping mTLS validation. Credential provisioning is not part of
this API. Tokens must be reacquired before expiration; requested leases may not
outlive the token's remaining authority.

Card retrieval uses `discover_agent`, including its redirect prohibition,
network policy, size limit, timeout, A2A 1.0 JSON-RPC validation, and same-authority
endpoint rule. The retrieved name and exact advertised endpoint must also match
the grant. The caller cannot submit capability schemas or an Agent Card body.
Card and registration responses must use identity content encoding; compressed
responses are rejected before decoding so expansion cannot bypass byte bounds.
Card and endpoint URLs must be printable ASCII without whitespace, credentials,
queries, or fragments; explicit ports must be integers from 1 through 65535.
Discovery requests do not inherit the supplied HTTP client's authentication,
default headers, cookies, or query parameters. Keep control-plane credentials out
of discovery transports and request hooks as well. Use network egress controls
appropriate to the deployment in addition to discovery policy.

`provenance` is an opaque build/attestation reference, **not** proof of signing.
For cryptographic provenance, set the grant's `trust_policy_ref`, configure the
catalog's `provenance_verifier`, and supply the confidential detached `signature`.
Verification remains catalog-owned. The catalog binds the admitted instance to
its authenticated subject/issuer, owner, environment and deployment metadata.

## Startup and deployment automation

1. Start the agent's A2A listener with its Agent Card available but readiness
   false. Do not accept ordinary invocations before admission.
2. Submit a registration request using a new instance identity for every process
   incarnation. Keep the logical agent identity stable across deployments.
3. Require `result.is_ready_at(time.time())`, save its generation and private lease handle, then
   expose readiness. Do not log or persist a raw transport response in CI output.
4. Renew before expiry using one serialized renewal loop. Refresh credentials
   independently. Each new operation gets a new idempotency key and the last
   confirmed generation.
5. Clear local readiness and drain the catalog instance before waiting for
   bounded in-flight work; deregister after that work finishes.

```python
import time
import httpx
from conducto.registration.client import RegistrationClient
from examples.registration_hooks import startup_request, renewal_request

# acquire_deployment_token is an async application-owned credential callback.
async with httpx.AsyncClient() as http:
    client = RegistrationClient(
        endpoint="https://registration.example",
        http_client=http,
        token_provider=acquire_deployment_token,
        max_attempts=2,
    )
    request = startup_request(
        instance_id="invoice-pod-unique-incarnation",
        card_url="https://invoice.example/.well-known/agent-card.json",
    )
    admission = await client.send(request)
    if not admission.is_ready_at(time.time()):
        raise RuntimeError(admission.code.value)  # safe reason, not a wire body
    current = admission
    # Set readiness here. The host schedules this before the current lease expires.
    renewal = renewal_request(request, admission, current)
    current = await client.send(renewal)
    # If renewal fails, clear readiness and reconcile; never assume authority extended.
```

`examples/registration_hooks.py` is repository example code, not an installed
package module. It also supplies drain/removal helpers. Production hosts own the
readiness flag, scheduling, local request rejection, and bounded in-flight work.
Do not run competing renewal loops for one instance.

## Wire contract and concurrency

POST JSON to `/registration/v1/{register|renew|drain|deregister|status|revoke}`.
The body `operation` must match the path. Requests carry `protocol_version="1"`,
owner, environment, logical/instance IDs, idempotency key, epoch `issued_at`,
correlation ID, and expected generation. Registration additionally carries card
URL, deployment ID/type, provenance and lease seconds. Renewal carries lease
seconds; renew/drain/deregister/status require the private lease handle.
Revocation requires administrator authority instead of the workload's handle.

The in-memory catalog serializes identity, receipts, tombstones and instance
mutations under its existing lock. Instance generations are separate from the
logical agent's metadata generation. Create uses zero; every effective managed
mutation advances the instance generation. Status does not extend authority.
Status accepts an unknown/stale expected generation and returns the current one,
so a lost renewal response can be reconciled without guessing a new write version.
Drain affects only one instance; healthy siblings remain discoverable. Removal
and revocation are terminal for that instance identity.

An exact retry with the same key returns the same effective receipt, including
the original private admission grant, without renewing the lease. Conflicting
reuse returns `idempotency_conflict`. A receipt superseded by a later generation,
expiry, removal or administrative lifecycle change cannot claim fresh admission.
Generation races return explicit non-success; clients do not guess or increment
generations. Tombstones prevent old requests from resurrecting removed instances.
Replay-window checks reject requests older than 300 seconds by default or dated
in the future. Do not change timestamps when retrying an ambiguous request.

The catalog and its bounded idempotency journal are process-memory reference
implementations, **not durable or multi-process storage**. Multiple frontend
objects can safely share one catalog. Do not run independent ASGI workers and
claim one coherent catalog. Restarting the catalog loses both registrations and
receipts; restore neither in isolation. Capacity exhaustion fails closed rather
than evicting replay protection. `AgentCatalog(managed_capacity=10_000)` bounds
identities and ordinary receipts; one additional terminal receipt per identity is
reserved so a full journal cannot prevent removal or administrator revocation.
Persistent storage and external catalog adapters
are out of scope.

## Failure, readiness, and revocation

The service performs no automatic retries. Card retrieval is bounded by the
discovery timeout. The ASGI adapter additionally bounds request duration and body
size. The client defaults to one attempt, allows at most three, and retries only
transient failures using the **identical** request. No retry changes authority,
idempotency keys, or expected generations. A stale generation is not retried.

`card_unavailable`, `catalog_unavailable`, `service_unavailable`, and
`audit_unavailable` are non-success. No cached catalog admission is fabricated
when storage is unavailable. A timeout or lost response can mean an operation
committed; retry the identical request within its replay window, then use
authenticated status to reconcile. Do not mark ready from an ambiguous response.
A successful receipt is only valid until its reported expiry and may be
administratively revoked sooner; it is not a permanent readiness assertion.

The host must become non-ready if it cannot maintain its lease. Catalog
snapshots exclude expired instances immediately using the injected clock,
without waiting for a scheduler or registration service. Schedule
`await service.expire()` to record attributable expiration evidence. Fake clocks
exercise the same path without sleeps. A crash requires no shutdown call.

Administrator `revoke` is instance-scoped and audited as the administrator.
The existing trusted catalog APIs can also quarantine, disable, revoke, or remove
a whole logical agent; managed leases cannot bypass those decisions. Direct
catalog mutation is privileged application code, never a model-facing tool.

## Evidence and confidential values

Registration carries Conducto correlation IDs and W3C `traceparent`/`tracestate`
through the service into card retrieval; deployment bearer credentials are
**never forwarded to the agent**. Dedicated `conducto.registration.server`
spans are distinct from A2A invocation spans. Required audit intent precedes
mutation, and outcomes identify actor, agent, instance, operation, generation,
reason and trace IDs. Expiration identifies the originally admitted actor.

The sink must acknowledge evidence synchronously; buffered acceptance is not
allowed. Audit failure after a commit produces a non-success response; it cannot
roll back a published catalog snapshot. Supervise expiration audit failures and
use a durable sink in production. The reference sink is only for local tests.

Tokens, signatures, card bodies, private endpoint URLs, and lease handles are not
included in Conducto evidence. Results normally serialize without handles;
`SecretStr` also masks representations. The authenticated HTTP response necessarily
delivers the admission handle once (or on an exact retry), with `Cache-Control:
no-store`. This confidential grant is the exception to ordinary result
serialization. Never log `request_document`/`result_document` outputs or enable
HTTP body/header tracing for this endpoint. Configure application/server/proxy
access logging to avoid private request destinations and credentials.
