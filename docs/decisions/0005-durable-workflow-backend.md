# ADR 0005: Use Temporal behind a Conducto-owned workflow contract

**Status:** Accepted

## Context

Story 7.2 needs durable execution for fixed, governed workflows that can outlive
a process: dependency ordering, activity retry, approval waits, deadlines,
cancellation, compensation, recovery, and definition upgrades. Conducto must
not reproduce workflow history, replay, timers, worker coordination, or crash
recovery when a mature durable engine supplies them.

The engine is an implementation detail. A workflow definition continues to use
Conducto capability discovery and invocation, authority attenuation, budgets,
task lineage, typed outcomes, audit, and trace correlation. Backend workflow
IDs, histories, retry exceptions, deployment topology, and search metadata are
opaque adapter details.

## Decision

Conducto will use **Temporal** as the default durable-workflow backend for
Story 7.2. Dapr Workflow is not a required or supported backend in Story 7.2;
it remains a revisitable deployment alternative, rather than a public API
choice. An in-memory adapter remains the required backend for unit and
conformance tests, so normal CI never needs a Temporal or Dapr service.

Conducto will also maintain its own append-only domain ledger for workflow
identity, step/activity intent and acceptance, policy/authority snapshots,
approval decisions, terminal typed outcome, and audit correlation. Temporal
history is operational recovery evidence, not the domain audit system of
record. The ledger must not duplicate opaque Temporal history payloads or store
credentials, raw approval secrets, or unbounded capability inputs/outputs.

Each capability or other external side effect runs as an Activity, never in
deterministically replayed workflow code. Activities receive a stable
idempotency key derived from the Conducto workflow, step, and attempt
acceptance boundary. A retry can occur before acceptance; an outcome with
uncertain acceptance must become a typed `acceptance_uncertain` failure rather
than being silently redirected or repeated.

## Conducto adapter contract for Story 7.2

The following is the minimum backend-neutral contract. It is a contract design,
not a promise that these names are public imports before Story 7.2.

| Operation | Required behavior |
|---|---|
| `start(definition, input, context)` | Creates a versioned execution and returns stable Conducto workflow/run IDs plus an opaque backend reference. Repeated start with the same caller idempotency key returns the same execution. |
| `signal(workflow_id, event)` / `approve(workflow_id, approval)` | Delivers a schema-validated, idempotent external event. Approval validation and audit occur at the Conducto boundary. |
| `query(workflow_id)` | Returns a bounded status/checkpoint snapshot without exposing backend history or implementation exceptions. |
| `cancel(workflow_id, reason)` | Requests parent cancellation and deterministically propagates it to running activities. Terminal cancellation is a typed outcome. |
| `resume(workflow_id)` | Reconciles a previously started execution after worker recovery; it never creates a second execution. |
| `get_result(workflow_id)` | Returns one immutable typed terminal success, failure, cancellation, or compensated outcome. |

The versioned definition/reference and input must identify a workflow name,
semantic definition version, schema digest, immutable input, and caller
idempotency key. The public identity model contains stable Conducto
`workflow_id`, `run_id`, `step_id`, and `activity_id`; the backend reference is
opaque and safe diagnostic metadata is bounded, credential-free, and
allowlisted.

An activity request must carry its idempotency key, immutable task lineage,
deadline, attenuated authority, shared budgets, cancellation handle, and
correlation identifiers for workflow, step, activity, gateway, model,
approval, audit, and trace. Timer and event operations must expose only their
contractual deadline/event semantics. A backend that cannot implement a
required operation must fail explicitly; it must not simulate success with a
backend-specific fallback.

The adapter capability matrix is explicit:

| Capability | In-memory | Temporal default | Dapr Workflow alternative |
|---|---|---|---|
| Required CI conformance | Yes | No service in required CI | No service in required CI |
| Durable timer, approval event, retry, recovery | Deterministic simulation | Required | Evaluated, not selected |
| Reverse compensation | Deterministic simulation | Conducto workflow policy | Conducto workflow policy |
| Long-running definition upgrade | Contract simulation | Required versioning discipline | Not in Story 7.2 scope |
| Backend history/query metadata | No | Opaque diagnostic reference only | Opaque diagnostic reference only |

## Conformance evidence and evaluation

The backend comparison used one bounded Conducto-facing workflow with fake
capabilities: discover an eligible gateway capability; execute two dependent
activities using one lineage and budget; wait for an idempotent approval; use a
durable timer; inject retryable pre-acceptance and non-retryable accepted
failures; restart a worker; cancel a running activity; compensate a completed
step in reverse order; resume an older definition version; and assert all
correlation IDs. Tests use virtual/deterministic time and never use a
wall-clock sleep.

| Criterion | Temporal | Dapr Workflow | Result |
|---|---|---|---|
| Python and .NET maturity | First-party SDKs and established replay model | Supported SDKs, but orchestration operational behavior is coupled to Dapr runtime releases | Temporal |
| Replay and side-effect boundary | Strict workflow/activity separation and replay tooling | Activity separation is available, but replay and state-store behavior need more deployment-specific proof | Temporal |
| Approval, timer, retry, cancellation, compensation | Signals, durable timers, retries, cancellation; compensation remains Conducto policy | External events, durable timers, retries, cancellation; compensation remains Conducto policy | Temporal |
| Restart and long-running upgrade | Worker recovery and explicit workflow versioning are first-class | Recovery depends on sidecar, Scheduler, Placement, and state-store topology; upgrade evidence is less direct | Temporal |
| Local and required-CI testing | Local server for opt-in spike; in-memory adapter for CI | Local sidecar plus Scheduler, Placement, and state store; in-memory adapter still required | Temporal |
| Serialization and diagnostics | Deterministic workflow restrictions; histories/search attributes are backend-private | Durable state and component configuration are backend-private | Tie |
| Operations and lock-in | Temporal service/database and worker deployment; history/API lock-in | Sidecars, Scheduler, Placement, state-store components, and configuration lock-in | Temporal is lower-risk for this use |
| Security and tenancy | Namespace, worker identity, and payload/secret discipline needed | Dapr identity, component secrets, state-store ACLs, and sidecar policy needed | Tie; Conducto owns policy/audit |

The Dapr experiment is rejected for the default because its required local and
production topology adds Dapr control-plane and state-store concerns without a
clear benefit for a single durable workflow backend. This does not reject Dapr
for service integration generally, nor rule out a future optional adapter.

## Reproducible local spike procedures

These procedures are opt-in integration evidence, not required CI commands.
They use only local containers and fake capabilities; do not supply cloud
credentials, model providers, or production secrets.

Temporal’s supported local development server is started and removed with:

```bash
docker run --rm --name conducto-temporal -p 7233:7233 temporalio/auto-setup:1.28.1
docker stop conducto-temporal
```

The spike records the exact `temporalio` Python package and Temporal .NET SDK
versions in its lockfiles, runs a Python worker/client and a .NET worker/client
against `localhost:7233`, and captures history with the matching `temporal`
CLI. Workflow code may only use deterministic Temporal APIs; network,
capability, model, and clock side effects belong in activities. The service
image, SDK packages, histories, search attributes, and deployment metadata are
Temporal-specific and must not enter Conducto contracts.

Dapr Workflow’s local proof requires a Dapr CLI/runtime version recorded with
the spike, Docker, a local state store, Placement, and Scheduler:

```bash
dapr init
dapr run --app-id conducto-dapr-worker --dapr-http-port 3500 -- python worker.py
dapr uninstall --all --yes
```

The corresponding Python and .NET workers use only fake capabilities and prove
the same event, timer, retry, cancellation, restart, and accepted-side-effect
scenarios. The experiment must additionally stop/restart the sidecar and
state-store, then remove all initialized containers/components. Dapr workflow
state, component configuration, sidecar endpoints, Scheduler/Placement
topology, and state-store implementation are Dapr-private.

Both procedures must record image digests, CLI and SDK versions, commands,
observed outcomes, limitations, and failed experiments beside the retained
adapter conformance tests. No unrecorded prototype may become a production API.

## Consequences

- Story 7.2 implements only the neutral contract and the Temporal adapter; it
  must not export `temporalio` or .NET/Dapr types from `conducto`.
- Required tests exercise the in-memory adapter and deterministic fakes.
  Temporal integration tests are explicitly opt-in and separately provision
  their local service.
- Workflow authors obey backend determinism restrictions through the adapter;
  side effects, approval validation, capability invocation, and audit are
  Conducto activities/boundaries.
- The domain ledger is mandatory for audit-grade Conducto facts and cross-backend
  portability; backend history remains a recovery/diagnostic aid.

## Alternatives considered

- **Dapr Workflow as the default:** rejected for Story 7.2 because the
  sidecar, Scheduler, Placement, and state-store topology create extra
  operational variables without a compensating workflow-contract advantage.
- **Both backends now:** rejected because maintaining two production adapters
  before a stable Conducto contract would widen the unsupported surface.
- **A custom durable engine:** rejected because replay, history, timers,
  recovery, and coordination are not Conducto differentiators.
- **No domain ledger:** rejected because backend history is neither a stable
  Conducto audit contract nor portable across a future backend change.

## Exit criteria for reconsideration

Revisit this decision if Temporal cannot meet the in-memory contract with
deterministic tests, fails Python/.NET worker parity, materially blocks
Conducto security/identity integration, exceeds agreed operational cost, or if
Dapr demonstrates equivalent recovery and version-upgrade behavior with a
lower measured operational burden. A change requires the same conformance
workflow, recorded versioned evidence, a migration plan for opaque backend
references, and an updated ADR.

## Related code and evidence

- `conducto.core.gateway.AgentGateway` owns discovery, opaque binding, and
  governed invocation semantics.
- `conducto.core.run_context.RunContext` owns deadline, cancellation,
  delegation lineage, and budgets.
- `conducto.security` owns approval and audit boundaries.
- `docs/gateway-and-discovery.md` documents gateway revalidation and the
  accepted-side-effect failover boundary.
- Story 7.2 will add the retained in-memory and Temporal adapter conformance
  tests; this decision does not add a backend SDK dependency or a production
  workflow API.
