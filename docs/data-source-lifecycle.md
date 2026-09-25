# Data source lifecycle

A declared data source has two owners, and they are deliberately separate.

- **A deployment owns existence.** It provisions the index or corpus, ingests
  content, indexes it, verifies it is queryable, and retires it.
- **An agent owns reading.** It retrieves through the governed retrieval path
  defined by [`@retriever` and `RetrievalQuery`](./capabilities-and-tools.md)
  and never provisions, ingests, or tears down the source it reads.

The lifecycle contract lives in `conducto.resources`. Core carries no vector
database, search service, or storage dependency; concrete behavior comes from
adapters. `conducto.resources.adapters.InMemoryDataSourceBackend` is the
reference implementation and requires no network access.

## Why the fused form is wrong

A typical retrieval example uploads a file, creates a vector store, polls for
indexing, binds a search tool to the resulting identifier, runs one query, then
deletes both resources in a `finally` block with suppressed exceptions. That
fuses four concerns and fails in production three ways: the agent cannot outlive
an index created seconds earlier, a partially indexed corpus is
indistinguishable from a complete one, and suppressed cleanup leaves orphaned
resources with no typed signal.

Conducto separates them:

```python
from conducto.resources import ContentBatch, ContentItem, DataSourceLifecycle, ProvisioningConfig
from conducto.resources.adapters import InMemoryDataSourceBackend

backend = InMemoryDataSourceBackend()
lifecycle = DataSourceLifecycle(backend=backend)

binding = await lifecycle.provision(
    ProvisioningConfig(data_source="policy_corpus", backend_kind="in_memory")
)
await lifecycle.ingest(binding, ContentBatch((ContentItem.from_text("Twenty days of leave."),)))
await lifecycle.index(binding)
await lifecycle.verify_on_start("policy_corpus")
```

No agent participates. The agent later reads the populated source by name.

## Provisioning

`ProvisioningConfig` is a frozen, JSON-safe request carrying the data source
name, a descriptive `backend_kind`, and backend parameters. Its `fingerprint` is
a deterministic digest of that content, so re-provisioning identical
configuration is idempotent and returns the same binding and revision.

`ProvisionedBinding` is an opaque handle: a `binding_id`, a `revision`, and the
configuration fingerprint. It carries no endpoint, connection string, or
credential and is never handed to a model.

`describe()` returns a `DataSourceDescription` with the observable
`DataSourceState` (`ABSENT`, `PROVISIONED`, `RETIRED`) and `IndexingState`
(`EMPTY`, `INDEXING`, `INDEXED`, `PARTIAL`, `FAILED`). Indexing is an explicit
state, not an implicit side effect of ingestion.

## Ingestion

`ContentItem` carries a stable `content_id` and a `body_digest` derived from its
text and metadata. `ContentBatch` sorts items by content identity, rejects
duplicate identifiers and empty batches, and exposes a `digest` over the whole
batch. Re-ingesting a batch with the same digest is a no-op.

`IngestionProgress` reports `submitted_count`, `accepted_count`,
`indexed_count`, `pending_count`, sorted `failed_content_ids`, and the resulting
`IndexingState`. Partial ingestion raises `PartialIngestionError`, which carries
that progress; it is never reported as success.

Indexing never promotes a partial corpus to a complete one. While content
refused by an earlier batch is still outstanding, `index()` fails with
`IngestionError` and reason `index_incomplete`, and the source stays `PARTIAL`
so readiness cannot pass over an incomplete corpus.

## Retirement

`retire()` raises `RetirementError` when the backend cannot remove the resource
and when the binding is stale or absent. Cleanup exceptions are never
suppressed, so an orphaned resource always produces a typed, attributable
signal.

## Readiness check policy

`ReadinessCheck` is a `Flag`, so check moments combine:

| Policy | Meaning |
| --- | --- |
| `ReadinessCheck.ON_START` | Default. Verify at host startup; cheap and fail-fast. |
| `ReadinessCheck.ON_INVOKE` | Verify before serving a query; catches sources retired or emptied after startup. |
| `ON_START \| ON_INVOKE` | Recommended production setting. |
| `ReadinessCheck.NONE` | Explicit, documented opt-out for in-memory sources and test fixtures. |

A declared data source with no explicit policy resolves to `ON_START`. `NONE` is
never reachable by omission; it must be selected.

`ReadinessPolicy.readiness_ttl` bounds how long an affirmative verdict may be
reused by `ON_INVOKE`:

- A zero duration probes on every invocation.
- A non-zero duration reuses a cached affirmative verdict until it expires.
- Only affirmative verdicts are cached. A negative or failed verdict is never
  cached and never suppresses the next check. A reused verdict is returned with
  `from_cache=True` and its original `evaluated_at`, so callers can always tell a
  cached check from a fresh probe.
- The cache is keyed by data source identity and is invalidated by
  re-provisioning, re-ingestion, indexing, and retirement of that source.

There is no background or periodic polling, and a failed verdict never triggers
automatic reprovisioning or re-ingestion.

## Failure semantics

Check timing and failure handling are separate axes.

- An `ON_START` failure fails host startup and holds the host not-ready. It does
  not degrade to serving. Pass `lifecycle.startup_check([...])` as the
  `on_startup` hook of `A2AHostLifecycle`; a failure raises `A2AStartupError`
  and `is_ready()` stays `False`.
- An `ON_INVOKE` failure refuses that invocation with `DataSourceNotReadyError`.
  Wrap the retriever in `ReadinessCheckedRetriever` to enforce it. It never
  returns an empty result set, a partial result set, or a success-shaped result
  with a degraded marker. Answering over a knowledge base known to be
  unavailable is the failure this contract exists to prevent.

Every lifecycle failure derives from `DataSourceLifecycleError`, is attributed
to the named data source, and carries a stable snake_case `reason`. Its message
and `to_dict()` payload disclose no backend endpoint, credential, or raw
exception text. An unexpected backend exception — a connection error or client
failure carrying an endpoint or key in its message — is mapped to a generic
`ReadinessProbeError` with reason `readiness_probe_failed`; the original
exception is preserved only as `__cause__` for local debugging.

Provisioning, ingestion, indexing, and readiness probes all accept a
`LifecycleBudget` carrying an optional timeout and a `CancellationState`. A probe
that exceeds its budget raises `ReadinessProbeError` with reason
`readiness_timeout`; it is never treated as a pass.

`ReadinessCheckedRetriever` derives that bound from the active
`RunContext` through `run_context_budget()`, so an invocation-time probe honours
the run's remaining deadline and cancellation state instead of running
unbounded. Pass `budget_factory` to supply the bound explicitly instead.

## Conformance fixtures

`conducto.testing` exports deterministic fixtures for adapter authors:

```python
from conducto.testing import assert_data_source_lifecycle_conformance
from conducto.resources.adapters import InMemoryDataSourceBackend

await assert_data_source_lifecycle_conformance(InMemoryDataSourceBackend())
```

`ManualClock` drives TTL expiry and deadline tests without wall-clock sleeps.
