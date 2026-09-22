# Python logging

Conducto's Python SDK uses the standard-library [`logging`](https://docs.python.org/3/library/logging.html)
API behind a stable structured-event contract. It is safe to use from a
library: importing `conducto` does not add root handlers, change application
levels, or otherwise configure global logging.

Events use the `conducto.events` logger, which is a child of `conducto`.
Applications own logging configuration. They can attach their existing
handlers to `conducto`, or use the SDK's opt-in helper for local development.

## Quick start

```python
from conducto.core.logging import configure_logging

configure_logging(format="json")
```

`configure_logging()` adds one handler owned by Conducto to the `conducto`
logger, sets that logger's level, and disables propagation from that logger to
avoid duplicate output. It does not change the root logger. Calling it again
replaces only the prior handler created by this helper; application-attached
handlers remain intact.

Use `format="development"` for concise human-readable output:

```python
configure_logging(format="development")
```

## Integrating with application logging

For production, attach an application-owned handler and formatter instead of
calling `configure_logging()`:

```python
import logging

from conducto.core.logging import JsonFormatter

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("conducto")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
```

The SDK adds a package-level `NullHandler` on import. This prevents Python's
fallback `lastResort` handler from writing warnings and errors to stderr when
the application has not configured logging, while normal propagation still
allows application-owned ancestor handlers to receive events.

## Event contract

All Conducto events have `schema_version: "1"` and a versioned `event` name.
JSON output is canonical: ASCII-only, compact, sorted keys, and UTC
timestamps with millisecond precision.

| Event                                         |                         Level | Meaning                                                                                          |
|-----------------------------------------------|------------------------------:|--------------------------------------------------------------------------------------------------|
| `conducto.agent.registered.v1`                |                        `INFO` | A local agent entered the registry.                                                              |
| `conducto.agent.discovered.v1`                |                       `DEBUG` | Local-agent discovery metadata was read.                                                         |
| `conducto.model.selected.v1`                  |                        `INFO` | A model configuration was selected for routing.                                                  |
| `conducto.delegation.turn_started.v1`         |                        `INFO` | A bounded model decision turn began against one toolbox snapshot.                                |
| `conducto.delegation.tool_completed.v1`       |                        `INFO` | A gateway-mediated tool call produced a safe outcome category.                                   |
| `conducto.delegation.completed.v1`            |                        `INFO` | A delegation loop reached a typed terminal outcome.                                              |
| `conducto.capability.arguments_validated.v1`  |                        `INFO` | Capability arguments passed or failed validation.                                                |
| `conducto.capability.invocation_started.v1`   |                        `INFO` | Local capability execution began.                                                                |
| `conducto.capability.invocation_completed.v1` |                        `INFO` | Local capability execution completed.                                                            |
| `conducto.capability.invocation_failed.v1`    | `INFO`, `WARNING`, or `ERROR` | Validation/target failures are normal control flow; unexpected capability exceptions are errors. |
| `conducto.capability.invocation_timed_out.v1` |                     `WARNING` | The configured invocation deadline elapsed.                                                      |
| `conducto.capability.invocation_cancelled.v1` |                        `INFO` | A capability cooperatively reported cancellation.                                                |

The fields below are emitted when applicable:

| Field               | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| `schema_version`    | Structured event schema version, currently `"1"`.                                               |
| `event`             | Stable, versioned event name.                                                                   |
| `outcome`           | One of `success`, `failure`, `timeout`, or `cancelled`.                                         |
| `correlation_id`    | Caller-provided invocation/route ID. If omitted, Conducto creates a UUID.                       |
| `trace_id`          | Active OpenTelemetry trace ID when the optional tracing extra and provider are configured.       |
| `span_id`           | Active OpenTelemetry span ID when the optional tracing extra and provider are configured.        |
| `agent_id`          | Published local agent name.                                                                     |
| `capability_id`     | Published capability name.                                                                      |
| `duration_ms`       | Capability execution duration in milliseconds.                                                  |
| `error_category`    | Stable category, not exception text.                                                            |
| `provider`          | Selected model provider identifier.                                                             |
| `model_reference`   | Effective model reference.                                                                      |
| `resolution_source` | One of `call_override`, `run_override`, `agent_default`, or `runtime_default`.                  |
| `agent_count`       | Number of registered or discovered local agents.                                                |
| `input_tokens`      | Prompt/input tokens reported by the provider, when available.                                   |
| `output_tokens`     | Completion/output tokens reported by the provider, when available.                              |
| `total_tokens`      | Total tokens reported by the provider, when available.                                          |
| `loop_id`           | Runtime-generated identifier for one isolated delegation loop.                                  |
| `turn`              | Completed model turns; `0` when the loop ends before its first model call, otherwise one-based. |
| `tool_call_id`      | Model-supplied stable call identifier after validation.                                         |
| `snapshot_revision` | Registry revision captured for the decision's immutable toolbox.                                |

Stable error categories currently include `target_not_found`,
`argument_validation`, `timeout`, `capability_exception`,
`unsupported_return_value`, and `internal_error`. Invocation result envelopes
continue to retain the original exception for application diagnostics, but
default event records omit exception text and stack traces.

## Correlation and concurrency

`OrchestratorAgent.route()` and `OrchestratorAgent.invoke()` bind correlation,
agent, and capability IDs using `contextvars`. Context stays isolated between
concurrent asyncio tasks and is automatically propagated into synchronous
capabilities run with `asyncio.to_thread`.

When `conducto-ai[opentelemetry]` is installed and the application has supplied
an OpenTelemetry tracer provider, Conducto also adds the active `trace_id` and
`span_id` to safe structured log records. Logging remains independent of any
exporter: missing OpenTelemetry dependencies or provider configuration simply
omit those fields and do not change invocation behavior.

Application code can bind additional non-sensitive context around Conducto
work:

```python
from conducto.core.logging import log_context

with log_context(request_id="request-123", tenant_id="tenant-a"):
    result = await orchestrator.invoke("AuditAgent", "review", {}, correlation_id="corr-123")
```

Nested contexts restore the prior values on exit. Context keys named
`approval`, `arguments`, `authorization`, `credential`, `credentials`,
`model_response`, `prompt`, `result`, `results`, `secret`, or `token` are
rejected.

## OpenTelemetry tracing

Conducto uses explicit, optional OpenTelemetry spans at SDK-owned boundaries:
gateway discovery/invocation, MCP server/list/call, A2A client/server, token
exchange, authorization/approval, model completion, capability invocation, and
delegation turns. The stable semantic fixture is checked in at
`docs/semantic-fixtures/opentelemetry-spans.v1.json`.

The SDK does not install a global tracer provider, exporter, sampler,
propagator, resource, or auto-instrumentation hook at import time. Applications
own that lifecycle. `configure_in_memory_tracing()` is a documented helper for
tests and local diagnostics; production applications should configure their own
provider and exporters. Conducto propagates W3C `traceparent`/`tracestate` at
supported HTTP/A2A boundaries, ignores malformed or oversized remote context,
and records only the bounded `conducto.invalid_remote_context` attribute.
Baggage is not forwarded by default.

## OpenTelemetry Logs export

`OpenTelemetryLogBridge` (`conducto.core.otel_logs`) bridges these same
versioned events to OpenTelemetry Logs and OTLP-compatible backends. It does
not create a second event taxonomy, and it never replaces or duplicates
Story 2.3 security audit evidence: audit events remain a separate,
application-owned durable stream, and accepting an OpenTelemetry log record
never satisfies mandatory audit delivery.

Constructing the bridge is explicit and requires an application-supplied
OpenTelemetry `LoggerProvider`. Importing `conducto` never installs a global
provider, handler, processor, exporter, resource, or environment
configuration; the application owns exporter endpoints, credentials, TLS,
headers, resource detectors, sampling/filtering, and provider shutdown.

```python
import logging

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
# Any OTLP exporter the application chooses to install, for example:
# from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

from conducto.core.otel_logs import OpenTelemetryLogBridge

provider = LoggerProvider(
    resource=Resource.create({"service.name": "my-service", "deployment.environment": "prod"})
)
# provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=...)))

bridge = OpenTelemetryLogBridge(provider, level=logging.INFO)
...
bridge.close(flush=True)  # detaches the handler; never shuts down `provider`
provider.shutdown()  # the application owns and shuts down its own provider
```

Console/JSON output from `configure_logging` and the OpenTelemetry bridge can
both be attached to the `conducto` logger at the same time without
duplicating event generation: each handler independently receives the same
underlying event record.

`OpenTelemetryLogBridge` owns only the handler it creates:

- `close()` is idempotent, detaches the owned handler, and never shuts down a
  caller-owned `LoggerProvider`.
- `close(flush=True)` additionally performs at most one bounded
  `force_flush` call against the provider.
- Translation or export failures are counted on `bridge.dropped_count` and
  reported only through the local `conducto.otel_logs` diagnostic logger,
  never by raising back into application code or recursing through the
  bridge itself.

Each mapped OpenTelemetry log record uses the stable Conducto event name as
both `body` and `event_name`, a severity derived from the Python logging
level, and only the bounded, low-cardinality `conducto.*` attributes listed
in `docs/semantic-fixtures/opentelemetry-logs.v1.json`. Trace/span
correlation is derived by the OpenTelemetry SDK itself from the active span
context (see `start_span`/`configure_in_memory_tracing`); an absent span
simply yields an uncorrelated record. `configure_in_memory_logs()` is a
documented test helper analogous to `configure_in_memory_tracing()`.

Sampled or disabled diagnostic telemetry never affects authorization,
approval, mandatory audit acceptance, invocation results, or audit event
ordering — those remain governed entirely by `conducto.security`.

## Privacy and sensitive data

Lifecycle events do **not** log prompts, model responses, capability
arguments, result bodies, credentials, tokens, approval data, provider
configuration secrets, exception text, or tracebacks. Provider/model events
only include the provider identifier, effective model reference, and
resolution source. Delegation events add only stable IDs, turn/revision
counters, and safe outcome categories; they do not add tool arguments or
results.

The low-level `emit_event()` API supports a payload only when both safeguards
are explicitly enabled:

```python
from conducto.core.logging import configure_logging, emit_event

configure_logging(format="json", include_sensitive_data=True)
emit_event(
    "my_application.debug.v1",
    payload={"request": "sensitive diagnostic data"},
    include_sensitive_data=True,
)
```

Payloads are emitted only to Conducto's separate `conducto.sensitive` logger
and its opt-in handler, never to ordinary application handlers on `conducto`.
This is intended only for controlled development diagnostics. Treat payload
logging as a security decision: avoid it in production and ensure any
destination, retention policy, and access controls are appropriate.

## Compatibility

Event names and documented fields are public compatibility contracts within
their schema version. Additive fields may be introduced in a compatible
release. Renaming, removing, or changing the meaning of an existing field or
event requires a new schema/event version. The committed JSON golden fixture
pins the current formatter output.
