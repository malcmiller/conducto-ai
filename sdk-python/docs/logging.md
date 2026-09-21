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
from conducto import configure_logging

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

from conducto import JsonFormatter

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

| Field               | Description                                                                    |
|---------------------|--------------------------------------------------------------------------------|
| `schema_version`    | Structured event schema version, currently `"1"`.                              |
| `event`             | Stable, versioned event name.                                                  |
| `outcome`           | One of `success`, `failure`, `timeout`, or `cancelled`.                        |
| `correlation_id`    | Caller-provided invocation/route ID. If omitted, Conducto creates a UUID.      |
| `agent_id`          | Published local agent name.                                                    |
| `capability_id`     | Published capability name.                                                     |
| `duration_ms`       | Capability execution duration in milliseconds.                                 |
| `error_category`    | Stable category, not exception text.                                           |
| `provider`          | Selected model provider identifier.                                            |
| `model_reference`   | Effective model reference.                                                     |
| `resolution_source` | One of `call_override`, `run_override`, `agent_default`, or `runtime_default`. |
| `agent_count`       | Number of registered or discovered local agents.                               |
| `input_tokens`      | Prompt/input tokens reported by the provider, when available.                  |
| `output_tokens`     | Completion/output tokens reported by the provider, when available.             |
| `total_tokens`      | Total tokens reported by the provider, when available.                         |
| `loop_id`           | Runtime-generated identifier for one isolated delegation loop.                  |
| `turn`              | One-based model decision turn within the loop.                                  |
| `tool_call_id`       | Model-supplied stable call identifier after validation.                         |
| `snapshot_revision`  | Registry revision captured for the decision's immutable toolbox.                |

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

Application code can bind additional non-sensitive context around Conducto
work:

```python
from conducto import log_context

with log_context(request_id="request-123", tenant_id="tenant-a"):
    result = await orchestrator.invoke("AuditAgent", "review", {}, correlation_id="corr-123")
```

Nested contexts restore the prior values on exit. Context keys named
`approval`, `arguments`, `authorization`, `credential`, `credentials`,
`model_response`, `prompt`, `result`, `results`, `secret`, or `token` are
rejected.

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
from conducto import configure_logging, emit_event

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
