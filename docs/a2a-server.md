# A2A ASGI server host

`conducto.a2a.A2AASGI` is the smallest complete inbound A2A 1.0 ASGI protocol
host for one Conducto agent. It owns HTTP/ASGI protocol adaptation, exact
route matching, Agent Card and mounted endpoint consistency, official A2A SDK
JSON-RPC dispatch, protocol parsing and standard protocol errors, and task
repository wiring. It does not own Conducto authorization, approvals, audit,
model resolution, capability invocation, or authentication. Those concerns
remain with `Runtime`, `A2ARuntimeHandler`, and the application-owned identity
resolver. Production hardening and process hosting remain application concerns
and later roadmap work.

The optional dependency lives in the `a2a-server` extra:

```bash
uv sync --extra a2a-server
# or
pip install "conducto-ai[a2a-server]"
```

Importing `conducto` or any client transport module never imports Starlette
or the official SDK's server routes. Constructing `A2AASGI` without the extra
raises `A2ADependencyError` with installation guidance.

## Construction

```python
from conducto.a2a import A2AASGI
from conducto.a2a import A2ARuntimeHandler
from conducto.transport.tasks import InMemoryTaskRepository

runtime_handler = A2ARuntimeHandler(
    runtime=runtime,
    agent=agent,
    identity_resolver=resolve_identity,
)
app = A2AASGI(
    agent=agent,
    endpoint_url="https://agent.example/a2a",
    task_repository=InMemoryTaskRepository(),
    request_handler=runtime_handler,
)
```

Constructing `A2AASGI` never starts a listener, event loop, thread, or
subprocess. The returned object is an ASGI callable; the embedding
application owns the ASGI server, TLS termination, and process supervision.
Two instances never share task or lifecycle state, so one process can host
multiple agents or multiple isolated deployments of the same agent.

`agent.get_agent_card(endpoint_url, ...)` produces the published card, which
is validated against the pinned A2A 1.0 profile
(`conducto.core.a2a_profile.parse_agent_card`) before the adapter starts
routing requests. A card that fails profile validation fails construction
with `ValueError`, not a runtime 500.

## Routes

- `GET /.well-known/agent-card.json` serves the reflected `AgentCard`.
- The JSON-RPC endpoint is mounted at the path component of `endpoint_url`
  (`/a2a` in the example above), so the card-advertised endpoint and the
  actual mounted route are always the same URL.

Both routes are produced by the official A2A SDK's
`a2a.server.routes.create_agent_card_routes` and `create_jsonrpc_routes`;
this adapter does not hand-roll parallel routing, parsing, or dispatch.

## Supported operations

The pinned profile's non-streaming operations are supported end to end
against the injected `TaskRepository`
(`conducto.transport.tasks.TaskRepository`):

- `SendMessage` creates a new task (or continues an existing one when the
  message carries a `task_id`) and invokes the request-handler seam exactly
  once per accepted message.
- `GetTask` returns a persisted task snapshot.
- `ListTasks` returns one deterministic page, honoring `page_size` and
  `page_token` exactly as implemented by the repository.
- `CancelTask` cancels a non-terminal task through the repository.

A2A 1.0 renamed the 0.3 slash-style JSON-RPC method names (for example
`message/send`) to gRPC-style service method names (for example
`SendMessage`); the pinned profile and this adapter use only the 1.0 names,
matching the official SDK's own dispatcher and client transport.

Streaming message send, task subscription, push notification configuration,
and extended Agent Cards all return the standard A2A
`UnsupportedOperationError`. Unsupported media types (including any message
part that is not a plain text part), unknown JSON-RPC methods, malformed
JSON, and missing or mismatched `A2A-Version` headers all return the
corresponding pinned protocol error from the official SDK. A request that
declares a required A2A extension the card does not advertise in
`capabilities.extensions` is rejected with `ExtensionSupportRequiredError`.

Task transitions are atomic and repository-backed: continuing a stale,
missing, or terminal task is rejected before the request-handler seam runs; a
continuation whose `context_id` disagrees with the task's stored context is
rejected without invoking the seam; and every accepted message atomically
claims the task (transitioning it to `TASK_STATE_WORKING` through the
repository's compare-and-transition contract) before the seam runs, so
concurrent continuations of the same task can never both invoke the seam. A
request-handler failure transitions the task to `TASK_STATE_FAILED` through
the same contract rather than leaking an exception to the caller, and a
successful outcome's status, artifacts, and metadata are persisted onto the
task atomically before it is returned.

## Canonical runtime binding

```python
from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2ARuntimeHandler,
)


async def resolve_identity(
    request: A2AAuthenticationRequest,
) -> A2AAuthenticatedIdentity:
    authorization = await authenticate_incoming_request(
        authorization_header=request.headers.get("authorization"),
        validator=validator,
        policy=trust_policy,
        task_id=request.task_id,
        correlation_id=request.correlation_id,
        trace_headers=dict(request.headers),
    )
    return A2AAuthenticatedIdentity(authorization)


handler = A2ARuntimeHandler(
    runtime=runtime,
    agent=agent,
    identity_resolver=resolve_identity,
)
```

`A2AASGI` never calls a reflected agent's capability methods directly. Every
accepted message is delegated to the injected `A2ARequestHandler`.
`A2ARuntimeHandler` is the standard implementation: it resolves the advertised
skill to an opaque runtime-bound capability binding, revalidates registration,
generation, schema, lifecycle, health, and binding integrity, and safely renews
expired server-owned bindings only while the registered target is unchanged and
active. It then dispatches only through `Runtime.invoke()` or
`Runtime.resume_approval()`. Argument validation,
guardrails, required audit delivery, timeout, cancellation, model resolution,
serialization, and typed results are therefore identical to local invocation.

The identity resolver is injected and may compose with
`authenticate_incoming_request(...)`; the runtime adapter never acquires or
validates credentials itself. Raw headers are visible only to that resolver.
It returns immutable authorization facts, an optional capability allowlist and
delegation budget, and an optional authenticated approval decision. Request
metadata is intersected with a resolver-owned capability allowlist. A
request-supplied budget is used only when the resolver did not establish one,
and its depth and call counts are capped by handler-owned limits. Transport
data therefore cannot broaden resolver-owned authority. Transport timeouts are
also capped by the handler's `max_timeout`.

### Invocation envelope

The message contains exactly one `text/plain` part whose text is JSON:

```json
{
  "skillId": "conducto-0123456789abcdef",
  "arguments": {"value": 7}
}
```

Optional message metadata uses the `x-conducto` object:

```json
{
  "x-conducto": {
    "correlationId": "correlation-1",
    "deadline": 1790114400.0,
    "timeoutSeconds": 10.0,
    "modelReference": "runtime-model",
    "allowedCapabilities": ["echo"],
    "budget": {"maxDepth": 4, "calls": 8, "tokens": 1000, "cost": 1.0},
    "metadata": {"classification": "internal"}
  }
}
```

`task_id`, `context_id`, `message_id`, JSON-RPC request ID, correlation ID,
reference-task lineage, and safe metadata are copied into immutable run
metadata. Sensitive metadata keys are rejected by `RunConfig`. Scopes and
principal identity come only from the resolver. Caller deadlines are converted
to the runtime timeout budget, and model references still pass through normal
runtime policy and provider resolution.

The handler atomically coalesces duplicate request or message identifiers
within a hashed authenticated-principal namespace.
Concurrent replays share the first execution result; conflicting reuse is
rejected and never executes business logic. Active task cancellation signals
the same `CancellationState` observed by local runtime invocation.

`conducto.a2a.invocation_result_to_task` maps every public
`InvocationResult` family to a pinned task state, fixed safe status message,
reason code, optional success artifact, and credential-free provenance.
Exceptions, tracebacks, arguments, and failed result values are never emitted.

## Task repository

`A2AASGI` accepts any `TaskRepository` implementation
(`conducto.transport.tasks.TaskRepository`), the same protocol already used
by the A2A client transport. `InMemoryTaskRepository` is suitable for tests
and single-process deployments; production task databases are out of scope
for this story.

## What this story does not cover

- Bearer/mTLS extraction or production authentication implementations; those
  are supplied through the identity resolver.
- Production rate limiting, proxy trust, or extensive body/header limits.
- Liveness/readiness/drain policy.
- Uvicorn/FastAPI examples or separate-process acceptance; this story is
  exercised entirely through an in-process ASGI transport in tests.
