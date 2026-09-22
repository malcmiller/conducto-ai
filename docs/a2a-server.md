# A2A ASGI server host

`conducto.a2a.A2AASGI` is the smallest complete inbound A2A 1.0 ASGI protocol
host for one Conducto agent. It owns HTTP/ASGI protocol adaptation, exact
route matching, Agent Card and mounted endpoint consistency, official A2A SDK
JSON-RPC dispatch, protocol parsing and standard protocol errors, and task
repository wiring. It does not own Conducto authorization, approvals, audit,
model resolution, capability invocation, authentication, production
hardening, or process hosting; those remain with the runtime, a later story,
and the application that mounts this adapter.

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
from conducto.transport.tasks import InMemoryTaskRepository

app = A2AASGI(
    agent=agent,
    endpoint_url="https://agent.example/a2a",
    task_repository=InMemoryTaskRepository(),
    request_handler=my_request_handler,
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

- `message/send` creates a new task (or continues an existing one when the
  message carries a `task_id`) and invokes the request-handler seam exactly
  once per accepted message.
- `tasks/get` returns a persisted task snapshot.
- `tasks/list` returns one deterministic page, honoring `page_size` and
  `page_token` exactly as implemented by the repository.
- `tasks/cancel` cancels a non-terminal task through the repository.

Streaming message send, task subscription, push notification configuration,
and extended Agent Cards all return the standard A2A
`UnsupportedOperationError`. Unsupported media types, unknown JSON-RPC
methods, malformed JSON, and missing or mismatched `A2A-Version` headers all
return the corresponding pinned protocol error from the official SDK. A
request that declares a required A2A extension the card does not advertise
in `capabilities.extensions` is rejected with
`ExtensionSupportRequiredError`.

Task transitions are atomic and repository-backed: continuing a stale,
missing, or terminal task is rejected before the request-handler seam runs,
and a request-handler failure transitions the task to `TASK_STATE_FAILED`
through the repository's compare-and-transition contract rather than leaking
an exception to the caller.

## The request-handler seam

```python
from a2a.types.a2a_pb2 import Message
from conducto.a2a import A2ARequestHandler
from conducto.core.invocation_results import InvocationResult


class MyRequestHandler(A2ARequestHandler):
    async def handle_message(
        self, message: Message, *, task_id: str, context_id: str
    ) -> InvocationResult:
        ...
```

`A2AASGI` never calls a reflected agent's capability methods directly. Every
accepted message is delegated once to the injected `A2ARequestHandler`, and
the returned `InvocationResult` is mapped onto the A2A task lifecycle by
`conducto.a2a.invocation_result_to_task`. Binding this seam to the canonical
Conducto runtime — so an inbound A2A message is invoked through the same
governed contract as every other invocation path — is delivered by a later
story; this story only defines and exercises the seam with a deterministic
fake implementation.

## Task repository

`A2AASGI` accepts any `TaskRepository` implementation
(`conducto.transport.tasks.TaskRepository`), the same protocol already used
by the A2A client transport. `InMemoryTaskRepository` is suitable for tests
and single-process deployments; production task databases are out of scope
for this story.

## What this story does not cover

- Conducto authorization, approvals, audit, model resolution, or capability
  invocation — the request-handler seam exists so a later story can bind
  those without changing this adapter.
- Bearer/mTLS extraction or authentication.
- Production rate limiting, proxy trust, or extensive body/header limits.
- Liveness/readiness/drain policy.
- Uvicorn/FastAPI examples or separate-process acceptance; this story is
  exercised entirely through an in-process ASGI transport in tests.
