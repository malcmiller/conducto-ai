# Host an agent for another process

Use this pattern when an agent must run independently from its caller.

## Install the server extra

```bash
pip install "conducto-ai[a2a-server]"
```

Install any model adapter separately, for example:

```bash
pip install "conducto-ai[a2a-server,ollama]"
```

## Build the ASGI application

The application must provide an identity resolver. The resolver authenticates
the request and returns immutable identity and authority facts.

```python
from conducto import Runtime
from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    create_a2a_app,
)
from conducto.security import AuthorizationContext, Principal

from my_agents import WeatherAgent


async def resolve_identity(
    request: A2AAuthenticationRequest,
) -> A2AAuthenticatedIdentity:
    # Development example only. Production code must validate a token and,
    # when required, an already verified mTLS peer identity.
    return A2AAuthenticatedIdentity(
        AuthorizationContext(
            principal=Principal(
                subject_id="local-development",
                issuer="local-development",
                audience="weather-agent",
                scopes=frozenset({"weather:read"}),
            ),
            task_id=request.task_id,
            correlation_id=request.correlation_id,
        )
    )


agent = WeatherAgent()
runtime = Runtime()

app = create_a2a_app(
    agent=agent,
    runtime=runtime,
    public_url="http://127.0.0.1:8001",
    identity_resolver=resolve_identity,
)
```

Save this as `app.py`, then use an application-owned ASGI server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

Conducto exposes:

```text
GET  /.well-known/agent-card.json
POST /a2a
```

The current adapter does not yet provide production readiness, drain, proxy,
or broad HTTP-limit policy. Those concerns belong to the transport-hardening
work and the embedding deployment.

## What `create_a2a_app()` does

- Generates and serves the Agent Card.
- Derives the `/a2a` endpoint from `public_url`.
- Creates the runtime-backed A2A handler.
- Uses an isolated in-memory task repository unless one is supplied.
- Maps every accepted request into `Runtime`.

It does not start Uvicorn, configure TLS, acquire credentials, or supervise
the process.

## Advanced composition

Use `A2AASGI` directly only when supplying a custom task repository and
request-handler composition. Most applications should use
`create_a2a_app()`.

For exact routes, message envelopes, task behavior, and identity contracts,
see [A2A ASGI server host](../a2a-server.md).
