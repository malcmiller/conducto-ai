# Host an agent for another process

Use this pattern when an agent must run independently from its caller.

## Install the server extra

```bash
pip install "conducto-ai[a2a-server]"
```

`a2a-server` installs only the optional application-owned server/example
surface: the official A2A HTTP server adapter, Uvicorn, FastAPI, and their
Starlette dependencies. It does not make any web framework a core Conducto
dependency. Install model adapters separately, for example:

```bash
pip install "conducto-ai[a2a-server,ollama]"
```

## Build the ASGI application

The application must provide an identity resolver. The resolver authenticates
the request and returns immutable identity and authority facts.

```python
from conducto import Runtime
from conducto.a2a import (
    StaticTokenIdentityResolver,
    create_a2a_app,
)
from conducto.security import Principal

from my_agents import WeatherAgent

identity_resolver = StaticTokenIdentityResolver(
    {
        "replace-with-a-secret-token": Principal(
            subject_id="local-client",
            issuer="local-development",
            audience="weather-agent",
            scopes=frozenset({"weather:read"}),
        )
    }
)


agent = WeatherAgent()
runtime = Runtime()

app = create_a2a_app(
    agent=agent,
    runtime=runtime,
    public_url="http://127.0.0.1:8001",
    identity_resolver=identity_resolver,
)
```

`StaticTokenIdentityResolver` is suitable for a small local deployment or
application-owned API-key map. It compares bearer tokens without returning or
logging them. For production OIDC-backed deployments, configure the existing
`TrustPolicy` and `JWKSKeyResolver` for the issuer and compose them with
`JWTBearerTokenValidator`:

```python
from conducto.a2a import JWTBearerIdentityResolver
from conducto.security import JWTBearerTokenValidator, TrustPolicy

identity_resolver = JWTBearerIdentityResolver(
    validator=JWTBearerTokenValidator(application_jwks_resolver),
    policy=application_trust_policy,
)
```

The same provider-neutral resolver supports Microsoft Entra ID, Auth0, Okta,
or any other OIDC-compliant issuer; only the application's trust-policy and
JWKS configuration changes.

Save this as `app.py`, then use an application-owned ASGI server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

Bind to loopback for local development. Production listener addresses, TLS,
reverse proxies, and process supervision are deployment decisions, not
Conducto configuration.

Conducto exposes:

```text
GET  /.well-known/agent-card.json
POST /a2a
```

## Installed-package examples

The wheel ships a deterministic example module with no model download,
credential, or external identity-provider requirement. Build and install the
optional server extra, then start Agent B and Agent A in separate terminals:

```bash
pip install "conducto-ai[a2a-server]"

CONDUCTO_DEMO_PORT=8002 \
CONDUCTO_DEMO_AGENT=agent-b \
CONDUCTO_DEMO_COMPOSITION=fastapi \
python -m uvicorn conducto.examples.a2a_hosting:app --host 127.0.0.1 --port 8002
```

```bash
CONDUCTO_DEMO_PORT=8001 \
CONDUCTO_DEMO_AGENT=agent-a \
CONDUCTO_DEMO_DOWNSTREAM_CARD_URL=http://127.0.0.1:8002/.well-known/agent-card.json \
python -m uvicorn conducto.examples.a2a_hosting:app --host 127.0.0.1 --port 8001
```

Agent A is a direct `A2AASGI` application. Agent B mounts the identical
Conducto application in FastAPI and forwards the embedding application's
lifespan to Conducto's bounded `startup()`, `drain()`, and `aclose()` methods.
Both publish the same `/.well-known/agent-card.json` and `/a2a` surface.

In a third terminal, run the installed-wheel orchestrator. It discovers Agent
A's public card, invokes its public `delegate` capability, and Agent A
discovers and invokes Agent B:

```bash
python -m conducto.examples.a2a_orchestrator \
  --agent-card-url http://127.0.0.1:8001/.well-known/agent-card.json \
  --value receipt-7 \
  --correlation-id local-chain-1
```

PowerShell:

```powershell
$env:CONDUCTO_DEMO_PORT = "8002"
$env:CONDUCTO_DEMO_AGENT = "agent-b"
$env:CONDUCTO_DEMO_COMPOSITION = "fastapi"
python -m uvicorn conducto.examples.a2a_hosting:app --host 127.0.0.1 --port 8002
```

The example identity resolver accepts `x-demo-scopes` only to demonstrate
deterministic local scope denials. It is not authentication and must not be
deployed as an authentication mechanism.

## Composition and ownership boundaries

| Concern | Owner |
| --- | --- |
| Capability metadata, Agent Card, inbound A2A dispatch, runtime, authentication result mapping, authorization, approval, limits, audit, task mapping, drain | Conducto ASGI application |
| Listener, event loop, worker model, process-local shutdown signal | Uvicorn or another ASGI server |
| Additional HTTP routes and middleware | Embedding FastAPI or Starlette application |
| TLS termination, forwarding headers, network exposure | Reverse proxy or ingress |
| Restarts, health policy, rolling deployment, log collection | Process supervisor or platform |

Mount the returned ASGI object; do not recreate the Agent Card or `/a2a`
routes, call reflected capability methods from framework routes, or replace the
Conducto lifespan with framework-only cleanup. Framework middleware may add
application concerns, but authentication, admission limits, runtime dispatch,
auditing, and result mapping continue through the mounted Conducto app.

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
