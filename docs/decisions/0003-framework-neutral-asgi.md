# ADR 0003: Use framework-neutral ASGI hosting

**Status:** Accepted

## Context

Remote agents need Agent Card and A2A JSON-RPC endpoints. FastAPI is a useful
application framework, but making it the Conducto contract would couple the
SDK to one framework and make core use depend on server packages.

## Decision

Conducto exposes an ASGI application through `create_a2a_app()` and an advanced
`A2AASGI` adapter. Applications may run or compose it with Uvicorn, Hypercorn,
FastAPI, Starlette, or another conforming host.

Conducto does not start listeners or own process supervision, TLS termination,
or reverse proxies.

## Consequences

- The server dependency remains optional.
- FastAPI composition is possible without becoming a core abstraction.
- ASGI lifespan and routing behavior require explicit tests.
- Applications remain responsible for deployment operation.

## Alternatives considered

- A FastAPI-only public API. Rejected because it makes one framework mandatory.
- A built-in server command that owns Uvicorn. Rejected because process and
  deployment lifecycle belong to the application.
- Handwritten socket/HTTP handling. Rejected because ASGI provides a standard
  integration boundary.

## Related code and evidence

- `conducto.a2a.create_a2a_app`
- `conducto.a2a.A2AASGI`
- A2A ASGI and public import tests
- [A2A server guide](../a2a-server.md)
