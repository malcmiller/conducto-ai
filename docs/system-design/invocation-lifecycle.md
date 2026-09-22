# Invocation lifecycle

## Plain-language flow

A request does not jump directly to an agent method. Conducto turns it into a
validated, authorized, bounded invocation.

```text
find target
  → bind exact capability
  → validate request
  → check identity and policy
  → establish limits
  → execute
  → serialize a typed outcome
```

## Local invocation

```mermaid
sequenceDiagram
    participant App
    participant Runtime
    participant Security
    participant Agent

    App->>Runtime: invoke(agent, capability, arguments)
    Runtime->>Runtime: reflect and validate arguments
    Runtime->>Security: scopes, approval, audit
    Security-->>Runtime: allowed or typed outcome
    Runtime->>Agent: invoke validated capability
    Agent-->>Runtime: value or failure
    Runtime-->>App: InvocationResult
```

## Gateway invocation

The gateway adds selection and an opaque binding:

```mermaid
sequenceDiagram
    participant Caller
    participant Gateway
    participant Runtime
    participant Agent

    Caller->>Gateway: lookup agent and capability
    Gateway-->>Caller: opaque binding
    Caller->>Gateway: invoke binding
    Gateway->>Runtime: revalidate target and authority
    Runtime->>Agent: invoke capability
    Agent-->>Runtime: result
    Runtime-->>Caller: typed result
```

The binding prevents model-facing or remote callers from holding mutable agent
objects or arbitrary callables.

## Inbound A2A invocation

```mermaid
sequenceDiagram
    participant Client
    participant ASGI as A2A ASGI adapter
    participant Handler as A2A runtime handler
    participant Identity as Identity resolver
    participant Tasks as Task repository
    participant Runtime

    Client->>ASGI: JSON-RPC SendMessage
    ASGI->>Tasks: create or claim task
    ASGI->>Handler: validated message and transport facts
    Handler->>Identity: authenticate request
    Identity-->>Handler: immutable authority
    Handler->>Runtime: canonical invoke
    Runtime-->>Handler: InvocationResult
    Handler-->>ASGI: mapped A2A task
    ASGI->>Tasks: persist terminal state
    ASGI-->>Client: JSON-RPC task response
```

Raw credentials stop at the identity resolver. The runtime receives identity
facts, not tokens.

## Deadline and cancellation behavior

Deadlines can only become shorter through nested calls. Cancellation uses a
shared cooperative state so the runtime, provider, gateway, and capability can
observe the same request to stop.

Different events remain distinct:

- caller coroutine cancellation;
- explicit application cancellation;
- runtime timeout;
- remote A2A task cancellation; and
- later operational drain/shutdown.

Adapters must map these events explicitly rather than hiding them as generic
failures.

## Result behavior

The runtime always returns an `InvocationResult` subtype. Protocol adapters
map these typed results into safe protocol outcomes. Raw exceptions never
become public wire messages.

See [runtime and invocation](../runtime-and-invocation.md) and
[A2A server](../a2a-server.md) for exact behavior.
