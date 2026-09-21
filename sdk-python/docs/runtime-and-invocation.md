# Runtime and invocation

`Runtime` is the composition root for execution. It owns long-lived
collaborators and creates invocation-scoped state; it is not the home for
agent reflection, registry indexing, provider implementations, or workflow
business logic.

## Runtime-owned collaborators

A runtime composes:

- `ProviderRegistry` and `ModelResolver`
- `AgentRegistry`
- model and gateway policy
- `SecurityPipeline`
- gateway limits, preferred providers, and binding integrity keys
- capability execution locks

Applications should construct and retain the runtime. Package import creates
none of these resources.

## `RunContext`

Each invocation receives a `RunContext` containing:

- run, task, parent-task, and correlation identifiers
- target agent identity
- resolved model binding and policy context
- authorization context
- deadline and cancellation state
- allowed capabilities
- delegation path and shared budget
- ordered model-call provenance and aggregate usage
- invocation-scoped model and agent gateways

The active context is propagated with `contextvars`. Use
`get_run_context()` when absence is valid and `require_run_context()` when the
operation must be inside a runtime invocation. Do not cache a context or its
gateways beyond the invocation.

## Invocation pipeline

`Runtime.invoke()` delegates mechanics to `invocation.invoke_agent()`:

1. derive or attenuate authorization from the parent context
2. resolve the requested decorated capability
3. calculate the effective deadline and cancellation behavior
4. resolve the model when the agent or call requires one
5. create and activate the run context
6. validate arguments with the reflected Pydantic model
7. enforce authorization, scopes, approval, and required audit delivery
8. execute synchronous work on the bounded worker path or await asynchronous
   work
9. serialize supported return values
10. attach correlation, lineage, model provenance, and usage
11. return a typed result envelope

Gateway calls re-enter this same pipeline after binding checks. There is no
lighter internal dispatch path for delegated capabilities.

## Deadlines and cancellation

Run and call timeouts become monotonic deadlines. Child work receives the
earliest applicable deadline. Model calls cap provider timeouts by the
remaining run time.

Cancellation is cooperative for asynchronous and provider work. Synchronous
Python cannot be forcibly stopped safely; its result is discarded if the
public invocation has already timed out or been cancelled. Synchronous
capabilities use runtime-scoped locks rather than process globals.

## Security and approval

`SecurityPipeline` runs before protected business logic. Missing identity,
insufficient scopes, approval requirements, replay, invalid signatures, and
required audit delivery failures become explicit invocation outcomes.

`resume_approval()` and `resume_approval_token()` re-enter the normal
invocation path after the application-owned approval state is validated.
Security stores, key custody, and durable audit persistence remain application
responsibilities. See [security and governance](./security-and-governance.md).

## Typed results

Normal protocol outcomes are values rather than exceptions:

- `InvocationSuccess`
- validation and target-not-found failures
- authorization, approval, and audit failures
- binding, stale-binding, lifecycle, and schema failures
- timeout and cancellation
- budget and delegation failures
- capability execution or unsupported-return failures

Programming/configuration errors can still raise exceptions where continuing
would hide an invalid setup. Callers should branch on result type rather than
parse messages or logs.

## Provenance and logging

`InvocationMetadata` preserves run/task lineage, correlation, model calls, and
usage without storing provider clients or credentials. Parent and child
metadata are combined in execution order.

Structured logging is diagnostic and deliberately excludes prompts, model
responses, capability arguments/results, credentials, approval material, and
tracebacks by default. Audit evidence is a separate security contract. See
[logging](./logging.md) for the event schema.
