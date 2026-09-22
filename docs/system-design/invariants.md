# System invariants

These rules are more important than individual class shapes. A change that
breaks one needs an explicit architecture decision and corresponding tests.

## Execution

1. **All public capability execution enters `Runtime`.**
   Adapters and gateways never call capability methods directly.
2. **Bindings are opaque and revalidated.**
   Models and remote callers never receive mutable agents or callables.
3. **Local behavior is the reference.**
   A transport preserves validation, security, cancellation, serialization,
   and typed result semantics.

## Authority

4. **Authority never increases.**
   Child scopes, capabilities, deadlines, and budgets are inherited or reduced.
5. **Discovery is not authorization.**
   Agent Cards and catalogs describe availability; runtime policy grants use.
6. **Credentials stop at their boundary.**
   Tokens and private material do not enter Agent Cards, bindings, run metadata,
   results, logs, traces, or task artifacts.
7. **Required audit fails closed.**
   Protected work does not execute if required evidence cannot be recorded.

## State and concurrency

8. **Public snapshots are immutable.**
   Registries, catalogs, tasks, and metadata do not expose mutable internals.
9. **Terminal tasks never restart.**
   Follow-up work creates or continues only valid non-terminal work.
10. **A request executes at most once under its idempotency contract.**
    Duplicate and racing requests cannot duplicate capability side effects.
11. **Independent runs remain isolated.**
    Principals, model references, budgets, cancellation, logs, and providers do
    not leak across concurrent calls.

## Failures and resources

12. **Public failures are typed and sanitized.**
    No raw exception, credential, or sensitive payload crosses a public API.
13. **Resources have explicit owners.**
    The component that creates a provider client, listener, task store, or
    background resource must document who closes it.
14. **Limits are bounded and validated.**
    Payloads, concurrency, retries, deadlines, and delegation loops do not grow
    without an explicit limit.
15. **No required test depends on external infrastructure.**
    Stable tests use deterministic local fixtures; real providers remain opt-in.

## Protocol and packaging

16. **Agent Cards advertise only implemented behavior.**
17. **Wire changes update conformance fixtures intentionally.**
18. **Optional integrations remain optional.**
    Importing core Conducto does not import provider, MCP, ASGI, or cloud
    dependencies.
19. **Installed behavior is tested from the built wheel.**
    Source-checkout success is not enough for public imports or extras.

## Enforcement map

| Invariant area | Primary code | Primary evidence |
|---|---|---|
| Runtime-only execution | `conducto.core.runtime*` | runtime, gateway, A2A, and MCP tests |
| Authority attenuation | `run_context`, `security`, `a2a.runtime` | security, delegation, A2A runtime tests |
| Task atomicity | `transport.tasks`, `a2a.asgi` | A2A transport and ASGI tests |
| Typed failures | `invocation_results`, protocol mappings | golden result-mapping fixtures |
| Optional imports | package `__init__` modules | public API and release-extra tests |
| Installed behavior | release workflows and smoke script | wheel quick start and smoke test |

When changing one of these areas, update both the implementation and its
evidence.
