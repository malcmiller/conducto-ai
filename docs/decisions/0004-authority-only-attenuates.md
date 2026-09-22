# ADR 0004: Allow authority only to attenuate

**Status:** Accepted

## Context

Agent calls can nest across local and remote boundaries. Each child call may
receive scopes, capability allowlists, deadlines, call limits, token limits,
and cost limits from several sources. Choosing the broader value would let a
transport or child call gain authority.

## Decision

Effective child authority is the intersection or stricter bound of parent,
authenticated, request, and server-owned limits. Already consumed budget is
never restored. Request-local ledgers do not mutate resolver-owned ledgers.

## Consequences

- Missing child restrictions inherit the bounded parent value.
- Explicit child restrictions may reduce but never increase authority.
- Attenuation must use remaining budget snapshots, not original configured
  limits.
- Concurrency tests must prove ledgers do not leak or restore resources.

## Alternatives considered

- Let request metadata replace authenticated authority. Rejected because the
  caller could self-grant authority.
- Ignore stricter request limits. Rejected because callers must be able to
  constrain one invocation below their maximum grant.

## Related code and evidence

- `AuthorizationContext`
- `DelegationBudget`
- `delegate_context`
- `A2ARuntimeHandler`
- Security, delegation, gateway, and A2A runtime tests
