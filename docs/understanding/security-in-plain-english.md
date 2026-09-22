# Security in plain English

## The central rule

A child call may keep or reduce the caller's authority. It may never gain more.

Think of authority as a travel pass. If the original pass allows weather
lookups only, a downstream agent cannot turn it into permission to approve
payments.

## Five separate checks

| Check | Question |
|---|---|
| Identity | Who is making the request? |
| Scope | Which capabilities may they use? |
| Approval | Does a human or external decision need to approve this action? |
| Budget | How much delegation, time, tokens, or cost remains? |
| Audit | Can required evidence of the decision be recorded? |

These checks are separate so passing one does not silently pass the others.

## Discovery is still not authorization

An Agent Card is public service metadata. It must not contain credentials,
private keys, tokens, or permission grants. Authorization happens when a
request enters the runtime.

## Limits travel with the work

Conducto carries deadlines, cancellation, allowed capabilities, and delegation
budgets through child calls. When two limits meet, the stricter one wins.

```text
caller allows 10 calls
request asks for 3 calls
effective limit is 3 calls
```

Already consumed authority is never restored at a network boundary.

## Failures are explicit

Conducto returns typed outcomes for conditions such as:

- invalid arguments;
- target not found;
- authorization denied;
- approval required;
- deadline exceeded;
- cancellation;
- audit unavailable; and
- capability or internal failure.

Public failures do not expose raw exceptions, credentials, or sensitive
arguments.

## Under the hood

- `AuthorizationContext` carries authenticated principal facts.
- Security decorators such as `require_scope` declare capability policy.
- `SecurityPipeline` evaluates guardrails.
- `DelegationBudget` provides a shared, thread-safe budget ledger.
- `authenticate_incoming_request()` combines application-owned token and mTLS
  validation facts at the A2A boundary.
- `A2AAuthenticatedIdentity` passes immutable authority into the runtime.

See [security and governance](../security-and-governance.md) for exact
contracts.

Next: understand the difference between identity, roles, exact scopes,
capability allowlists, approvals, and admin authority in
[authentication, authorization, and scopes](./authentication-and-scopes.md).
