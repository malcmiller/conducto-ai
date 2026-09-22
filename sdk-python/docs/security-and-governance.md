# Security and governance

Capability authorization is enforced in `conducto.security`, before capability
business logic and independently of any transport. Use `Principal` and
`AuthorizationContext` to pass authenticated identity facts at invocation time.

Scopes are opaque, case-sensitive values. Multiple declarations are cumulative
and every scope must be present. `require_approval(role, condition)` accepts an
application callback receiving the immutable context and validated argument
mapping. The callback is never an expression string; callback failure fails
closed.

The runtime pipeline resolves the target, establishes `RunContext`, checks
identity and scopes, validates arguments, evaluates approval conditions, and
only then invokes the capability. A protected call without context returns a
typed authorization failure. A call requiring human approval returns an
`InvocationApprovalRequired` result containing an immutable, display-safe
`ApprovalChallenge`.

`InMemoryApprovalStore` is a thread-safe reference implementation for tests and
single-process applications. Production applications own durable persistence,
distributed locking, authentication, and transport integration. Approval
payloads intentionally do not contain credentials, raw tokens, claims,
protected arguments, or tracebacks.

## MCP stdio identity

MCP export never relaxes these rules. An `McpStdioServer` requires an explicit
`Principal` or principal resolver, so there is no anonymous privileged default
and protected capabilities fail closed without identity. `tools/list` returns
only policy-admitted capabilities eligible for that principal, every
`tools/call` runs through `Runtime.invoke()` with the same authorization,
approval, and audit behavior, and approval-required results stay non-success
with a display-safe challenge reference. Mapped MCP failures carry fixed safe
messages and reason codes, never exceptions, tracebacks, credentials,
arguments, bindings, or internal endpoints. See
[MCP tool export](./mcp-export.md).

## Portable approval tokens

Approval decisions can cross process boundaries as compact RFC 7515 JWS tokens
using the pinned `conducto.approval+jwt;v1` type and ES256. Applications own
private-key custody through `ES256Signer`; runtimes receive only trusted public
keys through `StaticApprovalKeyResolver` or an application resolver. Configure
one current signing key and retain overlapping active verification keys during
rotation. Revoked keys must stop verifying immediately, according to the
application's resolver cache policy.

Keep clocks synchronized. The verifier uses injected UTC time, bounded skew,
and a maximum 15-minute lifetime. Replay stores must be durable and provide an
atomic consume operation for `jti`; the challenge store must provide an atomic
approved-to-completed transition. Tokens, private keys, complete claims, and
protected capability arguments must not be logged or included in result
envelopes.

## Security audit evidence

`conducto.security` provides a vendor-neutral, versioned audit envelope for
security evidence. It is intentionally independent of Python `logging`,
OpenTelemetry, SIEM products, and diagnostic filtering. Applications attach an
asynchronous `AuditSink` through `AuditEmitter`; applications own durable
storage, retention, access control, encryption, integrity protection, export,
and deletion.

The v1 taxonomy includes authorization allow/deny (including missing context
and insufficient scope reason codes), approval requested/approved/denied,
signature verification/rejection, replay rejection, and protected execution
accepted/started/completed/failed. Events contain stable identifiers for the
subject, issuer, audience, task, challenge, agent, capability, policy,
correlation, resource, and optional trace/span IDs. `sequence` guarantees
causal ordering for a task only; concurrent tasks have no global ordering.
`event_id` and `idempotency_key` let sinks detect duplicate deliveries.

Events exclude token bodies, credentials, signatures, private keys, prompts,
arguments, result bodies, approval payloads, exception details, and
tracebacks. Extension fields are scalar-only and reject prohibited names such
as `token`, `signature`, `payload`, `argument`, or `traceback` before delivery.
Within a schema version, only additive optional fields are allowed; breaking
changes require a new schema version and compliant sinks reject unknown
versions.

Pre-execution authorization, approval-request, and execution-accepted evidence
are required. The default `AuditDeliveryPolicy` is fail-closed: protected
business logic cannot begin until its required event is accepted by the sink or
the explicitly bounded buffer. Fail-open is an explicit application-owned
choice for noncritical events only; failures still produce an emergency
diagnostic signal and are never silently downgraded. Timeouts, buffers, and
retry attempts are bounded; each buffered event is retried once at `flush`,
then accounted as dropped. `InMemoryAuditSink` and `FailingAuditSink` are
network-free reference and conformance fixtures, not production persistence.

## OAuth token exchange and mTLS transport security

`conducto.security.trust` defines immutable, versioned `TrustPolicy` snapshots
(`IssuerPolicy`, `AudiencePolicy`, `ScopePolicy`, `ClockSkewPolicy`,
`CertificatePolicy`). A snapshot's `version` is bound to every `ValidatedIdentity`
produced under it, so configuration refresh cannot mutate in-flight
authorization.

`conducto.security.tokens` defines provider-neutral `TokenAcquirer` and
`TokenValidator` protocols, an RFC 8693 `TokenExchangeRequest`/`AcquiredToken`
pair, and `JWTBearerTokenValidator`, a strict RS256/ES256 bearer-token
validator built directly on `cryptography` primitives. Algorithms, issuers,
audiences, and verification keys are always resolved from the trust policy and
an application-owned `JWKSKeyResolver`, never from the token under validation.
`attenuate_scopes` enforces that a nested call's requested authority is always
a subset of both the caller's incoming scopes and the destination's allowed
scopes, raising `ScopeAttenuationError` otherwise. `TokenCache` bounds cache
size and lifetime, refreshes before expiry using a configured skew, and uses a
per-key lock so concurrent callers sharing one cache key perform one bounded
acquisition instead of a refresh stampede.

`conducto.transport.tls` builds application-owned client/server
`ssl.SSLContext` objects for mTLS. Hostname verification and certificate-chain
validation are always enabled; there is no parameter or code path that
disables them. Certificate and key material is supplied as PEM bytes by the
application and loaded through private, immediately-removed temporary files.

`conducto.transport.auth` binds these contracts to one inbound request:
`authenticate_incoming_request` treats mTLS workload authentication and OAuth
delegated-subject authorization as independent checks — a trusted client
certificate never substitutes for bearer-token validation and vice versa — and
returns a `Principal`/`AuthorizationContext` pair reusable by Story 2
guardrails. `build_delegated_token_request` attenuates scopes before building
an outgoing RFC 8693 exchange request for a nested call. Supplying an
`AuditEmitter` records token-exchange, token-validation, mTLS, and delegation
outcomes without logging protected material.

These contracts are deterministic, local-fixture reference implementations:
tests build local RSA/EC keys and sign fixture JWTs directly, without Azure,
MSAL, Authlib, or internet access. Provider-specific flows (Azure Identity,
MSAL, on-behalf-of) are expected to implement `TokenAcquirer`/`TokenValidator`
behind these same contracts in a later adapter, without changing this module.
