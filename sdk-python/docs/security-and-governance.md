# Security and governance

Capability authorization is enforced in `conducto.security`, before capability
business logic and independently of any transport. Use `Principal` and
`AuthorizationContext` to pass authenticated identity facts at invocation time.

Scopes are opaque, case-sensitive values. Multiple declarations are cumulative
and every scope must be present. `require_approval(role, condition)` accepts an
application callback receiving the immutable context and validated argument
mapping. The callback is never an expression string; callback failure fails
  to close.

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
