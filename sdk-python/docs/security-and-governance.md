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

## Portable approval tokens

Approval decisions can cross process boundaries as compact RFC 7515 JWS tokens
using the pinned `conducto.approval+jwt;v1` type and ES256. Applications own
private-key custody through `ES256Signer`; runtimes receive only trusted public
keys through `StaticApprovalKeyResolver` or an application resolver. Configure
one current signing key and retain overlapping active verification keys during
rotation. Revoked keys must stop verifying immediately according to the
application's resolver cache policy.

Keep clocks synchronized. The verifier uses injected UTC time, bounded skew,
and a maximum 15-minute lifetime. Replay stores must be durable and provide an
atomic consume operation for `jti`; the challenge store must provide an atomic
approved-to-completed transition. Tokens, private keys, complete claims, and
protected capability arguments must not be logged or included in result
envelopes.
