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
