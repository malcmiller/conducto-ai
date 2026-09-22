# Authentication, authorization, and scopes

Security terms are easy to blend together. Conducto keeps them separate so
that proving who a caller is does not silently grant permission to do
everything.

## The short version

| Concept | Question it answers |
|---|---|
| Authentication | Who is making the request? |
| Principal | What verified identity facts may policy use? |
| Authorization | May this principal perform this operation? |
| Scope | Which exact permission strings does the principal hold? |
| Role | Which application-defined group labels describe the principal? |
| Capability allowlist | Which agent capabilities may this run discover or delegate to? |
| Approval | Must a separate authorized decision allow this particular action? |

Passing one check does not bypass the others.

## The application authenticates; Conducto carries verified facts

Conducto does not own user accounts, passwords, login pages, or identity
provider configuration. The application validates a credential and constructs
immutable identity facts:

```python
from conducto.security import AuthorizationContext, Principal


authorization = AuthorizationContext(
    principal=Principal(
        subject_id="user-123",
        issuer="https://identity.example",
        audience="inventory-agent",
        roles=frozenset({"inventory-reader"}),
        scopes=frozenset({"inventory:read"}),
    ),
    task_id="task-42",
    correlation_id="request-42",
)
```

Raw tokens and credentials do not belong in `Principal`,
`AuthorizationContext`, agent metadata, logs, or Agent Cards.

For inbound A2A requests, the application supplies an identity resolver. It
may validate OAuth tokens and mTLS evidence before returning an
`A2AAuthenticatedIdentity`. A trusted network connection or client certificate
does not automatically grant application scopes.

## Scopes are exact permissions

A capability declares required scopes with `@require_scope`:

```python
from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.security import require_scope


@a2a_agent(
    name="InventoryAgent",
    version="1.0.0",
    description="Manages inventory.",
)
class InventoryAgent(BaseAgent):
    @a2a_capability(
        name="inventory.adjust",
        description="Adjusts stock for one SKU.",
    )
    @require_scope("inventory:write")
    def adjust(self, sku: str, quantity: int) -> dict[str, object]:
        return {"sku": sku, "quantity": quantity}
```

Conducto's scope rules are deliberately simple:

- values are opaque, exact, and case-sensitive;
- every required scope must be present;
- repeated `@require_scope` declarations are cumulative;
- `"inventory:*"` does not match `"inventory:write"`; and
- `"*"`, `"admin"`, and an `admin` role are not automatic bypasses.

A protected capability invoked without an authorization context fails closed.
An unprotected capability has no scope requirement, so a local application may
invoke it without constructing identity. An inbound A2A request still passes
through the required application-owned identity resolver.

## Roles do not grant scopes

Roles are verified labels available to application policy and approval flows.
Conducto does not contain a built-in role-to-scope mapping.

For example, this principal does **not** satisfy
`@require_scope("inventory:write")`:

```python
Principal(
    subject_id="user-123",
    issuer="https://identity.example",
    audience="inventory-agent",
    roles=frozenset({"admin"}),
    scopes=frozenset(),
)
```

If an identity provider says an administrator should receive all current
permissions, the application must map that trusted role or claim to the
complete explicit scope set.

## What “full admin access” means today

An administrator can receive all known scopes:

```python
admin = Principal(
    subject_id="admin-1",
    issuer="https://identity.example",
    audience="conducto",
    roles=frozenset({"admin"}),
    scopes=frozenset(
        {
            "inventory:read",
            "inventory:write",
            "billing:read",
            "billing:write",
        }
    ),
)
```

This is full **explicit** scope authority, not a superuser bypass. When a new
protected scope is introduced, the application's admin mapping must be updated
if administrators should receive it.

Even a fully scoped administrator does not automatically bypass:

- `@require_approval`;
- capability allowlists;
- destination trust policy;
- deadlines or cancellation;
- delegation depth, call, token, or cost budgets; or
- required security-audit delivery.

This separation prevents one broad label from disabling unrelated controls.

## Capability allowlists are separate from scopes

Scopes answer whether a principal has permission. `allowed_capabilities`
limits which capabilities a run may discover or call through gateway and
delegation flows.

Allowlist entries may be a capability ID such as `"inventory.adjust"` or an
exact agent-qualified ID such as `"InventoryAgent:inventory.adjust"`.

| Value | Meaning at gateway and A2A authority boundaries |
|---|---|
| `None` | No additional capability-allowlist restriction |
| `frozenset()` | No capability is allowed |
| `frozenset({"inventory.adjust"})` | Only matching capability IDs are allowed |
| `frozenset({"InventoryAgent:inventory.adjust"})` | Only that exact agent capability is allowed |

`None` does not skip required scopes or other security checks. It only means
that this particular allowlist adds no restriction.

For direct application-owned `Runtime.invoke()` calls, the application already
selects the root target. The allowlist carried by the resulting run context
constrains gateway discovery and child calls. The inbound A2A handler also
checks its effective allowlist against the requested root capability before
runtime invocation.

## A request can only reduce authority

For inbound A2A, resolver-owned and request-supplied capability allowlists are
intersected:

```text
authenticated: {"inventory.read", "inventory.adjust"}
requested:     {"inventory.read"}
effective:     {"inventory.read"}
```

If either side omits a restriction, the other side's restriction remains. If
both omit it, the effective allowlist is `None`. A request can never add
authority that the authenticated identity resolver did not grant.

Nested calls follow the same principle:

- child scopes must be a subset of parent scopes;
- child roles must be a subset of parent roles;
- child capability allowlists must preserve or reduce the parent's list; and
- child deadlines and budgets may only become tighter.

## Approval is not an admin override

`@require_approval("reviewer")` means the action requires a separate approval
decision associated with that role. Merely placing `"reviewer"` or `"admin"`
in the invoking principal's roles does not mark the request approved.

Approval challenges and decisions follow their own authenticated, replay-safe
lifecycle. This supports separation of duties: a caller may have permission to
request an action without having authority to approve it.

## A practical policy checklist

For each protected capability, decide:

1. Which credential authenticates the caller?
2. Which issuer and audience are trusted?
3. Which exact scopes does the capability require?
4. How does the application map trusted claims or roles to those scopes?
5. Should a capability allowlist narrow discovery or delegation?
6. Does the action require separate approval?
7. Which deadline and delegation budgets apply?
8. Must audit delivery succeed before execution?

For implementation contracts, token validation, mTLS, approvals, audit
evidence, and A2A identity resolution, continue to
[Security and governance](../security-and-governance.md).
