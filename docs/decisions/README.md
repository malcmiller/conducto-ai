# Architecture decisions

Architecture Decision Records explain choices that should not be reversed by
accident.

| ADR | Decision | Status |
|---|---|---|
| [0001](./0001-canonical-runtime.md) | All capability execution uses one canonical runtime | Accepted |
| [0002](./0002-capability-first-boundaries.md) | Agents declare capabilities but do not own infrastructure | Accepted |
| [0003](./0003-framework-neutral-asgi.md) | ASGI is the hosting contract; web frameworks remain optional | Accepted |
| [0004](./0004-authority-only-attenuates.md) | Nested authority and budgets only attenuate | Accepted |

## ADR format

New decisions should state:

- **Status**
- **Context**
- **Decision**
- **Consequences**
- **Alternatives considered**
- **Related code and tests**

Use an ADR for durable architectural direction, not an ordinary implementation
detail.
