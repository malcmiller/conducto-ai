# Orchestration and delegation

Conducto supports three distinct coordination modes. Keeping them separate
prevents a known target, a top-level routing choice, and a nested model/tool
loop from collapsing into one ambiguous API.

The `conducto.core.delegation` package owns the public loop and immutable
configuration/outcome contracts. Its state-machine models, child-result
mapping, argument validation, and execution coordination have separate
implementation boundaries. Tool discovery remains in `gateway_tools`;
provider-native decision parsing remains in `provider`. Neither delegation
nor a model output owns target authorization.

## Direct orchestration

Use `OrchestratorAgent.invoke()` when the application already knows the agent
and capability. The orchestrator resolves the registered target and delegates
execution to `Runtime`.

Direct invocation is preferred for deterministic application workflows. It
does not spend a model call selecting a target.

## Top-level model routing

Use `OrchestratorAgent.route()` when a model must select one registered local
capability from a natural-language request.

The orchestrator:

1. captures one deterministic routing metadata snapshot
2. builds a constrained routing JSON Schema
3. requests structured output from the selected model
4. parses an `agent_id`, `capability_id`, and argument mapping
5. invokes the selected target through the normal runtime path
6. merges routing-model usage and provenance into the final result

`route()` is the top-level routing API for choosing the first target. It is not
the capability gateway and does not manage recursive tool loops.

Register models on `ProviderRegistry`, pass that registry to `Runtime`, and
construct `OrchestratorAgent(runtime=runtime, model_reference="router")`.
Neither agents nor routing calls accept raw provider clients or provider model
configurations. A call selects a different registered model through
`model_reference` or `RunConfig`, preserving normal policy and provenance.
Use `invoke()` for direct calls; there is no `invoke_capability()` alias.

## Nested delegation

An agent opts into model-selected child calls with immutable
`DelegationConfig` and explicitly calls `BaseAgent.run_delegation()` from an
active capability.

```python
class ResearchAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            delegation_config=DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"docs.search"})),)
                ),
                max_model_turns=4,
                max_tool_calls=2,
            )
        )
```

Each turn builds one toolbox snapshot and accepts exactly one structured
decision:

- a terminal response matching the requested Pydantic response type, or
- one tool call whose ID belongs to that snapshot

Malformed, mixed, unknown, stale, foreign, or replayed decisions fail with a
typed `DelegationOutcome`. Tool calls are sequential and are not implicitly
retried.

Model-selected arguments are immutable snapshots. Pre-dispatch validation
normalizes their frozen JSON containers without modifying them, so nested
arrays retain JSON array semantics. Integer, number, and boolean arguments
are validated explicitly; numeric bounds and `multipleOf` remain enforced
before a delegation call is reserved.

## Budgets and authority

The active `RunContext` carries the shared deadline, cancellation state,
delegation path, and remaining depth/call/token/cost budget. Child calls
inherit or reduce these values. They cannot extend the deadline, add scopes,
reset the path, or replenish a budget.

Cycle detection uses the agent/capability path. A repeated frame or exhausted
depth stops before dispatch. A duplicate model tool-call ID terminates the
loop and records the earlier safe envelope only in provenance.

## Child failures and fallback

Child invocation results stay typed. By default, a child validation,
authorization, approval, binding, lifecycle, timeout, cancellation, budget,
or execution failure terminates delegation.

An explicit `DelegationFallbackPolicy` may permit another model turn for
eligible safe failure categories. A later terminal response is represented as
a typed fallback success and retains the child failure in provenance; it is
not rewritten as an ordinary success.

## Choosing the right API

| Need | API |
|---|---|
| Known local target | `OrchestratorAgent.invoke()` or `Runtime.invoke()` |
| Model selects the initial registered target | `OrchestratorAgent.route()` |
| Agent discovers an authorized capability | `RunContext.gateway` |
| Agent runs a bounded model/tool loop | `BaseAgent.run_delegation()` |
| Application implements a fixed multi-step workflow | Explicit application workflow calling `Runtime`/gateway |

See [agent chaining](./agent-chaining.md) for a runnable installed-package
example and [gateway and discovery](./gateway-and-discovery.md) for binding
semantics.
