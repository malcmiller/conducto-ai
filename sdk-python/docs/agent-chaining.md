# Agent chaining

Conducto composes local agents through one `Runtime`, its `AgentGateway`, and
the bounded `run_delegation` loop. The runnable reference is
[`examples/agent_chaining.py`](../examples/agent_chaining.py). It uses a
deterministic local model, requires neither credentials nor network access, and
imports only public names from `conducto`.

```bash
uv build
uv run --no-project --with dist/conducto_ai-*.whl python examples/agent_chaining.py
uv run --no-project --with dist/conducto_ai-*.whl python scripts/smoke_test.py
```

On PowerShell, set `$wheel = (Get-ChildItem dist\*.whl | Select-Object -First
1).FullName` and substitute `$wheel` for the wheel path.

## Configuration and call modes

Declare a parent agent's allowed *capability families* with `ToolboxPolicy` and
`CapabilityUse`; do not declare a documentation agent ID or endpoint. A
`CapabilityUse` can be required or optional. Required discovery failure stops
before a model call; optional failure produces an empty or partial toolbox.

Use `OrchestratorAgent.route()` when a model must select the parent agent. Use
`OrchestratorAgent.invoke()` when the application already knows the target;
this avoids a routing decision, but the target may still opt into model-selected
delegation. Each `DelegationConfig` bounds turns, sequential tool calls, depth,
timeout, token and cost budgets, and serialized tool results.

## Lifecycle and snapshots

Register documentation providers in the `AgentRegistry` owned by the runtime.
Registering a compatible provider changes later discovery boundaries without
recreating callers. A toolbox is immutable for one model turn: a concurrent
registration, removal, disable, drain, or replacement cannot alter tools
already shown to that model. The gateway revalidates lifecycle, schema,
authority, deadlines, cycle/depth, and shared budget before dispatch, so an
accepted in-flight binding follows the registry lifecycle guarantees.

Adding a provider makes another compatible implementation discoverable.
Changing an agent's `ToolboxPolicy` changes what that parent is authorized to
discover; it is an application policy change, not provider registration.

## Security, failures, and troubleshooting

Tool identifiers are opaque and valid only for their originating snapshot.
Unknown, forged, stale, foreign, and replayed identifiers are typed delegation
failures and never dispatch business logic twice. Tool output is returned to a
model as structured untrusted data, not as executable instructions. Child
authority and deadlines are attenuated from the parent; approval, scope, and
mandatory audit enforcement remain in the normal invocation pipeline.

Every result retains correlation ID, run/task lineage, model provenance, and
ordered usage. Default structured logs omit prompts, responses, arguments,
results, credentials, approvals, bindings, and tracebacks. Inspect the typed
`DelegationOutcome.code` or `InvocationResult` instead of parsing logs:
malformed model output, provider failure, timeout, cancellation, unavailable
tools, budget exhaustion, child failure, and invalid terminal values all remain
non-success outcomes.

If a tool is unexpectedly absent, check its capability ID/tags, caller
authority, lifecycle/health, schema compatibility, and the active
`ToolboxPolicy`. If a model call fails before dispatch, check native structured
output/tool-calling support and the configured model reference.
