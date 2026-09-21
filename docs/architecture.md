# Architecture

Conducto is built as a layered framework. The core can be understood as five
interacting layers: metadata declaration, runtime reflection, local
orchestration, low-level invocation contracts, and provider-backed routing.

## Layered view

```text
+-------------------------------------------------------------------+
| Consumer code                                                     |
| - Agent subclasses                                               |
| - Handwritten capability methods                                  |
| - Orchestrator integration                                        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Declarative metadata layer                                         |
| decorators.py                                                     |
| @a2a_agent, @a2a_capability, @tool                                |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Reflection and card generation layer                               |
| agent.py, registration.py, parameter_schema.py, agent_card.py      |
| - BaseAgent public facade                                         |
| - decorated-method registration and parameter models              |
| - Agent Card construction and validation                          |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Orchestration and invocation layer                                 |
| orchestrator.py, registry.py, invocation.py                        |
| - OrchestratorAgent public facade                                 |
| - thread-safe registry and deterministic routing metadata          |
| - validation, execution, timeout, and cancellation handling        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Low-level contracts and serialization                              |
| invocation_results.py, serialization.py                            |
| - immutable invocation and routing result envelopes                |
| - canonical JSON-compatible result serialization                   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Runtime composition                                                |
| runtime.py, model_config.py, run_context.py                         |
| model_resolution.py, provider_registry.py, model_gateway.py         |
| - Runtime public facade and compatibility exports                  |
| - immutable run configuration and task-local execution context     |
| - provider ownership, model resolution, and policy evaluation      |
| - invocation-scoped model access and call provenance               |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Provider interface and routing contracts                           |
| provider.py                                                       |
| - ModelProvider protocol                                          |
| - StructuredOutputRequest                                         |
| - RoutingSelection                                                |
| - build_routing_schema()                                          |
+-------------------------------------------------------------------+
```

## Agent registration flow

When a subclass of `BaseAgent` is instantiated:

1. `BaseAgent.__init__()` resolves the agent metadata on the class.
2. `registration.py` scans the class for decorated methods.
3. Methods marked with `@a2a_capability` are registered as exposed capabilities.
4. Methods marked with `@tool` are registered as internal tools.
5. `parameter_schema.py` creates one Pydantic model per decorated method and
   derives the method's JSON schema from that same model.
6. The agent can now emit a standards-aligned Agent Card.

This is a reflection-first runtime model: the SDK does not require a separate registry file or a custom schema compiler.

## Agent Card generation

`BaseAgent.get_agent_card()` delegates to `agent_card.py`, which produces a
dictionary shaped according to the pinned A2A 1.0 Agent Card contract.
Important details:

- `supportedInterfaces[].protocolVersion` is pinned to `1.0`
  (`A2A_AGENT_CARD_SPEC_VERSION`)
- `supportedInterfaces[].protocolBinding` is `JSONRPC`
- `skills` contain each capability with a generated stable `id`
- optional `x-conducto.parameters` extension data carries the reflected JSON
  schema for consumer tooling
- security blocks are validated to prevent malformed card definitions

This makes the card both standards-aligned and practically useful for code generation and local dispatch.

## Orchestrator execution flow

`OrchestratorAgent` remains the public facade. `registry.py` owns atomic
registration and deterministic discovery, while `invocation.py` owns argument
validation and execution. A normal local flow looks like this:

1. Register one or more `BaseAgent` instances with `OrchestratorAgent.register_agent()`.
2. Call `route()` or `invoke()` with the agent name, capability, and arguments.
3. The invocation service validates arguments against the generated Pydantic parameter model.
4. It resolves the target capability by name or stable skill ID.
5. It executes the method in a safe wrapper.
6. `serialization.py` serializes the result to a canonical JSON-compatible structure.
7. It returns a typed result from `invocation_results.py` instead of a raw Python object.

This makes it possible to treat capability execution as a controlled protocol event rather than a free-form function call.

## Runtime composition

`Runtime` is the stable orchestration facade rather than the implementation
home for every runtime concern:

- `runtime_errors.py` defines the stable runtime exception hierarchy.
- `model_config.py` owns immutable model references, precedence enums, run
  configuration, metadata validation, and recursive metadata freezing.
- `provider_registry.py` retains provider clients and configurations behind
  credential-free model references.
- `model_resolution.py` applies call, run, agent, and runtime precedence in
  that order, then evaluates capability compatibility and runtime policy.
- `run_context.py` owns task-local activation, cancellation, deadlines,
  invocation state, model-call recording, and credential-free metadata.
- `model_gateway.py` delegates provider completion and typed completion while
  recording ordered provenance and usage.

The facade composes these collaborators and retains compatibility re-exports,
so imports from `conducto`, `conducto.core`, and `conducto.core.runtime` remain
stable. Active contexts use `contextvars`; no global mutable context is shared
between concurrent asyncio tasks.

## Routing and model selection

The `route()` method in `OrchestratorAgent` uses a `ModelProvider` and a
`ModelConfiguration` to choose the best local capability based on structured
output. Each routing attempt takes one metadata snapshot and reuses it for both
the structured-output schema and the prompt message.

The routing contract is explicit:

- a structured request is built with `StructuredOutputRequest`
- `build_routing_schema()` constrains the model output
- `RoutingSelection` expects an `agent_id`, `capability_id`, and `arguments`
- `parse_routing_selection()` validates the provider's output and converts it to a typed model

This design keeps the orchestrator model-driven without embedding raw prompt logic into the runtime path.

## Security and trust boundaries

The architecture intentionally separates:

- trusted orchestrator logic
- untrusted runtime payloads
- prompt-supplied descriptions
- reflected capability metadata

For example, routing metadata is rendered into a prompt-safe context block with clear delimiters and escaping. This reduces the chance that untrusted description text is confused with orchestration instructions.

## Determinism goals

The codebase is designed around deterministic behavior:

- agent registration order is sorted
- capability names are keyed deterministically
- result serialization is canonicalized
- generated Agent Cards are stable and JSON-serializable
- tests include golden fixtures for key public contracts

This is important for acceptance tests, stable fixtures, and cross-language parity.
