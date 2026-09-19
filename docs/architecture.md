# Architecture

Conducto is built as a layered framework. The core can be understood as four interacting layers: metadata declaration, runtime reflection, local orchestration, and provider-backed routing.

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
| agent.py                                                          |
| - BaseAgent                                                       |
| - registered_methods                                              |
| - get_agent_card()                                                |
| - parameter schema generation                                     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Orchestration and invocation layer                                 |
| orchestrator.py                                                   |
| - registry of agents and capabilities                             |
| - argument validation                                             |
| - timeout/cancelation handling                                    |
| - deterministic routing metadata                                  |
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
2. The class is scanned for decorated methods.
3. Methods marked with `@a2a_capability` are registered as exposed capabilities.
4. Methods marked with `@tool` are registered as internal tools.
5. Pydantic models are created for each capability's arguments.
6. The agent can now emit a standards-aligned Agent Card.

This is a reflection-first runtime model: the SDK does not require a separate registry file or a custom schema compiler.

## Agent Card generation

`BaseAgent.get_agent_card()` produces a dictionary shaped according to the A2A Agent Card contract. Important details:

- `protocolVersion` is pinned to `0.3.0` (`A2A_AGENT_CARD_SPEC_VERSION`)
- `skills` contain each capability with a generated stable `id`
- `x-conducto.parameters` carries the reflected JSON schema for consumer tooling
- security blocks are validated to prevent malformed card definitions

This makes the card both standards-aligned and practically useful for code generation and local dispatch.

## Orchestrator execution flow

The orchestrator is responsible for runtime selection and invocation. A normal local flow looks like this:

1. Register one or more `BaseAgent` instances with `OrchestratorAgent.register_agent()`.
2. Call `route()` or `invoke()` with the agent name, capability, and arguments.
3. The orchestrator validates arguments against the generated Pydantic parameter model.
4. It resolves the target capability by name or stable skill ID.
5. It executes the method in a safe wrapper.
6. It serializes the result to a JSON-compatible structure.
7. It returns a typed `InvocationResult` instead of a raw Python object.

This makes it possible to treat capability execution as a controlled protocol event rather than a free-form function call.

## Routing and model selection

The `route()` method in `OrchestratorAgent` uses a `ModelProvider` and a `ModelConfiguration` to choose the best local capability based on structured output.

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
