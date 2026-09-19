# Security and governance

Conducto is built with a governance-first design. The repository's public contracts intentionally validate inputs, constrain discovery metadata, and provide typed execution outcomes rather than uncontrolled exceptions.

## Security model

The current implementation emphasizes:

- explicit schema validation for capability arguments
- typed invocation result envelopes
- runtime validation of A2A security metadata
- prompt-safe rendering of routing metadata
- provider-level contract validation for structured output

## Validation boundaries

### Capability arguments

When `OrchestratorAgent.invoke()` receives arguments, it first validates them against a generated Pydantic model for the target capability.

If validation fails, the result is an `InvocationValidationFailure` containing immutable detail objects rather than a raw stack trace.

This boundary matters because it prevents malformed arguments from reaching execution logic and turns validation into a protocol result.

### A2A Agent Cards

`BaseAgent.get_agent_card()` validates:

- the URL is absolute and uses `http` or `https`
- the preferred transport is not empty or whitespace-padded
- the agent name, version, and description are present and valid
- security scheme definitions are structurally valid
- `security_requirements` are shaped correctly

The code also rejects unknown security types and malformed scope declarations.

## Prompt-safety considerations

The orchestrator serializes agent descriptions and capability metadata into a routing context string. That payload is wrapped in explicit markers and uses escaped bracket characters to reduce the chance that an untrusted description is mistaken for control instructions.

The implementation is designed to respect the principle that agent-authored text is data, not executable logic.

## Provider trust boundaries

The `ModelProvider` contract requires structured output. Before invoking a provider route, the SDK validates that the selected provider supports structured output and that the schema is not empty.

If the provider produces malformed structured data, `MalformedStructuredOutputError` is raised and turned into a `RoutingFailure` result.

## Result handling and safe failure

The orchestrator does not expose arbitrary exceptions to callers as the primary protocol behavior. Instead, it converts:

- unsupported return values
- capability exceptions
- validation failure
- timeouts
- canceled tasks
- missing targets

into typed results.

This makes the calling layer more predictable and more suitable for multi-agent orchestration flows, governance checks, and UI-driven approvals.

## Governance posture

The system is intentionally conservative:

- names and descriptions are required for published metadata
- tool metadata is kept separate from public capability metadata
- capabilities are deterministic and discoverable
- routing metadata is explicit and inspectable

That helps maintain an auditable interface for agent discovery and invocation while keeping the SDK compact.
