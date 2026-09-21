# Providers and models

The provider subsystem separates a model's public reference, its
application-owned client, selection policy, and invocation-scoped use. Agent
code depends on Conducto contracts rather than vendor SDK types.

## Provider contract

`ModelProvider` is a structural protocol. A provider advertises immutable
`ProviderCapabilities` and implements asynchronous `complete(...)`.

Provider-neutral request and result types include:

- `ChatMessage` and typed content parts
- `GenerationOptions`
- `StructuredOutputRequest`
- model-facing tools and `ToolResultMessage`
- `ProviderResult`, `Usage`, finish reason, and acceptance state
- typed diagnostics and provider error categories

`validate_provider_contract()` and the reusable conformance helpers verify
capability claims, structured output, schema support, usage, cancellation, and
error behavior. A provider that cannot satisfy a required capability fails
before dispatch.

## Provider registration

`ProviderRegistry` binds a credential-free `ModelReference` to:

- a provider client
- immutable model configuration
- provider type
- ownership metadata
- availability evaluation

Trusted application code can register a preconstructed client or register an
allowlisted `ProviderFactory` and typed `ProviderClientConfig`. Construction
happens outside the registry lock and publication is atomic.

Agents never receive credentials, factories, or raw registry entries. See
[provider registration](./provider-registration.md) for replacement,
deregistration, snapshots, ownership, and typed failures.

## Model resolution

`ModelResolver` applies one stable precedence order:

1. call override
2. run override
3. agent default
4. runtime default

It resolves the selected reference through `ProviderRegistry`, verifies
required provider capabilities, and evaluates the optional application
`ModelPolicy`. Its public `ResolvedModel` contains only credential-free
identity, provider, and resolution source; the client binding remains
runtime-private.

No fallback to a lower-precedence model occurs merely because the selected
reference is unknown, unavailable, denied, or incompatible. Those conditions
are explicit failures.

## Invocation-scoped model access

Within a capability, `require_run_context().models.require()` returns a
`ModelGateway` bound to that run. The gateway:

- resolves an optional per-call model reference
- caps provider timeout by the run deadline
- cooperates with cancellation
- requires native structured-output semantics
- validates typed responses
- records ordered model provenance and usage
- emits safe model-selection and usage events

Provider access remains behind the registry/runtime, while client shutdown
follows the binding's caller-owned or runtime-owned declaration. A
`ModelGateway` is a short-lived capability to use a provider under current run
policy, not a client wrapper to store globally.

## Structured output and tools

`StructuredOutputRequest` is mandatory at the Conducto provider boundary.
Provider responses must satisfy the requested schema and declared schema
dialect/features. Required structured output never silently degrades to prose
parsing.

Tool descriptions are provider-neutral. The delegation layer projects
gateway-authorized tools into the model call and converts typed child
invocation results back into bounded `ToolResultMessage` values.

## Adapter guidance

Provider adapters should use supported vendor clients for protocol mechanics
while keeping those types behind `ModelProvider`. Optional dependencies must
not make `import conducto` construct clients, load credentials, contact a
network, or require every provider SDK.

Required tests use deterministic fake providers. Live model, daemon, and cloud
tests remain explicitly opt-in.
