# Provider type and model registration (Story 6.2)

`ProviderRegistry` maps two distinct, credential-free registration operations
onto immutable, published bindings. Applications register **how** to build a
provider family once, then bind **one configured client** to each
credential-free model reference agent and runs use.

Model resolution (`ProviderRegistry.resolve`) only ever selects an
already-published binding — it never constructs a client on the call path.

## Registering a provider type (factory) once

Trusted application code registers an allowlisted factory for one provider
family. The registry never imports modules, scans entry points, or executes
configuration-supplied code; the factory is a plain object your code
constructs and hands to the registry.

```python
from conducto import ProviderClientConfig, ProviderRegistry


class OllamaFactory:
    def create(self, configuration: ProviderClientConfig) -> "OllamaClient":
        return OllamaClient(
            endpoint=configuration.endpoint,
            credential_ref=configuration.credential_ref,
            **configuration.provider_defaults,
        )


registry = ProviderRegistry()
registry.register_provider_type("ollama", OllamaFactory())
```

`ProviderClientConfig` is the single typed source of truth for
connection-affecting settings: `endpoint`, `credential_ref` (an opaque
reference such as an environment variable or secret-store key — never the raw
secret), `transport`, `tls`, `proxy`, and `provider_defaults`.

## Binding a model reference to a factory-constructed client

```python
from conducto import ModelConfiguration

registry.register_provider(
    "local-llama",
    provider_type="ollama",
    configuration=ProviderClientConfig(endpoint="http://localhost:11434"),
    model_configuration=ModelConfiguration(provider="ollama", model="llama3"),
)
```

The factory runs exactly once, outside any registry lock, so a slow or
failing construction never blocks concurrent resolution or unrelated
registrations. The registry only publishes the binding after construction,
structural validation, and capability validation all succeed — other runs
never observe a partially constructed binding. The resulting client is marked
`ProviderOwnership.RUNTIME_OWNED` by default (the registry constructed it).

## Binding a preconstructed client

Applications may also hand the registry an already-built client that
satisfies the Story 6.1 structural provider protocol (`capabilities` plus a
callable `complete(...)`), without any inheritance requirement.

```python
registry.register_client(
    "custom-model",
    my_custom_client,
    ModelConfiguration(provider="custom", model="v1"),
    ownership=ProviderOwnership.CALLER_OWNED,  # default; pass RUNTIME_OWNED to transfer
)
```

Supplying `connection_config` alongside a preconstructed client is rejected
with `ContradictoryProviderConfigurationError`, since the client was already
built with its own settings — there must be exactly one source of truth for
endpoint, credential, proxy, TLS, and transport configuration.

## Ownership, replacement, and deregistration

- `ownership` (`ProviderOwnership.RUNTIME_OWNED` or `CALLER_OWNED`) is carried
  on every binding for Story 6.3 shutdown orchestration. This story only
  records the declaration; it does not close clients.
- Duplicate registration for the same model reference or provider type fails
  with `DuplicateModelReferenceError` / `DuplicateProviderTypeError` unless
  `replace=True` is passed. Replacement is atomic: a run that already resolved
  the previous binding keeps it; later `resolve()` calls see the replacement.
- `registry.deregister_model(reference)` and
  `registry.deregister_provider_type(provider_type)` remove a binding.
  Deregistration prevents *later* resolution; it never invalidates a binding a
  call already received from `resolve()`.

## Inspecting the registry safely

```python
snapshot = registry.snapshot()
snapshot.models          # tuple[ModelBindingSnapshot, ...], sorted by reference
snapshot.provider_types  # tuple[ProviderType, ...], sorted
```

Snapshots, and the `list_models()` / `list_provider_types()` helpers behind
them, never expose the registry's internal dictionaries, and never carry a
client, factory, credential, or connection configuration — only
reference/provider-type identifiers, ownership, and a point-in-time
availability boolean.

## Typed failures

| Error                                     | Raised when                                                             |
|-------------------------------------------|-------------------------------------------------------------------------|
| `DuplicateModelReferenceError`            | Reference already registered, `replace=False`                           |
| `DuplicateProviderTypeError`              | Provider type already registered, `replace=False`                       |
| `UnknownModelReferenceError`              | `resolve`/`deregister_model` for an unregistered reference              |
| `UnknownProviderTypeError`                | `register_provider`/`deregister_provider_type` for an unregistered type |
| `ProviderUnavailableError`                | Binding resolved but marked unavailable                                 |
| `ContradictoryProviderConfigurationError` | `connection_config` supplied alongside a preconstructed client          |
| `ProviderTypeMismatchError`               | `model_configuration.provider` does not match `provider_type`           |
| `ProviderFactoryValidationError`          | Factory has no callable `create(configuration)`                         |
| `ProviderClientValidationError`           | Client lacks `capabilities` or a callable `complete(...)`               |
| `ProviderConstructionError`               | The factory raised while constructing a client                          |
| `IncompatibleProviderCapabilitiesError`   | Client does not advertise a required capability                         |

All of the above derive from `ProviderRegistrationError` (itself a
`ConductoError`); the duplicate/contradictory/mismatch errors also derive from
`ValueError`, and the factory/client validation errors also derive from
`TypeError`, for compatibility with common exception-handling idioms.
Construction failures never echo the underlying provider exception's message
into the raised error; the original exception remains available as
`__cause__` for local debugging only.

## Migrating from `ProviderRegistry.register(...)`

`register(reference, client, configuration, *, available=True, replace=False)`
still works but is deprecated (`DeprecationWarning`) and now delegates to
`register_client(...)` with `ownership=ProviderOwnership.CALLER_OWNED` — the
same assumption the old API implicitly made, since it only ever accepted an
already-built client.

| Before                                                 | After                                                                                             |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `registry.register(ref, client, config)`               | `registry.register_client(ref, client, config)`                                                   |
| `registry.register(ref, client, config, replace=True)` | `registry.register_client(ref, client, config, replace=True)`                                     |
| *(no factory path existed)*                            | `registry.register_provider_type(...)` once, then `registry.register_provider(...)` per reference |

Call → run → agent → runtime model-reference precedence, concurrent run
isolation, and Story 6.1 provider request/result semantics are unchanged by
this migration.
