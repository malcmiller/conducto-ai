# Ollama provider

`conducto.providers.OllamaProvider` lets Conducto applications use an
existing Ollama daemon through the shared provider, structured-output,
lifecycle, ownership, and native-tool contracts. It does not install, start,
stop, reconfigure, pull, or delete Ollama models.

## Tested versions and profiles

The adapter is implemented against the official Ollama Python client and
Ollama's documented chat `format` JSON Schema and native `tools` fields.

| Profile          | Server           | Python client | Tool calls  | Call IDs                                                           | Streaming |
|------------------|------------------|---------------|-------------|--------------------------------------------------------------------|-----------|
| `terminal-json`  | `>=0.6.0` tested | `ollama>=0.5` | No          | N/A                                                                | No        |
| `llama3.1-tools` | `>=0.6.0` tested | `ollama>=0.5` | Yes, opt-in | Preserved when supplied; otherwise synthesized once by the adapter | No        |
| `qwen2.5-tools`  | `>=0.6.0` tested | `ollama>=0.5` | Yes, opt-in | Preserved when supplied; otherwise synthesized once by the adapter | No        |

The CPU-friendly example recommendation is `llama3.1:8b` or another small
tool-capable model already present in your Ollama installation. Conducto never
downloads it automatically.

Capabilities are profile-specific:

| Capability                                   | `terminal-json`                                                            | Tool-capable profiles                 |
|----------------------------------------------|----------------------------------------------------------------------------|---------------------------------------|
| Terminal JSON Schema through Ollama `format` | Yes                                                                        | Yes                                   |
| Independent Conducto JSON validation         | Yes                                                                        | Yes                                   |
| Native Ollama `tools`                        | No                                                                         | Yes                                   |
| Tool-result messages                         | No                                                                         | Yes                                   |
| Usage counters                               | Uses reported prompt/eval counts only                                      | Uses reported prompt/eval counts only |
| Cancellation                                 | Propagates task cancellation and pre-dispatch cancellation state           | Same                                  |
| Context/output limits                        | Configurable via `context_window`, `max_output_tokens`, and options        | Same                                  |
| Response body limit                          | Enforced while reading HTTP response bytes for configuration-owned clients | Same                                  |

Known limitations:

- Streaming is advertised as `False` until Conducto defines provider streaming
  request/result, cancellation, usage, and partial tool-call semantics.
- Terminal structured output supports the bounded schema subset Conducto can
  validate locally: `object`, `array`, scalar `type` values, `properties`,
  `items`, `required`, `additionalProperties`, `enum`, `const`, length/item
  bounds, and local `$ref`. `oneOf`, `anyOf`, and `allOf` are rejected unless a
  future profile proves native and local support for that terminal schema.
- Native tool selection does not require `SchemaFeature.ONE_OF`.
- Ollama model quality varies. The adapter only claims protocol behavior for
  pinned profiles proven by deterministic tests and opt-in smoke tests.

## Installation

```bash
uv add "conducto-ai[ollama]"
```

Importing `conducto` or `conducto.providers` does not import the optional
Ollama client, read credentials, or contact a network. Configuration-owned
providers validate the optional Ollama extra, then use a small bounded HTTP
adapter for Ollama's documented endpoints because the official Python client
does not expose a maximum response-body limit. Preconstructed clients remain
supported for applications that want to provide their own official-client
lifecycle.

Both adapters use shared private bounded HTTP and normalization infrastructure
under `conducto.providers`. Configured credentials are resolved only during
client construction and sent as actual bearer credentials on outbound HTTP
requests. Diagnostics and logs omit authorization headers, response bodies,
and underlying exception text; redaction never replaces the wire credential.
Import model contracts from `conducto.core.provider` and registration
contracts from `conducto.core.provider_registry`.

Use `await provider.aclose()` to release a configuration-owned asynchronous
HTTP client, or let `Runtime.aclose()` manage it through the provider registry.
The adapter has no synchronous `close()` shortcut. Borrowed clients remain
application-owned and are never closed by the adapter. The `ollama` extra
explicitly declares the HTTPX transport dependency.

## Registration

Configuration-owned registration keeps endpoint, authentication, TLS, proxy,
transport, keep-alive, and model defaults single-sourced in
`ProviderClientConfig`:

```python
from conducto.core.provider import ModelConfiguration
from conducto.core.provider_registry import ProviderClientConfig, ProviderRegistry
from conducto.providers import ollama_provider_factory

registry = ProviderRegistry()
registry.register_provider_type("ollama", ollama_provider_factory)
registry.register_provider(
    "local-llama",
    provider_type="ollama",
    configuration=ProviderClientConfig(
        endpoint="http://localhost:11434",
        provider_defaults={
            "model": "llama3.1:8b",
            "profile": "terminal-json",
            "keep_alive": "5m",
            "context_window": 8192,
            "max_output_tokens": 512,
            "seed": 7,
            "sampling_top_p": 0.9,
        },
    ),
    model_configuration=ModelConfiguration(provider="ollama", model="llama3.1:8b"),
)
```

Preconstructed clients are also supported:

```python
from conducto.core.provider import ModelConfiguration
from conducto.core.provider_registry import ProviderRegistry
from conducto.providers import OllamaProvider

client = OllamaProvider(model="llama3.1:8b", client=my_async_ollama_client)
registry = ProviderRegistry()
registry.register_client(
    "local-llama",
    client,
    ModelConfiguration(provider="ollama", model="llama3.1:8b"),
)
```

Do not pass endpoint, authentication, TLS, proxy, or transport settings to
`OllamaProvider` when supplying a ready client. The provider rejects that
contradictory configuration so the connection source of truth remains clear.

## Endpoint and transport examples

Localhost:

```python
ProviderClientConfig(
    endpoint="http://localhost:11434",
    provider_defaults={"model": "llama3.1:8b"},
)
```

Host-to-container on Docker Desktop:

```python
ProviderClientConfig(
    endpoint="http://host.docker.internal:11434",
    provider_defaults={"model": "llama3.1:8b"},
)
```

TLS reverse proxy with certificate verification still enabled:

```python
ProviderClientConfig(
    endpoint="https://ollama.example.internal",
    tls={"verify": "C:/certs/internal-ca.pem"},
    provider_defaults={"model": "llama3.1:8b"},
)
```

Optional bearer authentication through an environment-backed credential
reference:

```powershell
$env:CONDUCTO_OLLAMA_TOKEN = "<token from your reverse proxy>"
```

```python
ProviderClientConfig(
    endpoint="https://ollama.example.internal",
    credential_ref="CONDUCTO_OLLAMA_TOKEN",
    provider_defaults={"model": "llama3.1:8b"},
)
```

The token value is never stored in the registry. Authorization headers, query
secrets, prompts, model outputs, structured payloads, and server error bodies
are redacted from provider diagnostics by default.

## Native tools

Use a tool-capable profile only after proving the selected server/client/model
combination in your environment:

```python
from conducto.providers import OLLAMA_TOOL_CAPABLE_PROFILE, OllamaProvider

provider = OllamaProvider(
    endpoint="http://localhost:11434",
    model="llama3.1:8b",
    profile=OLLAMA_TOOL_CAPABLE_PROFILE,
)
```

For each turn, Conducto sends exactly the `ProviderToolDefinition` snapshot
provided by the runtime as Ollama `tools`. Tool results are translated to
bounded `role="tool"` messages. Returned function names resolve only through
that exact snapshot. Mixed terminal/tool output, multiple calls, unknown or
ambiguous tool names, malformed arguments, and tool calls from profiles that
do not advertise tool support fail before capability execution.

## Readiness and smoke tests

Readiness checks are explicit:

```python
await provider.check_readiness(model="llama3.1:8b")
```

They distinguish connection/readiness failures, authentication failures,
missing local models, incompatible server versions, timeouts, and unsupported
capabilities.

Required CI uses deterministic fake clients and never starts Ollama, downloads
models, needs a GPU, or requires internet access. To run opt-in smoke tests
against your own already running daemon:

```powershell
$env:CONDUCTO_OLLAMA_SMOKE = "1"
$env:CONDUCTO_OLLAMA_ENDPOINT = "http://localhost:11434"
$env:CONDUCTO_OLLAMA_MODEL = "llama3.1:8b"
$env:CONDUCTO_OLLAMA_TOOL_MODEL = "llama3.1:8b"
uv run pytest -m ollama
```

If any prerequisite is missing, the smoke tests skip with the missing setting
or package requirement and do not count as required-CI coverage.
