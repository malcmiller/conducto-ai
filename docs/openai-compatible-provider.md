# OpenAI-compatible provider (vLLM and LM Studio)

`conducto.providers.OpenAICompatibleProvider` lets Conducto applications use
an already running vLLM or LM Studio server through the shared provider,
structured-output, lifecycle, ownership, and native-tool contracts. It does
not install, start, stop, reconfigure, or supervise vLLM or LM Studio, and it
never downloads or manages model weights.

"OpenAI-compatible" does not mean universally compatible. Only the pinned
profiles below have been conformance tested. Pointing the adapter at an
untested server or an untested API surface is unsupported and may fail in
ways this adapter does not detect.

## Tested versions and profiles

The adapter sends the documented OpenAI chat completions request shape
(`POST /chat/completions`) with a JSON Schema `response_format`, and reads
model availability from `GET /models`. vLLM additionally exposes a documented
`GET /version` endpoint used for an optional compatibility check; LM Studio
does not document a version endpoint, so LM Studio profiles skip that check
entirely.

| Profile | Server family | Server version | Official client | Tool calls | Call IDs | Streaming |
|---------|---------------|-----------------|------------------|------------|----------|-----------|
| `vllm-terminal-json` | vLLM | `>=0.6.0` tested | `openai>=1.0` | No | N/A | No |
| `vllm-tools` | vLLM | `>=0.6.0` tested | `openai>=1.0` | Yes, opt-in | Preserved when supplied; otherwise synthesized once by the adapter | No |
| `lm-studio-terminal-json` | LM Studio | `>=0.3.0` tested | `openai>=1.0` | No | N/A | No |
| `lm-studio-tools` | LM Studio | `>=0.3.0` tested | `openai>=1.0` | Yes, opt-in | Preserved when supplied; otherwise synthesized once by the adapter | No |

Capabilities are profile-specific:

| Capability | `*-terminal-json` | `*-tools` |
|------------|--------------------|-----------|
| Native `response_format: json_schema` | Yes | Yes |
| Independent Conducto JSON validation | Yes | Yes |
| Native OpenAI-shaped `tools` | No | Yes |
| Tool-result messages | No | Yes |
| Parallel/multiple tool calls per turn | N/A | Always disabled (`parallel_tool_calls: false`) |
| Usage counters | Uses reported `prompt_tokens`/`completion_tokens`/`total_tokens` only | Same |
| Cancellation | Propagates task cancellation and pre-dispatch cancellation state | Same |
| Context/output limits | Configurable via `context_window`, `max_output_tokens` | Same |
| Response body limit | Enforced while reading HTTP response bytes for configuration-owned clients | Same |
| Server version check | vLLM: bounded `/version` check. LM Studio: skipped (no documented endpoint) | Same |

Known limitations:

- Streaming is advertised as `False` until Conducto defines provider streaming
  request/result, cancellation, usage, and partial tool-call semantics.
- Terminal structured output and native tool argument schemas support a
  narrower bounded subset than other Conducto adapters, matching OpenAI
  `strict` JSON Schema response-format constraints: `object`, `array`, scalar
  `type` values, `properties`, `items`, `required`, `additionalProperties`,
  `enum`, and local `$ref`. `const`, string/array length bounds, and
  `oneOf`/`anyOf`/`allOf` are rejected before dispatch.
- Native tool selection does not require `SchemaFeature.ONE_OF`.
- LM Studio profiles never verify server version compatibility because LM
  Studio does not document a version endpoint; only model availability is
  checked during readiness.
- vLLM and LM Studio model quality varies by loaded model. The adapter only
  claims protocol behavior for pinned profiles proven by deterministic tests
  and opt-in smoke tests, not model output quality.
- This adapter does not implement Ollama-specific APIs; use
  `conducto.providers.OllamaProvider` for Ollama.

## Installation

```bash
uv add "conducto-ai[openai]"
```

Importing `conducto` or `conducto.providers` does not import the optional
`openai` client, read credentials, or contact a network. Configuration-owned
providers validate the optional `openai` extra, then use a small bounded HTTP
adapter for the documented OpenAI-compatible endpoints because the official
Python client does not expose a maximum response-body limit. Preconstructed
clients (including a real `openai.AsyncOpenAI` instance) are also supported
for applications that want their own official-client lifecycle.

The adapter shares private bounded HTTP, schema traversal, deadline, and
normalization helpers with Ollama. Configured credentials become actual
outbound bearer authorization; only diagnostics are redacted. Diagnostic
events never publish credentials, authorization headers, response bodies, or
underlying exception text. Model contracts live in `conducto.core.provider`;
registry and construction contracts live in `conducto.core.provider_registry`.

Configuration-owned HTTP clients have asynchronous lifetimes: use
`await provider.aclose()` or runtime-managed registry shutdown. There is no
synchronous `close()` shortcut that could leave the connection pool open.
Borrowed clients remain application-owned. The `openai` extra explicitly
declares the HTTPX transport dependency.

## Registration

Configuration-owned registration keeps endpoint, authentication, TLS, proxy,
transport, and model defaults single-sourced in `ProviderClientConfig`:

```python
from conducto.core.provider import ModelConfiguration
from conducto.core.provider_registry import ProviderClientConfig, ProviderRegistry
from conducto.providers import openai_compatible_provider_factory

registry = ProviderRegistry()
registry.register_provider_type("openai_compatible", openai_compatible_provider_factory)
registry.register_provider(
    "local-vllm",
    provider_type="openai_compatible",
    configuration=ProviderClientConfig(
        endpoint="http://localhost:8000/v1",
        provider_defaults={
            "model": "my-served-model",
            "profile": "vllm-terminal-json",
            "context_window": 8192,
            "max_output_tokens": 512,
            "seed": 7,
            "sampling_top_p": 0.9,
        },
    ),
    model_configuration=ModelConfiguration(provider="openai_compatible", model="my-served-model"),
)
```

Preconstructed clients are also supported, including the real official
client:

```python
import openai
from conducto.core.provider import ModelConfiguration
from conducto.core.provider_registry import ProviderRegistry
from conducto.providers import OpenAICompatibleProvider

official_client = openai.AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
client = OpenAICompatibleProvider(
    model="my-loaded-model",
    profile="lm-studio-terminal-json",
    client=official_client,
)
registry = ProviderRegistry()
registry.register_client(
    "local-lm-studio",
    client,
    ModelConfiguration(provider="openai_compatible", model="my-loaded-model"),
)
```

Do not pass endpoint, authentication, organization, project, TLS, proxy,
transport, or timeout settings to `OpenAICompatibleProvider` when supplying a
ready client. The provider rejects that contradictory configuration so the
connection source of truth remains clear.

## Endpoint and transport examples

Localhost vLLM:

```python
ProviderClientConfig(
    endpoint="http://localhost:8000/v1",
    provider_defaults={"model": "my-served-model", "profile": "vllm-terminal-json"},
)
```

Localhost LM Studio:

```python
ProviderClientConfig(
    endpoint="http://localhost:1234/v1",
    provider_defaults={"model": "my-loaded-model", "profile": "lm-studio-terminal-json"},
)
```

Host-to-container on Docker Desktop:

```python
ProviderClientConfig(
    endpoint="http://host.docker.internal:8000/v1",
    provider_defaults={"model": "my-served-model", "profile": "vllm-terminal-json"},
)
```

TLS reverse proxy with certificate verification still enabled:

```python
ProviderClientConfig(
    endpoint="https://vllm.example.internal/v1",
    tls={"verify": "C:/certs/internal-ca.pem"},
    provider_defaults={"model": "my-served-model", "profile": "vllm-terminal-json"},
)
```

Optional bearer authentication through an environment-backed credential
reference:

```powershell
$env:CONDUCTO_OPENAI_COMPATIBLE_TOKEN = "<token from your reverse proxy>"
```

```python
ProviderClientConfig(
    endpoint="https://vllm.example.internal/v1",
    credential_ref="CONDUCTO_OPENAI_COMPATIBLE_TOKEN",
    provider_defaults={"model": "my-served-model", "profile": "vllm-terminal-json"},
)
```

The token value is never stored in the registry. Authorization headers,
organization/project headers, query secrets, prompts, model outputs,
structured payloads, and server error bodies are redacted from provider
diagnostics by default.

## Native tools

Use a tool-capable profile only after proving the selected server/client/model
combination in your environment:

```python
from conducto.providers import VLLM_TOOL_CAPABLE_PROFILE, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    endpoint="http://localhost:8000/v1",
    model="my-served-model",
    profile=VLLM_TOOL_CAPABLE_PROFILE,
)
```

For each turn, Conducto sends exactly the `ProviderToolDefinition` snapshot
provided by the runtime as native OpenAI-shaped `tools`, with
`parallel_tool_calls` always set to `false`. Tool results are translated to
bounded `role="tool"` messages carrying the same `tool_call_id` the server
returned. Returned function names resolve only through that exact snapshot.
Mixed terminal/tool output, multiple calls, unknown or ambiguous tool names,
malformed JSON arguments, and tool calls from profiles that do not advertise
tool support fail before capability execution.

## Readiness and smoke tests

Readiness checks are explicit:

```python
await provider.check_readiness(model="my-served-model")
```

They distinguish connection/readiness failures, authentication failures,
missing models, incompatible server versions (vLLM only), timeouts, and
unsupported capabilities.

Required CI uses deterministic fake clients and never starts vLLM or LM
Studio, downloads models, needs a GPU, or requires internet access. To run
opt-in smoke tests against your own already running servers:

```powershell
$env:CONDUCTO_VLLM_SMOKE = "1"
$env:CONDUCTO_VLLM_ENDPOINT = "http://localhost:8000/v1"
$env:CONDUCTO_VLLM_MODEL = "my-served-model"
$env:CONDUCTO_VLLM_TOOL_MODEL = "my-served-model"
uv run pytest -m vllm

$env:CONDUCTO_LM_STUDIO_SMOKE = "1"
$env:CONDUCTO_LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"
$env:CONDUCTO_LM_STUDIO_MODEL = "my-loaded-model"
$env:CONDUCTO_LM_STUDIO_TOOL_MODEL = "my-loaded-model"
uv run pytest -m lm_studio
```

If any prerequisite is missing, the smoke tests skip with the missing setting
or package requirement and do not count as required-CI coverage.
