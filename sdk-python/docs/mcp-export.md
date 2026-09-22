# MCP tool export

`conducto.mcp` projects canonical Conducto capabilities into Model Context
Protocol (MCP) tools. Agent authors keep declaring public behavior once with
`@a2a_capability`; there is no `@mcp_tool` decorator, no reflection path, and
no second schema declaration. The internal `@tool` marker is never exported.

The optional dependency lives in the `mcp` extra:

```bash
uv sync --extra mcp
# or
pip install "conducto-ai[mcp]"
```

Importing `conducto` never imports the MCP SDK. Requesting the stdio server
without the extra raises `McpDependencyError` with installation guidance.

## Export policy

Export is application-owned, immutable for one server instance, and
default-deny. `McpExportPolicy` allowlists exact `agent_id:capability_id`
identities or bounded capability queries and may assign an explicit MCP alias.
It cannot change canonical schemas, required scopes, approval policy, or
callable behavior.

```python
from conducto import AgentRegistry, Runtime
from conducto.mcp import McpCapabilityQuery, McpExportPolicy, McpExportRule, McpToolExporter

registry = AgentRegistry()
registry.register(weather_agent)

exporter = McpToolExporter(
    runtime=Runtime(agent_registry=registry),
    policy=McpExportPolicy(
        rules=(McpExportRule("WeatherAgent", "temperature", alias="weather_now"),),
        queries=(McpCapabilityQuery(agent_id="WeatherAgent", tags=frozenset({"public"}), limit=8),),
    ),
    registry=registry,
)
```

A rule or query that matches nothing, a query that exceeds its limit, or a
post-normalization name collision without distinct aliases fails exporter
construction with a typed error. Tool names, descriptions, schema size, and
tool count are bounded, and descriptions are treated as untrusted data.

## Naming and schemas

The default tool name is `normalize(agent_id)__normalize(capability_id)`:
lowercase ASCII, with every other character folded to `_` and runs collapsed.

Input schemas are the canonical Conducto schemas. An output schema is
published only when the capability return shape maps onto the supported
subset, wrapped as `{"type": "object", "properties": {"result": ...}}`.
The supported subset covers objects, arrays, enums, optional fields, and
bounds. Unsupported keywords, recursive references, ambiguous unions,
non-string mapping keys, and unmappable return types fail exporter
construction with `McpSchemaProjectionError` instead of silently broadening.

## Stdio identity and invocation

A stdio session must be constructed with an explicit immutable `Principal` or
a principal resolver. There is no anonymous privileged default, so protected
capabilities fail closed without identity, and `tools/list` exposes only
policy-admitted capabilities that the configured principal may call.

```python
from conducto.mcp import McpStdioServer
from conducto.security import Principal

server = McpStdioServer(
    exporter=exporter,
    principal=Principal(
        subject_id="local-operator",
        issuer="https://local.invalid",
        audience="conducto",
    ),
    call_timeout=30.0,
)
await server.serve_stdio()
```

Every `tools/call` dispatches the selected canonical capability through
`Runtime.invoke()`. The adapter never calls reflected methods directly and
keeps no parallel validation, authorization, approval, audit, timeout,
cancellation, or serialization behavior. MCP cancellation propagates into the
invocation, execution is capped by the earliest MCP, application, or Conducto
deadline, and Conducto correlation, lineage, logging, and audit evidence are
preserved. The correlation identifier and mapped reason code are returned in
tool result metadata under `ai.conducto/correlationId` and
`ai.conducto/reasonCode`.

## Result and error mapping

`invocation_result_to_tool_outcome()` maps every public `InvocationResult`
family to a safe MCP tool result. The mapping is pinned by
[`tests/golden/fixtures/mcp/mcp_result_mapping.json`](../tests/golden/fixtures/mcp/mcp_result_mapping.json).

- `InvocationSuccess` becomes structured content plus deterministic text.
- Validation, target, authorization, approval-required, audit, binding,
  lifecycle, schema, budget, timeout, cancellation, delegation,
  unsupported-return, and capability failures become non-success tool results
  with a fixed safe message and reason code.
- Approval-required stays non-success with a safe challenge reference; MCP
  never approves it automatically.
- Unknown tool names and draining sessions are protocol errors, so framing
  failures stay distinguishable from a capability's typed failure.

Messages never contain Python exceptions, tracebacks, credentials, prompts,
arguments, approval content, bindings, or internal endpoints.

## Lifecycle

`McpStdioServer` exposes `serve()` for supplied streams, `serve_stdio()` for
standard streams, `is_ready`, `drain()`, and an asynchronous idempotent
`aclose()`. Draining rejects new calls, allows accepted work only for the
configured grace period, propagates cancellation according to policy, and
releases the owned SDK server. The embedding application owns process launch,
standard-stream plumbing, principal selection, and supervision; see
[`examples/mcp_stdio_server.py`](../examples/mcp_stdio_server.py).

Streamable HTTP, network authentication, MCP resources/prompts/sampling, a
general MCP client, and dynamic mutation of a running server's tool list are
out of scope here.
