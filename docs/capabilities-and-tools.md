# Capabilities and tools

Conducto uses the word **tool** in more than one context. The important
distinction is between the method's declared contract and the temporary format
used to show work to a model or MCP client.

## Short version

| Concept | Who can discover it? | How it executes |
|---|---|---|
| `@a2a_capability` | Applications and, when admitted by policy, other agents, A2A callers, models, or MCP clients | Through the governed `Runtime` capability path |
| `@tool` | Code inspecting that specific `BaseAgent` instance | No automatic execution path currently exists |
| Model-facing tool | The active model call only | Resolves an opaque binding to a policy-approved capability |
| MCP tool | An authenticated MCP client, subject to export policy | Maps back to a policy-approved capability and enters `Runtime` |

An `@a2a_capability` is not automatically public to everyone. It is a
**publishable governed contract**. Registration, discovery policy,
authorization, scopes, approvals, and transport policy still decide who may
discover or invoke it.

## `@a2a_capability`: an agent-facing contract

Use `@a2a_capability` when the operation may be invoked through Conducto:

```python
from conducto import BaseAgent, a2a_agent, a2a_capability


@a2a_agent(
    name="InventoryAgent",
    version="1.0.0",
    description="Provides inventory information.",
)
class InventoryAgent(BaseAgent):
    @a2a_capability(
        name="inventory.lookup",
        description="Returns stock for one SKU.",
    )
    def lookup(self, sku: str) -> dict[str, object]:
        return {"sku": sku, "quantity": 12}
```

The reflected capability can participate in:

- local `Runtime.invoke()` calls;
- local registry and gateway discovery;
- an A2A Agent Card and inbound A2A invocation;
- a bounded model toolbox, when `ToolboxPolicy` admits it; and
- MCP export, when `McpExportPolicy` admits it.

Every invocation still enters the canonical runtime. Discoverability never
grants authority by itself.

### Declaring capability policy metadata

Capabilities can declare immutable authorization, execution, side-effect, and
data-sensitivity metadata alongside their published contract:

```python
from decimal import Decimal

from conducto import (
    a2a_capability,
    budget,
    classification,
    requires_scope,
    side_effect,
    timeout,
)


@a2a_capability(name="create_refund", description="Creates a customer refund.")
@requires_scope("refund.write")
@side_effect("writes_external_system")
@timeout(seconds=5)
@budget(max_model_calls=0, max_cost_usd=Decimal("2.50"))
@classification("confidential")
def create_refund(order_id: str, amount: float) -> str:
    return f"refund:{order_id}:{amount}"
```

These decorators may be stacked in any order with `@a2a_capability`.
`requires_scope` also installs the existing runtime authorization guardrail.
The declared metadata is available through immutable capability descriptors and
is published in Conducto's optional Agent Card extension; it does not alter A2A
standard fields or grant additional authority.

## Declaring external data sources

Declare external systems as metadata-only data sources, then bind capabilities to
them by stable name. The registry stores no connector, callable, endpoint, or
credential; trusted runtime configuration resolves each opaque binding:

```python
from conducto import (
    BaseAgent,
    DataSourceRegistry,
    a2a_capability,
    data_source,
    requires_scope,
    uses_data_source,
)
from conducto.core.registry import AgentRegistry


@data_source(
    name="fabric_customer_ontology",
    kind="fabric_ontology",
    description="Customer ontology metadata.",
    read_scopes={"customer.read"},
)
class CustomerOntology:
    pass


@data_source(
    name="onelake_customer_documents",
    kind="onelake",
    read_scopes={"customer.documents.read"},
)
class CustomerDocuments:
    pass


class CustomerAgent(BaseAgent):
    @a2a_capability(name="lookup_customer", description="Looks up a customer.")
    @requires_scope("customer.read")
    @uses_data_source("fabric_customer_ontology", "onelake_customer_documents")
    def lookup_customer(self, customer_id: str) -> str:
        return customer_id


sources = DataSourceRegistry()
sources.register(CustomerOntology)
sources.register(CustomerDocuments)
registry = AgentRegistry(data_sources=sources)
registry.register(CustomerAgent())
```

`kind` is descriptive metadata, not a request to install or contact a provider.
The same contract can describe Fabric ontology and OneLake/blob storage, as well
as SharePoint, SQL warehouses, CRM systems, or document stores. No cloud package
or credential is required for local declaration and snapshot inspection.
Capability snapshots and Agent Card extensions contain dependencies in sorted
name order. Connector binding IDs remain in trusted registry and audit metadata,
not in model-facing tool definitions or Agent Card dependency lists.

## `@tool`: internal export metadata

Use `@tool` only when code needs to identify a method as an internal export on
that agent instance:

```python
from conducto import BaseAgent, a2a_agent, tool


@a2a_agent(
    name="InventoryAgent",
    version="1.0.0",
    description="Provides inventory information.",
)
class InventoryAgent(BaseAgent):
    @tool(
        name="normalize_sku",
        description="Normalizes one SKU for internal processing.",
    )
    def normalize_sku(self, sku: str) -> str:
        return sku.strip().upper()
```

Conducto reflects this method into `agent.tools`, including its generated
argument schema. It does **not**:

- add the method to the Agent Card;
- register it as a gateway capability;
- expose it through A2A or MCP;
- make it eligible for a delegation `ToolboxPolicy`; or
- provide a public runtime API that automatically invokes it.

At present, agent code can call the Python method normally, and application
code can inspect `agent.tools`. Direct Python calls do not pass through the
governed capability invocation pipeline. If no code needs the reflected
metadata, prefer an ordinary private helper method instead of `@tool`.

## Why capabilities appear as “tools” to models

A model provider uses **tool** as a protocol term: a named JSON-schema
operation the model may request. Conducto builds those definitions from
policy-approved `@a2a_capability` exports, not from methods decorated only with
`@tool`.

```text
@a2a_capability
    → registry or catalog
    → policy-filtered opaque binding
    → model-facing tool definition
    → model requests the opaque tool ID
    → binding is revalidated
    → Runtime invokes the capability
```

The same naming issue occurs with MCP. Conducto projects admitted capabilities
as MCP tools, but internal `@tool` methods are never MCP exports.

## Choosing the right declaration

| Requirement | Declaration |
|---|---|
| Another agent or application may invoke it | `@a2a_capability` |
| It may be hosted over A2A | `@a2a_capability` |
| It may be offered to a model under bounded policy | `@a2a_capability` plus `ToolboxPolicy` |
| It may be exported to MCP under explicit policy | `@a2a_capability` plus `McpExportPolicy` |
| Internal code needs reflected method metadata | `@tool` |
| It is only an implementation helper | An ordinary method, usually private |

A method may carry both decorators because Conducto preserves both metadata
records. In that case it is externally publishable because it is a capability;
adding `@tool` does not make the capability private.
