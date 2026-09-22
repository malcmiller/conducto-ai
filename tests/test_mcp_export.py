"""Export policy, naming, projection, and dispatch coverage for MCP tools."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from importlib import metadata
from typing import Any

import pytest

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability, tool
from conducto.mcp import (
    MCP_EXTRA,
    McpCapabilityQuery,
    McpDependencyError,
    McpExportError,
    McpExportPolicy,
    McpExportRule,
    McpNameCollisionError,
    McpPolicyError,
    McpSchemaProjectionError,
    McpToolExporter,
    McpToolNotFoundError,
    default_tool_name,
    normalize_tool_name,
    require_mcp_dependency,
)
from conducto.security import Principal, require_approval, require_scope

BILLING_AGENT = "Billing Agent"


@a2a_agent(name=BILLING_AGENT, version="1.0", description="Billing capabilities.", tags=("sales",))
class BillingAgent(BaseAgent):
    """Agent exposing public, protected, and internal behavior."""

    @a2a_capability(name="quote", description="Quote a price for billable units.")
    def quote(self, units: int) -> dict[str, int]:
        """Return the quoted total."""
        return {"total": units * 3}

    @a2a_capability(name="refund", description="Refund a settled order.")
    @require_scope("billing.write")
    def refund(self, order_id: str) -> dict[str, str]:
        """Return the refunded order state."""
        return {"order": order_id, "state": "refunded"}

    @a2a_capability(name="close", description="Close a billing account.")
    @require_approval("billing-approver")
    def close(self, account_id: str) -> dict[str, str]:
        """Return the closed account state."""
        return {"account": account_id, "state": "closed"}

    @tool(name="internal_rate", description="Internal rate helper.")
    def internal_rate(self, units: int) -> int:
        """Return an internal rate that is never exported."""
        return units


@a2a_agent(name="Billing-Agent", version="1.0", description="Colliding agent identity.")
class CollidingAgent(BaseAgent):
    """Agent whose identity normalizes to the same MCP name."""

    @a2a_capability(name="quote", description="Quote a price.")
    def quote(self, units: int) -> dict[str, int]:
        """Return the quoted total."""
        return {"total": units}


def _principal(*scopes: str) -> Principal:
    """Return a stdio principal holding the requested scopes."""
    return Principal(
        subject_id="operator-1",
        issuer="https://issuer.invalid",
        audience="conducto",
        scopes=frozenset(scopes),
    )


def _exporter(policy: McpExportPolicy, *agents: BaseAgent) -> McpToolExporter:
    """Build an exporter over freshly constructed agents."""
    return McpToolExporter(
        runtime=Runtime(),
        policy=policy,
        agents=agents or (BillingAgent(),),
    )


def test_policy_requires_at_least_one_allowlist_entry() -> None:
    with pytest.raises(McpPolicyError, match="at least one capability"):
        McpExportPolicy()


def test_policy_rejects_duplicate_and_unnormalized_entries() -> None:
    with pytest.raises(McpPolicyError, match="one capability twice"):
        McpExportPolicy(
            rules=(
                McpExportRule(BILLING_AGENT, "quote"),
                McpExportRule(BILLING_AGENT, "quote"),
            )
        )
    with pytest.raises(McpPolicyError, match="normalized MCP tool name"):
        McpExportRule(BILLING_AGENT, "quote", alias="Billing Quote")


def test_default_deny_policy_exports_only_allowlisted_capabilities() -> None:
    exporter = _exporter(McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "quote"),)))

    assert [definition.name for definition in exporter.tools] == ["billing_agent__quote"]


def test_internal_tools_are_never_exported() -> None:
    exporter = _exporter(
        McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "quote"),)),
    )

    assert all("internal" not in definition.name for definition in exporter.tools)
    with pytest.raises(McpPolicyError, match="matches no registered capability"):
        _exporter(McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "internal_rate"),)))


def test_aliases_and_default_names_are_deterministic() -> None:
    exporter = _exporter(
        McpExportPolicy(
            rules=(
                McpExportRule(BILLING_AGENT, "refund", alias="billing_refund"),
                McpExportRule(BILLING_AGENT, "quote"),
            )
        )
    )

    assert normalize_tool_name("Billing -- Agent!") == "billing_agent"
    assert default_tool_name(BILLING_AGENT, "quote") == "billing_agent__quote"
    assert [definition.name for definition in exporter.tools] == [
        "billing_agent__quote",
        "billing_refund",
    ]


def test_normalized_name_collisions_are_rejected() -> None:
    policy = McpExportPolicy(
        rules=(
            McpExportRule(BILLING_AGENT, "quote"),
            McpExportRule("Billing-Agent", "quote"),
        )
    )

    with pytest.raises(McpNameCollisionError, match="configure distinct aliases"):
        _exporter(policy, BillingAgent(), CollidingAgent())


def test_distinct_aliases_resolve_a_normalized_collision() -> None:
    policy = McpExportPolicy(
        rules=(
            McpExportRule(BILLING_AGENT, "quote"),
            McpExportRule("Billing-Agent", "quote", alias="legacy_billing_quote"),
        )
    )

    exporter = _exporter(policy, BillingAgent(), CollidingAgent())

    assert [definition.name for definition in exporter.tools] == [
        "billing_agent__quote",
        "legacy_billing_quote",
    ]


def test_bounded_queries_admit_tagged_capabilities_and_enforce_limits() -> None:
    exporter = _exporter(McpExportPolicy(queries=(McpCapabilityQuery(BILLING_AGENT, limit=5),)))

    assert [definition.name for definition in exporter.tools] == [
        "billing_agent__close",
        "billing_agent__quote",
        "billing_agent__refund",
    ]
    with pytest.raises(McpPolicyError, match="exceeds its bound"):
        _exporter(McpExportPolicy(queries=(McpCapabilityQuery(BILLING_AGENT, limit=1),)))
    with pytest.raises(McpPolicyError, match="matches no registered capability"):
        _exporter(
            McpExportPolicy(
                queries=(McpCapabilityQuery(BILLING_AGENT, tags=frozenset({"missing"})),)
            )
        )


def test_unmatched_allowlist_entries_fail_construction() -> None:
    with pytest.raises(McpPolicyError, match="matches no registered capability"):
        _exporter(McpExportPolicy(rules=(McpExportRule("Unknown Agent", "quote"),)))
    with pytest.raises(McpPolicyError, match="matches no registered capability"):
        _exporter(McpExportPolicy(queries=(McpCapabilityQuery("Unknown Agent"),)))


def test_configured_bounds_are_enforced() -> None:
    rules = (McpExportRule(BILLING_AGENT, "quote"), McpExportRule(BILLING_AGENT, "refund"))

    with pytest.raises(McpExportError, match="tool bound"):
        _exporter(McpExportPolicy(rules=rules, max_tools=1))
    with pytest.raises(McpExportError, match="character bound"):
        _exporter(McpExportPolicy(rules=rules[:1], max_name_length=4))
    with pytest.raises(McpExportError, match="character bound"):
        _exporter(McpExportPolicy(rules=rules[:1], max_description_length=4))
    with pytest.raises(McpSchemaProjectionError, match="byte bound"):
        _exporter(McpExportPolicy(rules=rules[:1], max_schema_bytes=16))


def test_exporter_requires_a_snapshot_source() -> None:
    with pytest.raises(McpExportError, match="agents or a registry snapshot source"):
        McpToolExporter(
            runtime=Runtime(),
            policy=McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "quote"),)),
        )


def test_registry_snapshots_supply_canonical_metadata() -> None:
    registry = AgentRegistry()
    registry.register(BillingAgent())

    exporter = McpToolExporter(
        runtime=Runtime(),
        policy=McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "quote"),)),
        registry=registry,
    )
    definition = exporter.tool("billing_agent__quote")

    assert definition.description == "Quote a price for billable units."
    assert definition.input_schema_dict()["properties"]["units"]["type"] == "integer"
    assert definition.output_schema_dict() == {
        "type": "object",
        "properties": {"result": {"type": "object", "additionalProperties": {"type": "integer"}}},
        "required": ["result"],
        "additionalProperties": False,
    }


def test_protected_tools_are_filtered_by_the_configured_principal() -> None:
    exporter = _exporter(
        McpExportPolicy(
            rules=(
                McpExportRule(BILLING_AGENT, "quote"),
                McpExportRule(BILLING_AGENT, "refund"),
            )
        )
    )

    assert [definition.name for definition in exporter.list_tools(None)] == ["billing_agent__quote"]
    assert [definition.name for definition in exporter.list_tools(_principal("other"))] == [
        "billing_agent__quote"
    ]
    assert [definition.name for definition in exporter.list_tools(_principal("billing.write"))] == [
        "billing_agent__quote",
        "billing_agent__refund",
    ]


def test_protected_tool_calls_fail_closed_without_identity() -> None:
    exporter = _exporter(
        McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "refund"),)),
    )

    async def call() -> Any:
        return await exporter.call_tool(
            "billing_agent__refund",
            {"order_id": "order-1"},
            principal=None,
            task_id="task-1",
        )

    with pytest.raises(McpToolNotFoundError):
        asyncio.run(call())


def test_tool_calls_dispatch_through_the_runtime() -> None:
    exporter = _exporter(
        McpExportPolicy(
            rules=(
                McpExportRule(BILLING_AGENT, "quote"),
                McpExportRule(BILLING_AGENT, "close"),
            )
        )
    )

    async def call(name: str, arguments: dict[str, Any]) -> Any:
        return await exporter.call_tool(
            name,
            arguments,
            principal=_principal(),
            task_id="task-1",
        )

    success = asyncio.run(call("billing_agent__quote", {"units": 4}))
    invalid = asyncio.run(call("billing_agent__quote", {"units": "four"}))
    approval = asyncio.run(call("billing_agent__close", {"account_id": "a-1"}))

    assert success.structured_content == {"result": {"total": 12}}
    assert success.reason_code == "ok"
    assert invalid.is_error and invalid.reason_code == "invalid_arguments"
    assert approval.is_error and approval.reason_code == "approval_required"


def test_unknown_tool_names_are_reported_as_lookup_failures() -> None:
    exporter = _exporter(McpExportPolicy(rules=(McpExportRule(BILLING_AGENT, "quote"),)))

    with pytest.raises(McpToolNotFoundError, match="Unknown MCP tool"):
        exporter.tool("missing")


def test_missing_official_sdk_reports_actionable_installation_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError("mcp")

    monkeypatch.setattr(metadata, "version", missing)

    with pytest.raises(McpDependencyError, match=r"install conducto-ai\[mcp\]") as error:
        require_mcp_dependency()

    assert error.value.extra == MCP_EXTRA


def test_importing_conducto_does_not_import_the_official_mcp_sdk() -> None:
    script = (
        "import sys; import conducto; import conducto.mcp; "
        "print('mcp' in sys.modules or 'mcp.types' in sys.modules)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"
