import asyncio
import json

import pytest

from conducto import (
    AgentRegistry,
    BaseAgent,
    CapabilityUse,
    CapabilityUseRequirement,
    GatewayFailureCode,
    Runtime,
    ToolboxPolicy,
    ToolboxStatus,
    a2a_agent,
    a2a_capability,
    build_toolbox,
)
from conducto.core.runtime import use_run_context


@a2a_agent(name="AlphaSearch", version="1.2.0", description="Search metadata.", tags=("search",))
class AlphaSearch(BaseAgent):
    @a2a_capability(name="lookup", description="Looks up a value.", tags=("read",))
    def lookup(self, query: str) -> str:
        return f"alpha:{query}"


@a2a_agent(name="BravoSearch", version="2.0.0", description="Another search provider.")
class BravoSearch(BaseAgent):
    @a2a_capability(name="lookup", description="Looks up a value.", tags=("read",))
    def lookup(self, query: str) -> str:
        return f"bravo:{query}"


@a2a_agent(name="Diagnostics", version="1.0.0", description="Diagnostics agent.")
class Diagnostics(BaseAgent):
    @a2a_capability(name="diagnose", description="Diagnoses a failure.")
    def diagnose(self, code: str) -> str:
        return f"diagnosis:{code}"


def _runtime(registry: AgentRegistry, **kwargs: object) -> Runtime:
    return Runtime(agent_registry=registry, **kwargs)  # type: ignore[arg-type]


def test_capability_use_rejects_invalid_declarations() -> None:
    with pytest.raises(ValueError, match="capability_ids or tags"):
        CapabilityUse()
    with pytest.raises(ValueError, match="empty identifiers"):
        CapabilityUse(capability_ids=frozenset({""}))
    with pytest.raises(ValueError, match="limit must be positive"):
        CapabilityUse(capability_ids=frozenset({"lookup"}), limit=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        CapabilityUse(capability_ids=frozenset({"lookup"}), limit=10_000)


def test_toolbox_policy_rejects_duplicate_declarations() -> None:
    use = CapabilityUse(capability_ids=frozenset({"lookup"}))
    with pytest.raises(ValueError, match="Duplicate capability use"):
        ToolboxPolicy(uses=(use, use))


def test_empty_policy_produces_empty_success_toolbox() -> None:
    registry = AgentRegistry()
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, ToolboxPolicy())
            assert result.ok
            assert result.snapshot is not None
            assert len(result.snapshot) == 0

    asyncio.run(exercise())


def test_two_providers_yield_deterministic_distinct_tool_bindings() -> None:
    registry = AgentRegistry()
    registry.register(BravoSearch())
    registry.register(AlphaSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), limit=10),))

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.status is ToolboxStatus.SUCCESS
            assert result.snapshot is not None
            assert len(result.snapshot) == 2
            tool_ids = {tool.tool_id for tool in result.snapshot}
            assert len(tool_ids) == 2
            payload = json.dumps([tool.to_dict() for tool in result.snapshot])
            assert "callable" not in payload
            assert "endpoint" not in payload

    asyncio.run(exercise())


def test_knowledge_and_diagnostic_agents_share_one_capability_contract() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    knowledge_policy = ToolboxPolicy(
        uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), tags=frozenset({"read"})),)
    )
    diagnostic_policy = ToolboxPolicy(
        uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), tags=frozenset({"read"})),)
    )
    runtime = _runtime(registry)

    async def exercise() -> None:
        knowledge_ctx = runtime.create_run_context(agent_id="Knowledge")
        with use_run_context(knowledge_ctx):
            knowledge_result = await build_toolbox(knowledge_ctx.gateway, knowledge_policy)
        diagnostic_ctx = runtime.create_run_context(agent_id="Diagnostic")
        with use_run_context(diagnostic_ctx):
            diagnostic_result = await build_toolbox(diagnostic_ctx.gateway, diagnostic_policy)
        assert knowledge_result.ok and diagnostic_result.ok
        assert knowledge_result.snapshot is not None
        assert diagnostic_result.snapshot is not None
        assert len(knowledge_result.snapshot) == 1
        assert len(diagnostic_result.snapshot) == 1

    asyncio.run(exercise())


def test_missing_required_capability_fails_before_a_model_call() -> None:
    registry = AgentRegistry()
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(
        uses=(
            CapabilityUse(
                capability_ids=frozenset({"documentation.search"}),
                requirement=CapabilityUseRequirement.REQUIRED,
            ),
        )
    )

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.status is ToolboxStatus.REQUIRED_CAPABILITY_UNAVAILABLE
            assert result.snapshot is None
            assert result.failure is not None
            assert result.failure.code is GatewayFailureCode.NO_MATCH

    asyncio.run(exercise())


def test_missing_optional_capability_yields_partial_toolbox() -> None:
    registry = AgentRegistry()
    registry.register(Diagnostics())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(
        uses=(
            CapabilityUse(capability_ids=frozenset({"diagnose"})),
            CapabilityUse(capability_ids=frozenset({"documentation.search"})),
        )
    )

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.ok
            assert result.snapshot is not None
            assert len(result.snapshot) == 1
            assert len(result.skipped) == 1
            assert result.skipped[0].failure.code is GatewayFailureCode.NO_MATCH

    asyncio.run(exercise())


def test_unauthorized_capabilities_never_appear_in_projected_toolbox() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(
        agent_id="Caller",
        allowed_capabilities=frozenset({"other"}),
    )
    policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"lookup"})),))

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.ok
            assert result.snapshot is not None
            assert len(result.snapshot) == 0

    asyncio.run(exercise())


def test_toolbox_enforces_max_tool_count() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    registry.register(BravoSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(
        uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), limit=10),),
        max_tools=1,
    )

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.status is ToolboxStatus.LIMIT_EXCEEDED
            assert result.snapshot is None

    asyncio.run(exercise())


def test_toolbox_enforces_total_serialized_size_limit() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    registry.register(BravoSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(
        uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), limit=10),),
        max_total_bytes=8,
    )

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.status is ToolboxStatus.LIMIT_EXCEEDED

    asyncio.run(exercise())


def test_toolbox_snapshot_resolves_only_its_own_tool_ids() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"lookup"})),))

    async def exercise() -> None:
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.snapshot is not None
            tool = result.snapshot.tools[0]
            assert result.snapshot.resolve(tool.tool_id) is tool.binding
            assert result.snapshot.resolve("unknown-tool-id") is None
            assert tool.tool_id in result.snapshot
            assert "unknown-tool-id" not in result.snapshot

    asyncio.run(exercise())


def test_concurrent_runs_do_not_share_toolbox_snapshots() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    runtime = _runtime(registry)
    policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"lookup"})),))

    async def build_for(agent_id: str) -> tuple[str, ...]:
        context = runtime.create_run_context(agent_id=agent_id)
        with use_run_context(context):
            result = await build_toolbox(context.gateway, policy)
            assert result.snapshot is not None
            return tuple(tool.tool_id for tool in result.snapshot)

    async def exercise() -> None:
        first, second = await asyncio.gather(build_for("First"), build_for("Second"))
        assert first == second  # Same registry state, deterministic tool IDs.

    asyncio.run(exercise())


def test_registering_new_provider_changes_a_later_snapshot_only() -> None:
    registry = AgentRegistry()
    registry.register(AlphaSearch())
    runtime = _runtime(registry)
    context = runtime.create_run_context(agent_id="Caller")
    policy = ToolboxPolicy(uses=(CapabilityUse(capability_ids=frozenset({"lookup"}), limit=10),))

    async def exercise() -> None:
        with use_run_context(context):
            first_result = await build_toolbox(context.gateway, policy)
            assert first_result.snapshot is not None
            assert len(first_result.snapshot) == 1

            registry.register(BravoSearch())

            second_result = await build_toolbox(context.gateway, policy)
            assert second_result.snapshot is not None
            assert len(second_result.snapshot) == 2
            assert (
                second_result.snapshot.registry_revision > first_result.snapshot.registry_revision
            )

    asyncio.run(exercise())
