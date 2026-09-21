"""Acceptance coverage for the public three-agent chaining example."""

from __future__ import annotations

import asyncio

import pytest
from examples.agent_chaining import DocumentationAgent, build_application

from conducto import (
    BaseAgent,
    CapabilityUse,
    InvocationSuccess,
    ToolboxPolicy,
    a2a_agent,
    a2a_capability,
    build_toolbox,
)
from conducto.core.runtime import use_run_context

pytestmark = pytest.mark.acceptance


def test_three_agent_paths_preserve_correlation_and_model_order() -> None:
    async def exercise() -> None:
        orchestrator, model = build_application()
        no_tool = await orchestrator.route("What is Conducto?", correlation_id="chain-no-tool")
        direct = await orchestrator.invoke(
            "DiagnosticAgent",
            "diagnose",
            {"question": "Why did the build fail?"},
            correlation_id="chain-direct",
        )
        delegated = await orchestrator.route(
            "What does product documentation say about approvals?",
            correlation_id="chain-routed",
        )

        assert all(isinstance(result, InvocationSuccess) for result in (no_tool, direct, delegated))
        assert no_tool.metadata is not None
        assert direct.metadata is not None
        assert delegated.metadata is not None
        assert no_tool.metadata.correlation_id == "chain-no-tool"
        assert direct.metadata.correlation_id == "chain-direct"
        assert delegated.metadata.correlation_id == "chain-routed"
        assert len(no_tool.metadata.model_calls) == 2
        assert len(direct.metadata.model_calls) == 2
        assert [call.purpose for call in delegated.metadata.model_calls] == [
            "routing",
            "delegation_turn",
            "delegation_turn",
        ]
        assert model.requests[0].structured_output.name == "conducto_capability_selection"
        assert any(request.tools for request in model.requests)

    asyncio.run(exercise())


def test_provider_replacement_is_visible_only_at_a_later_toolbox_boundary() -> None:
    @a2a_agent(name="DocumentationAgent", version="2.0.0", description="Replacement docs.")
    class ReplacementDocumentationAgent(BaseAgent):
        @a2a_capability(
            name="documentation.search",
            description="Searches replacement documentation.",
            tags=("documentation", "search"),
            model_required=False,
        )
        def search(self, query: str) -> dict[str, str]:
            return {"source": "replacement-docs", "excerpt": query}

    async def exercise() -> None:
        orchestrator, _model = build_application()
        runtime = orchestrator.runtime
        registry = runtime.agent_registry
        context = runtime.create_run_context(agent_id="KnowledgeAgent")
        policy = ToolboxPolicy(
            uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
        )

        with use_run_context(context):
            initial = await build_toolbox(context.gateway, policy)
            assert initial.snapshot is not None
            initial_binding = initial.snapshot.tools[0].binding
            initial_payload = initial.snapshot.as_model_payload()

            replaced = registry.register(ReplacementDocumentationAgent(), replace=True)
            assert isinstance(replaced, DocumentationAgent)

            later = await build_toolbox(context.gateway, policy)
            assert later.snapshot is not None
            assert initial.snapshot.as_model_payload() == initial_payload
            assert later.snapshot.registry_revision > initial.snapshot.registry_revision
            assert later.snapshot.tools[0].binding.registration_generation != (
                initial_binding.registration_generation
            )

        result = await orchestrator.invoke(
            "KnowledgeAgent",
            "answer_product",
            {"question": "What do docs say about approvals?"},
            correlation_id="chain-replaced-provider",
        )
        assert isinstance(result, InvocationSuccess)

    asyncio.run(exercise())
