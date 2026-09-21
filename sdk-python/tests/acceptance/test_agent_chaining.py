"""Acceptance coverage for the public three-agent chaining example."""

from __future__ import annotations

import asyncio

import pytest
from examples.agent_chaining import build_application

from conducto import InvocationSuccess

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
