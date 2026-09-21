"""Three-agent local chaining example using only the public ``conducto`` API.

Run from a built wheel with:

    uv run --no-project --with dist/conducto_ai-*.whl python examples/agent_chaining.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from conducto import (
    AgentRegistry,
    BaseAgent,
    CapabilityUse,
    ChatMessage,
    DelegationConfig,
    FakeModelRequest,
    InvocationSuccess,
    ModelConfiguration,
    OrchestratorAgent,
    ProviderCapabilities,
    ProviderRegistry,
    ProviderResult,
    Runtime,
    StructuredOutputRequest,
    ToolboxPolicy,
    Usage,
    a2a_agent,
    a2a_capability,
)


class ProductAnswer(BaseModel):
    """Validated terminal response produced by a parent agent."""

    answer: str
    evidence: str | None = None


class ChainingModel:
    """Deterministic local model that chooses routing, tools, and terminal values."""

    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calling=True,
        usage_reporting=True,
    )

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[FakeModelRequest] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: Any,
        structured_output: StructuredOutputRequest,
        tools: Sequence[Mapping[str, Any]] = (),
        tool_results: Sequence[Any] = (),
        effective_deadline: float | None = None,
    ) -> ProviderResult:
        self.calls += 1
        self.requests.append(
            FakeModelRequest(
                message_roles=tuple(message.role for message in messages),
                options=options,
                structured_output=structured_output,
                tools=tuple(tools),
                tool_results=tuple(tool_results),
                effective_deadline=effective_deadline,
            )
        )
        question = next(
            (message.content for message in messages if message.role == "user"),
            "",
        )
        if structured_output.name == "conducto_capability_selection":
            agent_id = "DiagnosticAgent" if "fail" in question.lower() else "KnowledgeAgent"
            return ProviderResult(
                structured={
                    "agent_id": agent_id,
                    "capability_id": (
                        "diagnose" if agent_id == "DiagnosticAgent" else "answer_product"
                    ),
                    "arguments": {"question": question},
                },
                usage=Usage(total_tokens=1),
                accepted=True,
            )
        response: dict[str, Any]
        if "what is conducto" in question.lower():
            response = {"answer": "Conducto coordinates typed local agents.", "evidence": None}
        elif tool_results:
            response = {
                "answer": "The documentation evidence was incorporated.",
                "evidence": "documentation.search",
            }
        else:
            response = {
                "type": "tool_call",
                "call_id": f"docs-{self.calls}",
                "tool_id": str(tools[0]["id"]),
                "arguments": {"query": question},
            }
            return ProviderResult(structured=response, usage=Usage(total_tokens=1), accepted=True)
        return ProviderResult(
            structured={"type": "terminal", "response": response},
            usage=Usage(total_tokens=1),
            accepted=True,
        )


@a2a_agent(name="DocumentationAgent", version="1.0.0", description="Searches product docs.")
class DocumentationAgent(BaseAgent):
    @a2a_capability(
        name="documentation.search",
        description="Returns untrusted searchable documentation evidence.",
        tags=("documentation", "search"),
        model_required=False,
    )
    def search(self, query: str) -> dict[str, str]:
        return {"source": "local-docs", "excerpt": f"Evidence for: {query}"}


class _AnsweringAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            delegation_config=DelegationConfig(
                toolbox=ToolboxPolicy(
                    uses=(CapabilityUse(capability_ids=frozenset({"documentation.search"})),)
                ),
                max_model_turns=3,
                max_tool_calls=1,
            )
        )

    async def _answer(self, question: str) -> ProductAnswer:
        outcome = await self.run_delegation(
            (ChatMessage(role="user", content=question),),
            response_type=ProductAnswer,
        )
        if not outcome.ok or outcome.value is None:
            raise RuntimeError(f"Delegation failed: {outcome.code.value}")
        return outcome.value


@a2a_agent(
    name="KnowledgeAgent",
    version="1.0.0",
    description="Answers product questions.",
    default_model="local-model",
    model_required=True,
)
class KnowledgeAgent(_AnsweringAgent):
    @a2a_capability(name="answer_product", description="Answers a product question.")
    async def answer_product(self, question: str) -> ProductAnswer:
        return await self._answer(question)


@a2a_agent(
    name="DiagnosticAgent",
    version="1.0.0",
    description="Diagnoses product failures.",
    default_model="local-model",
    model_required=True,
)
class DiagnosticAgent(_AnsweringAgent):
    @a2a_capability(name="diagnose", description="Diagnoses a product failure.")
    async def diagnose(self, question: str) -> ProductAnswer:
        return await self._answer(question)


def build_application() -> tuple[OrchestratorAgent, ChainingModel]:
    """Build the local registry, one runtime, and the three public agents."""
    model = ChainingModel()
    providers = ProviderRegistry()
    providers.register(
        "local-model",
        model,
        ModelConfiguration(provider="fake", model="local-model"),
    )
    documentation_registry = AgentRegistry()
    documentation_registry.register(DocumentationAgent())
    runtime = Runtime(provider_registry=providers, agent_registry=documentation_registry)
    orchestrator = OrchestratorAgent(runtime=runtime, model_reference="local-model")
    orchestrator.register_agent(KnowledgeAgent())
    orchestrator.register_agent(DiagnosticAgent())
    return orchestrator, model


async def main() -> None:
    """Show no-tool, direct, and model-selected delegated paths."""
    orchestrator, _model = build_application()

    no_tool = await orchestrator.route(
        "What is Conducto?",
        correlation_id="chaining-no-tool",
    )
    direct = await orchestrator.invoke(
        "DiagnosticAgent",
        "diagnose",
        {"question": "Why did my build fail?"},
        correlation_id="chaining-direct",
    )
    delegated = await orchestrator.route(
        "What does product documentation say about approvals?",
        correlation_id="chaining-routed",
    )
    assert all(isinstance(result, InvocationSuccess) for result in (no_tool, direct, delegated))
    print("agent chaining passed: no-tool, direct, and model-selected delegation.")


if __name__ == "__main__":
    asyncio.run(main())
