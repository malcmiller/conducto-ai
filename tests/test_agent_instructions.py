import asyncio

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.instructions import resolve_instruction_chain


def test_instruction_precedence_is_stable() -> None:
    assert resolve_instruction_chain(
        ("runtime one", "runtime two"),
        "agent",
        "capability",
    ) == ("runtime one", "runtime two", "agent", "capability")


def test_agent_and_capability_instructions_are_metadata() -> None:
    @a2a_agent(
        description="An instructed agent.",
        instructions="Act as an analyst.",
        publish_instructions=True,
    )
    class InstructedAgent(BaseAgent):
        @a2a_capability(
            name="analyze",
            description="Analyze input.",
            instructions="Use concise findings.",
        )
        def analyze(self) -> str:
            return "ok"

    agent = InstructedAgent()
    assert agent.agent_metadata.instructions == "Act as an analyst."
    assert agent.capabilities["analyze"].capability is not None
    assert agent.capabilities["analyze"].capability.instructions == "Use concise findings."
    card = agent.get_agent_card("https://example.test/agent")
    extension = card["capabilities"]["extensions"][0]["params"]["x-conducto"]
    assert extension["instructions"] == "Act as an analyst."


def test_instructions_are_not_published_by_default() -> None:
    @a2a_agent(description="Private instructions.", instructions="Keep this private.")
    class PrivateAgent(BaseAgent):
        pass

    card = PrivateAgent().get_agent_card("https://example.test/agent")
    extension = card["capabilities"]["extensions"][0]["params"]["x-conducto"]
    assert "instructions" not in extension


def test_runtime_context_records_policy_and_declared_instructions() -> None:
    @a2a_agent(
        description="An instructed agent.",
        instructions="Agent guidance.",
        model_required=False,
    )
    class InstructedAgent(BaseAgent):
        @a2a_capability(
            name="work",
            description="Does work.",
            instructions="Capability guidance.",
            model_required=False,
        )
        def work(self) -> str:
            return "done"

    runtime = Runtime(policy_instructions=("Runtime policy.",))
    result = asyncio.run(runtime.invoke(InstructedAgent(), "work", {}))
    assert result.metadata is not None
    assert result.metadata.instruction_chain == (
        "Runtime policy.",
        "Agent guidance.",
        "Capability guidance.",
    )
