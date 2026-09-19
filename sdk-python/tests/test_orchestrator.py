import pytest

from conducto import BaseAgent, OrchestratorAgent, a2a_agent, a2a_capability


def test_orchestrator_registers_agents_deterministically_and_renders_prompt_context() -> None:
    @a2a_agent(name="BravoAgent", version="1.0.0", description="Second local agent.")
    class BravoAgent(BaseAgent):
        @a2a_capability(name="greet", description="Greets a person.")
        def greet(self, name: str) -> str:
            return f"hello {name}"

    @a2a_agent(name="AlphaAgent", version="1.0.0", description="First local agent.")
    class AlphaAgent(BaseAgent):
        @a2a_capability(name="lookup", description="Looks up data.")
        def lookup(self, query: str) -> str:
            return query

    orchestrator = OrchestratorAgent()
    orchestrator.register_agent(BravoAgent())
    orchestrator.register_agent(AlphaAgent())

    assert [agent.agent_metadata.name for agent in orchestrator.registered_agents] == [
        "AlphaAgent",
        "BravoAgent",
    ]
    assert [entry["name"] for entry in orchestrator.get_routing_metadata()] == [
        "AlphaAgent",
        "BravoAgent",
    ]

    prompt_context = orchestrator.get_routing_prompt_context()
    assert "[BEGIN UNTRUSTED LOCAL AGENT DATA]" in prompt_context
    assert "[END UNTRUSTED LOCAL AGENT DATA]" in prompt_context
    assert "AlphaAgent" in prompt_context
    assert "BravoAgent" in prompt_context


def test_orchestrator_rejects_duplicates_and_supports_replacement_removal() -> None:
    @a2a_agent(name="SharedAgent", version="1.0.0", description="Keeps state.")
    class SharedAgent(BaseAgent):
        @a2a_capability(name="ping", description="Pings.")
        def ping(self) -> str:
            return "pong"

    orchestrator = OrchestratorAgent()
    first = SharedAgent()
    orchestrator.register_agent(first)

    with pytest.raises(ValueError, match="already registered"):
        orchestrator.register_agent(SharedAgent())

    second = SharedAgent()
    assert orchestrator.register_agent(second, replace=True) is first
    assert orchestrator.get_agent_by_name("SharedAgent") is second

    removed = orchestrator.remove_agent("SharedAgent")
    assert removed is second
    assert len(orchestrator) == 0


def test_orchestrator_rejects_capability_name_conflicts() -> None:
    @a2a_agent(name="FirstAgent", version="1.0.0", description="A first agent.")
    class FirstAgent(BaseAgent):
        @a2a_capability(name="lookup", description="Looks up values.")
        def lookup(self, value: str) -> str:
            return value

    @a2a_agent(name="SecondAgent", version="1.0.0", description="A second agent.")
    class SecondAgent(BaseAgent):
        @a2a_capability(name="lookup", description="Looks up values too.")
        def lookup(self, value: str) -> str:
            return value

    orchestrator = OrchestratorAgent()
    orchestrator.register_agent(FirstAgent())

    with pytest.raises(ValueError, match="Capability name conflict"):
        orchestrator.register_agent(SecondAgent())

    replacement = SecondAgent()
    assert orchestrator.register_agent(replacement, replace=True) is None
    assert orchestrator.get_agent_by_name("SecondAgent") is replacement
    assert "lookup" in orchestrator.registered_capabilities
