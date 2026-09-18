from conducto import BaseAgent, a2a_agent, a2a_capability, tool


def test_base_agent_registers_decorated_methods() -> None:
    @a2a_agent(name="RegistryAgent", version="1.2.3", description="Example agent")
    class RegistryAgent(BaseAgent):
        @a2a_capability(name="greet", description="Greets a person.")
        def greet(self, name: str, count: int = 1) -> str:
            return f"{name} {count}"

        @tool(name="secret", description="Internal helper.")
        def secret(self, value: int) -> int:
            return value * 2

    agent = RegistryAgent()

    assert agent.agent_metadata.name == "RegistryAgent"
    assert agent.capabilities["greet"].callable == agent.greet
    assert agent.tools["secret"].callable == agent.secret
    assert agent.capabilities["greet"].parameter_schema["properties"]["name"]["type"] == "string"
    assert agent.capabilities["greet"].parameter_schema["required"] == ["name"]
