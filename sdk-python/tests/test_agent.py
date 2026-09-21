import json

import pytest

from conducto import (
    A2A_AGENT_CARD_SPEC_VERSION,
    A2A_JSONRPC_BINDING,
    A2A_PROTOCOL_VERSION,
    BaseAgent,
    a2a_agent,
    a2a_capability,
    tool,
)


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


def test_base_agent_generates_deterministic_a2a_agent_card() -> None:
    @a2a_agent(name="RegistryAgent", version="1.2.3", description="Example agent")
    class RegistryAgent(BaseAgent):
        @a2a_capability(name="greet", description="Greets a person.")
        def greet(self, name: str) -> str:
            return name

    agent = RegistryAgent()
    card = agent.get_agent_card("https://example.test/a2a")

    assert A2A_AGENT_CARD_SPEC_VERSION == A2A_PROTOCOL_VERSION == "1.0"
    assert card["supportedInterfaces"] == [
        {
            "protocolBinding": A2A_JSONRPC_BINDING,
            "protocolVersion": A2A_AGENT_CARD_SPEC_VERSION,
            "tenant": "",
            "url": "https://example.test/a2a",
        }
    ]
    assert card["skills"][0]["id"].startswith("conducto-")
    assert card["skills"][0]["name"] == "greet"
    skill_id = card["skills"][0]["id"]
    extension = card["capabilities"]["extensions"][0]
    assert extension["uri"] == "https://conducto.ai/a2a/extensions/parameters/v1"
    assert extension["required"] is False
    assert (
        extension["params"]["x-conducto"]["parameters"][skill_id]["properties"]["name"]["type"]
        == "string"
    )
    serialized = agent.get_agent_card_json("https://example.test/a2a")
    assert serialized == agent.get_agent_card_json("https://example.test/a2a")
    assert json.loads(serialized) == card


def test_agent_card_rejects_incomplete_metadata_and_endpoint() -> None:
    class UndescribedAgent(BaseAgent):
        @a2a_capability(name="lookup")
        def lookup(self, value: str) -> str:
            return value

    agent = UndescribedAgent()
    with pytest.raises(ValueError, match="absolute http or https URL"):
        agent.get_agent_card("/a2a")

    class NoDescriptionAgent(BaseAgent):
        pass

    with pytest.raises(ValueError, match="requires a description"):
        NoDescriptionAgent().get_agent_card("https://example.test/a2a")


def test_agent_card_validates_security_objects_and_uses_standard_security_field() -> None:
    class SecureAgent(BaseAgent):
        """A secure agent."""

        @a2a_capability(name="lookup", description="Finds a value.")
        def lookup(self, value: str) -> str:
            return value

    agent = SecureAgent()
    card = agent.get_agent_card(
        "https://example.test/a2a",
        security_schemes={
            "oauth": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "scopes": {},
                        "tokenUrl": "https://example.test/token",
                    }
                },
            }
        },
        security_requirements=[{"oauth": ["read"]}],
    )
    assert card["securityRequirements"] == [{"schemes": {"oauth": {"list": ["read"]}}}]
    assert "security" not in card

    with pytest.raises(ValueError, match="scopes must be a sequence"):
        agent.get_agent_card(
            "https://example.test/a2a",
            security_requirements=[{"oauth": "read"}],
        )
    with pytest.raises(ValueError, match="unsupported type"):
        agent.get_agent_card(
            "https://example.test/a2a",
            security_schemes={"unknown": {"type": "unknown"}},
        )
    with pytest.raises(ValueError, match="requires a scope-description mapping"):
        agent.get_agent_card(
            "https://example.test/a2a",
            security_schemes={
                "oauth": {
                    "type": "oauth2",
                    "flows": {
                        "clientCredentials": {
                            "scopes": {"read": 1},
                            "tokenUrl": "https://example.test/token",
                        }
                    },
                }
            },
        )


def test_agent_card_rejects_invalid_hostname_and_transport_whitespace() -> None:
    class Agent(BaseAgent):
        """An agent."""

    agent = Agent()
    with pytest.raises(ValueError, match="absolute http or https URL"):
        agent.get_agent_card("http://:80")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        agent.get_agent_card(
            "https://example.test/a2a",
            preferred_transport=" JSONRPC ",
        )
    with pytest.raises(ValueError, match="only supports JSONRPC"):
        agent.get_agent_card(
            "https://example.test/a2a",
            preferred_transport="GRPC",
        )


def test_agent_card_rejects_unsupported_media_modes() -> None:
    class Agent(BaseAgent):
        """An agent."""

    with pytest.raises(ValueError, match="unsupported media type"):
        Agent().get_agent_card(
            "https://example.test/a2a",
            default_input_modes=["image/png"],
        )
