from decimal import Decimal

import pytest

from conducto import (
    BaseAgent,
    a2a_agent,
    a2a_capability,
    budget,
    classification,
    requires_scope,
    side_effect,
    timeout,
    tool,
)
from conducto.core.decorators import (
    AgentMetadata,
    CapabilityBudget,
    CapabilityPolicyMetadata,
    ExportMetadata,
    MethodMetadata,
    PolicyMetadataError,
    get_agent_metadata,
    get_method_metadata,
)
from conducto.core.registry import AgentRegistry


def test_a2a_agent_uses_defaults_and_strips_metadata_text() -> None:
    @a2a_agent
    class DefaultAgent:
        """The agent description."""

    metadata = get_agent_metadata(DefaultAgent)
    assert metadata == AgentMetadata(
        name="DefaultAgent",
        version="0.1.0",
        description="The agent description.",
    )

    @a2a_agent(name="  Example Agent  ", version=" 2.3.1 ", description="  Example desc  ")
    class CustomAgent:
        pass

    assert get_agent_metadata(CustomAgent) == AgentMetadata(
        name="Example Agent",
        version="2.3.1",
        description="Example desc",
    )

    with pytest.raises(TypeError, match="can only decorate classes"):

        @a2a_agent
        def not_a_class() -> None:
            pass

    with pytest.raises(ValueError, match="version cannot be empty"):

        class InvalidVersionAgent:
            pass

        a2a_agent(version="   ")(InvalidVersionAgent)

    with pytest.raises(ValueError, match="Decorator metadata cannot contain empty text"):

        class InvalidNameAgent:
            pass

        a2a_agent(name="  ")(InvalidNameAgent)


def test_method_decorators_store_metadata_for_regular_and_descriptor_methods() -> None:
    class Example:
        @a2a_capability(name="lookup", description="Find a value.")
        @tool(name="lookup-tool", description="Internal lookup.")
        def regular(self, value: int) -> int:
            return value

        @classmethod
        @tool(name="count", description="Count values.")
        def class_method(cls, value: int) -> int:
            return value

        @staticmethod
        @a2a_capability(name="double", description="Double it.")
        def static_method(value: int) -> int:
            return value

    assert get_method_metadata(Example.regular) == MethodMetadata(
        capability=ExportMetadata(name="lookup", description="Find a value."),
        tool=ExportMetadata(name="lookup-tool", description="Internal lookup."),
    )
    assert get_method_metadata(Example.class_method) == MethodMetadata(
        tool=ExportMetadata(name="count", description="Count values."),
    )
    assert get_method_metadata(Example.static_method) == MethodMetadata(
        capability=ExportMetadata(name="double", description="Double it."),
    )


def test_metadata_accessors_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="only decorate callables"):
        get_method_metadata(123)

    class InvalidMethodMetadata:
        __conducto_method_metadata__: object = None

    InvalidMethodMetadata.__conducto_method_metadata__ = object()
    with pytest.raises(TypeError, match="Invalid Conducto method metadata"):
        get_method_metadata(InvalidMethodMetadata)

    class InvalidAgentMetadata:
        __conducto_agent_metadata__: object = None

    InvalidAgentMetadata.__conducto_agent_metadata__ = object()
    with pytest.raises(TypeError, match="Invalid Conducto agent metadata"):
        get_agent_metadata(InvalidAgentMetadata)


@pytest.mark.parametrize("capability_outermost", [False, True])
def test_capability_policy_decorators_stack_in_any_order(capability_outermost: bool) -> None:
    def decorate(function: object) -> object:
        decorators = (
            a2a_capability(name="refund", description="Creates a refund."),
            requires_scope("refund.write", "refund.read"),
            side_effect("writes_external_system"),
            timeout(seconds=5),
            budget(max_model_calls=0, max_tool_calls=1, max_cost_usd=Decimal("2.50")),
            classification("confidential"),
        )
        ordered = decorators if capability_outermost else tuple(reversed(decorators))
        decorated = function
        for decorator in reversed(ordered):
            decorated = decorator(decorated)
        return decorated

    class RefundAgent(BaseAgent):
        """Creates customer refunds."""

        create_refund = decorate(lambda self: "created")

    agent = RefundAgent()
    registered = agent.capabilities["refund"]
    expected = CapabilityPolicyMetadata(
        required_scopes=("refund.read", "refund.write"),
        side_effect="writes_external_system",
        timeout_seconds=5.0,
        budget=CapabilityBudget(max_model_calls=0, max_tool_calls=1, max_cost_usd=Decimal("2.50")),
        data_classification="confidential",
    )
    assert registered.policy == expected

    descriptor = AgentRegistry()
    descriptor.register(agent)
    capability = descriptor.snapshot().capabilities[0]
    assert capability.policy == expected
    assert capability.required_scopes == ("refund.read", "refund.write")

    card = agent.get_agent_card("https://example.test/a2a")
    skill_id = card["skills"][0]["id"]
    policy = card["capabilities"]["extensions"][0]["params"]["x-conducto"]["capabilityPolicies"]
    assert policy == {
        skill_id: {
            "requiredScopes": ["refund.read", "refund.write"],
            "sideEffect": "writes_external_system",
            "timeoutSeconds": 5.0,
            "budget": {"maxModelCalls": 0, "maxToolCalls": 1, "maxCostUsd": "2.50"},
            "dataClassification": "confidential",
        }
    }


@pytest.mark.parametrize(
    ("decorator", "message"),
    [
        (lambda: requires_scope(), "at least one"),
        (lambda: requires_scope(" "), "non-empty"),
        (lambda: side_effect(" "), "non-empty"),
        (lambda: timeout(seconds=0), "finite positive"),
        (lambda: timeout(seconds=float("inf")), "finite positive"),
        (lambda: budget(), "at least one"),
        (lambda: budget(max_model_calls=-1), "non-negative"),
        (lambda: budget(max_tool_calls=True), "non-negative"),
        (lambda: budget(max_cost_usd=Decimal("-0.01")), "non-negative"),
        (lambda: classification(" "), "non-empty"),
    ],
)
def test_capability_policy_decorators_reject_invalid_arguments(
    decorator: object, message: str
) -> None:
    with pytest.raises(PolicyMetadataError, match=message):
        decorator()


def test_capability_without_policy_keeps_agent_card_contract_unchanged() -> None:
    class Agent(BaseAgent):
        """Looks up values."""

        @a2a_capability(name="lookup", description="Looks up a value.")
        def lookup(self) -> str:
            return "value"

    card = Agent().get_agent_card("https://example.test/a2a")
    extension = card["capabilities"]["extensions"][0]["params"]["x-conducto"]
    assert "capabilityPolicies" not in extension
