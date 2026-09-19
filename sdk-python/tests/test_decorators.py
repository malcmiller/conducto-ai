import pytest

from conducto import a2a_agent, a2a_capability, tool
from conducto.core.decorators import (
    AgentMetadata,
    ExportMetadata,
    MethodMetadata,
    get_agent_metadata,
    get_method_metadata,
)


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
