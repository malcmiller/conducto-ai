"""Decorated-method discovery and export registration."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .decorators import AgentMetadata, ExportMetadata, get_agent_metadata, get_method_metadata
from .parameter_schema import (
    ParameterSchemaError,
    build_parameter_model,
    build_parameter_schema,
)


class AgentRegistrationError(ValueError):
    """Raised when an agent or decorated method cannot be registered."""


@dataclass(frozen=True, slots=True)
class RegisteredMethod:
    """A reflected method and the metadata registered for its exports.

    Attributes:
        attribute_name: Attribute name on the agent owning the reflected method.
        callable: Bound callable that implements the capability or tool.
        parameter_schema: Generated JSON schema for the callable arguments.
        parameter_model: Generated Pydantic model for validating arguments.
        capability: Capability metadata, if the method is exported as a capability.
        tool: Tool metadata, if the method is exported as a tool.
    """

    attribute_name: str
    callable: Callable[..., Any]
    parameter_schema: dict[str, Any]
    parameter_model: type[BaseModel]
    capability: ExportMetadata | None = None
    tool: ExportMetadata | None = None


def register_decorated_methods(
    agent: Any,
) -> tuple[
    dict[str, RegisteredMethod],
    dict[str, RegisteredMethod],
    dict[str, RegisteredMethod],
]:
    """Discover decorated methods and build each parameter model exactly once.

    Args:
        agent: Agent instance providing reflection hooks and export registration.

    Returns:
        A tuple of (registered_methods, capabilities, tools).

    Raises:
        AgentRegistrationError: If a decorated method is not callable or its schema cannot be built.
    """
    registered_methods: dict[str, RegisteredMethod] = {}
    capabilities: dict[str, RegisteredMethod] = {}
    tools: dict[str, RegisteredMethod] = {}

    agent_type = type(agent)
    for attribute_name, declared_value in resolved_attributes(agent_type):
        metadata = get_method_metadata(declared_value)
        if metadata is None:
            continue

        bound_method = getattr(agent, attribute_name)
        if not callable(bound_method):
            raise AgentRegistrationError(
                f"{type(agent).__name__}.{attribute_name} is decorated but is not callable"
            )

        try:
            parameter_model = build_parameter_model(agent_type, attribute_name, bound_method)
            parameter_schema = build_parameter_schema(
                agent_type,
                attribute_name,
                parameter_model,
            )
        except ParameterSchemaError as error:
            raise AgentRegistrationError(str(error)) from error.__cause__

        registered = RegisteredMethod(
            attribute_name=attribute_name,
            callable=bound_method,
            parameter_schema=parameter_schema,
            parameter_model=parameter_model,
            capability=resolve_export_metadata(
                attribute_name,
                bound_method,
                metadata.capability,
            ),
            tool=resolve_export_metadata(
                attribute_name,
                bound_method,
                metadata.tool,
            ),
        )
        if registered.capability is not None:
            capability_name = registered.capability.name
            assert capability_name is not None
            add_export(
                agent_type,
                capabilities,
                capability_name,
                registered,
                export_kind="capability",
            )
        if registered.tool is not None:
            tool_name = registered.tool.name
            assert tool_name is not None
            add_export(
                agent_type,
                tools,
                tool_name,
                registered,
                export_kind="tool",
            )
        registered_methods[attribute_name] = registered

    return registered_methods, capabilities, tools


def resolved_attributes(agent_type: type[Any]) -> list[tuple[str, Any]]:
    """Return effective callable attributes in deterministic name order.

    Args:
        agent_type: Concrete agent type whose attributes should be reflected.

    Returns:
        Callable attributes of the class hierarchy sorted by attribute name.
    """
    resolved: dict[str, Any] = {}
    for agent_class in reversed(agent_type.__mro__):
        if agent_class is object:
            continue
        for attribute_name, value in agent_class.__dict__.items():
            if attribute_name.startswith("__"):
                continue
            if not (callable(value) or isinstance(value, (classmethod, staticmethod))):
                continue
            resolved[attribute_name] = value
    return sorted(resolved.items(), key=lambda item: item[0])


def resolve_agent_metadata(agent_type: type[Any]) -> AgentMetadata:
    """Resolve declared metadata or stable class-derived defaults.

    Args:
        agent_type: Concrete agent type to inspect.

    Returns:
        The effective agent metadata for the type.
    """
    metadata = get_agent_metadata(agent_type)
    if metadata is not None:
        return metadata
    return AgentMetadata(
        name=agent_type.__name__,
        version="0.1.0",
        description=(
            inspect.cleandoc(agent_type.__dict__["__doc__"])
            if isinstance(agent_type.__dict__.get("__doc__"), str)
            else None
        ),
    )


def resolve_export_metadata(
    attribute_name: str,
    method: Callable[..., Any],
    metadata: ExportMetadata | None,
) -> ExportMetadata | None:
    """Fill omitted export fields from the method declaration.

    Args:
        attribute_name: Attribute name on the agent.
        method: Underlying method being exported.
        metadata: Declared export metadata, if any.

    Returns:
        The normalized export metadata or None when no export is declared.
    """
    if metadata is None:
        return None
    return ExportMetadata(
        name=metadata.name or attribute_name,
        description=metadata.description or inspect.getdoc(method),
        model_required=metadata.model_required,
        tags=metadata.tags,
        instructions=metadata.instructions,
    )


def add_export(
    agent_type: type[Any],
    registry: dict[str, RegisteredMethod],
    export_name: str,
    method: RegisteredMethod,
    *,
    export_kind: str,
) -> None:
    """Add an export while preserving duplicate-name diagnostics.

    Args:
        agent_type: Concrete agent type that owns the export.
        registry: Mutable export registry to update.
        export_name: Name assigned to the export.
        method: Registered method entry to store.
        export_kind: Kind of export being registered, such as "capability".

    Raises:
        AgentRegistrationError: If the export name duplicates an existing entry.
    """
    existing = registry.get(export_name)
    if existing is not None:
        raise AgentRegistrationError(
            f"Duplicate {export_kind} name '{export_name}' on "
            f"{agent_type.__name__}.{existing.attribute_name} and "
            f"{agent_type.__name__}.{method.attribute_name}"
        )
    registry[export_name] = method
