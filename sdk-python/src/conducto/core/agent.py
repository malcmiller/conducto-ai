"""Agent registration and parameter-schema generation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import ConfigDict, PydanticSchemaGenerationError, create_model

from .decorators import (
    AgentMetadata,
    ExportMetadata,
    get_agent_metadata,
    get_method_metadata,
)


class AgentRegistrationError(ValueError):
    """Raised when an agent contains an invalid decorated method.

    This includes unsupported method signatures, unresolved or missing type
    annotations, invalid parameter schemas, and duplicate export names.
    """


@dataclass(frozen=True, slots=True)
class RegisteredMethod:
    """A reflected method and the metadata registered for its exports.

    Attributes:
        attribute_name: The method's attribute name on the agent class.
        callable: The bound callable exposed by the agent instance.
        parameter_schema: The JSON schema for the method's parameters.
        capability: Resolved A2A capability metadata, if applicable.
        tool: Resolved Conducto tool metadata, if applicable.
    """

    attribute_name: str
    callable: Callable[..., Any]
    parameter_schema: dict[str, Any]
    capability: ExportMetadata | None = None
    tool: ExportMetadata | None = None


class BaseAgent:
    """Base class for reflected Conducto agents.

    Subclasses can use: func:`conducto.core.decorators.a2a_agent`,
    func:`conducto.core.decorators.a2a_capability`, and
    func:`conducto.core.decorators.tool` to declare agent and method
    metadata. Construction reflects those declarations, resolves export
    names and descriptions, and generates strict Pydantic JSON schemas for
    method parameters.

    Raises:
        AgentRegistrationError: If a decorated method cannot be registered.
    """

    def __init__(self) -> None:
        """Initialize the agent and register its decorated methods."""
        self.agent_metadata = self._resolve_agent_metadata()
        self._registered_methods: dict[str, RegisteredMethod] = {}
        self._capabilities: dict[str, RegisteredMethod] = {}
        self._tools: dict[str, RegisteredMethod] = {}
        self._register_decorated_tools()

    @property
    def registered_methods(self) -> tuple[RegisteredMethod, ...]:
        """Return all decorated methods in deterministic attribute order."""
        return tuple(self._registered_methods.values())

    @property
    def capabilities(self) -> dict[str, RegisteredMethod]:
        """Return the registered A2A capabilities keyed by export name."""
        return dict(self._capabilities)

    @property
    def tools(self) -> dict[str, RegisteredMethod]:
        """Return the registered Conducto tools keyed by export name."""
        return dict(self._tools)

    def _register_decorated_tools(self) -> None:
        """Reflect and register every decorated method on the agent.

        Subclasses may override this method when they need custom
        registration behavior but should preserve the registration
        invariants enforced by the base implementation.

        Raises:
            AgentRegistrationError: If a decorated attribute is not callable,
                its signature cannot be represented, its parameter schema
                cannot be generated, or an export name is duplicated.
        """
        registered_methods: dict[str, RegisteredMethod] = {}
        capabilities: dict[str, RegisteredMethod] = {}
        tools: dict[str, RegisteredMethod] = {}

        for attribute_name, declared_value in self._resolved_attributes():
            metadata = get_method_metadata(declared_value)
            if metadata is None:
                continue

            bound_method = getattr(self, attribute_name)
            if not callable(bound_method):
                raise AgentRegistrationError(
                    f"{type(self).__name__}.{attribute_name} is decorated "
                    "but is not callable"
                )

            registered = RegisteredMethod(
                attribute_name=attribute_name,
                callable=bound_method,
                parameter_schema=self._build_parameter_schema(
                    attribute_name,
                    bound_method,
                ),
                capability=self._resolve_export_metadata(
                    attribute_name,
                    bound_method,
                    metadata.capability,
                ),
                tool=self._resolve_export_metadata(
                    attribute_name,
                    bound_method,
                    metadata.tool,
                ),
            )

            if registered.capability is not None:
                capability_name = registered.capability.name
                assert capability_name is not None
                self._add_export(
                    capabilities,
                    capability_name,
                    registered,
                    export_kind="capability",
                )

            if registered.tool is not None:
                tool_name = registered.tool.name
                assert tool_name is not None
                self._add_export(
                    tools,
                    tool_name,
                    registered,
                    export_kind="tool",
                )

            registered_methods[attribute_name] = registered

        self._registered_methods = registered_methods
        self._capabilities = capabilities
        self._tools = tools

    def _resolved_attributes(self) -> list[tuple[str, Any]]:
        """Return effective class attributes after applying inheritance.

        Base-class declarations are collected first, and subclass declarations
        replace attributes with the same name. The resulting attributes are
        sorted by name to make registration deterministic.
        """
        resolved: dict[str, Any] = {}

        # Base classes are applied first. Subclass declarations then replace
        # inherited attributes, including decorated methods.
        for agent_class in reversed(type(self).__mro__):
            if agent_class is object:
                continue

            for attribute_name, value in agent_class.__dict__.items():
                if attribute_name.startswith("__"):
                    continue
                if not (callable(value) or isinstance(value, (classmethod, staticmethod))):
                    continue
                resolved[attribute_name] = value

        return sorted(resolved.items(), key=lambda item: item[0])

    def _build_parameter_schema(
        self,
        attribute_name: str,
        method: Callable[..., Any],
    ) -> dict[str, Any]:
        """Build a strict JSON schema for a method's parameters.

        Variadic, positional-only, and unannotated parameters are not
        supported. Parameters without defaults are required; parameters with
        defaults retain those defaults in the generated schema.

        Args:
            attribute_name: The method's attribute name, used in errors and
                the generated model name.
            method: The bound callable whose signature should be modeled.

        Returns:
            A Pydantic JSON schema with extra parameters forbidden.

        Raises:
            AgentRegistrationError: If annotations cannot be resolved, a
                parameter is unsupported or unannotated, or Pydantic cannot
                generate the schema.
        """
        signature = inspect.signature(method)

        try:
            type_hints = get_type_hints(method)
        except (NameError, TypeError) as error:
            raise AgentRegistrationError(
                f"Could not resolve type annotations for "
                f"{type(self).__name__}.{attribute_name}: {error}"
            ) from error

        fields: dict[str, tuple[Any, Any]] = {}

        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            }:
                raise AgentRegistrationError(
                    f"{type(self).__name__}.{attribute_name} uses unsupported "
                    f"parameter '{parameter.name}' of kind "
                    f"{parameter.kind.description}"
                )

            annotation = type_hints.get(parameter.name)
            if annotation is None:
                raise AgentRegistrationError(
                    f"{type(self).__name__}.{attribute_name} parameter "
                    f"'{parameter.name}' must have a type annotation"
                )

            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            fields[parameter.name] = (annotation, default)

        model_name = (
            f"{type(self).__name__}_{attribute_name}_Parameters"
        )

        try:
            parameter_model = create_model(
                model_name,
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
            return parameter_model.model_json_schema()
        except PydanticSchemaGenerationError as error:
            raise AgentRegistrationError(
                f"Could not generate a parameter schema for "
                f"{type(self).__name__}.{attribute_name}: {error}"
            ) from error

    def _resolve_agent_metadata(self) -> AgentMetadata:
        """Resolve declared agent metadata or provide default metadata.

        Returns:
            Explicit metadata declared on the concrete class. If none is
            declared, metadata using the class name, version ``"0.1.0"``,
            and class docstring is returned.
        """
        metadata = get_agent_metadata(type(self))
        if metadata is not None:
            return metadata

        return AgentMetadata(
            name=type(self).__name__,
            version="0.1.0",
            description=inspect.getdoc(type(self)),
        )

    @staticmethod
    def _resolve_export_metadata(
        attribute_name: str,
        method: Callable[..., Any],
        metadata: ExportMetadata | None,
    ) -> ExportMetadata | None:
        """Resolve omitted export fields from a method and its attribute name.

        Args:
            attribute_name: The method's attribute name is used as the fallback
                export name.
            method: The method whose docstring supplies the fallback
                description.
            metadata: The declared export metadata, if any.

        Returns:
            Resolved export metadata, or ``None`` when no export was declared.
        """
        if metadata is None:
            return None

        return ExportMetadata(
            name=metadata.name or attribute_name,
            description=metadata.description or inspect.getdoc(method),
        )

    def _add_export(
        self,
        registry: dict[str, RegisteredMethod],
        export_name: str,
        method: RegisteredMethod,
        *,
        export_kind: str,
    ) -> None:
        """Add an export to a registry, rejecting duplicate names.

        Args:
            registry: The ability or tool registry to update.
            export_name: The name under which the method is exposed.
            method: The registered method to add.
            export_kind: A label used in duplicate-name error messages.

        Raises:
            AgentRegistrationError: If ``export_name`` is already registered.
        """
        existing = registry.get(export_name)
        if existing is not None:
            raise AgentRegistrationError(
                f"Duplicate {export_kind} name '{export_name}' on "
                f"{type(self).__name__}.{existing.attribute_name} and "
                f"{type(self).__name__}.{method.attribute_name}"
            )

        registry[export_name] = method