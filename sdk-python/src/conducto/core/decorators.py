"""Decorators and metadata models for Conducto agents and methods."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar, cast, overload

_AGENT_METADATA_ATTRIBUTE = "__conducto_agent_metadata__"
_METHOD_METADATA_ATTRIBUTE = "__conducto_method_metadata__"

ExportKind = Literal["capability", "tool"]
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AgentMetadata:
    """Metadata declared on an A2A agent class.

    Attributes:
        name: The agent's published name.
        version: The agent's published version.
        description: An optional description of the agent.
        default_model: Credential-free model reference used as the agent
            default when no call-level or run-level override is supplied.
        model_required: Whether capabilities require a resolved model unless
            they explicitly opt out.
        tags: Opaque discovery tags inherited by the agent's abilities.
    """

    name: str
    version: str
    description: str | None = None
    default_model: str | None = None
    model_required: bool = False
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """Optional name and description for an exported method.

    Attributes:
        name: The published export name, if one was provided.
        description: The published export description, if one was provided.
        model_required: Per-export model requirement. ``None`` inherits the
            agent requirement; ``False`` declares deterministic execution.
        tags: Opaque discovery tags for this export.
    """

    name: str | None = None
    description: str | None = None
    model_required: bool | None = None
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MethodMetadata:
    """Metadata declared by decorators on a method.

    Attributes:
        capability: A2A capability metadata, when declared.
        tool: Internal Conducto tool metadata, when declared.
    """

    capability: ExportMetadata | None = None
    tool: ExportMetadata | None = None


@overload
def a2a_agent(
    cls: T,
    *,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    default_model: str | None = None,
    model_required: bool = False,
    tags: tuple[str, ...] = (),
) -> T: ...


@overload
def a2a_agent(
    *,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    default_model: str | None = None,
    model_required: bool = False,
    tags: tuple[str, ...] = (),
) -> Callable[[T], T]: ...


def a2a_agent(
    cls: T | None = None,
    *,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    default_model: str | None = None,
    model_required: bool = False,
    tags: tuple[str, ...] = (),
) -> T | Callable[[T], T]:
    """Declare a class as an A2A agent.

    This decorator stores: class:`AgentMetadata` directly on the decorated
    class. If ``name`` is omitted, the class name is used. If
    ``description`` is omitted, the class docstring is used.

    Args:
        cls: The class to decorate when using ``@a2a_agent`` without
            parentheses.
        name: The published agent name. Defaults to the class name.
        version: The published agent version. Defaults to ``"0.1.0"``.
        description: An optional published description. Defaults to the
            class docstring.
        default_model: Optional credential-free model reference for the agent.
            The runtime resolves it through its provider registry.
        model_required: Whether exports require a model by default. Individual
            capabilities and tools may override this setting.
        tags: Opaque discovery tags inherited by capabilities.

    Returns:
        The decorated class, or a decorator when called with keyword
        arguments.

    Raises:
        TypeError: If the decorated value is not a class.
        ValueError: If ``version``, ``name``, or ``description`` is empty
            or contains only whitespace.
    """

    def decorate(agent_class: T) -> T:
        """Attach agent metadata to a class and return it.

        Args:
            agent_class: The class to decorate.

        Returns:
            The decorated class.

        Raises:
            TypeError: If the decorator target is not a class.
        """
        if not inspect.isclass(agent_class):
            raise TypeError("@a2a_agent can only decorate classes")

        agent_name = _normalize_optional_text(name) or agent_class.__name__
        agent_version = _require_text(version, "version")
        agent_description = _normalize_optional_text(description) or inspect.getdoc(agent_class)

        metadata = AgentMetadata(
            name=agent_name,
            version=agent_version,
            description=agent_description,
            default_model=_normalize_optional_text(default_model),
            model_required=model_required,
            tags=_normalize_tags(tags),
        )
        setattr(agent_class, _AGENT_METADATA_ATTRIBUTE, metadata)
        return agent_class

    if cls is None:
        return decorate
    return decorate(cls)


def a2a_capability(
    *,
    name: str | None = None,
    description: str | None = None,
    model_required: bool | None = None,
    tags: tuple[str, ...] = (),
) -> Callable[[F], F]:
    """Expose a method as an A2A capability.

    The metadata is attached to the unwrapped function, allowing this
    decorator to be used with regular methods, ``classmethod``, and
    ``staticmethod`` declarations. Existing tool metadata on the method is
    preserved.

    Args:
        name: An optional published capability name.
        description: An optional published capability description.
        model_required: Whether the capability requires a model. ``None``
            inherits the agent setting; ``False`` explicitly permits
            deterministic execution without a configured model.
        tags: Opaque tags used by capability discovery.

    Returns:
        A decorator that preserves and returns the decorated callable.

    Raises:
        TypeError: If the decorated value is not callable.
        ValueError: If ``name`` or ``description`` is empty or contains only
            whitespace.
    """

    return _export_decorator(
        kind="capability",
        name=name,
        description=description,
        model_required=model_required,
        tags=tags,
    )


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    model_required: bool | None = None,
    tags: tuple[str, ...] = (),
) -> Callable[[F], F]:
    """Register a method as an internal Conducto tool.

    The metadata is attached to the unwrapped function, allowing this
    decorator to be used with regular methods, ``classmethod``, and
    ``staticmethod`` declarations. Existing capability metadata on the
    method is preserved.

    Args:
        name: An optional tool name.
        description: An optional tool description.
        model_required: Whether the tool requires a model. ``None`` inherits
            the agent setting; ``False`` explicitly permits deterministic
            execution without a configured model.
        tags: Opaque tags attached to the tool metadata.

    Returns:
        A decorator that preserves and returns the decorated callable.

    Raises:
        TypeError: If the decorated value is not callable.
        ValueError: If ``name`` or ``description`` is empty or contains only
            whitespace.
    """

    return _export_decorator(
        kind="tool",
        name=name,
        description=description,
        model_required=model_required,
        tags=tags,
    )


def get_agent_metadata(agent_class: type) -> AgentMetadata | None:
    """Return metadata declared directly on an agent class.

    Inherited metadata is not returned; only metadata stored in the class's
    own ``__dict__`` is considered.

    Args:
        agent_class: The class whose metadata should be inspected.

    Returns:
        The class's: class:`AgentMetadata`, or ``None`` when no metadata was
        declared directly on the class.

    Raises:
        TypeError: If the stored metadata has an unexpected type.
    """

    metadata = agent_class.__dict__.get(_AGENT_METADATA_ATTRIBUTE)
    if metadata is None:
        return None
    if not isinstance(metadata, AgentMetadata):
        raise TypeError("Invalid Conducto agent metadata")
    return metadata


def get_method_metadata(value: Any) -> MethodMetadata | None:
    """Return Conducto metadata attached to a method or descriptor.

    Args:
        value: A callable, ``classmethod``, or ``staticmethod`` to inspect.

    Returns:
        The callable's: class:`MethodMetadata`, or ``None`` when no metadata
        is attached.

    Raises:
        TypeError: If ``value`` is not callable, or the stored metadata has an
            unexpected type.
    """

    target = _decorator_target(value)
    metadata = getattr(target, _METHOD_METADATA_ATTRIBUTE, None)

    if metadata is None:
        return None
    if not isinstance(metadata, MethodMetadata):
        raise TypeError("Invalid Conducto method metadata")
    return metadata


def _export_decorator(
    *,
    kind: ExportKind,
    name: str | None,
    description: str | None,
    model_required: bool | None,
    tags: tuple[str, ...],
) -> Callable[[F], F]:
    """Create a decorator for one of the supported method export kinds."""

    export = ExportMetadata(
        name=_normalize_optional_text(name),
        description=_normalize_optional_text(description),
        model_required=model_required,
        tags=_normalize_tags(tags),
    )

    def decorate(value: F) -> F:
        """Attach export metadata to a callable and return it unchanged.

        Args:
            value: The callable being decorated.

        Returns:
            The original callable with metadata attached.
        """
        target = _decorator_target(value)
        current = get_method_metadata(target) or MethodMetadata()

        metadata = (
            replace(current, capability=export)
            if kind == "capability"
            else replace(current, tool=export)
        )

        setattr(target, _METHOD_METADATA_ATTRIBUTE, metadata)
        return value

    return decorate


def _decorator_target(value: Any) -> Callable[..., Any]:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__

    if not callable(value):
        raise TypeError("Conducto method decorators can only decorate callables")

    return cast(Callable[..., Any], inspect.unwrap(value))


def _require_text(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        raise ValueError("Decorator metadata cannot contain empty text")
    return normalized


def _normalize_tags(values: tuple[str, ...]) -> frozenset[str]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Decorator tags must be non-empty strings")
    return frozenset(value.strip() for value in values)
