"""Decorators and metadata models for Conducto agents and methods."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from math import isfinite
from typing import Any, Literal, TypeVar, cast, overload

from conducto.security.guardrails import require_scope

_AGENT_METADATA_ATTRIBUTE = "__conducto_agent_metadata__"
_METHOD_METADATA_ATTRIBUTE = "__conducto_method_metadata__"

ExportKind = Literal["capability", "tool"]
SideEffectKind = Literal[
    "read_only",
    "writes_external_system",
    "sends_message",
    "mutates_state",
]
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class PolicyMetadataError(ValueError):
    """Raised when capability policy metadata has an invalid declaration."""


@dataclass(frozen=True, slots=True)
class RetrieverMetadata:
    """Immutable behavior declared for a retriever capability.

    Attributes:
        citations_required: Whether every returned document needs a citation.
    """

    citations_required: bool = True


@dataclass(frozen=True, slots=True)
class CapabilityBudget:
    """Immutable execution limits declared for a capability.

    Attributes:
        max_model_calls: Maximum model calls permitted during execution.
        max_tool_calls: Maximum nested tool calls permitted during execution.
        max_cost_usd: Maximum USD cost permitted during execution.
    """

    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_cost_usd: Decimal | None = None

    def to_dict(self) -> dict[str, int | str]:
        """Return a deterministic JSON-safe budget representation.

        Returns:
            Declared budget limits using Agent Card extension field names.
        """
        budget: dict[str, int | str] = {}
        if self.max_model_calls is not None:
            budget["maxModelCalls"] = self.max_model_calls
        if self.max_tool_calls is not None:
            budget["maxToolCalls"] = self.max_tool_calls
        if self.max_cost_usd is not None:
            budget["maxCostUsd"] = str(self.max_cost_usd)
        return budget


@dataclass(frozen=True, slots=True)
class CapabilityPolicyMetadata:
    """Immutable governance metadata declared for a capability.

    Attributes:
        required_scopes: Exact authorization scopes required for invocation.
        side_effect: Declared external or stateful behavior of the capability.
        timeout_seconds: Maximum execution time declared by the capability.
        budget: Optional execution budget declared by the capability.
        data_classification: Sensitivity classification for capability data.
    """

    required_scopes: tuple[str, ...] = ()
    side_effect: str | None = None
    timeout_seconds: float | None = None
    budget: CapabilityBudget | None = None
    data_classification: str | None = None

    @property
    def is_empty(self) -> bool:
        """Return whether no policy metadata was declared."""
        return (
            not self.required_scopes
            and self.side_effect is None
            and self.timeout_seconds is None
            and self.budget is None
            and self.data_classification is None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe policy representation.

        Returns:
            Declared policy metadata using Agent Card extension field names.
        """
        policy: dict[str, Any] = {}
        if self.required_scopes:
            policy["requiredScopes"] = list(self.required_scopes)
        if self.side_effect is not None:
            policy["sideEffect"] = self.side_effect
        if self.timeout_seconds is not None:
            policy["timeoutSeconds"] = self.timeout_seconds
        if self.budget is not None:
            policy["budget"] = self.budget.to_dict()
        if self.data_classification is not None:
            policy["dataClassification"] = self.data_classification
        return policy


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
        tags: Opaque discovery tags inherited by the agent's capabilities.
        instructions: Optional persona and behavioral instructions for the
            agent. Composed into the system role of runtime-issued model
            calls after runtime-owned policy instructions and before any
            capability instructions. Callers cannot inject, replace, or
            suppress this text through invocation arguments.
        publish_instructions: Whether ``instructions`` may be published on
            the agent card. Defaults to ``False``, so instructions remain
            unpublished unless explicitly opted in.
    """

    name: str
    version: str
    description: str | None = None
    default_model: str | None = None
    model_required: bool = False
    tags: frozenset[str] = frozenset()
    instructions: str | None = None
    publish_instructions: bool = False


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """Optional name and description for an exported method.

    Attributes:
        name: The published export name, if one was provided.
        description: The published export description, if one was provided.
        model_required: Per-export model requirement. ``None`` inherits the
            agent requirement; ``False`` declares deterministic execution.
        tags: Opaque discovery tags for this export.
        instructions: Optional capability-level instructions that refine the
            agent and runtime policy instructions. Composed last in the
            resolved instruction chain; never removes or replaces the
            instructions that precede it.
        policy: Immutable governance metadata declared for the method.
        output_schema: Optional explicit structured-output JSON Schema used
            instead of deriving one from the capability return annotation.
    """

    name: str | None = None
    description: str | None = None
    model_required: bool | None = None
    tags: frozenset[str] = frozenset()
    instructions: str | None = None
    policy: CapabilityPolicyMetadata = CapabilityPolicyMetadata()
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MethodMetadata:
    """Metadata declared by decorators on a method.

    Attributes:
        capability: A2A capability metadata, when declared.
        tool: Internal Conducto tool metadata, when declared.
        policy: Governance metadata that is applied to any declared exports.
        retriever: Retrieval-specific metadata when this is a retriever capability.
    """

    capability: ExportMetadata | None = None
    tool: ExportMetadata | None = None
    policy: CapabilityPolicyMetadata = CapabilityPolicyMetadata()
    retriever: RetrieverMetadata | None = None


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
    instructions: str | None = None,
    publish_instructions: bool = False,
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
    instructions: str | None = None,
    publish_instructions: bool = False,
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
    instructions: str | None = None,
    publish_instructions: bool = False,
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
        instructions: Optional persona and behavioral instructions for the
            agent. The runtime composes these into the system role of every
            model call it issues, after runtime-owned policy instructions
            and before any capability instructions. Callers cannot inject,
            replace, or suppress this text through invocation arguments.
        publish_instructions: Whether ``instructions`` may be published on
            the agent card. Defaults to ``False``; instructions remain
            unpublished unless explicitly opted in.

    Returns:
        The decorated class, or a decorator when called with keyword
        arguments.

    Raises:
        TypeError: If the decorated value is not a class.
        ValueError: If ``version``, ``name``, ``description``, or
            ``instructions`` is empty or contains only whitespace.
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
            instructions=_normalize_optional_text(instructions),
            publish_instructions=publish_instructions,
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
    instructions: str | None = None,
    output_schema: Mapping[str, Any] | None = None,
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
        instructions: Optional capability-level instructions that refine
            the agent's instructions. Composed last in the resolved
            instruction chain, after runtime policy and agent instructions;
            never removes or replaces the instructions that precede it.
        output_schema: Optional explicit structured-output JSON Schema. When
            omitted, Conducto derives a schema from Pydantic return annotations.

    Returns:
        A decorator that preserves and returns the decorated callable.

    Raises:
        TypeError: If the decorated value is not callable.
        ValueError: If ``name``, ``description``, or ``instructions`` is
            empty or contains only whitespace.
    """

    return _export_decorator(
        kind="capability",
        name=name,
        description=description,
        model_required=model_required,
        tags=tags,
        instructions=instructions,
        output_schema=output_schema,
    )


def retriever(
    *,
    name: str | None = None,
    description: str | None = None,
    citations: bool = True,
    model_required: bool | None = False,
    tags: tuple[str, ...] = (),
    instructions: str | None = None,
) -> Callable[[F], F]:
    """Expose a backend-neutral retriever as a governed capability.

    Retriever capabilities use the same registration, policy, gateway, and
    invocation path as ``@a2a_capability`` while adding query, result, citation,
    and payload-free provenance validation.

    Args:
        name: Optional published retriever capability name.
        description: Optional published capability description.
        citations: Whether every returned document must include a citation.
        model_required: Whether invocation requires a configured model. Defaults
            to ``False`` because retrieval backends are normally deterministic.
        tags: Opaque tags used by capability discovery.
        instructions: Optional capability-level instructions.

    Returns:
        A decorator that preserves and returns the decorated callable.

    Raises:
        TypeError: If ``citations`` is not a boolean or the target is not callable.
        ValueError: If textual decorator metadata is blank.
    """
    if not isinstance(citations, bool):
        raise TypeError("retriever citations must be a boolean")
    capability_decorator = _export_decorator(
        kind="capability",
        name=name,
        description=description,
        model_required=model_required,
        tags=tags,
        instructions=instructions,
        output_schema=None,
    )

    def decorate(value: F) -> F:
        decorated = capability_decorator(value)
        target = _decorator_target(decorated)
        metadata = get_method_metadata(target)
        assert metadata is not None
        setattr(
            target,
            _METHOD_METADATA_ATTRIBUTE,
            replace(
                metadata,
                retriever=RetrieverMetadata(citations_required=citations),
            ),
        )
        return decorated

    return decorate


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    model_required: bool | None = None,
    tags: tuple[str, ...] = (),
    instructions: str | None = None,
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
        instructions: Optional tool-level instructions that refine the
            agent's instructions in the resolved instruction chain.

    Returns:
        A decorator that preserves and returns the decorated callable.

    Raises:
        TypeError: If the decorated value is not callable.
        ValueError: If ``name``, ``description``, or ``instructions`` is
            empty or contains only whitespace.
    """

    return _export_decorator(
        kind="tool",
        name=name,
        description=description,
        model_required=model_required,
        tags=tags,
        instructions=instructions,
        output_schema=None,
    )


def requires_scope(*scopes: str) -> Callable[[F], F]:
    """Declare exact authorization scopes required to invoke a capability.

    The declaration also composes with Conducto's existing runtime scope
    guardrail, so local invocations enforce the metadata immediately.

    Args:
        *scopes: One or more non-empty, case-sensitive scope values.

    Returns:
        A decorator that attaches immutable scope policy metadata.

    Raises:
        PolicyMetadataError: If no scopes are supplied or a scope is blank.
    """
    normalized = _normalize_scopes(scopes)

    def decorate(value: F) -> F:
        require_scope(*normalized)(value)
        return _attach_policy(
            value,
            lambda policy: replace(
                policy,
                required_scopes=tuple(sorted(set(policy.required_scopes) | set(normalized))),
            ),
        )

    return decorate


def side_effect(kind: SideEffectKind | str) -> Callable[[F], F]:
    """Declare the external or stateful effect of a capability.

    Args:
        kind: A non-empty side-effect classification. Built-in classifications
            include ``read_only``, ``writes_external_system``,
            ``sends_message``, and ``mutates_state``.

    Returns:
        A decorator that attaches immutable side-effect policy metadata.

    Raises:
        PolicyMetadataError: If ``kind`` is not a non-empty string.
    """
    normalized = _require_policy_text(kind, "side_effect kind")
    return _policy_decorator(lambda policy: replace(policy, side_effect=normalized))


def timeout(*, seconds: float) -> Callable[[F], F]:
    """Declare the maximum execution time for a capability.

    Args:
        seconds: A finite positive timeout in seconds.

    Returns:
        A decorator that attaches immutable timeout policy metadata.

    Raises:
        PolicyMetadataError: If ``seconds`` is not a finite positive number.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        raise PolicyMetadataError("timeout seconds must be a finite positive number")
    try:
        normalized = float(seconds)
    except OverflowError as error:
        raise PolicyMetadataError("timeout seconds must be a finite positive number") from error
    if not isfinite(normalized) or normalized <= 0:
        raise PolicyMetadataError("timeout seconds must be a finite positive number")
    return _policy_decorator(lambda policy: replace(policy, timeout_seconds=normalized))


def budget(
    *,
    max_model_calls: int | None = None,
    max_tool_calls: int | None = None,
    max_cost_usd: Decimal | None = None,
) -> Callable[[F], F]:
    """Declare execution budget limits for a capability.

    Args:
        max_model_calls: Optional non-negative limit on model calls.
        max_tool_calls: Optional non-negative limit on nested tool calls.
        max_cost_usd: Optional finite non-negative USD cost limit.

    Returns:
        A decorator that attaches immutable budget policy metadata.

    Raises:
        PolicyMetadataError: If no limit is supplied or a limit is invalid.
    """
    if max_model_calls is None and max_tool_calls is None and max_cost_usd is None:
        raise PolicyMetadataError("budget requires at least one limit")
    _validate_budget_limit(max_model_calls, "max_model_calls")
    _validate_budget_limit(max_tool_calls, "max_tool_calls")
    if max_cost_usd is not None and (
        not isinstance(max_cost_usd, Decimal) or not max_cost_usd.is_finite() or max_cost_usd < 0
    ):
        raise PolicyMetadataError("max_cost_usd must be a finite non-negative Decimal")
    declared_budget = CapabilityBudget(max_model_calls, max_tool_calls, max_cost_usd)
    return _policy_decorator(lambda policy: replace(policy, budget=declared_budget))


def classification(level: str) -> Callable[[F], F]:
    """Declare the sensitivity classification of a capability's data.

    Args:
        level: A non-empty data classification level.

    Returns:
        A decorator that attaches immutable classification policy metadata.

    Raises:
        PolicyMetadataError: If ``level`` is not a non-empty string.
    """
    normalized = _require_policy_text(level, "classification level")
    return _policy_decorator(lambda policy: replace(policy, data_classification=normalized))


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
    instructions: str | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> Callable[[F], F]:
    """Create a decorator for one of the supported method export kinds."""

    declared_export = ExportMetadata(
        name=_normalize_optional_text(name),
        description=_normalize_optional_text(description),
        model_required=model_required,
        tags=_normalize_tags(tags),
        instructions=_normalize_optional_text(instructions),
        output_schema=output_schema,
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
        export = replace(declared_export, policy=current.policy)

        metadata = (
            replace(current, capability=export)
            if kind == "capability"
            else replace(current, tool=export)
        )

        setattr(target, _METHOD_METADATA_ATTRIBUTE, metadata)
        return value

    return decorate


def _policy_decorator(
    update: Callable[[CapabilityPolicyMetadata], CapabilityPolicyMetadata],
) -> Callable[[F], F]:
    """Create a decorator that updates immutable capability policy metadata."""

    def decorate(value: F) -> F:
        return _attach_policy(value, update)

    return decorate


def _attach_policy(
    value: F,
    update: Callable[[CapabilityPolicyMetadata], CapabilityPolicyMetadata],
) -> F:
    """Attach policy metadata to a callable while preserving export metadata."""
    target = _decorator_target(value)
    current = get_method_metadata(target) or MethodMetadata()
    policy = update(current.policy)
    metadata = replace(
        current,
        capability=(
            replace(current.capability, policy=policy) if current.capability is not None else None
        ),
        tool=replace(current.tool, policy=policy) if current.tool is not None else None,
        policy=policy,
    )
    setattr(target, _METHOD_METADATA_ATTRIBUTE, metadata)
    return value


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


def _normalize_scopes(scopes: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and canonically order declared capability scopes."""
    if not scopes:
        raise PolicyMetadataError("requires_scope requires at least one non-empty scope")
    normalized = tuple(scope.strip() if isinstance(scope, str) else "" for scope in scopes)
    if any(not scope for scope in normalized):
        raise PolicyMetadataError("requires_scope requires non-empty string scopes")
    return tuple(sorted(set(normalized)))


def _require_policy_text(value: str, field: str) -> str:
    """Validate one non-empty policy classification string."""
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise PolicyMetadataError(f"{field} must be a non-empty string")
    return normalized


def _validate_budget_limit(value: int | None, field: str) -> None:
    """Validate one optional non-negative integer budget limit."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise PolicyMetadataError(f"{field} must be a non-negative integer")
