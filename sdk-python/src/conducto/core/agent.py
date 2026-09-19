"""Agent registration, parameter schemas, and A2A Agent Card generation."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticInvalidForJsonSchema,
    PydanticSchemaGenerationError,
    create_model,
)

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


# The card shape is pinned independently of the SDK version, so Python and .NET
# exporters can share fixtures. This is the A2A specification version used by
# the Agent Card JSON schema.
A2A_AGENT_CARD_SPEC_VERSION = "0.3.0"
_DEFAULT_INPUT_MODES = ("text",)
_DEFAULT_OUTPUT_MODES = ("text",)


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
    parameter_model: type[BaseModel]
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

    def get_agent_card(
        self,
        url: str,
        *,
        preferred_transport: str = "JSONRPC",
        security_schemes: Mapping[str, Any] | None = None,
        security_requirements: Sequence[Mapping[str, Sequence[str]]] | None = None,
        default_input_modes: Sequence[str] = _DEFAULT_INPUT_MODES,
        default_output_modes: Sequence[str] = _DEFAULT_OUTPUT_MODES,
        capabilities: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Generate a standards-conformant A2A Agent Card.

        ``parameter_schema`` is not a standard A2A ``AgentSkill`` field. It is
        therefore carried in ``x-conducto.parameters`` so consumers can use
        reflected schemas without making the card invalid against the A2A
        schema.

        Args:
            url: The absolute HTTP (S) endpoint serving the agent.
            preferred_transport: A2A transport identifier, normally
                ``"JSONRPC"``.
            security_schemes: A2A security scheme definitions.
            security_requirements: A2A security requirements.
            default_input_modes: MIME-like modes accepted by the agent.
            default_output_modes: MIME-like modes produced by the agent.
            capabilities: A2A capability flags.

        Raises:
            AgentRegistrationError: If the endpoint, metadata, modes, or
                security definitions are incomplete or invalid.
        """
        self._validate_card_metadata(url, preferred_transport)
        input_modes = self._validate_modes(default_input_modes, "input")
        output_modes = self._validate_modes(default_output_modes, "output")

        normalized_security_schemes = self._validate_security_schemes(
            security_schemes
        )
        normalized_security = self._validate_security_requirements(
            security_requirements
        )

        agent_capabilities = {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        }
        if capabilities is not None and not isinstance(capabilities, Mapping):
            raise AgentRegistrationError("capabilities must be a mapping")
        if capabilities is not None:
            unknown = set(capabilities) - set(agent_capabilities)
            if unknown:
                raise AgentRegistrationError(
                    "Unsupported A2A capability flag(s): "
                    + ", ".join(sorted(unknown))
                )
            if any(not isinstance(value, bool) for value in capabilities.values()):
                raise AgentRegistrationError("A2A capability flags must be booleans")
            agent_capabilities.update(capabilities)

        skills: list[dict[str, Any]] = []
        parameter_schemas: dict[str, dict[str, Any]] = {}
        for capability_name, registered in self._capabilities.items():
            metadata = registered.capability
            assert metadata is not None
            description = metadata.description
            if not description:
                raise AgentRegistrationError(
                    f"{type(self).__name__}.{registered.attribute_name} capability "
                    "description is required for an A2A Agent Card"
                )

            skill_id = self._skill_id(capability_name)
            skills.append(
                {
                    "id": skill_id,
                    "name": capability_name,
                    "description": description,
                    "tags": [skill_id],
                    "inputModes": list(input_modes),
                    "outputModes": list(output_modes),
                }
            )
            parameter_schemas[skill_id] = registered.parameter_schema

        card: dict[str, Any] = {
            "protocolVersion": A2A_AGENT_CARD_SPEC_VERSION,
            "name": self.agent_metadata.name,
            "description": self.agent_metadata.description,
            "url": url,
            "preferredTransport": preferred_transport,
            "version": self.agent_metadata.version,
            "capabilities": agent_capabilities,
            "defaultInputModes": list(input_modes),
            "defaultOutputModes": list(output_modes),
            "skills": skills,
            "securitySchemes": normalized_security_schemes,
            "security": normalized_security,
            "x-conducto": {
                "parameters": parameter_schemas,
                "skillIdStrategy": "conducto-<sha256(agent-name:capability-name)[:16]>",
            },
        }
        return card

    def get_agent_card_json(self, url: str, **kwargs: Any) -> str:
        """Serialize an Agent Card canonically for golden fixtures."""
        return json.dumps(
            self.get_agent_card(url, **kwargs),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _validate_card_metadata(self, url: str, preferred_transport: str) -> None:
        if not isinstance(url, str):
            raise AgentRegistrationError(
                "Agent Card url must be an absolute http or https URL"
            )
        if not self._is_absolute_http_url(url):
            raise AgentRegistrationError(
                "Agent Card url must be an absolute http or https URL"
            )
        if not isinstance(preferred_transport, str) or not preferred_transport.strip():
            raise AgentRegistrationError("Agent Card preferred_transport cannot be empty")
        if preferred_transport != preferred_transport.strip():
            raise AgentRegistrationError(
                "Agent Card preferred_transport cannot contain surrounding whitespace"
            )
        if not self.agent_metadata.name.strip():
            raise AgentRegistrationError("Agent Card agent name cannot be empty")
        if not self.agent_metadata.version.strip():
            raise AgentRegistrationError("Agent Card agent version cannot be empty")
        if not self.agent_metadata.description:
            raise AgentRegistrationError(
                f"{type(self).__name__} requires a description for an A2A Agent Card"
            )

    @staticmethod
    def _is_absolute_http_url(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = urlparse(value)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)

    @staticmethod
    def _validate_modes(modes: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
            raise AgentRegistrationError(f"default_{label}_modes must be a sequence")
        normalized = tuple(mode.strip() for mode in modes if isinstance(mode, str))
        if len(normalized) != len(modes) or not normalized or any(not mode for mode in normalized):
            raise AgentRegistrationError(
                f"default_{label}_modes must contain non-empty strings"
            )
        return normalized

    @staticmethod
    def _validate_security_schemes(
        security_schemes: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if security_schemes is None:
            return {}
        if not isinstance(security_schemes, Mapping):
            raise AgentRegistrationError("security_schemes must be a mapping")

        validated: dict[str, Any] = {}
        for name, scheme in security_schemes.items():
            if not isinstance(name, str) or not name.strip():
                raise AgentRegistrationError(
                    "security_schemes names must be non-empty strings"
                )
            if not isinstance(scheme, Mapping):
                raise AgentRegistrationError(
                    f"security scheme '{name}' must be an object"
                )
            scheme_type = scheme.get("type")
            if scheme_type == "apiKey":
                if not isinstance(scheme.get("name"), str) or not scheme["name"].strip():
                    raise AgentRegistrationError(
                        f"apiKey security scheme '{name}' requires a non-empty name"
                    )
                if scheme.get("in") not in {"header", "query", "cookie"}:
                    raise AgentRegistrationError(
                        f"apiKey security scheme '{name}' requires in=header, query, or cookie"
                    )
            elif scheme_type == "http":
                if not isinstance(scheme.get("scheme"), str) or not scheme["scheme"].strip():
                    raise AgentRegistrationError(
                        f"http security scheme '{name}' requires a non-empty scheme"
                    )
            elif scheme_type == "oauth2":
                flows = scheme.get("flows")
                if not isinstance(flows, Mapping) or not flows:
                    raise AgentRegistrationError(
                        f"oauth2 security scheme '{name}' requires non-empty flows"
                    )
                for flow_name, flow in flows.items():
                    if flow_name not in {
                        "authorizationCode",
                        "clientCredentials",
                        "implicit",
                        "password",
                    } or not isinstance(flow, Mapping):
                        raise AgentRegistrationError(
                            f"oauth2 security scheme '{name}' has an invalid flow"
                        )
                    scopes = flow.get("scopes")
                    if not isinstance(scopes, Mapping) or any(
                        not isinstance(scope, str) or not isinstance(description, str)
                        for scope, description in scopes.items()
                    ):
                        raise AgentRegistrationError(
                            f"oauth2 security scheme '{name}' flow '{flow_name}' "
                            "requires a scope-description mapping"
                        )
                    if flow_name in {"authorizationCode", "implicit"}:
                        authorization_url = flow.get("authorizationUrl")
                        if not BaseAgent._is_absolute_http_url(authorization_url):
                            raise AgentRegistrationError(
                                f"oauth2 security scheme '{name}' flow '{flow_name}' "
                                "requires an absolute authorizationUrl"
                            )
                    if flow_name != "implicit":
                        token_url = flow.get("tokenUrl")
                        if not BaseAgent._is_absolute_http_url(token_url):
                            raise AgentRegistrationError(
                                f"oauth2 security scheme '{name}' flow '{flow_name}' "
                                "requires an absolute tokenUrl"
                        )
            elif scheme_type == "openIdConnect":
                connect_url = scheme.get("openIdConnectUrl")
                if not BaseAgent._is_absolute_http_url(connect_url):
                    raise AgentRegistrationError(
                        f"openIdConnect security scheme '{name}' requires an absolute URL"
                    )
            else:
                raise AgentRegistrationError(
                    f"security scheme '{name}' has unsupported type {scheme_type!r}"
                )
            validated[name] = dict(scheme)
        return validated

    @staticmethod
    def _validate_security_requirements(
        security_requirements: Sequence[Mapping[str, Sequence[str]]] | None,
    ) -> list[dict[str, list[str]]]:
        if security_requirements is None:
            return []
        if isinstance(security_requirements, (str, bytes)) or not isinstance(
            security_requirements, Sequence
        ):
            raise AgentRegistrationError("security_requirements must be a sequence")

        validated: list[dict[str, list[str]]] = []
        for index, requirement in enumerate(security_requirements):
            if not isinstance(requirement, Mapping) or not requirement:
                raise AgentRegistrationError(
                    f"security requirement {index} must be a non-empty object"
                )
            normalized: dict[str, list[str]] = {}
            for scheme_name, scopes in requirement.items():
                if not isinstance(scheme_name, str) or not scheme_name.strip():
                    raise AgentRegistrationError(
                        f"security requirement {index} has an invalid scheme name"
                    )
                if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
                    raise AgentRegistrationError(
                        f"security requirement '{scheme_name}' scopes must be a sequence"
                    )
                if any(not isinstance(scope, str) for scope in scopes):
                    raise AgentRegistrationError(
                        f"security requirement '{scheme_name}' scopes must be strings"
                    )
                normalized[scheme_name] = list(scopes)
            validated.append(normalized)
        return validated

    def _skill_id(self, capability_name: str) -> str:
        """Return a stable, collision-resistant ID for a reflected capability."""
        import hashlib

        value = f"{self.agent_metadata.name}:{capability_name}".encode("utf-8")
        return f"conducto-{hashlib.sha256(value).hexdigest()[:16]}"

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
                parameter_model=self._build_parameter_model(
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

    def _build_parameter_model(
        self,
        attribute_name: str,
        method: Callable[..., Any],
    ) -> type[BaseModel]:
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
        try:
            return create_model(
                f"{type(self).__name__}_{attribute_name}_Parameters",
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
        except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema) as error:
            raise AgentRegistrationError(
                f"Could not generate a parameter schema for "
                f"{type(self).__name__}.{attribute_name}: {error}"
            ) from error

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
        return self._build_parameter_model(
            attribute_name,
            method,
        ).model_json_schema()

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
            description=(
                inspect.cleandoc(type(self).__dict__["__doc__"])
                if isinstance(type(self).__dict__.get("__doc__"), str)
                else None
            ),
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