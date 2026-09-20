"""Public BaseAgent facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from .agent_card import (
    A2A_AGENT_CARD_SPEC_VERSION,
    DEFAULT_INPUT_MODES,
    DEFAULT_OUTPUT_MODES,
    build_agent_card,
    is_absolute_http_url,
    serialize_agent_card,
    stable_skill_id,
    validate_card_metadata,
    validate_modes,
    validate_security_requirements,
    validate_security_schemes,
)
from .decorators import AgentMetadata, ExportMetadata
from .model_config import AgentModelConfig, ModelReference, ModelRequirement
from .parameter_schema import (
    ParameterSchemaError,
    build_parameter_model,
    build_parameter_schema,
)
from .provider import ModelConfiguration
from .registration import (
    AgentRegistrationError,
    RegisteredMethod,
    add_export,
    register_decorated_methods,
    resolve_agent_metadata,
    resolve_export_metadata,
    resolved_attributes,
)

__all__ = [
    "A2A_AGENT_CARD_SPEC_VERSION",
    "AgentRegistrationError",
    "BaseAgent",
    "RegisteredMethod",
]

_DEFAULT_INPUT_MODES = DEFAULT_INPUT_MODES
_DEFAULT_OUTPUT_MODES = DEFAULT_OUTPUT_MODES


class BaseAgent:
    """Base class for reflected Conducto agents.

    Attributes:
        agent_metadata: Declared A2A metadata for the concrete agent type.
        _agent_config: Resolved model policy for the agent.
        _model_config: Optional provider model configuration supplied at construction.
        _registered_methods: Methods discovered by reflection and metadata registration.
        _capabilities: Capability exports keyed by capability name.
        _tools: Tool exports keyed by tool name.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfiguration | None = None,
        model_reference: ModelReference | str | None = None,
        agent_config: AgentModelConfig | None = None,
    ) -> None:
        self.agent_metadata = self._resolve_agent_metadata()
        declared_reference = model_reference or self.agent_metadata.default_model
        if declared_reference is None and model_config is not None:
            declared_reference = model_config.model
        if isinstance(declared_reference, str):
            declared_reference = ModelReference(declared_reference)
        self._agent_config = agent_config or AgentModelConfig(
            default_model=declared_reference,
            requirement=(
                ModelRequirement.REQUIRED
                if self.agent_metadata.model_required
                else ModelRequirement.NONE
            ),
        )
        self._model_config = model_config
        self._registered_methods: dict[str, RegisteredMethod] = {}
        self._capabilities: dict[str, RegisteredMethod] = {}
        self._tools: dict[str, RegisteredMethod] = {}
        self._register_decorated_tools()

    @property
    def agent_config(self) -> AgentModelConfig:
        return self._agent_config

    @property
    def model_config(self) -> ModelConfiguration | None:
        return self._model_config

    @property
    def registered_methods(self) -> tuple[RegisteredMethod, ...]:
        return tuple(self._registered_methods.values())

    @property
    def capabilities(self) -> dict[str, RegisteredMethod]:
        return dict(self._capabilities)

    @property
    def tools(self) -> dict[str, RegisteredMethod]:
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
        return build_agent_card(
            agent_type_name=type(self).__name__,
            metadata=self.agent_metadata,
            registered_capabilities=self._capabilities,
            url=url,
            preferred_transport=preferred_transport,
            security_schemes=security_schemes,
            security_requirements=security_requirements,
            default_input_modes=default_input_modes,
            default_output_modes=default_output_modes,
            capabilities=capabilities,
        )

    def get_agent_card_json(self, url: str, **kwargs: Any) -> str:
        return serialize_agent_card(self.get_agent_card(url, **kwargs))

    def _validate_card_metadata(self, url: str, preferred_transport: str) -> None:
        validate_card_metadata(
            type(self).__name__,
            self.agent_metadata,
            url,
            preferred_transport,
        )

    @staticmethod
    def _is_absolute_http_url(value: Any) -> bool:
        return is_absolute_http_url(value)

    @staticmethod
    def _validate_modes(modes: Sequence[str], label: str) -> tuple[str, ...]:
        return validate_modes(modes, label)

    @staticmethod
    def _validate_security_schemes(
        security_schemes: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return validate_security_schemes(security_schemes)

    @staticmethod
    def _validate_security_requirements(
        security_requirements: Sequence[Mapping[str, Sequence[str]]] | None,
    ) -> list[dict[str, list[str]]]:
        return validate_security_requirements(security_requirements)

    def _skill_id(self, capability_name: str) -> str:
        return stable_skill_id(self.agent_metadata.name, capability_name)

    def _register_decorated_tools(self) -> None:
        (
            self._registered_methods,
            self._capabilities,
            self._tools,
        ) = register_decorated_methods(self)

    def _resolved_attributes(self) -> list[tuple[str, Any]]:
        return resolved_attributes(type(self))

    def _build_parameter_model(
        self,
        attribute_name: str,
        method: Callable[..., Any],
    ) -> type[BaseModel]:
        try:
            return build_parameter_model(type(self), attribute_name, method)
        except ParameterSchemaError as error:
            raise AgentRegistrationError(str(error)) from error.__cause__

    def _build_parameter_schema(
        self,
        attribute_name: str,
        method: Callable[..., Any],
    ) -> dict[str, Any]:
        parameter_model = self._build_parameter_model(attribute_name, method)
        try:
            return build_parameter_schema(type(self), attribute_name, parameter_model)
        except ParameterSchemaError as error:
            raise AgentRegistrationError(str(error)) from error.__cause__

    def _resolve_agent_metadata(self) -> AgentMetadata:
        return resolve_agent_metadata(type(self))

    @staticmethod
    def _resolve_export_metadata(
        attribute_name: str,
        method: Callable[..., Any],
        metadata: ExportMetadata | None,
    ) -> ExportMetadata | None:
        return resolve_export_metadata(attribute_name, method, metadata)

    def _add_export(
        self,
        registry: dict[str, RegisteredMethod],
        export_name: str,
        method: RegisteredMethod,
        *,
        export_kind: str,
    ) -> None:
        add_export(
            type(self),
            registry,
            export_name,
            method,
            export_kind=export_kind,
        )
