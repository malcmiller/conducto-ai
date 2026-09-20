"""Pydantic parameter model and JSON schema generation."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticInvalidForJsonSchema,
    PydanticSchemaGenerationError,
    create_model,
)


class ParameterSchemaError(ValueError):
    """Raised when a callable cannot be represented by a parameter model."""


def build_parameter_model(
        agent_type: type[Any],
        attribute_name: str,
        method: Callable[..., Any],
) -> type[BaseModel]:
    """Build one strict Pydantic model for a reflected method.

    Args:
        agent_type: Type that owns the method being reflected.
        attribute_name: Attribute name of the callable on the agent.
        method: Callable whose parameters should be represented in the model.

    Returns:
        A strict Pydantic model that validates the callable's arguments.

    Raises:
        ParameterSchemaError: If the method cannot be introspected or represented.
    """
    signature = inspect.signature(method)
    try:
        type_hints = get_type_hints(method)
    except (NameError, TypeError) as error:
        raise ParameterSchemaError(
            f"Could not resolve type annotations for "
            f"{agent_type.__name__}.{attribute_name}: {error}"
        ) from error

    fields: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            raise ParameterSchemaError(
                f"{agent_type.__name__}.{attribute_name} uses unsupported "
                f"parameter '{parameter.name}' of kind {parameter.kind.description}"
            )
        annotation = type_hints.get(parameter.name)
        if annotation is None:
            raise ParameterSchemaError(
                f"{agent_type.__name__}.{attribute_name} parameter "
                f"'{parameter.name}' must have a type annotation"
            )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)

    try:
        return create_model(
            f"{agent_type.__name__}_{attribute_name}_Parameters",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
    except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema) as error:
        raise ParameterSchemaError(
            f"Could not generate a parameter schema for "
            f"{agent_type.__name__}.{attribute_name}: {error}"
        ) from error


def build_parameter_schema(
        agent_type: type[Any],
        attribute_name: str,
        parameter_model: type[BaseModel],
) -> dict[str, Any]:
    """Derive JSON schema from an already-created parameter model.

    Args:
        agent_type: Type that owns the method being reflected.
        attribute_name: Attribute name of the method being described.
        parameter_model: Pydantic model used to generate the JSON schema.

    Returns:
        The JSON schema dictionary describing the callable arguments.

    Raises:
        ParameterSchemaError: If the schema cannot be generated from the model.
    """
    try:
        return parameter_model.model_json_schema()
    except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema) as error:
        raise ParameterSchemaError(
            f"Could not generate a parameter schema for "
            f"{agent_type.__name__}.{attribute_name}: {error}"
        ) from error
