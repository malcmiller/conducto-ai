"""Capability structured-output contract derivation and validation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, PydanticInvalidForJsonSchema, TypeAdapter, ValidationError

from .gateway_models import canonical_json
from .provider import MalformedStructuredOutputError, StructuredOutputRequest


@dataclass(frozen=True, slots=True)
class CapabilityOutputContract:
    """Structured-output contract derived for one capability return value.

    Attributes:
        request: Provider-neutral schema request sent to the model provider.
        adapter: Optional Pydantic adapter used to validate a typed return
            value. Explicit schema-only overrides may not have an adapter.
    """

    request: StructuredOutputRequest
    adapter: TypeAdapter[Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """Return the detached JSON Schema represented by this contract.

        Returns:
            A mutable JSON-compatible copy suitable for publication.
        """
        return dict(self.request.json_schema)

    def validate(self, value: Any) -> Any:
        """Validate provider output and return the typed capability value.

        Args:
            value: Decoded provider structured output.

        Returns:
            The validated typed value when a return annotation is available;
            otherwise the original value after schema validation.

        Raises:
            MalformedStructuredOutputError: If provider output does not match
                the derived schema or the Pydantic return annotation.
        """
        if value is None:
            raise MalformedStructuredOutputError("Provider returned no structured response")
        if self.adapter is None:
            return value
        try:
            return self.adapter.validate_python(value, strict=True)
        except ValidationError as error:
            raise MalformedStructuredOutputError(
                "Provider returned malformed structured response"
            ) from error


def build_capability_output_contract(
    target: Callable[..., Any],
    *,
    name: str,
    schema_override: Mapping[str, Any] | None = None,
) -> CapabilityOutputContract | None:
    """Derive a capability output contract from its return annotation.

    Args:
        target: Capability callable whose return annotation is inspected.
        name: Stable provider request name used when a contract is derived.
        schema_override: Optional caller-supplied JSON Schema that replaces the
            derived schema while keeping typed validation when a Pydantic return
            annotation exists.

    Returns:
        A structured-output contract, or ``None`` when neither a Pydantic return
        annotation nor an explicit schema override is present.

    Raises:
        ValueError: If annotations cannot be resolved or the resulting schema is
            not JSON serializable.
    """
    annotation = _return_annotation(target)
    adapter = _annotation_adapter(annotation)
    if schema_override is None and adapter is None:
        return None
    schema = dict(schema_override) if schema_override is not None else _adapter_schema(adapter)
    _validate_schema(schema)
    return CapabilityOutputContract(
        request=StructuredOutputRequest(name=name, schema=schema),
        adapter=adapter,
    )


def build_return_schema(target: Callable[..., Any]) -> dict[str, Any] | None:
    """Return a deterministic JSON Schema for a callable return annotation.

    Args:
        target: Callable whose return annotation should be converted to JSON
            Schema.

    Returns:
        The JSON Schema for supported Pydantic-backed return annotations, or
        ``None`` when the callable has no structured return annotation.

    Raises:
        ValueError: If a present return annotation cannot be represented as a
            deterministic JSON Schema.
    """
    adapter = _annotation_adapter(_return_annotation(target))
    if adapter is None:
        return None
    schema = _adapter_schema(adapter)
    _validate_schema(schema)
    return schema


def _return_annotation(target: Callable[..., Any]) -> Any:
    try:
        return get_type_hints(target, include_extras=True).get("return", inspect.Signature.empty)
    except (NameError, TypeError) as error:
        raise ValueError(f"Could not resolve return annotation: {error}") from error


def _annotation_adapter(annotation: Any) -> TypeAdapter[Any] | None:
    if annotation is inspect.Signature.empty or annotation is None or annotation is type(None):
        return None
    if annotation is Any:
        return None
    try:
        adapter = TypeAdapter(annotation)
        schema = adapter.json_schema()
    except (PydanticInvalidForJsonSchema, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported capability output schema: {error}") from error
    if not _references_pydantic_model(annotation, schema):
        return None
    return adapter


def _adapter_schema(adapter: TypeAdapter[Any] | None) -> dict[str, Any]:
    if adapter is None:
        raise ValueError("Pydantic return annotation is required")
    return dict(adapter.json_schema())


def _references_pydantic_model(annotation: Any, schema: Mapping[str, Any]) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    defs = schema.get("$defs")
    if isinstance(defs, Mapping) and defs:
        return True
    return False


def _validate_schema(schema: Mapping[str, Any]) -> None:
    try:
        canonical_json(schema)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported capability output schema: {error}") from error
