"""Invocation-scoped model gateway and provider completion."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from .logging import MODEL_SELECTED, MODEL_USAGE_RECORDED, emit_event, log_context
from .model_config import ModelReference
from .provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ProviderResult,
    StructuredOutputRequest,
    complete_with_retries,
)
from .run_context import InvocationMetadata, ModelCallProvenance, RunContext
from .runtime_errors import MissingModelDefaultError

if TYPE_CHECKING:
    from .runtime import Runtime

ModelResponseT = TypeVar("ModelResponseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    """Provider result paired with credential-free invocation metadata."""

    result: ProviderResult
    metadata: InvocationMetadata


@dataclass(frozen=True, slots=True)
class ModelGatewayCollection:
    """Invocation-scoped entry point for policy-aware model access."""

    _runtime: Runtime = field(repr=False, compare=False)
    _context: RunContext = field(repr=False, compare=False)

    def require(
        self,
        reference: ModelReference | str | None = None,
    ) -> ModelGateway:
        self._context._invocation_state.require_active()
        resolved = self._runtime._resolve_for_call_binding(self._context, reference)
        if resolved is None:
            raise MissingModelDefaultError(f"Agent '{self._context.agent_id}' requires a model")
        return ModelGateway(self._runtime, self._context, reference)


@dataclass(frozen=True, slots=True)
class ModelGateway:
    """Constrained model API that enforces runtime policy and telemetry."""

    _runtime: Runtime = field(repr=False, compare=False)
    _context: RunContext = field(repr=False, compare=False)
    _reference: ModelReference | str | None = field(default=None, repr=False)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        structured_output: StructuredOutputRequest,
    ) -> ModelCallResult:
        task = self._context._invocation_state.begin_model_call()
        try:
            return await self._runtime.complete(
                self._context,
                messages,
                structured_output=structured_output,
                model=self._reference,
                purpose="capability",
            )
        finally:
            self._context._invocation_state.end_model_call(task)

    async def complete_typed(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_type: type[ModelResponseT],
    ) -> ModelResponseT:
        request = StructuredOutputRequest(
            name=response_type.__name__,
            schema=response_type.model_json_schema(),
        )
        call = await self.complete(messages, structured_output=request)
        if call.result.structured is None:
            raise MalformedStructuredOutputError("Provider returned no structured response")
        try:
            return response_type.model_validate(call.result.structured)
        except ValidationError as error:
            raise MalformedStructuredOutputError(
                "Provider returned malformed structured response"
            ) from error


async def complete_model_call(
    runtime: Runtime,
    context: RunContext,
    messages: Sequence[ChatMessage],
    *,
    structured_output: StructuredOutputRequest,
    model: ModelReference | str | None = None,
    purpose: str = "model_call",
) -> ModelCallResult:
    """Resolve and execute one provider call without mutating the run context."""
    required = frozenset({"structured_output"}) if structured_output.json_schema else frozenset()
    binding = runtime._resolve_for_call_binding(
        context,
        model,
        required_capabilities=required,
    )
    if binding is None:
        raise MissingModelDefaultError(f"Agent '{context.agent_id}' requires a model")
    if context.cancellation.cancelled:
        raise asyncio.CancelledError
    resolved = binding.model

    if context.deadline is None:
        timeout = binding.configuration.timeout
    else:
        remaining = context.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Run deadline exceeded")
        timeout = min(binding.configuration.timeout, remaining)

    options = GenerationOptions(
        model=binding.configuration.model,
        timeout=timeout,
        retries=binding.configuration.retries,
    )
    with log_context(
        correlation_id=context.correlation_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
    ):
        emit_event(
            MODEL_SELECTED,
            provider=resolved.provider,
            model_reference=str(resolved.reference),
            resolution_source=resolved.source.value,
            outcome="success",
        )
        result = await complete_with_retries(
            binding.client,
            messages,
            options=options,
            structured_output=structured_output,
            deadline=context.deadline,
        )
        emit_event(
            MODEL_USAGE_RECORDED,
            provider=resolved.provider,
            model_reference=str(resolved.reference),
            resolution_source=resolved.source.value,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            outcome="success",
        )
    model_call = ModelCallProvenance(
        purpose=purpose,
        model_reference=str(resolved.reference),
        provider=resolved.provider,
        resolution_source=resolved.source,
        usage=result.usage,
    )
    context._model_calls.append(model_call)
    metadata = InvocationMetadata(
        run_id=context.run_id,
        correlation_id=context.correlation_id,
        model_reference=str(resolved.reference),
        provider=resolved.provider,
        resolution_source=resolved.source,
        usage=result.usage,
        model_calls=(model_call,),
        attributes=context.metadata,
    )
    return ModelCallResult(result, metadata)
