"""Invocation-scoped model gateway and provider completion."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .logging import MODEL_SELECTED, MODEL_USAGE_RECORDED, emit_event, log_context
from .model_config import ModelReference
from .provider import (
    ChatMessage,
    GenerationOptions,
    MalformedStructuredOutputError,
    ModelConfiguration,
    ModelProvider,
    ProviderCallContext,
    ProviderResult,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    complete_with_retries,
)
from .run_context import InvocationMetadata, ModelCallProvenance, RunContext
from .runtime_errors import MissingModelDefaultError
from .telemetry import SPAN_MODEL_COMPLETE, start_span

if TYPE_CHECKING:
    from .model_resolution import ResolvedModel
    from .runtime import Runtime

ModelResponseT = TypeVar("ModelResponseT", bound=BaseModel)


class _ModelBinding(Protocol):
    @property
    def model(self) -> ResolvedModel: ...

    @property
    def client(self) -> ModelProvider: ...

    @property
    def configuration(self) -> ModelConfiguration: ...


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
        """Resolve and return a model gateway for the active invocation.

        Args:
            reference: Optional model override to bind to the gateway.

        Returns:
            A model gateway bound to the active run context.

        Raises:
            MissingModelDefaultError: If no model can be resolved for the invocation.
        """
        self._context.require_active()
        resolved = self._runtime.resolve_for_call(self._context, reference)
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
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        required_capabilities: frozenset[str] = frozenset(),
        effective_deadline: float | None = None,
        purpose: str = "capability",
        clock: Callable[[], float] = time.monotonic,
    ) -> ModelCallResult:
        """Execute a provider call through the active runtime context.

        Args:
            messages: Conversation history to send to the model.
            structured_output: Native structured-output contract required by the call.
            tools: Provider-neutral tool definitions for this turn.
            tool_results: Bounded results from prior tool calls.
            required_capabilities: Additional provider capabilities required by this call.
            effective_deadline: Optional monotonic deadline, capped by the run deadline.
            purpose: Logical purpose recorded in model-call provenance.

        Returns:
            The provider result and associated invocation metadata.
        """
        task = self._context.begin_model_call()
        try:
            return await self._runtime.complete(
                self._context,
                messages,
                structured_output=structured_output,
                model=self._reference,
                tools=tools,
                tool_results=tool_results,
                required_capabilities=required_capabilities,
                effective_deadline=effective_deadline,
                purpose=purpose,
                clock=clock,
            )
        finally:
            self._context.end_model_call(task)

    async def complete_typed(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_type: type[ModelResponseT],
    ) -> ModelResponseT:
        """Execute a provider call and validate a typed structured response.

        Args:
            messages: Conversation history to send to the model.
            response_type: Pydantic model type expected from the provider.

        Returns:
            The validated typed response object.

        Raises:
            MalformedStructuredOutputError: If the provider response is missing or invalid.
        """
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
    context: RunContext,
    binding: _ModelBinding,
    messages: Sequence[ChatMessage],
    *,
    structured_output: StructuredOutputRequest,
    tools: Sequence[ProviderToolDefinition] = (),
    tool_results: Sequence[ToolResultMessage] = (),
    effective_deadline: float | None = None,
    purpose: str = "model_call",
    clock: Callable[[], float] = time.monotonic,
) -> ModelCallResult:
    """Execute one resolved provider call without mutating the run context."""
    if context.cancellation.cancelled:
        raise asyncio.CancelledError
    resolved = binding.model
    deadline = context.deadline
    if effective_deadline is not None:
        deadline = min(deadline, effective_deadline) if deadline is not None else effective_deadline

    if deadline is None:
        timeout = binding.configuration.timeout
    else:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Run deadline exceeded")
        timeout = min(binding.configuration.timeout, remaining)

    options = GenerationOptions(
        model=binding.configuration.model,
        timeout=timeout,
        retries=binding.configuration.retries,
    )
    with (
        start_span(
            SPAN_MODEL_COMPLETE,
            attributes={
                "conducto.agent.id": context.agent_id,
                "conducto.correlation_id": context.correlation_id,
                "conducto.run.id": context.run_id,
                "conducto.model.provider": resolved.provider,
                "conducto.model.reference": str(resolved.reference),
                "conducto.model.source": resolved.source.value,
            },
        ) as span,
        log_context(
            correlation_id=context.correlation_id,
            run_id=context.run_id,
            agent_id=context.agent_id,
        ),
    ):
        emit_event(
            MODEL_SELECTED,
            provider=resolved.provider,
            model_reference=str(resolved.reference),
            resolution_source=resolved.source.value,
            outcome="success",
        )
        completion = asyncio.create_task(
            complete_with_retries(
                binding.client,
                messages,
                options=options,
                structured_output=structured_output,
                deadline=deadline,
                tools=tools,
                tool_results=tool_results,
                effective_deadline=deadline,
                call_context=(
                    ProviderCallContext(
                        deadline=deadline,
                        cancelled=context.cancellation.cancelled,
                    )
                    if binding.client.capabilities.cancellation
                    else None
                ),
                clock=clock,
            )
        )
        try:
            while not completion.done():
                if context.cancellation.cancelled:
                    completion.cancel()
                    raise asyncio.CancelledError
                await asyncio.wait((completion,), timeout=0.01)
            result = await completion
        except TimeoutError:
            span.set_outcome("timeout", reason="timeout")
            raise
        except asyncio.CancelledError:
            span.set_outcome("cancelled", reason="cancellation")
            raise
        except Exception:
            span.set_error("provider_failure")
            raise
        finally:
            if not completion.done():
                completion.cancel()
                try:
                    await completion
                except asyncio.CancelledError:
                    pass
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
        span.set_outcome("success")
    model_call = ModelCallProvenance(
        purpose=purpose,
        model_reference=str(resolved.reference),
        provider=resolved.provider,
        resolution_source=resolved.source,
        usage=result.usage,
    )
    context.record_model_call(model_call)
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
