"""Deadline-bounded provider completion with acceptance-aware safe retries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from .configuration import GenerationOptions
from .errors import (
    MalformedStructuredOutputError,
    ProviderCancellationError,
    ProviderContentPolicyError,
    ProviderError,
    ProviderTimeoutError,
)
from .messages import ChatMessage
from .protocol import ModelProvider, ProviderCallContext, validate_provider_contract
from .results import ProviderResult
from .structured import StructuredOutputRequest, validate_structured_output
from .tools import ProviderToolDefinition, ToolResultMessage, _normalize_provider_tools


async def complete_with_retries(
    provider: ModelProvider,
    messages: Sequence[ChatMessage],
    *,
    options: GenerationOptions,
    structured_output: StructuredOutputRequest,
    deadline: float | None = None,
    tools: Sequence[ProviderToolDefinition] = (),
    tool_results: Sequence[ToolResultMessage] = (),
    effective_deadline: float | None = None,
    call_context: ProviderCallContext | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProviderResult:
    """Complete a provider request under timeout and safe-retry rules.

    Retryable provider failures and timeouts are retried only before a request
    is known to have been accepted. Provider capability validation occurs
    before the first request.

    Args:
        provider: Resolved provider client.
        messages: Provider-neutral conversation messages.
        options: Generation, timeout, and retry settings.
        structured_output: Required native structured-output contract.
        deadline: Optional monotonic deadline governing the whole request.
        tools: Provider-neutral tools available for this request.
        tool_results: Bounded results from prior tool calls.
        effective_deadline: Optional tighter monotonic deadline.
        call_context: Cancellation and deadline constraints that can only
            tighten the request's execution window.
        clock: Monotonic clock used for deterministic deadline enforcement.

    Returns:
        The normalized provider result.

    Raises:
        UnsupportedProviderCapabilityError: If structured output is unsupported.
        ProviderTimeoutError: If all permitted attempts time out.
        ProviderError: If a provider failure is not safely retryable or retries
            are exhausted.
    """
    normalized_tools = _normalize_provider_tools(tools)
    tool_aware = bool(normalized_tools) or bool(tool_results)
    if effective_deadline is not None:
        deadline = min(deadline, effective_deadline) if deadline is not None else effective_deadline
    if call_context is not None:
        if call_context.cancelled:
            raise ProviderCancellationError()
        if call_context.deadline is not None:
            deadline = (
                min(deadline, call_context.deadline)
                if deadline is not None
                else call_context.deadline
            )
        call_context = replace(call_context, deadline=deadline)
    validate_provider_contract(
        provider,
        structured_output=structured_output,
        tools=normalized_tools,
        tool_results=tool_results,
    )
    attempts = options.retries + 1
    for attempt in range(attempts):
        if deadline is not None and deadline <= clock():
            raise ProviderTimeoutError(attempted=attempt > 0)
        try:
            timeout_for_attempt: float | None = None
            if deadline is not None:
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("Run deadline exceeded")
                timeout_for_attempt = remaining
            if options.timeout is not None:
                timeout_for_attempt = (
                    min(options.timeout, timeout_for_attempt)
                    if timeout_for_attempt is not None
                    else options.timeout
                )
            request_kwargs: dict[str, Any] = {
                "options": options,
                "structured_output": structured_output,
            }
            if tool_aware:
                request_kwargs["tools"] = normalized_tools
                request_kwargs["tool_results"] = tool_results
            if tool_aware or deadline is not None:
                request_kwargs["effective_deadline"] = deadline
            if call_context is not None and provider.capabilities.cancellation:
                request_kwargs["call_context"] = call_context
            completion = provider.complete(messages, **request_kwargs)
            if timeout_for_attempt is not None:
                result = await asyncio.wait_for(completion, timeout_for_attempt)
            else:
                result = await completion
            if result.content_filtered:
                raise ProviderContentPolicyError(
                    accepted=result.accepted,
                    request_id=result.request_id,
                )
            if structured_output.required:
                if result.structured is None:
                    error = MalformedStructuredOutputError(
                        "Provider returned no required structured output",
                        accepted=result.accepted,
                        request_id=result.request_id,
                        usage=result.usage,
                    )
                    raise error
                try:
                    validate_structured_output(result.structured, structured_output)
                except MalformedStructuredOutputError as error:
                    error.usage = result.usage
                    error.accepted = result.accepted
                    error.request_id = result.request_id
                    error.attempted = True
                    raise
            return result
        except TimeoutError as error:
            timeout_error = ProviderTimeoutError()
            if attempt == attempts - 1 or (deadline is not None and deadline <= clock()):
                raise timeout_error from error
            await asyncio.sleep(0)
        except ProviderError as error:
            error.attempted = True
            if not error.retryable or error.accepted or attempt == attempts - 1:
                raise
            await asyncio.sleep(0)
    raise AssertionError("unreachable")
