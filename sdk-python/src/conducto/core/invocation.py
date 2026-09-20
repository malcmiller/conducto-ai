"""Capability argument validation and execution pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from .agent import BaseAgent
from .invocation_results import (
    InvocationCancelled,
    InvocationFailure,
    InvocationResult,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTimeout,
    InvocationValidationFailure,
    UnsupportedReturnValueError,
)
from .logging import (
    ARGUMENTS_VALIDATED,
    INVOCATION_CANCELLED,
    INVOCATION_COMPLETED,
    INVOCATION_FAILED,
    INVOCATION_STARTED,
    INVOCATION_TIMED_OUT,
    emit_event,
    log_context,
)
from .model_config import ModelReference, ModelRequirement, RunConfig
from .registration import RegisteredMethod
from .run_context import use_run_context
from .runtime import Runtime
from .serialization import freeze_mapping, serialize_result

__all__ = [
    "InvocationCancelled",
    "InvocationFailure",
    "InvocationResult",
    "InvocationSuccess",
    "InvocationTargetNotFound",
    "InvocationTimeout",
    "InvocationValidationFailure",
    "UnsupportedReturnValueError",
    "invoke_agent",
]

# Retain private names used by older internal integrations.
_serialize_result = serialize_result
_freeze_mapping = freeze_mapping


class _CapabilityExecutionError(Exception):
    """Wrap an exception raised by a capability before deadline handling."""

    def __init__(self, exception: Exception) -> None:
        super().__init__(str(exception))
        self.exception = exception


def _resolve_capability(
    agent: BaseAgent,
    capability: str | Callable[..., Any],
) -> tuple[str, RegisteredMethod] | None:
    if isinstance(capability, str):
        registered = agent.capabilities.get(capability)
        if registered is not None:
            return capability, registered
        for name, candidate in agent.capabilities.items():
            if agent._skill_id(name) == capability:
                return name, candidate
        return None

    bound_instance = getattr(capability, "__self__", None)
    if (
        bound_instance is not None
        and bound_instance is not agent
        and bound_instance is not type(agent)
    ):
        return None
    requested_function = getattr(capability, "__func__", capability)
    for name, candidate in agent.capabilities.items():
        candidate_function = getattr(candidate.callable, "__func__", candidate.callable)
        if candidate_function is requested_function:
            return name, candidate
    return None


def _validated_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("Invocation timeout must be a finite positive number")
    try:
        value = float(timeout)
    except (OverflowError, ValueError) as error:
        raise ValueError("Invocation timeout must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invocation timeout must be a finite positive number")
    return value


async def invoke_agent(
    runtime: Runtime,
    agent: BaseAgent,
    capability: str | Callable[..., Any],
    arguments: Mapping[str, Any],
    *,
    timeout: float | None = None,
    correlation_id: str = "",
    model_reference: ModelReference | str | None = None,
    run_config: RunConfig | None = None,
) -> InvocationResult:
    """Execute one capability through the shared runtime-owned pipeline."""
    if not isinstance(agent, BaseAgent):
        raise TypeError("Runtime.invoke() requires a BaseAgent instance")
    if not isinstance(arguments, Mapping):
        raise TypeError("Invocation arguments must be a mapping")

    timeout_value = _validated_timeout(timeout)
    correlation_id = correlation_id or runtime.new_correlation_id()
    resolved = _resolve_capability(agent, capability)
    capability_id = capability if isinstance(capability, str) else capability.__name__
    agent_id = agent.agent_metadata.name
    if resolved is None:
        with log_context(correlation_id=correlation_id, agent_id=agent_id):
            emit_event(
                INVOCATION_FAILED,
                level=20,
                outcome="failure",
                error_category="target_not_found",
            )
        return InvocationTargetNotFound(correlation_id, agent_id, str(capability_id))

    capability_name, registered = resolved
    capability_metadata = registered.capability
    capability_requirement = (
        capability_metadata.model_required if capability_metadata is not None else None
    )
    agent_model_config = (
        dataclasses.replace(
            agent.agent_config,
            requirement=(
                ModelRequirement.REQUIRED if capability_requirement else ModelRequirement.NONE
            ),
            required_capabilities=(
                agent.agent_config.required_capabilities if capability_requirement else frozenset()
            ),
        )
        if capability_requirement is not None
        else agent.agent_config
    )
    effective_run = run_config or RunConfig()
    if timeout_value is not None:
        effective_run = dataclasses.replace(effective_run, timeout=timeout_value)
    context = runtime.create_run_context(
        agent_id=agent_id,
        agent_config=agent_model_config,
        run_config=effective_run,
        call_override=model_reference,
        correlation_id=correlation_id,
    )
    invocation_timeout = context.timeout
    target = registered.callable
    parameter_model = registered.parameter_model
    execution_lock = runtime.capability_lock(agent, capability_name)
    model_log_context = (
        {
            "provider": context.model.provider,
            "model_reference": str(context.model.reference),
            "resolution_source": context.model.source.value,
        }
        if context.model is not None
        else {}
    )

    with (
        use_run_context(context),
        log_context(
            correlation_id=correlation_id,
            run_id=context.run_id,
            agent_id=agent_id,
            capability_id=capability_name,
            **model_log_context,
        ),
    ):
        try:
            validated = parameter_model.model_validate(dict(arguments))
        except ValidationError as error:
            emit_event(
                ARGUMENTS_VALIDATED,
                outcome="failure",
                error_category="argument_validation",
            )
            emit_event(
                INVOCATION_FAILED,
                outcome="failure",
                error_category="argument_validation",
            )
            return InvocationValidationFailure(
                correlation_id,
                tuple(freeze_mapping(item) for item in error.errors()),
                context.invocation_metadata(),
            )
        emit_event(ARGUMENTS_VALIDATED, outcome="success")

        async def execute() -> Any:
            call_arguments = {
                name: getattr(validated, name) for name in parameter_model.model_fields
            }
            try:
                if inspect.iscoroutinefunction(target):
                    return await target(**call_arguments)

                def run_sync() -> Any:
                    with execution_lock:
                        return target(**call_arguments)

                return await asyncio.to_thread(run_sync)
            except asyncio.CancelledError:
                raise
            except Exception as capability_error:
                raise _CapabilityExecutionError(capability_error) from capability_error

        started = time.perf_counter()
        emit_event(INVOCATION_STARTED)
        try:
            remaining = context.remaining_timeout()
            result = await asyncio.wait_for(execute(), timeout=remaining)
            serialized = serialize_result(result)
            emit_event(
                INVOCATION_COMPLETED,
                outcome="success",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return InvocationSuccess(
                correlation_id,
                serialized,
                metadata=context.invocation_metadata(),
            )
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            emit_event(
                INVOCATION_CANCELLED,
                outcome="cancelled",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return InvocationCancelled(correlation_id, context.invocation_metadata())
        except TimeoutError:
            assert invocation_timeout is not None
            emit_event(
                INVOCATION_TIMED_OUT,
                level=30,
                outcome="timeout",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_category="timeout",
            )
            return InvocationTimeout(
                correlation_id,
                invocation_timeout,
                context.invocation_metadata(),
            )
        except _CapabilityExecutionError as error:
            emit_event(
                INVOCATION_FAILED,
                level=40,
                outcome="failure",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_category="capability_exception",
            )
            return InvocationFailure(
                correlation_id,
                "Capability execution failed",
                error.exception,
                context.invocation_metadata(),
            )
        except UnsupportedReturnValueError as error:
            emit_event(
                INVOCATION_FAILED,
                level=30,
                outcome="failure",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_category="unsupported_return_value",
            )
            return InvocationFailure(
                correlation_id,
                str(error),
                error,
                context.invocation_metadata(),
            )
        except Exception as error:
            emit_event(
                INVOCATION_FAILED,
                level=40,
                outcome="failure",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_category="internal_error",
            )
            return InvocationFailure(
                correlation_id,
                "Capability execution failed",
                error,
                context.invocation_metadata(),
            )
