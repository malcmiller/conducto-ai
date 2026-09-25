"""Finite provider-neutral model/tool delegation state machine."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..gateway_models import ToolDescriptor, canonical_json
from ..gateway_tools import ToolboxSnapshot, build_toolbox
from ..logging import (
    DELEGATION_COMPLETED,
    DELEGATION_TOOL_COMPLETED,
    DELEGATION_TURN_STARTED,
    emit_event,
    log_context,
)
from ..provider import (
    ChatMessage,
    MalformedStructuredOutputError,
    ProviderError,
    ProviderTimeoutError,
    TerminalModelDecision,
    ToolCallModelDecision,
    ToolResultMessage,
    Usage,
    build_terminal_output_request,
    parse_model_decision,
)
from ..run_context import (
    InvocationMetadata,
    ModelCallProvenance,
    RunContext,
    aggregate_usage,
)
from ..runtime_errors import ModelResolutionError
from ..telemetry import SPAN_DELEGATION_TURN, start_span
from ._arguments import _arguments_match_schema
from ._fallback import ToolResultStatus
from ._models import (
    DelegationConfig,
    DelegationOutcome,
    DelegationOutcomeCode,
    DelegationProvenance,
    DelegationRequirement,
    DelegationToolCallRecord,
    ToolResultEnvelope,
)
from ._results import _to_tool_result

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


async def run_delegation(
    context: RunContext,
    messages: Sequence[ChatMessage],
    *,
    config: DelegationConfig,
    response_type: type[ResponseT],
    clock: Callable[[], float] = time.monotonic,
) -> DelegationOutcome[ResponseT]:
    """Run a finite sequential model/tool loop in the active invocation context."""
    context.require_active()
    loop_id = str(uuid.uuid4())
    deadline = _effective_deadline(context, config, clock)
    model_turns = 0
    tool_calls = 0
    consumed_tokens = 0
    consumed_cost = 0.0
    history = tuple(messages)
    seen_call_ids: dict[str, ToolResultEnvelope] = {}
    records: list[DelegationToolCallRecord] = []
    results: list[ToolResultEnvelope] = []
    ordered_model_calls: list[ModelCallProvenance] = []
    fallback_pending = False

    def finish(
        code: DelegationOutcomeCode,
        *,
        terminal_value: ResponseT | None = None,
        failure_code: str | None = None,
        usage: Usage | None = None,
    ) -> DelegationOutcome[ResponseT]:
        provenance = DelegationProvenance(
            loop_id=loop_id,
            correlation_id=context.correlation_id,
            parent_task_id=context.run_id,
            model_turns=model_turns,
            tool_calls=tuple(records),
            tool_results=tuple(results),
        )
        emit_event(
            DELEGATION_COMPLETED,
            outcome="success"
            if code
            in {
                DelegationOutcomeCode.SUCCESS,
                DelegationOutcomeCode.FALLBACK_SUCCESS,
            }
            else ("cancelled" if code is DelegationOutcomeCode.CANCELLATION else "failure"),
            error_category=None if terminal_value is not None else code.value,
            loop_id=loop_id,
            turn=model_turns,
        )
        base_metadata = context.invocation_metadata()
        metadata = InvocationMetadata(
            run_id=base_metadata.run_id,
            correlation_id=base_metadata.correlation_id,
            parent_run_id=base_metadata.parent_run_id,
            delegation_path=base_metadata.delegation_path,
            model_reference=base_metadata.model_reference,
            provider=base_metadata.provider,
            resolution_source=base_metadata.resolution_source,
            usage=usage or aggregate_usage(ordered_model_calls),
            model_calls=tuple(ordered_model_calls),
            attributes=base_metadata.attributes,
            instruction_chain=base_metadata.instruction_chain,
        )
        return DelegationOutcome(
            code=code,
            value=terminal_value,
            provenance=provenance,
            metadata=metadata,
            failure_code=failure_code,
        )

    with log_context(
        correlation_id=context.correlation_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
    ):
        while model_turns < config.max_model_turns:
            terminal = _preflight(context, deadline, clock)
            if terminal is not None:
                return finish(terminal)

            toolbox = await build_toolbox(context.gateway, config.toolbox)
            if not toolbox.ok or toolbox.snapshot is None:
                return finish(
                    DelegationOutcomeCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                    failure_code=toolbox.status.value,
                )
            snapshot = toolbox.snapshot
            model_turns += 1
            try:
                with start_span(
                    SPAN_DELEGATION_TURN,
                    attributes={
                        "conducto.agent.id": context.agent_id,
                        "conducto.correlation_id": context.correlation_id,
                        "conducto.run.id": context.run_id,
                        "conducto.delegation.loop_id": loop_id,
                        "conducto.delegation.turn": model_turns,
                    },
                ) as turn_span:
                    emit_event(
                        DELEGATION_TURN_STARTED,
                        outcome="success",
                        loop_id=loop_id,
                        turn=model_turns,
                        snapshot_revision=snapshot.registry_revision,
                    )
                    model_call = await context.models.require(config.model).complete(
                        history,
                        structured_output=build_terminal_output_request(response_type),
                        tools=snapshot.as_provider_tools(),
                        tool_results=tuple(
                            ToolResultMessage(
                                call_id=result.call_id,
                                status=result.status.value,
                                result=result.to_dict(),
                            )
                            for result in results
                        ),
                        effective_deadline=deadline,
                        purpose="delegation_turn",
                        clock=clock,
                    )
                    turn_span.set_outcome("success")
            except asyncio.CancelledError:
                if context.cancellation.cancelled:
                    return finish(DelegationOutcomeCode.CANCELLATION)
                raise
            except TimeoutError:
                return finish(DelegationOutcomeCode.DEADLINE_EXHAUSTED)
            except ProviderTimeoutError:
                return finish(
                    (
                        DelegationOutcomeCode.DEADLINE_EXHAUSTED
                        if deadline is not None and clock() >= deadline
                        else DelegationOutcomeCode.PROVIDER_FAILURE
                    ),
                    failure_code="ProviderTimeoutError",
                )
            except ModelResolutionError as error:
                return finish(
                    DelegationOutcomeCode.PROVIDER_FAILURE,
                    failure_code=type(error).__name__,
                )
            except MalformedStructuredOutputError as error:
                return finish(DelegationOutcomeCode.MALFORMED_DECISION, usage=error.usage)
            except ProviderError as error:
                return finish(
                    DelegationOutcomeCode.PROVIDER_FAILURE,
                    failure_code=type(error).__name__,
                )

            if context.cancellation.cancelled:
                return finish(DelegationOutcomeCode.CANCELLATION)
            ordered_model_calls.extend(model_call.metadata.model_calls)
            if config.token_budget is not None and model_call.result.usage.total_tokens is None:
                return finish(DelegationOutcomeCode.USAGE_UNKNOWN)
            if config.cost_budget is not None and model_call.result.usage.cost is None:
                return finish(DelegationOutcomeCode.USAGE_UNKNOWN)
            if model_call.result.usage.total_tokens is not None:
                consumed_tokens += model_call.result.usage.total_tokens
            if model_call.result.usage.cost is not None:
                consumed_cost += model_call.result.usage.cost
            if config.token_budget is not None and consumed_tokens > config.token_budget:
                return finish(DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED)
            if config.cost_budget is not None and consumed_cost > config.cost_budget:
                return finish(DelegationOutcomeCode.COST_BUDGET_EXHAUSTED)

            try:
                decision = parse_model_decision(
                    model_call.result,
                    response_type=response_type,
                    tools=snapshot.as_provider_tools(),
                )
            except MalformedStructuredOutputError:
                return finish(DelegationOutcomeCode.MALFORMED_DECISION)

            if isinstance(decision, TerminalModelDecision):
                if config.requirement is DelegationRequirement.REQUIRED and tool_calls == 0:
                    return finish(DelegationOutcomeCode.REQUIRED_DELEGATION_NOT_PERFORMED)
                try:
                    value = response_type.model_validate(decision.response)
                except ValidationError:
                    return finish(DelegationOutcomeCode.FINAL_OUTPUT_VALIDATION_FAILURE)
                return finish(
                    (
                        DelegationOutcomeCode.FALLBACK_SUCCESS
                        if fallback_pending
                        else DelegationOutcomeCode.SUCCESS
                    ),
                    terminal_value=value,
                )

            assert isinstance(decision, ToolCallModelDecision)
            if not _CALL_ID_PATTERN.fullmatch(decision.call_id):
                return finish(DelegationOutcomeCode.MALFORMED_DECISION)
            if decision.call_id in seen_call_ids:
                return finish(
                    DelegationOutcomeCode.REPLAYED_TOOL_CALL,
                    failure_code=seen_call_ids[decision.call_id].status.value,
                )
            if fallback_pending:
                return finish(
                    DelegationOutcomeCode.CHILD_FAILURE,
                    failure_code="fallback_must_be_terminal",
                )
            if tool_calls >= config.max_tool_calls:
                return finish(DelegationOutcomeCode.TOOL_CALL_LIMIT_EXHAUSTED)
            if len(context.delegation_path) >= config.max_depth:
                return finish(DelegationOutcomeCode.DEPTH_LIMIT_EXHAUSTED)
            if (
                config.token_budget is not None
                and consumed_tokens + config.tool_token_cost > config.token_budget
            ):
                return finish(DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED)
            if (
                config.cost_budget is not None
                and consumed_cost + config.tool_cost > config.cost_budget
            ):
                return finish(DelegationOutcomeCode.COST_BUDGET_EXHAUSTED)

            descriptor = _find_tool(snapshot, decision.tool_id)
            binding = snapshot.resolve(decision.tool_id)
            if descriptor is None or binding is None:
                return finish(DelegationOutcomeCode.UNKNOWN_TOOL_CALL)
            if not _arguments_match_schema(decision.arguments, descriptor.input_schema):
                return finish(DelegationOutcomeCode.INVALID_ARGUMENTS)

            tool_calls += 1
            remaining = _remaining(deadline, clock)
            if remaining is not None and remaining <= 0:
                return finish(DelegationOutcomeCode.DEADLINE_EXHAUSTED)
            try:
                child_result = await context.gateway.invoke(
                    binding,
                    decision.arguments,
                    timeout=remaining,
                    token_cost=config.tool_token_cost,
                    cost=config.tool_cost,
                )
            except asyncio.CancelledError:
                if context.cancellation.cancelled:
                    return finish(DelegationOutcomeCode.CANCELLATION)
                raise
            if context.cancellation.cancelled:
                return finish(DelegationOutcomeCode.CANCELLATION)

            envelope = _to_tool_result(decision.call_id, child_result)
            if len(canonical_json(envelope.to_dict()).encode("utf-8")) > config.max_result_bytes:
                envelope = ToolResultEnvelope(
                    decision.call_id,
                    ToolResultStatus.RESULT_TOO_LARGE,
                    reason_code="serialized_result_limit",
                )
            seen_call_ids[decision.call_id] = envelope
            results.append(envelope)
            child_metadata = getattr(child_result, "metadata", None)
            child_model_calls = (
                child_metadata.model_calls[len(context.model_calls()) :]
                if isinstance(child_metadata, InvocationMetadata)
                else ()
            )
            ordered_model_calls.extend(child_model_calls)
            fallback_allowed = config.fallback.permits(envelope.status)
            records.append(
                DelegationToolCallRecord(
                    turn=model_turns,
                    call_id=decision.call_id,
                    tool_id=decision.tool_id,
                    snapshot_revision=snapshot.registry_revision,
                    parent_task_id=context.run_id,
                    child_task_id=(
                        child_metadata.run_id
                        if isinstance(child_metadata, InvocationMetadata)
                        else None
                    ),
                    status=envelope.status,
                    fallback_allowed=fallback_allowed,
                    usage=aggregate_usage(child_model_calls),
                    model_calls=child_model_calls,
                )
            )
            emit_event(
                DELEGATION_TOOL_COMPLETED,
                outcome="success" if envelope.status is ToolResultStatus.SUCCESS else "failure",
                error_category=(
                    None if envelope.status is ToolResultStatus.SUCCESS else envelope.status.value
                ),
                loop_id=loop_id,
                turn=model_turns,
                tool_call_id=decision.call_id,
                snapshot_revision=snapshot.registry_revision,
            )
            if envelope.status is ToolResultStatus.RESULT_TOO_LARGE:
                return finish(DelegationOutcomeCode.RESULT_SIZE_EXHAUSTED)
            if envelope.status is ToolResultStatus.CANCELLATION:
                return finish(DelegationOutcomeCode.CANCELLATION)
            if envelope.status is ToolResultStatus.BUDGET_REJECTED:
                return finish(
                    DelegationOutcomeCode.SHARED_BUDGET_EXHAUSTED,
                    failure_code=envelope.reason_code,
                )
            if envelope.status is not ToolResultStatus.SUCCESS:
                if not fallback_allowed:
                    return finish(
                        DelegationOutcomeCode.CHILD_FAILURE,
                        failure_code=envelope.status.value,
                    )
                fallback_pending = True

            consumed_tokens += config.tool_token_cost
            consumed_cost += config.tool_cost

        return finish(DelegationOutcomeCode.TURN_LIMIT_EXHAUSTED)


def _effective_deadline(
    context: RunContext,
    config: DelegationConfig,
    clock: Callable[[], float],
) -> float | None:
    configured = clock() + config.timeout if config.timeout is not None else None
    candidates = tuple(
        value for value in (context.deadline, config.deadline, configured) if value is not None
    )
    return min(candidates) if candidates else None


def _remaining(deadline: float | None, clock: Callable[[], float]) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - clock())


def _preflight(
    context: RunContext,
    deadline: float | None,
    clock: Callable[[], float],
) -> DelegationOutcomeCode | None:
    if context.cancellation.cancelled:
        return DelegationOutcomeCode.CANCELLATION
    if deadline is not None and clock() >= deadline:
        return DelegationOutcomeCode.DEADLINE_EXHAUSTED
    return None


def _find_tool(snapshot: ToolboxSnapshot, tool_id: str) -> ToolDescriptor | None:
    return next((tool for tool in snapshot.tools if tool.tool_id == tool_id), None)
