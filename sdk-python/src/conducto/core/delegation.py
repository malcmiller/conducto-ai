"""Finite provider-neutral model/tool delegation state machine."""

from __future__ import annotations

import asyncio
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .gateway_models import ToolDescriptor, canonical_json, freeze_json, thaw_json
from .gateway_tools import ToolboxPolicy, ToolboxSnapshot, build_toolbox
from .invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationBudgetExhausted,
    InvocationCancelled,
    InvocationDelegationFailure,
    InvocationFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationSuccess,
    InvocationTargetNotFound,
    InvocationTargetUnavailable,
    InvocationTimeout,
    InvocationValidationFailure,
)
from .logging import (
    DELEGATION_COMPLETED,
    DELEGATION_TOOL_COMPLETED,
    DELEGATION_TURN_STARTED,
    emit_event,
    log_context,
)
from .model_config import ModelReference
from .provider import (
    ChatMessage,
    MalformedStructuredOutputError,
    ProviderError,
    ProviderTimeoutError,
    TerminalModelDecision,
    ToolCallModelDecision,
    ToolResultMessage,
    Usage,
    build_model_decision_schema,
    parse_model_decision,
)
from .run_context import (
    InvocationMetadata,
    ModelCallProvenance,
    RunContext,
    aggregate_usage,
)
from .runtime_errors import ModelResolutionError

__all__ = [
    "DelegationConfig",
    "DelegationFallbackPolicy",
    "DelegationOutcome",
    "DelegationOutcomeCode",
    "DelegationProvenance",
    "DelegationRequirement",
    "DelegationToolCallRecord",
    "ToolResultEnvelope",
    "ToolResultStatus",
    "run_delegation",
]

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class DelegationRequirement(StrEnum):
    """Whether a successful loop must execute at least one capability."""

    OPTIONAL = "optional"
    REQUIRED = "required"


class ToolResultStatus(StrEnum):
    """Safe model-facing classification of one child invocation result."""

    SUCCESS = "success"
    APPROVAL_REQUIRED = "approval_required"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    STALE_TARGET = "stale_target"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    BUDGET_REJECTED = "budget_rejected"
    DEPTH_REJECTED = "depth_rejected"
    CYCLE_REJECTED = "cycle_rejected"
    EXECUTION_FAILURE = "execution_failure"
    RESULT_TOO_LARGE = "result_too_large"


_NEVER_FALLBACK = frozenset(
    {
        ToolResultStatus.DENIED,
        ToolResultStatus.CANCELLATION,
        ToolResultStatus.BUDGET_REJECTED,
        ToolResultStatus.DEPTH_REJECTED,
        ToolResultStatus.CYCLE_REJECTED,
    }
)


@dataclass(frozen=True, slots=True)
class DelegationFallbackPolicy:
    """Explicit child failures for which the model may produce a fallback."""

    eligible_statuses: frozenset[ToolResultStatus] = frozenset()

    def __post_init__(self) -> None:
        statuses = frozenset(self.eligible_statuses)
        prohibited = statuses & _NEVER_FALLBACK
        if prohibited:
            raise ValueError(
                "Fallback cannot include security, cancellation, replay, budget, depth, or "
                f"cycle outcomes: {sorted(item.value for item in prohibited)!r}"
            )
        object.__setattr__(self, "eligible_statuses", statuses)

    def permits(self, status: ToolResultStatus) -> bool:
        """Return whether ``status`` may continue to a fallback model turn."""
        return status in self.eligible_statuses


@dataclass(frozen=True, slots=True)
class DelegationConfig:
    """Immutable limits and policy for one delegation loop."""

    toolbox: ToolboxPolicy = field(default_factory=ToolboxPolicy)
    requirement: DelegationRequirement = DelegationRequirement.OPTIONAL
    model: ModelReference | str | None = None
    max_model_turns: int = 8
    max_tool_calls: int = 8
    max_depth: int = 8
    timeout: float | None = None
    deadline: float | None = None
    token_budget: int | None = None
    cost_budget: float | None = None
    max_result_bytes: int = 64 * 1024
    tool_token_cost: int = 0
    tool_cost: float = 0.0
    fallback: DelegationFallbackPolicy = field(default_factory=DelegationFallbackPolicy)

    def __post_init__(self) -> None:
        if self.max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")
        if self.max_tool_calls < 0 or self.max_depth < 0:
            raise ValueError("max_tool_calls and max_depth cannot be negative")
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be a finite positive number")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        if self.token_budget is not None and self.token_budget < 0:
            raise ValueError("token_budget cannot be negative")
        if self.cost_budget is not None and (
            not math.isfinite(self.cost_budget) or self.cost_budget < 0
        ):
            raise ValueError("cost_budget must be finite and non-negative")
        if self.max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive")
        if self.tool_token_cost < 0:
            raise ValueError("tool_token_cost cannot be negative")
        if not math.isfinite(self.tool_cost) or self.tool_cost < 0:
            raise ValueError("tool_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ToolResultEnvelope:
    """Bounded JSON-safe result supplied to the next model turn."""

    call_id: str
    status: ToolResultStatus
    data: Any = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", freeze_json(self.data))

    def to_dict(self) -> dict[str, Any]:
        """Return the provider-neutral JSON representation."""
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "status": self.status.value,
        }
        if self.data is not None:
            payload["data"] = thaw_json(self.data)
        if self.reason_code is not None:
            payload["reason_code"] = self.reason_code
        return payload


@dataclass(frozen=True, slots=True)
class DelegationToolCallRecord:
    """Credential-free lineage and outcome for one accepted tool call."""

    turn: int
    call_id: str
    tool_id: str
    snapshot_revision: int
    parent_task_id: str
    child_task_id: str | None
    status: ToolResultStatus
    fallback_allowed: bool = False
    usage: Usage = field(default_factory=Usage)
    model_calls: tuple[ModelCallProvenance, ...] = ()


@dataclass(frozen=True, slots=True)
class DelegationProvenance:
    """Ordered safe provenance for one completed delegation loop."""

    loop_id: str
    correlation_id: str
    parent_task_id: str
    model_turns: int
    tool_calls: tuple[DelegationToolCallRecord, ...] = ()
    tool_results: tuple[ToolResultEnvelope, ...] = ()


class DelegationOutcomeCode(StrEnum):
    """Terminal result codes for the delegation state machine."""

    SUCCESS = "success"
    FALLBACK_SUCCESS = "fallback_success"
    MALFORMED_DECISION = "malformed_decision"
    UNKNOWN_TOOL_CALL = "unknown_tool_call"
    REPLAYED_TOOL_CALL = "replayed_tool_call"
    INVALID_ARGUMENTS = "invalid_arguments"
    REQUIRED_CAPABILITY_UNAVAILABLE = "required_capability_unavailable"
    REQUIRED_DELEGATION_NOT_PERFORMED = "required_delegation_not_performed"
    FINAL_OUTPUT_VALIDATION_FAILURE = "final_output_validation_failure"
    TURN_LIMIT_EXHAUSTED = "turn_limit_exhausted"
    TOOL_CALL_LIMIT_EXHAUSTED = "tool_call_limit_exhausted"
    DEPTH_LIMIT_EXHAUSTED = "depth_limit_exhausted"
    DEADLINE_EXHAUSTED = "deadline_exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    SHARED_BUDGET_EXHAUSTED = "shared_budget_exhausted"
    RESULT_SIZE_EXHAUSTED = "result_size_exhausted"
    PROVIDER_FAILURE = "provider_failure"
    CANCELLATION = "cancellation"
    CHILD_FAILURE = "child_failure"


@dataclass(frozen=True, slots=True)
class DelegationOutcome[ResponseT: BaseModel]:
    """Validated terminal value or a distinct typed loop failure."""

    code: DelegationOutcomeCode
    value: ResponseT | None
    provenance: DelegationProvenance
    metadata: InvocationMetadata
    failure_code: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether the loop produced an ordinary or fallback value."""
        return self.code in {
            DelegationOutcomeCode.SUCCESS,
            DelegationOutcomeCode.FALLBACK_SUCCESS,
        }


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
            usage=aggregate_usage(ordered_model_calls),
            model_calls=tuple(ordered_model_calls),
            attributes=base_metadata.attributes,
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
            emit_event(
                DELEGATION_TURN_STARTED,
                outcome="success",
                loop_id=loop_id,
                turn=model_turns,
                snapshot_revision=snapshot.registry_revision,
            )

            try:
                model_call = await context.models.require(config.model).complete(
                    history,
                    structured_output=build_model_decision_schema(response_type),
                    tools=snapshot.as_model_payload(),
                    tool_results=tuple(
                        ToolResultMessage(
                            call_id=result.call_id,
                            status=result.status.value,
                            result=result.to_dict(),
                        )
                        for result in results[-1:]
                    ),
                    effective_deadline=deadline,
                    purpose="delegation_turn",
                    clock=clock,
                )
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
            except ProviderError as error:
                return finish(
                    DelegationOutcomeCode.PROVIDER_FAILURE,
                    failure_code=type(error).__name__,
                )

            if context.cancellation.cancelled:
                return finish(DelegationOutcomeCode.CANCELLATION)
            ordered_model_calls.extend(model_call.metadata.model_calls)
            consumed_tokens += model_call.result.usage.total_tokens
            consumed_cost += model_call.result.usage.cost
            if config.token_budget is not None and consumed_tokens > config.token_budget:
                return finish(DelegationOutcomeCode.TOKEN_BUDGET_EXHAUSTED)
            if config.cost_budget is not None and consumed_cost > config.cost_budget:
                return finish(DelegationOutcomeCode.COST_BUDGET_EXHAUSTED)

            try:
                decision = parse_model_decision(model_call.result, response_type=response_type)
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


def _to_tool_result(call_id: str, result: InvocationResult) -> ToolResultEnvelope:
    if isinstance(result, InvocationSuccess):
        return ToolResultEnvelope(call_id, ToolResultStatus.SUCCESS, data=result.value)
    if isinstance(result, InvocationValidationFailure):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.INVALID_ARGUMENTS,
            data={"error_count": len(result.errors)},
        )
    if isinstance(result, InvocationApprovalRequired):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.APPROVAL_REQUIRED,
            reason_code="approval_required",
        )
    if isinstance(result, (InvocationAuthorizationFailure, InvocationAuditFailure)):
        return ToolResultEnvelope(call_id, ToolResultStatus.DENIED, reason_code=result.reason_code)
    if isinstance(result, InvocationStaleBinding):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.STALE_TARGET,
            reason_code="stale_binding",
        )
    if isinstance(
        result,
        (
            InvocationTargetNotFound,
            InvocationTargetUnavailable,
            InvocationSchemaMismatch,
            InvocationBindingFailure,
        ),
    ):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.UNAVAILABLE,
            reason_code=type(result).__name__,
        )
    if isinstance(result, InvocationTimeout):
        return ToolResultEnvelope(call_id, ToolResultStatus.TIMEOUT, reason_code="timeout")
    if isinstance(result, InvocationCancelled):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.CANCELLATION,
            reason_code="cancelled",
        )
    if isinstance(result, InvocationBudgetExhausted):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.BUDGET_REJECTED,
            reason_code=result.budget,
        )
    if isinstance(result, InvocationDelegationFailure):
        status = (
            ToolResultStatus.CYCLE_REJECTED
            if result.reason_code == "cycle_detected"
            else ToolResultStatus.DEPTH_REJECTED
        )
        return ToolResultEnvelope(call_id, status, reason_code=result.reason_code)
    if isinstance(result, InvocationFailure):
        return ToolResultEnvelope(
            call_id,
            ToolResultStatus.EXECUTION_FAILURE,
            reason_code="execution_failure",
        )
    raise TypeError(f"Unsupported invocation result: {type(result).__name__}")


def _arguments_match_schema(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> bool:
    """Validate model arguments against the bounded JSON Schema subset."""
    return _matches_schema(dict(arguments), schema, schema)


def _matches_schema(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if not reference.startswith("#/$defs/"):
            return False
        definition = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        return isinstance(definition, Mapping) and _matches_schema(value, definition, root)
    if "const" in schema and value != schema["const"]:
        return False
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, str) and value not in enum:
        return False
    for keyword in ("allOf",):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, str):
            if not all(
                isinstance(branch, Mapping) and _matches_schema(value, branch, root)
                for branch in branches
            ):
                return False
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(branches, str):
            matches = sum(
                isinstance(branch, Mapping) and _matches_schema(value, branch, root)
                for branch in branches
            )
            if matches < 1 or (keyword == "oneOf" and matches != 1):
                return False

    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_matches_schema(value, {**schema, "type": item}, root) for item in expected)
    if expected == "null":
        return value is None
    if expected == "boolean" and not isinstance(value, bool):
        return False
    if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        return False
    if expected == "number" and (isinstance(value, bool) or not isinstance(value, int | float)):
        return False
    if expected == "string":
        if not isinstance(value, str):
            return False
        if len(value) < int(schema.get("minLength", 0)):
            return False
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        pattern = schema.get("pattern")
        return not isinstance(pattern, str) or re.search(pattern, value) is not None
    if expected == "array":
        if not isinstance(value, list):
            return False
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        items = schema.get("items")
        return not isinstance(items, Mapping) or all(
            _matches_schema(item, items, root) for item in value
        )
    if expected == "object" or "properties" in schema:
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return False
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, str):
            return False
        if any(name not in value for name in required):
            return False
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            property_schema = properties.get(name)
            if isinstance(property_schema, Mapping):
                if not _matches_schema(item, property_schema, root):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, Mapping) and not _matches_schema(item, additional, root):
                return False
        return True
    if expected is None:
        return True
    return False
