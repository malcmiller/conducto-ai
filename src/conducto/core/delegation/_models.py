"""Immutable delegation configuration, model-facing results, and ordered provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from ..gateway_models import freeze_json, thaw_json
from ..gateway_tools import ToolboxPolicy
from ..model_config import ModelReference
from ..provider import Usage
from ..run_context import InvocationMetadata, ModelCallProvenance
from ._fallback import DelegationFallbackPolicy, ToolResultStatus


class DelegationRequirement(StrEnum):
    """Whether a successful loop must execute at least one capability."""

    OPTIONAL = "optional"
    REQUIRED = "required"


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
    USAGE_UNKNOWN = "usage_unknown"
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
