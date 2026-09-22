"""Explicit fallback eligibility that cannot bypass authority or cancellation."""

from dataclasses import dataclass
from enum import StrEnum


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
