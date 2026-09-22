"""Finite, governed model/tool delegation with explicit outcomes and fallback policy."""

from ._fallback import DelegationFallbackPolicy, ToolResultStatus
from ._loop import run_delegation
from ._models import (
    DelegationConfig,
    DelegationOutcome,
    DelegationOutcomeCode,
    DelegationProvenance,
    DelegationRequirement,
    DelegationToolCallRecord,
    ToolResultEnvelope,
)

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
