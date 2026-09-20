"""Public transport-independent security contracts."""

from .approval import (
    ApprovalChallenge,
    ApprovalDecision,
    ApprovalState,
    ApprovalStore,
    Clock,
    IdentifierGenerator,
    InMemoryApprovalStore,
)
from .context import AuthorizationContext, Principal
from .errors import (
    ApprovalExpiredError,
    ApprovalRequiredError,
    AuthorizationDeniedError,
    InsufficientScopeError,
    InvalidApprovalStateError,
    MissingAuthorizationContextError,
    PolicyEvaluationError,
    SecurityError,
    TaskCanceledError,
)
from .guardrails import (
    ApprovalRequirement,
    Guardrails,
    ScopeRequirement,
    discover_guardrails,
    get_guardrails,
    require_approval,
    require_scope,
)
from .pipeline import GuardrailResult, SecurityPipeline

__all__ = [
    "ApprovalChallenge",
    "ApprovalDecision",
    "ApprovalState",
    "ApprovalStore",
    "Clock",
    "IdentifierGenerator",
    "InMemoryApprovalStore",
    "AuthorizationContext",
    "Principal",
    "ApprovalRequirement",
    "Guardrails",
    "ScopeRequirement",
    "discover_guardrails",
    "get_guardrails",
    "require_approval",
    "require_scope",
    "GuardrailResult",
    "SecurityPipeline",
    "SecurityError",
    "MissingAuthorizationContextError",
    "InsufficientScopeError",
    "ApprovalRequiredError",
    "AuthorizationDeniedError",
    "ApprovalExpiredError",
    "TaskCanceledError",
    "InvalidApprovalStateError",
    "PolicyEvaluationError",
]
