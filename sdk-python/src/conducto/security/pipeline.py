"""Transport-independent authorization and approval interception pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .approval import ApprovalChallenge, ApprovalDecision, ApprovalStore, default_challenge
from .context import AuthorizationContext
from .errors import (
    AuthorizationDeniedError,
    InsufficientScopeError,
    InvalidApprovalStateError,
    MissingAuthorizationContextError,
    PolicyEvaluationError,
)
from .guardrails import discover_guardrails

_POLICY_FAILURE_TYPES = (Exception,)


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Typed result of pre-invocation authorization."""

    allowed: bool
    challenge: ApprovalChallenge | None = None
    error: Exception | None = None


class SecurityPipeline:
    """Apply guardrails in fixed order before capability business logic."""

    def __init__(self, store: ApprovalStore | None = None) -> None:
        self.store = store

    def check(
        self,
        target: Callable[..., Any],
        context: AuthorizationContext | None,
        arguments: Mapping[str, Any],
        *,
        agent_id: str = "",
        capability_id: str = "",
    ) -> GuardrailResult:
        """Check identity, scopes, and validated argument-dependent approval policy."""
        guardrails = discover_guardrails(target)
        if guardrails.scopes or guardrails.approvals:
            if context is None:
                return GuardrailResult(False, error=MissingAuthorizationContextError())
            missing = set(guardrails.scopes) - context.principal.scopes
            if missing:
                return GuardrailResult(
                    False, error=InsufficientScopeError("required scope missing")
                )
            for requirement in guardrails.approvals:
                try:
                    applies = requirement.condition is None or bool(
                        requirement.condition(context, arguments)
                    )
                # Conditions are application callbacks; every callback failure fails closed.
                except _POLICY_FAILURE_TYPES:
                    return GuardrailResult(
                        False, error=PolicyEvaluationError("approval policy failed")
                    )
                if not applies:
                    continue
                challenge = default_challenge(
                    context, agent_id=agent_id, capability_id=capability_id, role=requirement.role
                )
                if self.store is None:
                    return GuardrailResult(False, challenge=challenge)
                return GuardrailResult(False, challenge=self.store.create(challenge))
        return GuardrailResult(True)

    def resume(
        self,
            decision: ApprovalDecision,
        execute: Callable[[], Any],
    ) -> Any:
        """Consume an approved challenge at once, then execute its protected work.

        The store performs the compare-and-transition operation, so stale or
        duplicate decisions cannot run business logic twice.
        """
        if self.store is None:
            raise InvalidApprovalStateError("an approval store is required to resume")
        challenge = self.store.decide(decision)
        if not decision.approved:
            raise AuthorizationDeniedError("approval was denied")
        self.store.complete(challenge.approval_id)
        return execute()
