"""Transport-independent authorization and approval interception pipeline."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .approval import (
    ApprovalChallenge,
    ApprovalDecision,
    ApprovalStore,
    Clock,
    IdentifierGenerator,
    default_challenge,
)
from .approval_token import ApprovalTokenService
from .context import AuthorizationContext
from .errors import (
    ApprovalRequiredError,
    AuthorizationDeniedError,
    InsufficientScopeError,
    InvalidApprovalStateError,
    MissingAuthorizationContextError,
    PolicyEvaluationError,
)
from .guardrails import discover_guardrails


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Typed result of pre-invocation authorization.

    Attributes:
        allowed: Whether execution may proceed.
        challenge: Approval challenge when approval is required.
        error: Typed failure when execution is denied.
    """

    allowed: bool
    challenge: ApprovalChallenge | None = None
    error: Exception | None = None


class SecurityPipeline:
    """Apply guardrails in fixed order before capability business logic."""

    def __init__(
        self,
        store: ApprovalStore | None = None,
        *,
        clock: Clock | None = None,
        identifiers: IdentifierGenerator | None = None,
        token_service: ApprovalTokenService | None = None,
    ) -> None:
        """Initialize a pipeline with optional approval persistence.

        Args:
            store: Application-owned approval store. If omitted, approval
                challenges are returned without persistence.
            clock: Clock used to create challenge timestamps.
            identifiers: Identifier generator used for challenge IDs.
            token_service: Optional portable-token verifier and consumer.
        """
        self.store = store
        self.clock = clock
        self.identifiers = identifiers
        self.token_service = token_service

    async def resume_token(
        self,
        token: str,
        execute: Callable[[], Any],
        *,
        context: AuthorizationContext,
    ) -> Any:
        """Verify and atomically consume a portable approval token."""
        if self.token_service is None:
            raise InvalidApprovalStateError("an approval token service is required")
        return await self.token_service.consume_token_and_resume(
            token,
            execute,
            context=context,
        )

    def check(
        self,
        target: Callable[..., Any],
        context: AuthorizationContext | None,
        arguments: Mapping[str, Any],
        *,
        agent_id: str = "",
        capability_id: str = "",
        approved_approval_id: str | None = None,
    ) -> GuardrailResult:
        """Check identity, scopes, and approval policy.

        Args:
            target: Resolved capability callable.
            context: Authenticated execution context.
            arguments: Validated capability arguments.
            agent_id: Agent bound to the invocation.
            capability_id: Capability bound to the invocation.
            approved_approval_id: Approval ID used by an approved resume.

        Returns:
            A typed allowed, approval-required, or authorization-failure result.
        """
        guardrails = discover_guardrails(target)
        if approved_approval_id is not None:
            if self.store is None:
                return GuardrailResult(
                    False,
                    error=InvalidApprovalStateError("approval store is required"),
                )
            return GuardrailResult(True)
        if guardrails.scopes or guardrails.approvals:
            if not isinstance(context, AuthorizationContext):
                return GuardrailResult(False, error=MissingAuthorizationContextError())
            missing = set(guardrails.scopes) - context.principal.scopes
            if missing:
                return GuardrailResult(
                    False, error=InsufficientScopeError("required scope missing")
                )
            applicable_roles: list[str] = []
            for requirement in guardrails.approvals:
                # Application policy callbacks may raise arbitrary exceptions; fail closed.
                # noinspection PyBroadException
                try:
                    applies = requirement.condition is None or bool(
                        requirement.condition(context, arguments)
                    )
                # Conditions are application callbacks; every callback failure fails closed.
                except Exception:
                    return GuardrailResult(
                        False, error=PolicyEvaluationError("approval policy failed")
                    )
                if not applies:
                    continue
                applicable_roles.append(requirement.role)
            if applicable_roles:
                challenge = default_challenge(
                    context,
                    agent_id=agent_id,
                    capability_id=capability_id,
                    role=",".join(sorted(set(applicable_roles))),
                    required_roles=tuple(sorted(set(applicable_roles))),
                    identifiers=self.identifiers or (lambda: str(uuid.uuid4())),
                    clock=self.clock,
                )
                if self.store is None:
                    return GuardrailResult(False, challenge=challenge)
                return GuardrailResult(False, challenge=self.store.create(challenge))
        return GuardrailResult(True)

    async def resume(
        self,
        decision: ApprovalDecision,
        execute: Callable[[], Any],
        *,
        agent_id: str,
        capability_id: str,
        context: AuthorizationContext,
    ) -> Any:
        """Consume an approved challenge, then execute its protected work.

        Args:
            decision: Approval decision to apply.
            execute: Protected callback to run after consumption.
            agent_id: Agent bound to the invocation.
            capability_id: Capability bound to the invocation.
            context: Immutable authorization context for the invocation.

        Returns:
            The callback result, awaited when necessary.

        Raises:
            InvalidApprovalStateError: If the challenge is unknown, stale, or
                bound to a different invocation.
            AuthorizationDeniedError: If the decision denies approval.
            ApprovalRequiredError: If additional roles must still be approved.

        The store performs the compare-and-transition operation, so stale or
        duplicate decisions cannot run business logic twice.
        """
        if self.store is None:
            raise InvalidApprovalStateError("an approval store is required to resume")
        try:
            challenge = self.store.get(decision.approval_id)
        except KeyError as error:
            raise InvalidApprovalStateError("unknown approval id") from error
        if (
            challenge.agent_id != agent_id
            or challenge.capability_id != capability_id
            or challenge.task_id != context.task_id
            or challenge.correlation_id != context.correlation_id
            or (
                challenge.principal_subject_id
                and challenge.principal_subject_id != context.principal.subject_id
            )
        ):
            raise InvalidApprovalStateError("approval challenge binding does not match")
        challenge = self.store.decide(decision)
        if not decision.approved:
            raise AuthorizationDeniedError("approval was denied")
        if self.store.state(challenge.approval_id).value != "working":
            raise ApprovalRequiredError("additional approvals are required")
        self.store.complete(challenge.approval_id)
        result = execute()
        if inspect.isawaitable(result):
            return await result
        return result
