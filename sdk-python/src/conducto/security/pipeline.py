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
from .audit import (
    AuditCategory,
    AuditDecision,
    AuditEmitter,
    AuditEvent,
    AuditEventName,
    AuditOutcome,
    AuditSeverity,
)
from .context import AuthorizationContext
from .errors import (
    ApprovalExpiredError,
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
        audit_emitter: AuditEmitter | None = None,
    ) -> None:
        """Initialize a pipeline with optional approval persistence.

        Args:
            store: Application-owned approval store. If omitted, approval
                challenges are returned without persistence.
            clock: Clock used to create challenge timestamps.
            identifiers: Identifier generator used for challenge IDs.
            token_service: Optional portable-token verifier and consumer.
            audit_emitter: Optional application-owned security audit delivery boundary.
        """
        self.store = store
        self.clock = clock
        self.identifiers = identifiers
        self.token_service = token_service
        self.audit_emitter = audit_emitter

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

    async def check_async(
        self,
        target: Callable[..., Any],
        context: AuthorizationContext | None,
        arguments: Mapping[str, Any],
        *,
        agent_id: str = "",
        capability_id: str = "",
        approved_approval_id: str | None = None,
    ) -> GuardrailResult:
        """Check guardrails and deliver mandatory pre-execution evidence.

        Existing synchronous callers may retain ``check``; runtime invocation
        uses this method, so required audit acceptance is enforced before work.
        """
        from conducto.core.telemetry import (
            SPAN_SECURITY_APPROVAL,
            SPAN_SECURITY_AUTHORIZE,
            start_span,
        )

        span_name = (
            SPAN_SECURITY_APPROVAL if approved_approval_id is not None else SPAN_SECURITY_AUTHORIZE
        )
        with start_span(
            span_name,
            attributes={
                "conducto.agent.id": agent_id,
                "conducto.capability.id": capability_id,
                "conducto.task.id": context.task_id if context else "",
                "conducto.correlation_id": context.correlation_id if context else "",
            },
        ) as span:
            result = self.check(
                target,
                context,
                arguments,
                agent_id=agent_id,
                capability_id=capability_id,
                approved_approval_id=approved_approval_id,
            )
            if result.allowed:
                span.set_outcome("success")
            elif result.challenge is not None:
                span.set_outcome("approval_required", reason=result.challenge.reason_code)
            else:
                span.set_outcome(
                    "denied",
                    reason=getattr(result.error, "reason_code", "authorization_denied"),
                )
        if self.audit_emitter is None:
            return result
        guardrails = discover_guardrails(target)
        if not (guardrails.scopes or guardrails.approvals or approved_approval_id):
            return result
        if result.challenge is not None:
            await self._emit(
                AuditEventName.APPROVAL_REQUESTED,
                context,
                agent_id,
                capability_id,
                AuditDecision.REQUIRE_APPROVAL,
                AuditOutcome.PENDING,
                result.challenge.reason_code,
                challenge_id=result.challenge.approval_id,
                policy_version=result.challenge.policy_version,
                required=True,
            )
        elif result.allowed:
            await self._emit(
                AuditEventName.AUTHORIZATION_ALLOWED,
                context,
                agent_id,
                capability_id,
                AuditDecision.ALLOW,
                AuditOutcome.SUCCESS,
                "authorized",
                required=True,
            )
            await self._emit(
                AuditEventName.EXECUTION_ACCEPTED,
                context,
                agent_id,
                capability_id,
                AuditDecision.ALLOW,
                AuditOutcome.SUCCESS,
                "audit_accepted",
                required=True,
            )
        else:
            assert result.error is not None
            await self._emit(
                AuditEventName.AUTHORIZATION_DENIED,
                context,
                agent_id,
                capability_id,
                AuditDecision.DENY,
                AuditOutcome.REJECTED,
                getattr(result.error, "reason_code", "authorization_denied"),
                required=True,
            )
        return result

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
        except ApprovalExpiredError:
            await self._emit(
                AuditEventName.APPROVAL_EXPIRED,
                context,
                agent_id,
                capability_id,
                AuditDecision.DENY,
                AuditOutcome.REJECTED,
                "expired",
                challenge_id=decision.approval_id,
                required=False,
            )
            raise
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
        try:
            challenge = self.store.decide(decision)
        except ApprovalExpiredError:
            await self._emit(
                AuditEventName.APPROVAL_EXPIRED,
                context,
                agent_id,
                capability_id,
                AuditDecision.DENY,
                AuditOutcome.REJECTED,
                "expired",
                challenge_id=decision.approval_id,
                required=False,
            )
            raise
        if not decision.approved:
            await self._emit(
                AuditEventName.APPROVAL_DENIED,
                context,
                agent_id,
                capability_id,
                AuditDecision.DENY,
                AuditOutcome.REJECTED,
                decision.reason_code or "denied",
                challenge_id=challenge.approval_id,
                policy_version=challenge.policy_version,
                required=True,
            )
            raise AuthorizationDeniedError("approval was denied")
        if self.store.state(challenge.approval_id).value != "working":
            await self._emit(
                AuditEventName.APPROVAL_APPROVED,
                context,
                agent_id,
                capability_id,
                AuditDecision.REQUIRE_APPROVAL,
                AuditOutcome.PENDING,
                "additional_approval_required",
                challenge_id=challenge.approval_id,
                policy_version=challenge.policy_version,
                required=True,
            )
            raise ApprovalRequiredError("additional approvals are required")
        await self._emit(
            AuditEventName.APPROVAL_APPROVED,
            context,
            agent_id,
            capability_id,
            AuditDecision.ALLOW,
            AuditOutcome.SUCCESS,
            decision.reason_code or "approved",
            challenge_id=challenge.approval_id,
            policy_version=challenge.policy_version,
            required=True,
        )
        self.store.complete(challenge.approval_id)
        result = execute()
        if inspect.isawaitable(result):
            return await result
        return result

    async def cancel_approval(
        self,
        approval_id: str,
        *,
        agent_id: str,
        capability_id: str,
        context: AuthorizationContext,
    ) -> ApprovalChallenge:
        """Cancel a pending approval and emit its auditable lifecycle outcome."""
        if self.store is None:
            raise InvalidApprovalStateError("an approval store is required to cancel")
        try:
            challenge = self.store.cancel(approval_id)
        except ApprovalExpiredError:
            await self._emit(
                AuditEventName.APPROVAL_EXPIRED,
                context,
                agent_id,
                capability_id,
                AuditDecision.DENY,
                AuditOutcome.REJECTED,
                "expired",
                challenge_id=approval_id,
                required=False,
            )
            raise
        await self._emit(
            AuditEventName.APPROVAL_CANCELED,
            context,
            agent_id,
            capability_id,
            AuditDecision.DENY,
            AuditOutcome.REJECTED,
            "canceled",
            challenge_id=challenge.approval_id,
            policy_version=challenge.policy_version,
            required=False,
        )
        return challenge

    async def emit_execution(
        self,
        name: AuditEventName,
        context: AuthorizationContext | None,
        *,
        agent_id: str,
        capability_id: str,
        outcome: AuditOutcome,
        reason_code: str,
    ) -> None:
        """Emit lifecycle evidence after pre-execution acceptance."""
        await self._emit(
            name,
            context,
            agent_id,
            capability_id,
            AuditDecision.ALLOW,
            outcome,
            reason_code,
            required=False,
        )

    async def _emit(
        self,
        name: AuditEventName,
        context: AuthorizationContext | None,
        agent_id: str,
        capability_id: str,
        decision: AuditDecision,
        outcome: AuditOutcome,
        reason_code: str,
        *,
        challenge_id: str = "",
        policy_version: str = "1",
        required: bool,
    ) -> None:
        if self.audit_emitter is None:
            return
        principal = context.principal if context is not None else None
        from conducto.core.telemetry import current_trace_ids

        trace_ids = current_trace_ids()
        category = (
            AuditCategory.AUTHORIZATION
            if name.value.startswith("security.authorization")
            else AuditCategory.APPROVAL
            if name.value.startswith("security.approval")
            else AuditCategory.EXECUTION
        )
        await self.audit_emitter.emit(
            AuditEvent(
                event_name=name,
                category=category,
                decision=decision,
                outcome=outcome,
                reason_code=reason_code,
                subject_id=principal.subject_id if principal else "",
                issuer=principal.issuer if principal else "",
                audience=principal.audience if principal else "",
                task_id=context.task_id if context else "",
                challenge_id=challenge_id,
                agent_id=agent_id,
                capability_id=capability_id,
                policy_id="conducto.guardrails",
                policy_version=policy_version,
                correlation_id=context.correlation_id if context else "",
                trace_id=trace_ids.trace_id if trace_ids is not None else "",
                span_id=trace_ids.span_id if trace_ids is not None else "",
                resource=f"capability:{agent_id}:{capability_id}",
                severity=(
                    AuditSeverity.ERROR
                    if outcome in (AuditOutcome.FAILURE, AuditOutcome.REJECTED)
                    else AuditSeverity.INFO
                ),
            ),
            required=required,
        )
