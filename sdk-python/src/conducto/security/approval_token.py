"""Approval-token claims and replay-safe consumption."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from .approval import ApprovalDecision, ApprovalStore
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
from .crypto import ES256Verifier
from .errors import (
    ApprovalRequiredError,
    ApprovalTokenBindingError,
    ApprovalTokenError,
    ApprovalTokenReplayError,
    ApprovalTokenRoleError,
    AuthorizationDeniedError,
)


@dataclass(frozen=True, slots=True)
class ApprovalTokenClaims:
    """Portable claims binding one approval decision to one invocation."""

    issuer: str
    audience: str
    subject: str
    issued_at: int
    not_before: int
    expires_at: int
    token_id: str
    challenge_id: str
    task_id: str
    agent_id: str
    capability_id: str
    decision: str
    required_role: str
    policy_version: str = "1"
    contract_version: str = "1"

    def as_dict(self) -> dict[str, Any]:
        """Return the language-neutral claim names used on the wire."""
        return {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": self.subject,
            "iat": self.issued_at,
            "nbf": self.not_before,
            "exp": self.expires_at,
            "jti": self.token_id,
            "challenge_id": self.challenge_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "capability_id": self.capability_id,
            "decision": self.decision,
            "required_role": self.required_role,
            "policy_version": self.policy_version,
            "ver": self.contract_version,
        }


class ApprovalReplayStore(Protocol):
    """Atomic, durable token-ID consumption boundary."""

    def consume(self, token_id: str, expires_at: int) -> None:
        """Record a token ID or raise ``ApprovalTokenReplayError``."""


class InMemoryApprovalReplayStore:
    """Thread-safe reference replay store for deterministic tests."""

    def __init__(self) -> None:
        """Initialize an empty consumed-token set."""
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def consume(self, token_id: str, expires_at: int) -> None:
        """Atomically consume a token ID exactly once."""
        del expires_at
        with self._lock:
            if token_id in self._consumed:
                raise ApprovalTokenReplayError("approval token was already consumed")
            self._consumed.add(token_id)


@dataclass(frozen=True, slots=True)
class ApprovalVerificationResult:
    """Verified claims and the resulting approval decision."""

    claims: ApprovalTokenClaims
    decision: ApprovalDecision


class ApprovalTokenService:
    """Verify, bind, consume, and apply one approval token atomically."""

    def __init__(
        self,
        verifier: ES256Verifier,
        replay_store: ApprovalReplayStore,
        approval_store: ApprovalStore,
        audit_emitter: AuditEmitter | None = None,
    ) -> None:
        """Initialize token verification with application-owned stores."""
        self.verifier = verifier
        self.replay_store = replay_store
        self.approval_store = approval_store
        self.audit_emitter = audit_emitter

    def verify(self, token: str, *, challenge: Any) -> ApprovalVerificationResult:
        """Verify token cryptography and all challenge bindings."""
        raw = self.verifier.verify(token)
        claims = ApprovalTokenClaims(
            issuer=raw["iss"],
            audience=raw["aud"],
            subject=raw["sub"],
            issued_at=raw["iat"],
            not_before=raw["nbf"],
            expires_at=raw["exp"],
            token_id=raw["jti"],
            challenge_id=raw["challenge_id"],
            task_id=raw["task_id"],
            agent_id=raw["agent_id"],
            capability_id=raw["capability_id"],
            decision=raw["decision"],
            required_role=raw["required_role"],
            policy_version=raw["policy_version"],
            contract_version=raw["ver"],
        )
        if any(
            getattr(challenge, field) != value
            for field, value in (
                ("approval_id", claims.challenge_id),
                ("task_id", claims.task_id),
                ("agent_id", claims.agent_id),
                ("capability_id", claims.capability_id),
                ("required_role", claims.required_role),
                ("policy_version", claims.policy_version),
            )
        ):
            raise ApprovalTokenBindingError("approval token does not match challenge")
        if claims.required_role not in (challenge.required_roles or (challenge.required_role,)):
            raise ApprovalTokenRoleError("approval token role is not required")
        return ApprovalVerificationResult(
            claims,
            ApprovalDecision(
                claims.challenge_id,
                claims.decision == "approve",
                datetime.fromtimestamp(claims.issued_at, UTC),
                claims.subject,
                role=claims.required_role,
            ),
        )

    async def consume_and_resume(
        self,
        token: str,
        execute: Callable[[], Any],
        *,
        challenge: Any,
        context: AuthorizationContext,
    ) -> Any:
        """Verify and atomically consume a token before invoking protected work."""
        result = self.verify(token, challenge=challenge)
        await self._audit(
            AuditEventName.SIGNATURE_VERIFIED,
            context,
            challenge,
            AuditOutcome.SUCCESS,
            "signature_verified",
        )
        if (
            challenge.principal_subject_id
            and challenge.principal_subject_id != context.principal.subject_id
        ):
            raise ApprovalTokenBindingError("approval token subject is not intended")
        try:
            self.replay_store.consume(result.claims.token_id, result.claims.expires_at)
        except ApprovalTokenReplayError:
            await self._audit(
                AuditEventName.REPLAY_REJECTED,
                context,
                challenge,
                AuditOutcome.REJECTED,
                "replay",
            )
            raise
        self.approval_store.decide(result.decision)
        if not result.decision.approved:
            raise AuthorizationDeniedError("approval was denied")
        if self.approval_store.state(challenge.approval_id).value != "working":
            raise ApprovalRequiredError("additional approvals are required")
        self.approval_store.complete(challenge.approval_id)
        value = execute()
        if hasattr(value, "__await__"):
            return await value
        return value

    async def consume_token_and_resume(
        self,
        token: str,
        execute: Callable[[], Any],
        *,
        context: AuthorizationContext,
    ) -> Any:
        """Resolve the token's challenge, then perform atomic consumption."""
        try:
            raw = self.verifier.verify(token)
        except ApprovalTokenError:
            # Token bodies and verification exception details are deliberately
            # excluded from the event.
            if self.audit_emitter is not None:
                await self.audit_emitter.emit(
                    AuditEvent(
                        event_name=AuditEventName.SIGNATURE_REJECTED,
                        category=AuditCategory.CRYPTOGRAPHY,
                        decision=AuditDecision.DENY,
                        outcome=AuditOutcome.REJECTED,
                        reason_code="signature_verification_failed",
                        subject_id=context.principal.subject_id,
                        issuer=context.principal.issuer,
                        audience=context.principal.audience,
                        task_id=context.task_id,
                        correlation_id=context.correlation_id,
                        resource="approval-token",
                        severity=AuditSeverity.ERROR,
                    ),
                    required=False,
                )
            raise
        challenge = self.approval_store.get(raw["challenge_id"])
        return await self.consume_and_resume(
            token,
            execute,
            challenge=challenge,
            context=context,
        )

    async def _audit(
        self,
        name: AuditEventName,
        context: AuthorizationContext,
        challenge: Any,
        outcome: AuditOutcome,
        reason_code: str,
    ) -> None:
        if self.audit_emitter is None:
            return
        await self.audit_emitter.emit(
            AuditEvent(
                event_name=name,
                category=AuditCategory.CRYPTOGRAPHY,
                decision=(
                    AuditDecision.ALLOW
                    if outcome == AuditOutcome.SUCCESS
                    else AuditDecision.DENY
                ),
                outcome=outcome,
                reason_code=reason_code,
                subject_id=context.principal.subject_id,
                issuer=context.principal.issuer,
                audience=context.principal.audience,
                task_id=context.task_id,
                challenge_id=challenge.approval_id,
                agent_id=challenge.agent_id,
                capability_id=challenge.capability_id,
                policy_id="conducto.approval-token",
                policy_version=challenge.policy_version,
                correlation_id=context.correlation_id,
                resource="approval-token",
                severity=(
                    AuditSeverity.ERROR if outcome != AuditOutcome.SUCCESS else AuditSeverity.INFO
                ),
            ),
            required=False,
        )
