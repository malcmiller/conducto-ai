"""Runtime-owned authorization and approval wiring for capability invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from conducto.security import ApprovalDecision, AuditDeliveryError, AuthorizationContext
from conducto.security.context import delegate_context
from conducto.security.errors import SecurityError

from .invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationResult,
)
from .model_config import ModelReference, RunConfig
from .run_context import CancellationState, DelegationBudget, get_run_context

if TYPE_CHECKING:
    from .agent import BaseAgent
    from .runtime import Runtime


async def invoke_capability(
    runtime: Runtime,
    agent: BaseAgent,
    capability: str | Callable[..., Any],
    arguments: Mapping[str, Any],
    *,
    timeout: float | None,
    correlation_id: str,
    model_reference: ModelReference | str | None,
    run_config: RunConfig | None,
    authorization: Any,
    allowed_capabilities: frozenset[str] | None,
    delegation_budget: DelegationBudget | None,
    cancellation: CancellationState | None,
) -> InvocationResult:
    """Attenuate inherited authorization and normalize security failures."""
    from .invocation import invoke_agent

    active_context = get_run_context()
    try:
        effective_authorization = delegate_context(
            active_context.authorization if active_context is not None else None,
            authorization,
        )
    except SecurityError as error:
        return InvocationAuthorizationFailure(
            correlation_id or runtime.new_correlation_id(), error.reason_code
        )
    try:
        return await invoke_agent(
            runtime,
            agent,
            capability,
            arguments,
            timeout=timeout,
            correlation_id=correlation_id,
            model_reference=model_reference,
            run_config=run_config,
            authorization=effective_authorization,
            security_pipeline=runtime.security_pipeline,
            allowed_capabilities=allowed_capabilities,
            delegation_budget=delegation_budget,
            cancellation=cancellation,
        )
    except AuditDeliveryError as error:
        return InvocationAuditFailure(
            correlation_id or runtime.new_correlation_id(), error.reason_code
        )
    except SecurityError as error:
        return InvocationAuthorizationFailure(
            correlation_id or runtime.new_correlation_id(), error.reason_code
        )


async def resume_approved_capability(
    runtime: Runtime,
    agent: BaseAgent,
    capability: str | Callable[..., Any],
    arguments: Mapping[str, Any],
    decision: ApprovalDecision,
    *,
    authorization: AuthorizationContext,
    timeout: float | None,
    model_reference: ModelReference | str | None,
    run_config: RunConfig | None,
    allowed_capabilities: frozenset[str] | None,
    delegation_budget: DelegationBudget | None,
    cancellation: CancellationState | None,
) -> InvocationResult:
    """Resume a persisted approval through the same invocation pipeline."""
    from .invocation import invoke_agent

    async def execute() -> InvocationResult:
        return await invoke_agent(
            runtime,
            agent,
            capability,
            arguments,
            timeout=timeout,
            correlation_id=authorization.correlation_id,
            model_reference=model_reference,
            run_config=run_config,
            authorization=authorization,
            security_pipeline=runtime.security_pipeline,
            approved_approval_id=decision.approval_id,
            allowed_capabilities=allowed_capabilities,
            delegation_budget=delegation_budget,
            cancellation=cancellation,
        )

    capability_name = capability if isinstance(capability, str) else ""
    if not isinstance(capability, str):
        requested = getattr(capability, "__func__", capability)
        for name, registered in agent.capabilities.items():
            candidate = getattr(registered.callable, "__func__", registered.callable)
            if candidate is requested:
                capability_name = name
                break
    try:
        return cast(
            InvocationResult,
            await runtime.security_pipeline.resume(
                decision,
                execute,
                agent_id=agent.agent_metadata.name,
                capability_id=capability_name,
                context=authorization,
            ),
        )
    except SecurityError as error:
        if error.reason_code == "approval_required":
            assert runtime.security_pipeline.store is not None
            challenge = runtime.security_pipeline.store.get(decision.approval_id)
            return InvocationApprovalRequired(authorization.correlation_id, challenge)
        return InvocationAuthorizationFailure(authorization.correlation_id, error.reason_code)


async def resume_token_capability(
    runtime: Runtime,
    agent: BaseAgent,
    capability: str | Callable[..., Any],
    arguments: Mapping[str, Any],
    token: str,
    *,
    authorization: AuthorizationContext,
) -> InvocationResult:
    """Verify a portable approval token before invoking its bound capability."""
    from .invocation import invoke_agent

    if runtime.security_pipeline.token_service is None:
        return InvocationAuthorizationFailure(
            authorization.correlation_id, "invalid_state_transition"
        )
    claims = runtime.security_pipeline.token_service.verifier.verify(token)

    async def execute() -> InvocationResult:
        return await invoke_agent(
            runtime,
            agent,
            capability,
            arguments,
            correlation_id=authorization.correlation_id,
            authorization=authorization,
            security_pipeline=runtime.security_pipeline,
            approved_approval_id=claims["challenge_id"],
        )

    try:
        return cast(
            InvocationResult,
            await runtime.security_pipeline.resume_token(token, execute, context=authorization),
        )
    except SecurityError as error:
        return InvocationAuthorizationFailure(authorization.correlation_id, error.reason_code)
