"""Runtime-owned context construction and child authority attenuation."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from conducto.security import AuthorizationContext
from conducto.security.errors import SecurityError

from .model_config import RunConfig
from .model_resolution import _ResolvedModelBinding
from .run_context import (
    CancellationState,
    DelegationBudget,
    DelegationFrame,
    RunContext,
    get_run_context,
)

if TYPE_CHECKING:
    from .runtime import Runtime
    from .structured import CapabilityOutputContract


def build_run_context(
    runtime: Runtime,
    *,
    agent_id: str,
    run: RunConfig,
    binding: _ResolvedModelBinding | None,
    correlation_id: str,
    run_id: str,
    authorization: Any,
    allowed_capabilities: frozenset[str] | None,
    delegation_budget: DelegationBudget | None,
    delegation_frame: DelegationFrame | None,
    cancellation: CancellationState | None,
    instruction_chain: tuple[str, ...] = (),
    output_contract: CapabilityOutputContract | None = None,
) -> RunContext:
    """Build a context without amplifying its parent's deadlines or authority.

    Parent state is inherited only from a context owned by the same runtime.
    Model selection has already completed before this function is called.

    Args:
        runtime: Runtime that owns the created context.
        agent_id: Agent identifier associated with the run.
        run: Run-level timeout, metadata, and policy configuration.
        binding: Resolved model binding, if a model was selected.
        correlation_id: Correlation identifier for the run.
        run_id: Explicit run identifier, or empty to derive one.
        authorization: Authenticated authorization context, if any.
        allowed_capabilities: Capability set bounded by the parent context.
        delegation_budget: Root delegation limits when there is no parent.
        delegation_frame: Frame appended to the parent's delegation path.
        cancellation: Cancellation state for a root invocation.
        instruction_chain: Resolved, ordered instruction chain -- runtime
            policy, then agent, then capability instructions -- to record on
            the created context and compose into its model calls.
        output_contract: Optional structured-output contract for the active
            capability return value.

    Returns:
        A task-local run context associated with this runtime.
    """
    active_context = get_run_context()
    parent = (
        active_context
        if active_context is not None and active_context.belongs_to(runtime)
        else None
    )
    effective_timeout: float | None = run.timeout
    requested_deadline = (
        time.monotonic() + effective_timeout if effective_timeout is not None else None
    )
    deadline: float | None
    if parent is not None and parent.deadline is not None:
        deadline = (
            min(requested_deadline, parent.deadline)
            if requested_deadline is not None
            else parent.deadline
        )
        effective_timeout = max(0.0, deadline - time.monotonic())
    else:
        deadline = requested_deadline
    effective_run_id = run_id or (
        str(uuid.uuid4())
        if parent is not None
        else (
            authorization.task_id
            if isinstance(authorization, AuthorizationContext)
            else str(uuid.uuid4())
        )
    )
    if isinstance(authorization, AuthorizationContext) and authorization.correlation_id != (
        correlation_id or authorization.correlation_id
    ):
        raise SecurityError("authorization correlation_id does not match run context")
    parent_allowed = parent.allowed_capabilities if parent is not None else None
    effective_allowed: frozenset[str] | None
    if parent_allowed is not None:
        if allowed_capabilities is None:
            effective_allowed = parent_allowed
        elif not allowed_capabilities.issubset(parent_allowed):
            raise SecurityError("delegated capabilities are broader than their caller")
        else:
            effective_allowed = frozenset(allowed_capabilities)
    else:
        effective_allowed = (
            frozenset(allowed_capabilities) if allowed_capabilities is not None else None
        )
    path = parent.delegation_path if parent is not None else ()
    if delegation_frame is not None:
        path += (delegation_frame,)
    return RunContext(
        run_id=effective_run_id,
        correlation_id=correlation_id or str(uuid.uuid4()),
        model=binding.model if binding is not None else None,
        timeout=effective_timeout,
        deadline=deadline,
        cancellation=(
            parent.cancellation if parent is not None else (cancellation or CancellationState())
        ),
        metadata=run.metadata,
        agent_id=agent_id,
        parent_run_id=parent.run_id if parent is not None else None,
        delegation_path=path,
        allowed_capabilities=effective_allowed,
        delegation_budget=(
            parent.delegation_budget
            if parent is not None
            else (delegation_budget or DelegationBudget())
        ),
        policy_context=run,
        instruction_chain=instruction_chain,
        output_contract=output_contract,
        _runtime=runtime,
        _agent_registry=runtime.agent_registry,
        _binding=binding,
        authorization=authorization,
    )
