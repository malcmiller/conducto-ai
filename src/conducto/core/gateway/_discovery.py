"""Filter one immutable registry snapshot using compatibility and caller authority."""

import re

from ..gateway_models import (
    CapabilityDescriptor,
    DiscoveryQuery,
    RegistrationLifecycle,
    RegistrySnapshot,
)
from ..run_context import DelegationFrame, RunContext
from ._contracts import GatewayPolicy
from ._schema import _schema_compatible, _UnsupportedSchemaError, _validate_compatibility_schema


class _GatewayPolicyEvaluationError(Exception):
    """Internal signal for a failed application policy callback."""


class GatewayAuthorization:
    """Apply the same attenuated authority at discovery and invocation boundaries."""

    def __init__(self, context: RunContext, policy: GatewayPolicy | None) -> None:
        self._context = context
        self._policy = policy

    def permits(self, descriptor: CapabilityDescriptor, *, check_budget: bool = True) -> bool:
        """Check scopes, allowlists, application policy, budgets, and delegation cycles."""
        allowed = self._context.allowed_capabilities
        exact_id = f"{descriptor.agent_id}:{descriptor.capability_id}"
        if allowed is not None and not (descriptor.capability_id in allowed or exact_id in allowed):
            return False
        authorization = self._context.authorization
        scopes = authorization.principal.scopes if authorization is not None else frozenset()
        if not set(descriptor.required_scopes).issubset(scopes):
            return False
        if self._policy is not None:
            try:
                if not self._policy(self._context, descriptor):
                    return False
            except Exception as error:
                raise _GatewayPolicyEvaluationError from error
        if check_budget:
            budget = self._context.remaining_delegation_budget
            if budget.depth < 1 or budget.calls < 1 or budget.time == 0:
                return False
        frame = DelegationFrame(descriptor.agent_id, descriptor.capability_id)
        return frame not in self._context.delegation_path


def matching_candidates(
    query: DiscoveryQuery,
    snapshot: RegistrySnapshot,
    authorization: GatewayAuthorization,
) -> tuple[int, list[tuple[CapabilityDescriptor, int]], bool, bool]:
    """Return ordered compatible candidates without accessing mutable registrations."""
    if query.input_schema is not None:
        _validate_compatibility_schema(query.input_schema)
    if query.output_schema is not None:
        _validate_compatibility_schema(query.output_schema)
    matches: list[tuple[CapabilityDescriptor, int]] = []
    denied = False
    unsupported = False
    for agent in snapshot.agents:
        if agent.lifecycle is not RegistrationLifecycle.ACTIVE or not agent.healthy:
            continue
        if query.agent_id is not None and agent.agent_id != query.agent_id:
            continue
        if query.version_constraint and not _version_matches(
            agent.version, query.version_constraint
        ):
            continue
        for descriptor in agent.capabilities:
            if query.capability_ids and descriptor.capability_id not in query.capability_ids:
                continue
            if query.tags and not query.tags.issubset(descriptor.tags):
                continue
            try:
                input_compatible = _schema_compatible(
                    query.input_schema, descriptor.input_schema, output=False
                )
                output_compatible = _schema_compatible(
                    query.output_schema, descriptor.output_schema, output=True
                )
            except _UnsupportedSchemaError:
                unsupported = True
                continue
            if not input_compatible or not output_compatible:
                continue
            if not query.include_approval_required and descriptor.approval_required:
                continue
            if not authorization.permits(descriptor):
                denied = True
                continue
            matches.append((descriptor, agent.generation))
    matches.sort(
        key=lambda item: (
            item[0].capability_id,
            item[0].agent_id,
            item[0].agent_version,
            item[0].schema_digest,
        )
    )
    return snapshot.revision, matches, denied, unsupported


def _version_tuple(value: str) -> tuple[int, int, int, str]:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?([-.+].*)?\s*", value)
    if match is None:
        return 0, 0, 0, value
    return (
        int(match.group(1)),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
        match.group(4) or "",
    )


def _version_matches(version: str, constraint: str) -> bool:
    actual = _version_tuple(version)
    for clause in constraint.split(","):
        match = re.fullmatch(r"\s*(==|!=|>=|<=|>|<)?\s*(\S+)\s*", clause)
        if match is None:
            return False
        operator = match.group(1) or "=="
        expected = _version_tuple(match.group(2))
        comparisons = {
            "==": actual == expected,
            "!=": actual != expected,
            ">=": actual >= expected,
            "<=": actual <= expected,
            ">": actual > expected,
            "<": actual < expected,
        }
        if not comparisons[operator]:
            return False
    return True
