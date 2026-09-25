"""Deterministic precedence and composition for governed agent instructions.

Agents and capabilities may declare persona and behavioral instructions as
governed metadata instead of hand-managed module-level system prompts. This
module resolves those declared instructions, together with runtime-owned
policy instructions, into one deterministically ordered chain, and composes
that chain into the system role of runtime-issued model calls.

Precedence is fixed and non-negotiable: runtime policy instructions always
come first, then agent instructions, then capability instructions. Callers
cannot inject, replace, or suppress any part of this chain through
invocation arguments; the chain is assembled exclusively from trusted,
framework-resolved metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

from .provider import ChatMessage
from .runtime_errors import UntrustedSystemMessageError

__all__ = [
    "compose_system_message",
    "normalize_instruction",
    "resolve_instruction_chain",
]


def normalize_instruction(value: str | None) -> str | None:
    """Normalize declared instruction text.

    Args:
        value: Raw instruction text, or ``None`` when unset.

    Returns:
        The stripped instruction text, or ``None`` when ``value`` is ``None``
        or contains only whitespace.
    """
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_instruction_chain(
    policy_instructions: Sequence[str],
    agent_instructions: str | None,
    capability_instructions: str | None = None,
) -> tuple[str, ...]:
    """Resolve the deterministic, ordered instruction chain for one invocation.

    Precedence is fixed: runtime-owned policy instructions are applied first,
    then declared agent instructions, then declared capability instructions.
    Capability instructions may refine the chain but never remove or replace
    the runtime policy or agent instructions that precede them. Empty or
    whitespace-only entries are dropped; the caller-visible chain is limited
    to entries with actual content.

    Args:
        policy_instructions: Runtime-owned policy instructions, applied
            first and in the given order.
        agent_instructions: Declared agent instructions, applied second.
        capability_instructions: Declared capability instructions, applied
            last to refine the agent and policy instructions.

    Returns:
        An ordered, immutable tuple of non-empty instruction strings in
        runtime policy, agent, then capability precedence.
    """
    resolved: list[str] = []
    for entry in policy_instructions:
        normalized_policy = normalize_instruction(entry)
        if normalized_policy is not None:
            resolved.append(normalized_policy)
    for candidate in (agent_instructions, capability_instructions):
        normalized_candidate = normalize_instruction(candidate)
        if normalized_candidate is not None:
            resolved.append(normalized_candidate)
    return tuple(resolved)


def compose_system_message(
    messages: Sequence[ChatMessage],
    instruction_chain: Sequence[str],
) -> tuple[ChatMessage, ...]:
    """Prepend one composed system message for a resolved instruction chain.

    The framework exclusively owns the system role of a runtime-issued model
    call: the sole system-role message is always the one composed here from
    the trusted, resolved instruction chain. This guarantees that a caller
    cannot inject, replace, or suppress trusted instructions by supplying a
    competing ``system``-role message alongside its arguments.

    Args:
        messages: Conversation history for the provider request. Must not
            already contain a ``system``-role message; declare persona or
            behavioral text through ``@a2a_agent``/``@a2a_capability``
            ``instructions`` instead.
        instruction_chain: Resolved, ordered instruction chain to compose
            into the system role, as produced by
            :func:`resolve_instruction_chain`.

    Returns:
        The original messages unchanged when ``instruction_chain`` is empty;
        otherwise a new tuple with one system :class:`ChatMessage` -- built
        by joining the chain with blank lines -- prepended.

    Raises:
        UntrustedSystemMessageError: If ``messages`` already contains a
            ``system``-role message. The framework owns the system role, so
            capability code must not construct one directly.
    """
    if any(message.role == "system" for message in messages):
        raise UntrustedSystemMessageError(
            "Capability-supplied messages must not use the 'system' role; the "
            "runtime composes the sole system message from the resolved, "
            "trusted instruction chain."
        )
    if not instruction_chain:
        return tuple(messages)
    composed = "\n\n".join(instruction_chain)
    return (ChatMessage(role="system", content=composed), *messages)
