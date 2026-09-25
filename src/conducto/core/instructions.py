"""Trusted instruction resolution for governed runtime model calls."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .provider import ChatMessage


def normalize_instruction(value: str | None) -> str | None:
    """Normalize one optional instruction while rejecting blank declarations."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("Instructions cannot be empty")
    return normalized


def resolve_instruction_chain(
    runtime_policy: Iterable[str] = (),
    agent_instructions: str | None = None,
    capability_instructions: str | None = None,
) -> tuple[str, ...]:
    """Resolve instructions in immutable, deterministic precedence order.

    Runtime policy instructions are followed by agent instructions and then
    capability refinement. Empty values are omitted; no lower-precedence value
    can replace an earlier value.
    """
    chain = tuple(normalize_instruction(value) for value in runtime_policy)
    normalized_agent = normalize_instruction(agent_instructions)
    normalized_capability = normalize_instruction(capability_instructions)
    return tuple(
        value for value in (*chain, normalized_agent, normalized_capability) if value is not None
    )


def compose_system_message(
    messages: Sequence[ChatMessage],
    instructions: Sequence[str],
) -> tuple[ChatMessage, ...]:
    """Prepend the resolved instruction chain as one trusted system message."""
    if not instructions:
        return tuple(messages)
    content = "\n\n".join(instructions)
    return (ChatMessage(role="system", content=content), *messages)
