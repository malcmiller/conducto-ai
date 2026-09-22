"""Typed invocation seam bound by a later story to canonical Conducto dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from a2a.types.a2a_pb2 import Message

from conducto.core.invocation_results import InvocationResult


@runtime_checkable
class A2ARequestHandler(Protocol):
    """Abstract seam between the A2A ASGI host and Conducto capability dispatch.

    This story owns HTTP/ASGI protocol adaptation and task dispatch only; it never
    calls a reflected agent's capability methods directly. A later story binds this
    seam to the canonical Conducto runtime so accepted A2A messages are invoked
    through the same governed contract as every other invocation path.
    """

    async def handle_message(
        self, message: Message, *, task_id: str, context_id: str
    ) -> InvocationResult:
        """Invoke Conducto capability dispatch for one accepted A2A message.

        Args:
            message: Validated inbound A2A message.
            task_id: Server-assigned or continued A2A task identifier.
            context_id: A2A context identifier shared across related tasks.

        Returns:
            The invocation outcome to map onto the A2A task lifecycle.
        """
        ...


__all__ = ["A2ARequestHandler"]
