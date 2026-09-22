"""Typed invocation seam between A2A protocol dispatch and Conducto execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from a2a.types.a2a_pb2 import Message

from conducto.core.invocation_results import InvocationResult
from conducto.core.model_config import freeze_metadata


@dataclass(frozen=True, slots=True)
class A2ARequestContext:
    """Immutable transport facts associated with one accepted A2A request.

    Attributes:
        request_id: JSON-RPC request identifier normalized to text.
        headers: Inbound headers supplied only to the application-owned identity
            resolver. Runtime metadata never receives this mapping.
        method: Pinned A2A method name.
        safe_metadata: Credential-free integration metadata supplied by the
            embedding application.
    """

    request_id: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    method: str = "SendMessage"
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze request facts and reject missing replay identifiers."""
        if not self.request_id:
            raise ValueError("A2A request_id is required")
        object.__setattr__(
            self,
            "headers",
            MappingProxyType({str(key).lower(): str(value) for key, value in self.headers.items()}),
        )
        object.__setattr__(self, "safe_metadata", freeze_metadata(self.safe_metadata))


@runtime_checkable
class A2ARequestHandler(Protocol):
    """Abstract seam between the A2A ASGI host and Conducto capability dispatch.

    The ASGI host owns protocol adaptation and task dispatch only; it never calls
    reflected capability methods directly. Runtime-backed implementations bind this
    seam to the same governed invocation contract used by local calls.
    """

    async def handle_message(
        self,
        message: Message,
        *,
        task_id: str,
        context_id: str,
        request_context: A2ARequestContext,
    ) -> InvocationResult:
        """Invoke Conducto capability dispatch for one accepted A2A message.

        Args:
            message: Validated inbound A2A message.
            task_id: Server-assigned or continued A2A task identifier.
            context_id: A2A context identifier shared across related tasks.
            request_context: Immutable JSON-RPC and transport facts.

        Returns:
            The invocation outcome to map onto the A2A task lifecycle.
        """
        ...


@runtime_checkable
class A2ACancellableRequestHandler(Protocol):
    """Optional cancellation hook implemented by runtime-backed handlers."""

    async def cancel(self, task_id: str) -> None:
        """Request cancellation of the active invocation for ``task_id``."""
        ...


__all__ = ["A2ACancellableRequestHandler", "A2ARequestContext", "A2ARequestHandler"]
