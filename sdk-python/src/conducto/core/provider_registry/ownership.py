"""Provider ownership, immutable cleanup outcomes, and client identity records."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from ..model_config import ModelReference
from ..provider import ModelProvider

if TYPE_CHECKING:
    from .lifecycle import _ProviderLifecycle


class ProviderOwnership(StrEnum):
    """Declare which party is responsible for closing a provider client."""

    RUNTIME_OWNED = "runtime_owned"
    CALLER_OWNED = "caller_owned"


@dataclass(frozen=True, slots=True)
class ProviderCleanupFailure:
    """Safe diagnostic for one client that did not close.

    Attributes:
        references: Model references that intentionally shared the client.
        provider: Provider identity declared by the binding.
        cause_type: Exception class name without exception text.
        timed_out: Whether the configured cleanup bound expired.
    """

    references: tuple[ModelReference, ...]
    provider: str
    cause_type: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCleanupReport:
    """Immutable aggregate result of one provider cleanup pass."""

    closed: tuple[ModelReference, ...] = ()
    skipped_caller_owned: tuple[ModelReference, ...] = ()
    failures: tuple[ProviderCleanupFailure, ...] = ()


@dataclass(slots=True)
class _ProviderClientRecord:
    """Identity-based lifecycle state for one unique client."""

    client: ModelProvider
    ownership: ProviderOwnership
    provider: str
    references: set[ModelReference] = field(default_factory=set)
    retired_references: set[ModelReference] = field(default_factory=set)
    leases: int = 0
    close_started: bool = False
    closed: bool = False
    close_failure: ProviderCleanupFailure | None = None
    close_completion: Future[None] = field(default_factory=Future)

    def identities(self) -> tuple[ModelReference, ...]:
        """Return every known reference in deterministic order."""
        return tuple(sorted(self.references | self.retired_references, key=lambda item: item.value))


class ProviderLease:
    """One accepted runtime-mediated use of a provider binding.

    Agents and run contexts borrow bindings; they never receive a lease or
    own a client. Releasing a lease more than once has no additional effect.
    """

    def __init__(self, registry: _ProviderLifecycle, client_id: int) -> None:
        self._registry = registry
        self._client_id = client_id
        self._released = False

    async def release(self) -> None:
        """Release this accepted-use lease and retire eligible clients."""
        if self._released:
            return
        self._released = True
        await self._registry.release_lease(self._client_id)
