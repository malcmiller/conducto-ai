"""Backend-neutral provisioning contract for declared data sources.

Provisioning is owned by a deployment, never by an agent. This module defines
the create, describe, and retire contract, the opaque binding handle returned by
provisioning, the observable resource and indexing states, and the deadline and
cooperative cancellation bound applied to every lifecycle operation.

Notes:
    Nothing in this module performs I/O, imports a backend SDK, or resolves a
    credential. Concrete behavior is supplied by adapters that implement
    :class:`DataSourceProvisioner`.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, cast

from conducto.core.run_context import CancellationState

from ._identity import digest_text, freeze_payload, thaw_payload
from .errors import DataSourceLifecycleError, LifecycleCancelledError, LifecycleTimeoutError

__all__ = [
    "DataSourceDescription",
    "DataSourceProvisioner",
    "DataSourceState",
    "IndexingState",
    "LifecycleBudget",
    "ProvisionedBinding",
    "ProvisioningConfig",
    "resolve_budget",
]

T = TypeVar("T")


class DataSourceState(Enum):
    """Observable existence state of one declared data source."""

    ABSENT = "absent"
    PROVISIONED = "provisioned"
    RETIRED = "retired"


class IndexingState(Enum):
    """Observable indexing state of a provisioned data source.

    Indexing is an explicit state rather than an implicit side effect of
    ingestion. Content that has been accepted but not yet indexed reports
    :attr:`INDEXING`, and only :attr:`INDEXED` describes a queryable corpus.
    """

    EMPTY = "empty"
    INDEXING = "indexing"
    INDEXED = "indexed"
    PARTIAL = "partial"
    FAILED = "failed"


class LifecycleBudget:
    """Deadline and cooperative cancellation bound for lifecycle operations.

    Args:
        timeout_seconds: Optional positive, finite wall budget for the whole
            operation. ``None`` means unbounded.
        cancellation: Optional cooperative cancellation state shared with a
            caller or run context.
        clock: Monotonic timestamp source used to compute the deadline.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive and finite.

    Notes:
        A budget is consumed from construction time, so passing one budget to a
        sequence of operations bounds the sequence rather than each step.
    """

    __slots__ = ("_cancellation", "_clock", "_deadline", "_timeout_seconds")

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        cancellation: CancellationState | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds is not None:
            if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
                raise ValueError("timeout_seconds must be a number")
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive and finite")
        self._timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self._cancellation = cancellation
        self._clock = clock
        self._deadline = None if timeout_seconds is None else clock() + float(timeout_seconds)

    @property
    def timeout_seconds(self) -> float | None:
        """Return the configured total budget in seconds, if any."""
        return self._timeout_seconds

    @property
    def cancelled(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return self._cancellation is not None and self._cancellation.cancelled

    def remaining_seconds(self) -> float | None:
        """Return the seconds left in this budget.

        Returns:
            The remaining budget, ``0.0`` when it is exhausted, or ``None`` when
            the budget is unbounded.
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._clock())

    def check(self, *, data_source: str, operation: str) -> None:
        """Fail fast when the budget is exhausted or cancellation was requested.

        Args:
            data_source: Logical name used to attribute the failure.
            operation: Stable snake_case operation name, such as ``"ingest"``.

        Raises:
            LifecycleCancelledError: If cooperative cancellation was requested.
            LifecycleTimeoutError: If the deadline has already passed.
        """
        if self.cancelled:
            raise LifecycleCancelledError(
                f"Data-source {operation} was cancelled",
                data_source=data_source,
                reason=f"{operation}_cancelled",
            )
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise LifecycleTimeoutError(
                f"Data-source {operation} exceeded its deadline",
                data_source=data_source,
                reason=f"{operation}_timeout",
            )

    async def run(
        self,
        awaitable: Awaitable[T],
        *,
        data_source: str,
        operation: str,
    ) -> T:
        """Await one lifecycle step inside this budget.

        Args:
            awaitable: Backend work for a single lifecycle step.
            data_source: Logical name used to attribute a failure.
            operation: Stable snake_case operation name.

        Returns:
            The awaited result when the step completes inside the budget.

        Raises:
            LifecycleCancelledError: If cancellation was requested before the
                step started or while it was running.
            LifecycleTimeoutError: If the step exceeded the remaining budget.
        """
        try:
            self.check(data_source=data_source, operation=operation)
        except DataSourceLifecycleError:
            if isinstance(awaitable, Coroutine):
                awaitable.close()
            raise
        remaining = self.remaining_seconds()
        try:
            result = await asyncio.wait_for(awaitable, timeout=remaining)
        except TimeoutError as error:
            raise LifecycleTimeoutError(
                f"Data-source {operation} exceeded its deadline",
                data_source=data_source,
                reason=f"{operation}_timeout",
            ) from error
        self.check(data_source=data_source, operation=operation)
        return result


def resolve_budget(budget: LifecycleBudget | None) -> LifecycleBudget:
    """Return the supplied budget or an unbounded default.

    Args:
        budget: Caller-supplied budget, or ``None``.

    Returns:
        ``budget`` when provided, otherwise an unbounded budget with no
        cancellation state.
    """
    return budget if budget is not None else LifecycleBudget()


@dataclass(frozen=True, slots=True)
class ProvisioningConfig:
    """Backend-neutral request to create one declared data source.

    Attributes:
        data_source: Logical name of the declared data source.
        backend_kind: Connector category the deployment selected, such as
            ``in_memory`` or ``vector_index``.
        parameters: JSON-safe, non-secret backend parameters. Credentials,
            endpoints, and connection strings are resolved by trusted deployment
            configuration and must never be placed here.
    """

    data_source: str
    backend_kind: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and deeply freeze the configuration."""
        object.__setattr__(self, "data_source", _required_text(self.data_source, "data_source"))
        object.__setattr__(self, "backend_kind", _required_text(self.backend_kind, "backend_kind"))
        object.__setattr__(
            self,
            "parameters",
            cast(Mapping[str, object], freeze_payload(dict(self.parameters), field="parameters")),
        )

    @property
    def fingerprint(self) -> str:
        """Return the deterministic identity of this configuration.

        Returns:
            A stable digest over the data-source name, backend kind, and
            parameters. Two configurations with the same fingerprint describe the
            same resource, which makes re-provisioning idempotent.
        """
        return digest_text("cfg", self.data_source, self.backend_kind, self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe projection of this configuration."""
        return {
            "data_source": self.data_source,
            "backend_kind": self.backend_kind,
            "parameters": thaw_payload(self.parameters),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ProvisionedBinding:
    """Opaque handle to one provisioned data source.

    Attributes:
        data_source: Logical name of the declared data source.
        binding_id: Opaque backend-neutral identifier. It never encodes an
            endpoint, credential, or connection string.
        revision: Monotonic revision incremented by each distinct provisioning.
        fingerprint: Identity of the configuration that produced this binding.
    """

    data_source: str
    binding_id: str
    revision: int
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate binding identity fields."""
        object.__setattr__(self, "data_source", _required_text(self.data_source, "data_source"))
        binding_id = _required_text(self.binding_id, "binding_id")
        if not binding_id.isascii() or any(
            not (character.isalnum() or character in "._-") for character in binding_id
        ):
            raise ValueError(
                "binding_id must be an opaque identifier containing only letters, digits, "
                "'.', '_' or '-'"
            )
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "fingerprint", _required_text(self.fingerprint, "fingerprint"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be an integer")
        if self.revision < 1:
            raise ValueError("revision must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, credential-free projection of this binding."""
        return {
            "data_source": self.data_source,
            "binding_id": self.binding_id,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class DataSourceDescription:
    """Observable lifecycle state of one declared data source.

    Attributes:
        data_source: Logical name of the declared data source.
        state: Whether the source is absent, provisioned, or retired.
        indexing: Observable indexing state of its content.
        document_count: Number of indexed, queryable documents.
        pending_count: Number of accepted documents awaiting indexing.
        content_digest: Deterministic digest of applied content identities, or
            ``None`` when no content has been applied.
        binding: Opaque binding handle when the source is provisioned.
    """

    data_source: str
    state: DataSourceState
    indexing: IndexingState = IndexingState.EMPTY
    document_count: int = 0
    pending_count: int = 0
    content_digest: str | None = None
    binding: ProvisionedBinding | None = None

    def __post_init__(self) -> None:
        """Validate description invariants."""
        object.__setattr__(self, "data_source", _required_text(self.data_source, "data_source"))
        for name in ("document_count", "pending_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.state is not DataSourceState.PROVISIONED and self.binding is not None:
            raise ValueError("only a provisioned data source may carry a binding")
        if self.state is DataSourceState.PROVISIONED and self.binding is None:
            raise ValueError("a provisioned data source must carry a binding")

    @property
    def is_queryable(self) -> bool:
        """Return whether this source currently holds a queryable corpus."""
        return (
            self.state is DataSourceState.PROVISIONED
            and self.indexing is IndexingState.INDEXED
            and self.document_count > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, credential-free projection of this state."""
        return {
            "data_source": self.data_source,
            "state": self.state.value,
            "indexing": self.indexing.value,
            "document_count": self.document_count,
            "pending_count": self.pending_count,
            "content_digest": self.content_digest,
            "binding": None if self.binding is None else self.binding.to_dict(),
        }


class DataSourceProvisioner(Protocol):
    """Structural contract for deployment-owned data-source provisioning.

    Notes:
        Implementations are backend adapters. They never call a model, never
        execute a capability, and never expose a mutable backend client to a
        caller. Every method honours the supplied :class:`LifecycleBudget`.
    """

    async def provision(
        self,
        config: ProvisioningConfig,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ProvisionedBinding:
        """Create the described data source, or return the existing binding.

        Args:
            config: Backend-neutral provisioning configuration.
            budget: Optional deadline and cancellation bound.

        Returns:
            An opaque binding handle. Provisioning twice with an identical
            configuration fingerprint returns the same binding and revision.

        Raises:
            ProvisioningError: If the source could not be created.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...

    async def describe(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> DataSourceDescription:
        """Report the observable lifecycle state of one data source.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            Its observable state, which is ``ABSENT`` when it was never
            provisioned.

        Raises:
            ProvisioningError: If the state could not be determined.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...

    async def retire(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> DataSourceDescription:
        """Retire a provisioned data source and release its backend resources.

        Args:
            binding: Opaque handle returned by :meth:`provision`.
            budget: Optional deadline and cancellation bound.

        Returns:
            The retired state of the source.

        Raises:
            RetirementError: If retirement failed or the binding is stale. The
                failure is always raised; it is never suppressed.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...


def _required_text(value: str, field_name: str) -> str:
    """Return stripped non-empty text or raise a deterministic error."""
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized
