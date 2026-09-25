"""Typed, redacted failures for the declared data-source lifecycle.

Every public lifecycle failure is attributable to one named data source and to a
stable machine-readable reason code. Public messages and serialized diagnostics
never contain a backend endpoint, credential, connection string, or raw backend
exception text. An originating exception may remain attached as ``__cause__`` for
local tracebacks, but it is never rendered into the public message or the
:meth:`DataSourceLifecycleError.to_dict` projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingestion import IngestionProgress

__all__ = [
    "DataSourceLifecycleError",
    "DataSourceNotReadyError",
    "IngestionError",
    "LifecycleCancelledError",
    "LifecycleTimeoutError",
    "PartialIngestionError",
    "ProvisioningError",
    "ReadinessError",
    "ReadinessProbeError",
    "RetirementError",
]


class DataSourceLifecycleError(RuntimeError):
    """Base class for every typed data-source lifecycle failure.

    Args:
        message: Stable, redacted summary of the failure.
        data_source: Logical name of the data source the failure belongs to.
        reason: Stable snake_case reason code used by operators and tests.

    Attributes:
        data_source: Logical name the failure is attributable to.
        reason: Stable snake_case reason code.

    Raises:
        ValueError: If ``message``, ``data_source``, or ``reason`` is empty, or
            if ``reason`` is not a snake_case code.
    """

    __slots__ = ("data_source", "reason")

    def __init__(self, message: str, *, data_source: str, reason: str) -> None:
        normalized_message = _required_text(message, "message")
        normalized_source = _required_text(data_source, "data_source")
        normalized_reason = _required_text(reason, "reason")
        if not all(part.isalnum() for part in normalized_reason.split("_")):
            raise ValueError("reason must be a snake_case code")
        super().__init__(
            f"{normalized_message} (data_source={normalized_source}, reason={normalized_reason})"
        )
        self.data_source = normalized_source
        self.reason = normalized_reason

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic, credential-free diagnostic projection.

        Returns:
            The failure kind, owning data source, and stable reason code. No
            endpoint, credential, or raw backend exception text is included.
        """
        return {
            "error": type(self).__name__,
            "data_source": self.data_source,
            "reason": self.reason,
        }


class ProvisioningError(DataSourceLifecycleError):
    """Raised when creating or describing a data source fails."""


class IngestionError(DataSourceLifecycleError):
    """Raised when a content batch cannot be accepted or indexed."""


class PartialIngestionError(IngestionError):
    """Raised when a content batch was only partially ingested.

    Partial ingestion is an explicit failure. It is never reported through a
    success-shaped result, because a partially indexed corpus is not usable as a
    complete one.

    Args:
        message: Stable, redacted summary of the partial ingestion.
        data_source: Logical name of the partially populated data source.
        reason: Stable snake_case reason code.
        progress: Explicit ingestion progress describing what was applied.

    Attributes:
        progress: Ingestion progress captured when the batch failed.
    """

    __slots__ = ("progress",)

    def __init__(
        self,
        message: str,
        *,
        data_source: str,
        reason: str,
        progress: IngestionProgress,
    ) -> None:
        super().__init__(message, data_source=data_source, reason=reason)
        self.progress = progress


class RetirementError(DataSourceLifecycleError):
    """Raised when retiring a provisioned data source fails.

    Notes:
        Retirement failures are always surfaced. Cleanup exceptions are never
        suppressed, because a suppressed failure leaves an orphaned backend
        resource with no typed signal.
    """


class ReadinessError(DataSourceLifecycleError):
    """Base class for readiness verification failures."""


class DataSourceNotReadyError(ReadinessError):
    """Raised when a data source is absent, empty, or not queryable."""


class ReadinessProbeError(ReadinessError):
    """Raised when a readiness probe fails, times out, or is cancelled."""


class LifecycleTimeoutError(DataSourceLifecycleError):
    """Raised when a lifecycle operation exceeds its deadline."""


class LifecycleCancelledError(DataSourceLifecycleError):
    """Raised when a lifecycle operation observes cooperative cancellation."""


def _required_text(value: str, field: str) -> str:
    """Return stripped non-empty text or raise a deterministic error."""
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{field} must be a non-empty string")
    return normalized
