"""Deployment-owned coordination of the declared data-source lifecycle.

:class:`DataSourceLifecycle` binds provisioning, ingestion, and readiness into
one deployment-owned object. An agent never provisions, populates, or retires a
data source; it reads from one through governed retrieval, and the readiness gate
decides whether that read is allowed to proceed.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from conducto.core.retrieval import RetrievalQuery, RetrievalResult, RetrieverProtocol
from conducto.core.run_context import get_run_context

from .errors import ReadinessProbeError
from .ingestion import ContentBatch, DataSourceIngestor, IngestionProgress
from .provisioning import (
    DataSourceDescription,
    DataSourceProvisioner,
    LifecycleBudget,
    ProvisionedBinding,
    ProvisioningConfig,
)
from .readiness import ReadinessGate, ReadinessPolicy, ReadinessProbe, ReadinessVerdict

__all__ = [
    "DataSourceBackend",
    "DataSourceLifecycle",
    "ReadinessCheckedRetriever",
    "run_context_budget",
]


def run_context_budget(data_source: str) -> LifecycleBudget | None:
    """Derive a lifecycle budget from the active invocation context.

    Args:
        data_source: Logical name used to attribute a deadline failure.

    Returns:
        A budget carrying the run's remaining timeout and cooperative
        cancellation state, or ``None`` when no run context is active.

    Raises:
        ReadinessProbeError: If the active run deadline is already exhausted. A
            probe that cannot start inside its budget is a readiness failure
            rather than an assumed pass.
    """
    context = get_run_context()
    if context is None:
        return None
    try:
        remaining = context.remaining_timeout()
    except TimeoutError as error:
        raise ReadinessProbeError(
            "Readiness probe exceeded its deadline",
            data_source=data_source,
            reason="readiness_timeout",
        ) from error
    return LifecycleBudget(timeout_seconds=remaining, cancellation=context.cancellation)


class DataSourceBackend(DataSourceProvisioner, DataSourceIngestor, ReadinessProbe, Protocol):
    """Structural contract for a backend that owns a data source end to end."""


class DataSourceLifecycle:
    """Provision, populate, verify, and retire declared data sources.

    Args:
        backend: Adapter implementing provisioning, ingestion, and readiness.
        policy: Declarative readiness policy. Omitting it resolves to
            :attr:`~conducto.resources.readiness.ReadinessCheck.ON_START`.
        clock: Monotonic timestamp source used for readiness TTL accounting.

    Notes:
        Every mutation of a source, including a partial or failed ingestion,
        invalidates any cached affirmative readiness verdict for that source, so
        a stale verdict can never outlive the corpus it described.
    """

    __slots__ = ("_backend", "_gate")

    def __init__(
        self,
        *,
        backend: DataSourceBackend,
        policy: ReadinessPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._gate = ReadinessGate(probe=backend, policy=policy, clock=clock)

    @property
    def readiness_policy(self) -> ReadinessPolicy:
        """Return the resolved readiness policy applied by this lifecycle."""
        return self._gate.policy

    async def provision(
        self,
        config: ProvisioningConfig,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ProvisionedBinding:
        """Create the declared data source and invalidate cached readiness.

        Args:
            config: Backend-neutral provisioning configuration.
            budget: Optional deadline and cancellation bound.

        Returns:
            An opaque binding. Provisioning an identical configuration again
            returns the same binding and revision.

        Raises:
            ProvisioningError: If the source could not be created.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        try:
            return await self._backend.provision(config, budget=budget)
        finally:
            self._gate.invalidate(config.data_source)

    async def ingest(
        self,
        binding: ProvisionedBinding,
        batch: ContentBatch,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Apply one content batch and invalidate cached readiness.

        Args:
            binding: Opaque handle returned by provisioning.
            batch: Content batch with deterministic identity.
            budget: Optional deadline and cancellation bound.

        Returns:
            Explicit ingestion progress for the applied batch.

        Raises:
            IngestionError: If the batch could not be accepted.
            PartialIngestionError: If only part of the batch was applied.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        try:
            return await self._backend.ingest(binding, batch, budget=budget)
        finally:
            self._gate.invalidate(binding.data_source)

    async def index(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Index accepted content and invalidate cached readiness.

        Args:
            binding: Opaque handle returned by provisioning.
            budget: Optional deadline and cancellation bound.

        Returns:
            Progress whose state is ``INDEXED`` once the corpus is queryable.

        Raises:
            IngestionError: If indexing failed.
            PartialIngestionError: If only part of the content could be indexed.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        try:
            return await self._backend.index(binding, budget=budget)
        finally:
            self._gate.invalidate(binding.data_source)

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
            Its observable existence, indexing state, and counts.

        Raises:
            ProvisioningError: If the state could not be determined.
        """
        return await self._backend.describe(data_source, budget=budget)

    async def retire(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> DataSourceDescription:
        """Retire a provisioned data source and invalidate cached readiness.

        Args:
            binding: Opaque handle returned by provisioning.
            budget: Optional deadline and cancellation bound.

        Returns:
            The retired state of the source.

        Raises:
            RetirementError: If retirement failed. The failure is raised, never
                suppressed, so an orphaned backend resource is always visible.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        try:
            return await self._backend.retire(binding, budget=budget)
        finally:
            self._gate.invalidate(binding.data_source)

    async def verify_on_start(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict | None:
        """Verify one data source during host startup.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            The affirmative verdict, or ``None`` when the policy does not select
            ``ON_START``.

        Raises:
            DataSourceNotReadyError: If the source is not queryable.
            ReadinessProbeError: If the probe failed, timed out, or was cancelled.
        """
        return await self._gate.verify_on_start(data_source, budget=budget)

    async def verify_on_invoke(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict | None:
        """Verify one data source before serving an invocation.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            The affirmative verdict, possibly reused within ``readiness_ttl``, or
            ``None`` when the policy does not select ``ON_INVOKE``.

        Raises:
            DataSourceNotReadyError: If the source is not queryable. The
                invocation is refused rather than answered over an unavailable
                knowledge base.
            ReadinessProbeError: If the probe failed, timed out, or was cancelled.
        """
        return await self._gate.verify_on_invoke(data_source, budget=budget)

    def startup_check(
        self,
        data_sources: Iterable[str],
        *,
        budget: LifecycleBudget | None = None,
    ) -> Callable[[], Awaitable[None]]:
        """Build a host startup check over the named data sources.

        Args:
            data_sources: Logical names to verify, verified in stable name order.
            budget: Optional deadline and cancellation bound shared by the checks.

        Returns:
            An awaitable callable suitable as a host startup dependency check.
            It raises on the first failing source, which holds the host
            not-ready instead of letting it serve over a missing corpus.
        """
        names = tuple(sorted(set(data_sources)))

        async def check() -> None:
            """Verify every named data source or fail host startup."""
            for name in names:
                await self._gate.verify_on_start(name, budget=budget)

        return check


class ReadinessCheckedRetriever:
    """Applies the invocation-time readiness gate in front of a retriever.

    Args:
        retriever: Read-side retriever defined by the governed retrieval
            contract. It is never asked to provision, populate, or retire the
            source it reads.
        lifecycle: Deployment-owned lifecycle that owns the readiness policy.
        data_source: Logical name of the data source being read.
        budget_factory: Optional supplier of the bound applied to the readiness
            probe. When omitted, the bound is derived from the active
            :class:`~conducto.core.run_context.RunContext` so the probe honours
            the invocation's remaining deadline and cancellation state.

    Notes:
        A refused invocation raises a typed failure. It never degrades to an
        empty page, a partial page, or a success-shaped result carrying a
        degraded marker, because answering over a knowledge base known to be
        unavailable is precisely the failure this contract prevents.
    """

    __slots__ = ("_budget_factory", "_data_source", "_lifecycle", "_retriever")

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        lifecycle: DataSourceLifecycle,
        data_source: str,
        budget_factory: Callable[[], LifecycleBudget | None] | None = None,
    ) -> None:
        self._retriever = retriever
        self._lifecycle = lifecycle
        self._data_source = data_source
        self._budget_factory = budget_factory

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Verify readiness, then delegate to the wrapped retriever.

        Args:
            query: Validated backend-neutral retrieval query.

        Returns:
            The wrapped retriever's result page.

        Raises:
            DataSourceNotReadyError: If the source is not queryable.
            ReadinessProbeError: If the probe failed, timed out, or was
                cancelled, including when the active run deadline is already
                exhausted before the probe starts.
        """
        budget = (
            self._budget_factory()
            if self._budget_factory is not None
            else run_context_budget(self._data_source)
        )
        await self._lifecycle.verify_on_invoke(self._data_source, budget=budget)
        result = self._retriever.retrieve(query)
        if isinstance(result, RetrievalResult):
            return result
        return await result
