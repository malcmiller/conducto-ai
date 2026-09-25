"""In-memory reference implementation of the data-source lifecycle.

This adapter demonstrates the complete provisioning, ingestion, indexing,
readiness, and retirement contract without network access, credentials, or a
backend SDK. It is the reference used by conformance fixtures and by examples
that must stay deterministic.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from conducto.core.retrieval import RetrievalQuery, RetrievalResult, RetrievedDocument

from .._identity import digest_text
from ..errors import (
    DataSourceNotReadyError,
    IngestionError,
    PartialIngestionError,
    RetirementError,
)
from ..ingestion import ContentBatch, ContentItem, IngestionProgress
from ..provisioning import (
    DataSourceDescription,
    DataSourceState,
    IndexingState,
    LifecycleBudget,
    ProvisionedBinding,
    ProvisioningConfig,
    resolve_budget,
)
from ..readiness import ReadinessVerdict

__all__ = ["InMemoryDataSourceBackend", "InMemoryRetriever"]


@dataclass(slots=True)
class _SourceState:
    """Mutable in-memory state for one provisioned data source."""

    binding: ProvisionedBinding
    state: DataSourceState = DataSourceState.PROVISIONED
    indexing: IndexingState = IndexingState.EMPTY
    accepted: dict[str, ContentItem] = field(default_factory=dict)
    indexed: dict[str, ContentItem] = field(default_factory=dict)
    applied_digests: set[str] = field(default_factory=set)

    @property
    def content_digest(self) -> str | None:
        """Return the deterministic digest of applied content identities."""
        if not self.accepted:
            return None
        return digest_text(
            "content",
            [[item.content_id, item.body_digest] for item in sorted_items(self.accepted)],
        )


def sorted_items(items: dict[str, ContentItem]) -> tuple[ContentItem, ...]:
    """Return content items ordered deterministically by identity.

    Args:
        items: Content keyed by identity.

    Returns:
        The items sorted by ``content_id``.
    """
    return tuple(items[key] for key in sorted(items))


class InMemoryDataSourceBackend:
    """Deterministic in-process data-source backend.

    Args:
        reject_content_ids: Identities the backend refuses during ingestion, used
            to exercise partial-ingestion reporting.
        fail_index_for: Data-source names whose indexing step fails.
        fail_retire_for: Data-source names whose retirement fails.

    Notes:
        The adapter performs no I/O and holds no credentials. Rejection sets are
        test affordances only; they never silence a failure, because every
        refusal is reported through a typed error.
    """

    __slots__ = (
        "_clock",
        "_fail_index_for",
        "_fail_retire_for",
        "_lock",
        "_probe_count",
        "_reject_content_ids",
        "_revisions",
        "_sources",
    )

    def __init__(
        self,
        *,
        reject_content_ids: Iterable[str] = (),
        fail_index_for: Iterable[str] = (),
        fail_retire_for: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._sources: dict[str, _SourceState] = {}
        self._revisions: dict[str, int] = {}
        self._reject_content_ids = frozenset(reject_content_ids)
        self._fail_index_for = frozenset(fail_index_for)
        self._fail_retire_for = frozenset(fail_retire_for)
        self._probe_count = 0
        self._lock = asyncio.Lock()

    @property
    def probe_count(self) -> int:
        """Return how many readiness probes this backend has evaluated."""
        return self._probe_count

    async def provision(
        self,
        config: ProvisioningConfig,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ProvisionedBinding:
        """Create the data source, or return the existing identical binding.

        Args:
            config: Backend-neutral provisioning configuration.
            budget: Optional deadline and cancellation bound.

        Returns:
            An opaque binding handle whose revision only advances when the
            configuration fingerprint changes.

        Raises:
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=config.data_source, operation="provision")
        async with self._lock:
            existing = self._sources.get(config.data_source)
            if (
                existing is not None
                and existing.state is DataSourceState.PROVISIONED
                and existing.binding.fingerprint == config.fingerprint
            ):
                return existing.binding
            revision = self._revisions.get(config.data_source, 0) + 1
            self._revisions[config.data_source] = revision
            binding = ProvisionedBinding(
                data_source=config.data_source,
                binding_id=digest_text("bind", config.fingerprint, revision),
                revision=revision,
                fingerprint=config.fingerprint,
            )
            self._sources[config.data_source] = _SourceState(binding=binding)
            return binding

    async def describe(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> DataSourceDescription:
        """Report the observable state of one data source.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            Its existence, indexing state, and document counts. An unknown name
            is reported as ``ABSENT`` rather than raising.

        Raises:
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=data_source, operation="describe")
        async with self._lock:
            state = self._sources.get(data_source)
            if state is None:
                return DataSourceDescription(data_source=data_source, state=DataSourceState.ABSENT)
            return DataSourceDescription(
                data_source=data_source,
                state=state.state,
                indexing=state.indexing,
                document_count=len(state.indexed),
                pending_count=len(state.accepted) - len(state.indexed),
                content_digest=state.content_digest,
                binding=(state.binding if state.state is DataSourceState.PROVISIONED else None),
            )

    async def ingest(
        self,
        binding: ProvisionedBinding,
        batch: ContentBatch,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Accept one content batch, reporting a partial apply as a failure.

        Args:
            binding: Opaque handle returned by :meth:`provision`.
            batch: Content batch with deterministic identity.
            budget: Optional deadline and cancellation bound.

        Returns:
            Progress describing accepted, skipped, and pending content.

        Raises:
            IngestionError: If the binding is stale or the whole batch was
                refused.
            PartialIngestionError: If only part of the batch was applied.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=binding.data_source, operation="ingest")
        async with self._lock:
            state = self._require_live(binding, operation="ingest")
            if batch.digest in state.applied_digests:
                return self._progress(
                    state,
                    submitted_count=len(batch),
                    skipped_count=len(batch),
                )
            accepted = 0
            skipped = 0
            refused: list[str] = []
            for item in batch.items:
                resolved.check(data_source=binding.data_source, operation="ingest")
                if item.content_id in self._reject_content_ids:
                    refused.append(item.content_id)
                    continue
                current = state.accepted.get(item.content_id)
                if current is not None and current.body_digest == item.body_digest:
                    skipped += 1
                    continue
                state.accepted[item.content_id] = item
                state.indexed.pop(item.content_id, None)
                accepted += 1
            if accepted or skipped:
                state.indexing = (
                    IndexingState.INDEXING
                    if len(state.indexed) < len(state.accepted)
                    else state.indexing
                )
            if not refused:
                state.applied_digests.add(batch.digest)
            progress = self._progress(
                state,
                submitted_count=len(batch),
                accepted_count=accepted,
                skipped_count=skipped,
                failed_content_ids=tuple(refused),
                state_override=IndexingState.PARTIAL if refused else None,
            )
            if refused and (accepted or skipped):
                state.indexing = IndexingState.PARTIAL
                raise PartialIngestionError(
                    "Content batch was only partially ingested",
                    data_source=binding.data_source,
                    reason="partial_ingestion",
                    progress=progress,
                )
            if refused:
                state.indexing = IndexingState.FAILED
                raise IngestionError(
                    "Content batch was refused",
                    data_source=binding.data_source,
                    reason="ingest_rejected",
                )
            return progress

    async def index(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Index accepted content so the data source becomes queryable.

        Args:
            binding: Opaque handle returned by :meth:`provision`.
            budget: Optional deadline and cancellation bound.

        Returns:
            Progress whose state is ``INDEXED`` when every accepted item is
            queryable.

        Raises:
            IngestionError: If the binding is stale, no content was accepted, or
                indexing was configured to fail for this source.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=binding.data_source, operation="index")
        async with self._lock:
            state = self._require_live(binding, operation="index")
            if binding.data_source in self._fail_index_for:
                state.indexing = IndexingState.FAILED
                raise IngestionError(
                    "Indexing did not complete",
                    data_source=binding.data_source,
                    reason="index_failed",
                )
            if not state.accepted:
                state.indexing = IndexingState.EMPTY
                raise IngestionError(
                    "No content has been accepted for indexing",
                    data_source=binding.data_source,
                    reason="index_empty",
                )
            state.indexed = dict(state.accepted)
            state.indexing = IndexingState.INDEXED
            return self._progress(state, submitted_count=len(state.accepted))

    async def retire(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> DataSourceDescription:
        """Retire the data source and release its in-memory content.

        Args:
            binding: Opaque handle returned by :meth:`provision`.
            budget: Optional deadline and cancellation bound.

        Returns:
            The retired state of the source.

        Raises:
            RetirementError: If the binding is stale, the source is absent, or
                retirement was configured to fail for this source.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=binding.data_source, operation="retire")
        async with self._lock:
            state = self._sources.get(binding.data_source)
            if state is None or state.state is not DataSourceState.PROVISIONED:
                raise RetirementError(
                    "Data source is not provisioned",
                    data_source=binding.data_source,
                    reason="not_provisioned",
                )
            if state.binding != binding:
                raise RetirementError(
                    "Binding does not match the provisioned data source",
                    data_source=binding.data_source,
                    reason="stale_binding",
                )
            if binding.data_source in self._fail_retire_for:
                raise RetirementError(
                    "Data source retirement did not complete",
                    data_source=binding.data_source,
                    reason="retire_failed",
                )
            state.state = DataSourceState.RETIRED
            state.indexing = IndexingState.EMPTY
            state.accepted.clear()
            state.indexed.clear()
            state.applied_digests.clear()
            return DataSourceDescription(
                data_source=binding.data_source, state=DataSourceState.RETIRED
            )

    async def probe(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict:
        """Evaluate whether one data source is provisioned and queryable.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            An affirmative verdict only when the source is provisioned, indexed,
            and non-empty. Every other outcome is a negative verdict with a
            stable reason code.

        Raises:
            LifecycleTimeoutError: If the probe exceeded its budget.
            LifecycleCancelledError: If cancellation was requested.
        """
        resolved = resolve_budget(budget)
        resolved.check(data_source=data_source, operation="readiness_probe")
        self._probe_count += 1
        description = await self.describe(data_source, budget=resolved)
        evaluated_at = self._clock()
        if description.is_queryable:
            return ReadinessVerdict(
                data_source=data_source,
                ready=True,
                reason="indexed",
                evaluated_at=evaluated_at,
                revision=None if description.binding is None else description.binding.revision,
                document_count=description.document_count,
            )
        return ReadinessVerdict(
            data_source=data_source,
            ready=False,
            reason=_not_ready_reason(description),
            evaluated_at=evaluated_at,
            revision=None if description.binding is None else description.binding.revision,
            document_count=description.document_count,
        )

    def retriever(self, data_source: str) -> InMemoryRetriever:
        """Build a read-only retriever over one data source's indexed corpus.

        Args:
            data_source: Logical name of the declared data source.

        Returns:
            A retriever that reads the indexed corpus and never provisions,
            populates, or retires the source.
        """
        return InMemoryRetriever(backend=self, data_source=data_source)

    def indexed_documents(self, data_source: str) -> tuple[ContentItem, ...]:
        """Return the indexed corpus of one data source in stable order.

        Args:
            data_source: Logical name of the declared data source.

        Returns:
            Indexed items ordered by ``content_id``, or an empty tuple when the
            source is absent, retired, or not yet indexed.
        """
        state = self._sources.get(data_source)
        if state is None or state.state is not DataSourceState.PROVISIONED:
            return ()
        return sorted_items(state.indexed)

    def _require_live(self, binding: ProvisionedBinding, *, operation: str) -> _SourceState:
        """Return the live state for a binding or raise a typed ingestion error."""
        state = self._sources.get(binding.data_source)
        if state is None or state.state is not DataSourceState.PROVISIONED:
            raise IngestionError(
                "Data source is not provisioned",
                data_source=binding.data_source,
                reason=f"{operation}_not_provisioned",
            )
        if state.binding != binding:
            raise IngestionError(
                "Binding does not match the provisioned data source",
                data_source=binding.data_source,
                reason=f"{operation}_stale_binding",
            )
        return state

    def _progress(
        self,
        state: _SourceState,
        *,
        submitted_count: int,
        accepted_count: int = 0,
        skipped_count: int = 0,
        failed_content_ids: tuple[str, ...] = (),
        state_override: IndexingState | None = None,
    ) -> IngestionProgress:
        """Build explicit progress from current in-memory state."""
        return IngestionProgress(
            data_source=state.binding.data_source,
            binding_id=state.binding.binding_id,
            revision=state.binding.revision,
            state=state_override or state.indexing,
            submitted_count=submitted_count,
            accepted_count=accepted_count,
            skipped_count=skipped_count,
            indexed_count=len(state.indexed),
            pending_count=len(state.accepted) - len(state.indexed),
            failed_content_ids=failed_content_ids,
            content_digest=state.content_digest,
        )


class InMemoryRetriever:
    """Read-only retriever over one in-memory data source.

    Args:
        backend: Backend owning the indexed corpus.
        data_source: Logical name of the source this retriever reads.

    Notes:
        The retriever never creates, populates, or retires the source. It refuses
        to read an absent or unindexed corpus with a typed failure rather than
        returning an empty page that would be indistinguishable from a genuine
        no-match result.
    """

    __slots__ = ("_backend", "_data_source")

    def __init__(self, *, backend: InMemoryDataSourceBackend, data_source: str) -> None:
        self._backend = backend
        self._data_source = data_source

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Return matching documents from the indexed corpus.

        Args:
            query: Validated backend-neutral retrieval query.

        Returns:
            A deterministic page of matching documents ordered by identity.

        Raises:
            DataSourceNotReadyError: If the source is absent, retired, or not
                indexed.
        """
        description = await self._backend.describe(self._data_source)
        if not description.is_queryable:
            raise DataSourceNotReadyError(
                "Declared data source is not queryable",
                data_source=self._data_source,
                reason=_not_ready_reason(description),
            )
        needle = query.query.casefold()
        matches = [
            item
            for item in self._backend.indexed_documents(self._data_source)
            if needle in item.text.casefold()
        ][: query.limit]
        return RetrievalResult(
            documents=tuple(
                RetrievedDocument(
                    text=item.text,
                    source=self._data_source,
                    citation=item.content_id,
                    metadata=dict(item.metadata),
                )
                for item in matches
            )
        )


def _not_ready_reason(description: DataSourceDescription) -> str:
    """Return the stable reason code for a non-queryable description."""
    if description.state is DataSourceState.ABSENT:
        return "data_source_absent"
    if description.state is DataSourceState.RETIRED:
        return "data_source_retired"
    if description.indexing is IndexingState.PARTIAL:
        return "index_partial"
    if description.indexing is IndexingState.FAILED:
        return "index_failed"
    if description.document_count == 0 and description.indexing is IndexingState.INDEXED:
        return "index_empty"
    return "index_incomplete"
