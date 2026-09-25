"""Ingestion contract for populating a provisioned data source.

Ingestion is expressed as content batches with deterministic content identity,
an explicit indexing step, and explicit progress. A partially applied batch is
reported as a typed failure carrying its progress, never as a success-shaped
result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ._identity import digest_text, freeze_payload, thaw_payload
from .provisioning import IndexingState, LifecycleBudget, ProvisionedBinding

__all__ = [
    "ContentBatch",
    "ContentItem",
    "DataSourceIngestor",
    "IngestionProgress",
]


@dataclass(frozen=True, slots=True)
class ContentItem:
    """One unit of ingestible content with stable identity.

    Attributes:
        content_id: Deterministic identity of this content. Re-ingesting the same
            identity with the same body is a no-op rather than a duplicate.
        text: Content body handed to the backend for indexing.
        metadata: JSON-safe, non-secret metadata such as labels or document keys.
    """

    content_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity and deeply freeze metadata."""
        object.__setattr__(self, "content_id", _required_text(self.content_id, "content_id"))
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, object], freeze_payload(dict(self.metadata), field="metadata")),
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        metadata: Mapping[str, object] | None = None,
        content_id: str | None = None,
    ) -> ContentItem:
        """Build an item, deriving a deterministic identity when none is given.

        Args:
            text: Content body.
            metadata: Optional JSON-safe, non-secret metadata.
            content_id: Optional caller-owned stable identity, such as a
                business key. A digest over the body and metadata is derived when
                it is omitted.

        Returns:
            A frozen content item whose identity is reproducible across runs.

        Raises:
            ValueError: If ``text`` is empty or the metadata is not JSON-safe.
        """
        frozen_metadata = freeze_payload(dict(metadata or {}), field="metadata")
        derived = content_id or digest_text("doc", _required_text(text, "text"), frozen_metadata)
        return cls(
            content_id=derived, text=text, metadata=cast(Mapping[str, object], frozen_metadata)
        )

    @property
    def body_digest(self) -> str:
        """Return the deterministic digest of this item's body and metadata."""
        return digest_text("body", self.text, self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe projection of this item."""
        return {
            "content_id": self.content_id,
            "text": self.text,
            "metadata": thaw_payload(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ContentBatch:
    """An ordered, de-duplicated batch of content with a stable digest.

    Attributes:
        items: Batch content ordered deterministically by ``content_id``.
    """

    items: tuple[ContentItem, ...]

    def __init__(self, items: Iterable[ContentItem]) -> None:
        """Order and validate batch content.

        Args:
            items: Content items to apply as one batch.

        Raises:
            ValueError: If the batch is empty, contains a non-item value, or
                repeats a ``content_id``.
        """
        materialized = tuple(items)
        if not materialized:
            raise ValueError("a content batch must contain at least one item")
        if any(not isinstance(item, ContentItem) for item in materialized):
            raise ValueError("a content batch must contain ContentItem values")
        ordered = tuple(sorted(materialized, key=lambda item: item.content_id))
        identities = [item.content_id for item in ordered]
        if len(set(identities)) != len(identities):
            raise ValueError("a content batch must not repeat a content_id")
        object.__setattr__(self, "items", ordered)

    def __len__(self) -> int:
        """Return the number of items in this batch."""
        return len(self.items)

    @property
    def content_ids(self) -> tuple[str, ...]:
        """Return batch identities in deterministic order."""
        return tuple(item.content_id for item in self.items)

    @property
    def digest(self) -> str:
        """Return the deterministic identity of this batch's content.

        Returns:
            A digest over every item's identity and body. Two batches with the
            same digest carry identical content, which makes re-ingestion
            idempotent.
        """
        return digest_text(
            "batch",
            [[item.content_id, item.body_digest] for item in self.items],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe projection of this batch."""
        return {
            "digest": self.digest,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class IngestionProgress:
    """Explicit progress and completion state for applied content.

    Attributes:
        data_source: Logical name of the populated data source.
        binding_id: Opaque binding the content was applied to.
        revision: Provisioning revision the content was applied to.
        state: Observable indexing state after this operation.
        submitted_count: Items presented by the caller.
        accepted_count: Items newly accepted for indexing.
        skipped_count: Items already present with identical content identity.
        indexed_count: Indexed, queryable documents held by the source.
        pending_count: Accepted documents still awaiting indexing.
        failed_content_ids: Identities the backend refused, in stable order.
        content_digest: Deterministic digest of all applied content identities.
    """

    data_source: str
    binding_id: str
    revision: int
    state: IndexingState
    submitted_count: int
    accepted_count: int = 0
    skipped_count: int = 0
    indexed_count: int = 0
    pending_count: int = 0
    failed_content_ids: tuple[str, ...] = ()
    content_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate counters and order failed identities deterministically."""
        object.__setattr__(self, "data_source", _required_text(self.data_source, "data_source"))
        object.__setattr__(self, "binding_id", _required_text(self.binding_id, "binding_id"))
        for name in (
            "submitted_count",
            "accepted_count",
            "skipped_count",
            "indexed_count",
            "pending_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "failed_content_ids", tuple(sorted(self.failed_content_ids)))

    @property
    def is_complete(self) -> bool:
        """Return whether every submitted item is indexed and queryable."""
        return (
            self.state is IndexingState.INDEXED
            and not self.failed_content_ids
            and self.pending_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe projection of this progress."""
        return {
            "data_source": self.data_source,
            "binding_id": self.binding_id,
            "revision": self.revision,
            "state": self.state.value,
            "submitted_count": self.submitted_count,
            "accepted_count": self.accepted_count,
            "skipped_count": self.skipped_count,
            "indexed_count": self.indexed_count,
            "pending_count": self.pending_count,
            "failed_content_ids": list(self.failed_content_ids),
            "content_digest": self.content_digest,
            "is_complete": self.is_complete,
        }


class DataSourceIngestor(Protocol):
    """Structural contract for deployment-owned ingestion and indexing.

    Notes:
        Ingestion accepts content; indexing makes it queryable. The two steps are
        separate so a partially indexed corpus is never mistaken for a complete
        one.
    """

    async def ingest(
        self,
        binding: ProvisionedBinding,
        batch: ContentBatch,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Accept one content batch for a provisioned data source.

        Args:
            binding: Opaque handle returned by provisioning.
            batch: Content batch with deterministic identity.
            budget: Optional deadline and cancellation bound.

        Returns:
            Progress describing accepted, skipped, and pending content. Applying
            a batch whose digest was already applied is idempotent and reports
            the items as skipped.

        Raises:
            IngestionError: If the batch could not be accepted or the binding is
                stale.
            PartialIngestionError: If only part of the batch was applied. The
                failure carries the explicit progress of the partial apply.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...

    async def index(
        self,
        binding: ProvisionedBinding,
        *,
        budget: LifecycleBudget | None = None,
    ) -> IngestionProgress:
        """Index accepted content so the data source becomes queryable.

        Args:
            binding: Opaque handle returned by provisioning.
            budget: Optional deadline and cancellation bound.

        Returns:
            Progress whose state is ``INDEXED`` once every accepted item is
            queryable.

        Raises:
            IngestionError: If indexing failed or the binding is stale.
            PartialIngestionError: If only part of the accepted content could be
                indexed.
            LifecycleTimeoutError: If the deadline was exceeded.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...


def _required_text(value: str, field_name: str) -> str:
    """Return stripped non-empty text or raise a deterministic error."""
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized
