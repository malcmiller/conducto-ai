"""Tests for content identity, explicit indexing, and partial ingestion."""

from __future__ import annotations

import asyncio

import pytest

from conducto.core.run_context import CancellationState
from conducto.resources import (
    ContentBatch,
    ContentItem,
    IndexingState,
    IngestionError,
    LifecycleBudget,
    LifecycleCancelledError,
    PartialIngestionError,
    ProvisioningConfig,
)
from conducto.resources.adapters import InMemoryDataSourceBackend

_CONFIG = ProvisioningConfig(
    data_source="policy_corpus",
    backend_kind="in_memory",
    parameters={"dimension": 3},
)


def _batch() -> ContentBatch:
    """Return a deterministic three-item content batch."""
    return ContentBatch(
        (
            ContentItem(content_id="doc-2", text="Second policy document."),
            ContentItem(content_id="doc-1", text="First policy document."),
            ContentItem(content_id="doc-3", text="Third policy document."),
        )
    )


def test_content_identity_is_deterministic_and_order_independent() -> None:
    forward = _batch()
    reversed_batch = ContentBatch(tuple(reversed(forward.items)))

    assert forward.content_ids == ("doc-1", "doc-2", "doc-3")
    assert forward.digest == reversed_batch.digest
    assert ContentItem.from_text("First policy document.").content_id == (
        ContentItem.from_text("First policy document.").content_id
    )
    assert ContentItem.from_text("a").content_id != ContentItem.from_text("b").content_id

    changed = ContentBatch(
        (
            ContentItem(content_id="doc-1", text="First policy document, revised."),
            ContentItem(content_id="doc-2", text="Second policy document."),
            ContentItem(content_id="doc-3", text="Third policy document."),
        )
    )
    assert changed.digest != forward.digest


def test_an_empty_duplicated_or_untyped_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        ContentBatch(())
    with pytest.raises(ValueError, match="ContentItem"):
        ContentBatch([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must not repeat"):
        ContentBatch(
            (
                ContentItem(content_id="doc-1", text="One."),
                ContentItem(content_id="doc-1", text="Two."),
            )
        )


def test_indexing_is_an_explicit_state_rather_than_a_side_effect() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        ingested = await backend.ingest(binding, _batch())
        assert ingested.state is IndexingState.INDEXING
        assert ingested.accepted_count == 3
        assert ingested.indexed_count == 0
        assert ingested.pending_count == 3
        assert not ingested.is_complete
        assert not (await backend.probe("policy_corpus")).ready

        indexed = await backend.index(binding)
        assert indexed.state is IndexingState.INDEXED
        assert indexed.indexed_count == 3
        assert indexed.pending_count == 0
        assert indexed.is_complete
        assert (await backend.probe("policy_corpus")).ready

    asyncio.run(exercise())


def test_re_ingesting_identical_content_is_idempotent() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        first = await backend.ingest(binding, _batch())
        await backend.index(binding)
        repeat = await backend.ingest(binding, _batch())

        assert repeat.accepted_count == 0
        assert repeat.skipped_count == 3
        assert repeat.indexed_count == 3
        assert repeat.content_digest == first.content_digest
        description = await backend.describe("policy_corpus")
        assert description.document_count == 3
        assert description.is_queryable

    asyncio.run(exercise())


def test_partial_ingestion_is_reported_explicitly_and_never_as_success() -> None:
    backend = InMemoryDataSourceBackend(reject_content_ids=("doc-3",))

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        with pytest.raises(PartialIngestionError) as failure:
            await backend.ingest(binding, _batch())

        progress = failure.value.progress
        assert failure.value.reason == "partial_ingestion"
        assert progress.state is IndexingState.PARTIAL
        assert progress.accepted_count == 2
        assert progress.failed_content_ids == ("doc-3",)
        assert progress.is_complete is False

        description = await backend.describe("policy_corpus")
        assert description.indexing is IndexingState.PARTIAL
        assert not description.is_queryable
        verdict = await backend.probe("policy_corpus")
        assert verdict.ready is False
        assert verdict.reason == "index_partial"

    asyncio.run(exercise())


def test_indexing_refuses_to_promote_a_partial_corpus_to_a_complete_one() -> None:
    backend = InMemoryDataSourceBackend(reject_content_ids=("doc-3",))

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        with pytest.raises(PartialIngestionError):
            await backend.ingest(binding, _batch())

        with pytest.raises(IngestionError) as refused:
            await backend.index(binding)
        assert refused.value.reason == "index_incomplete"
        description = await backend.describe("policy_corpus")
        assert description.indexing is IndexingState.PARTIAL
        assert not description.is_queryable
        assert not (await backend.probe("policy_corpus")).ready

    asyncio.run(exercise())


def test_reprovisioning_clears_an_unresolved_partial_corpus() -> None:
    backend = InMemoryDataSourceBackend(reject_content_ids=("doc-3",))

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        with pytest.raises(PartialIngestionError):
            await backend.ingest(binding, _batch())

        replacement = await backend.provision(
            ProvisioningConfig(
                data_source="policy_corpus",
                backend_kind="in_memory",
                parameters={"dimension": 16},
            )
        )
        assert replacement.revision == binding.revision + 1
        description = await backend.describe("policy_corpus")
        assert description.indexing is IndexingState.EMPTY
        with pytest.raises(IngestionError) as empty:
            await backend.index(replacement)
        assert empty.value.reason == "index_empty"

    asyncio.run(exercise())


def test_a_fully_rejected_batch_fails_without_partial_state() -> None:
    backend = InMemoryDataSourceBackend(reject_content_ids=("doc-1", "doc-2", "doc-3"))

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        with pytest.raises(IngestionError) as failure:
            await backend.ingest(binding, _batch())
        assert failure.value.reason == "ingest_rejected"
        assert not isinstance(failure.value, PartialIngestionError)
        description = await backend.describe("policy_corpus")
        assert description.document_count == 0
        assert not description.is_queryable

    asyncio.run(exercise())


def test_indexing_failures_and_empty_corpora_are_typed_failures() -> None:
    failing = InMemoryDataSourceBackend(fail_index_for=("policy_corpus",))
    empty = InMemoryDataSourceBackend()

    async def exercise() -> None:
        failing_binding = await failing.provision(_CONFIG)
        await failing.ingest(failing_binding, _batch())
        with pytest.raises(IngestionError) as index_failure:
            await failing.index(failing_binding)
        assert index_failure.value.reason == "index_failed"
        assert (await failing.probe("policy_corpus")).reason == "index_failed"

        empty_binding = await empty.provision(_CONFIG)
        with pytest.raises(IngestionError) as empty_failure:
            await empty.index(empty_binding)
        assert empty_failure.value.reason == "index_empty"

    asyncio.run(exercise())


def test_ingestion_respects_cancellation_and_rejects_stale_bindings() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        cancellation = CancellationState()
        cancellation.cancel()
        with pytest.raises(LifecycleCancelledError) as cancelled:
            await backend.ingest(
                binding, _batch(), budget=LifecycleBudget(cancellation=cancellation)
            )
        assert cancelled.value.reason == "ingest_cancelled"

        replacement = await backend.provision(
            ProvisioningConfig(
                data_source="policy_corpus",
                backend_kind="in_memory",
                parameters={"dimension": 8},
            )
        )
        assert replacement.revision == 2
        with pytest.raises(IngestionError) as stale:
            await backend.ingest(binding, _batch())
        assert stale.value.reason == "ingest_stale_binding"

    asyncio.run(exercise())


def test_ingestion_progress_serializes_deterministically() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        binding = await backend.provision(_CONFIG)
        await backend.ingest(binding, _batch())
        progress = await backend.index(binding)
        payload = progress.to_dict()
        assert payload["state"] == "indexed"
        assert payload["is_complete"] is True
        assert payload["failed_content_ids"] == []
        assert payload["binding_id"] == binding.binding_id
        assert payload == progress.to_dict()

    asyncio.run(exercise())
