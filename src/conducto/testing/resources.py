"""Deterministic fixtures and conformance checks for the data-source lifecycle.

These helpers exercise a :class:`~conducto.resources.DataSourceBackend` through
the full provisioning, ingestion, indexing, readiness, and retirement contract
without network access, credentials, or wall-clock sleeps. Any adapter that
passes them honours the backend-neutral lifecycle contract.
"""

from __future__ import annotations

from typing import Any

from conducto.resources import (
    ContentBatch,
    ContentItem,
    DataSourceBackend,
    DataSourceState,
    IndexingState,
    IngestionError,
    ProvisioningConfig,
    RetirementError,
)

__all__ = [
    "LIFECYCLE_FIXTURE_VERSION",
    "ManualClock",
    "assert_data_source_lifecycle_conformance",
    "conformance_content_batch",
    "conformance_provisioning_config",
    "run_data_source_lifecycle_conformance",
]

LIFECYCLE_FIXTURE_VERSION = "1"
"""Version of the pinned lifecycle conformance fixtures."""

_CONFORMANCE_DATA_SOURCE = "conformance_corpus"


class ManualClock:
    """Deterministic monotonic clock advanced explicitly by a test.

    Args:
        start: Initial timestamp reported by the clock.

    Notes:
        A manual clock keeps readiness TTL behavior deterministic, so no test
        needs a wall-clock sleep to observe expiry.
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        """Return the current timestamp without advancing it."""
        return self._now

    def advance(self, seconds: float) -> float:
        """Advance the clock.

        Args:
            seconds: Non-negative number of seconds to advance.

        Returns:
            The new current timestamp.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        self._now += float(seconds)
        return self._now


def conformance_content_batch() -> ContentBatch:
    """Return the pinned conformance content batch.

    Returns:
        A three-item batch with explicit, deterministic content identities.
    """
    return ContentBatch(
        (
            ContentItem(content_id="doc-1", text="Conducto governs capability invocation."),
            ContentItem(content_id="doc-2", text="Data sources are declared as metadata."),
            ContentItem(content_id="doc-3", text="Provisioning is owned by a deployment."),
        )
    )


def conformance_provisioning_config(
    *,
    data_source: str = _CONFORMANCE_DATA_SOURCE,
    backend_kind: str = "in_memory",
) -> ProvisioningConfig:
    """Return the pinned conformance provisioning configuration.

    Args:
        data_source: Logical name to provision.
        backend_kind: Connector category selected by the deployment.

    Returns:
        A configuration whose fingerprint is stable across processes.
    """
    return ProvisioningConfig(
        data_source=data_source,
        backend_kind=backend_kind,
        parameters={"dimension": 3, "similarity": "cosine"},
    )


async def run_data_source_lifecycle_conformance(
    backend: DataSourceBackend,
    *,
    data_source: str = _CONFORMANCE_DATA_SOURCE,
) -> tuple[dict[str, Any], ...]:
    """Drive one backend through the full lifecycle and record each step.

    Args:
        backend: Adapter implementing provisioning, ingestion, and readiness.
        data_source: Logical name used for the conformance corpus.

    Returns:
        Deterministic, JSON-safe records for each lifecycle step in execution
        order.

    Raises:
        AssertionError: If a step that must fail unexpectedly succeeded.
        DataSourceLifecycleError: If a step that must succeed failed.
    """
    config = conformance_provisioning_config(data_source=data_source)
    batch = conformance_content_batch()
    steps: list[dict[str, Any]] = []

    absent = await backend.describe(data_source)
    steps.append({"step": "describe_absent", "state": absent.state.value})

    binding = await backend.provision(config)
    repeat = await backend.provision(config)
    steps.append(
        {
            "step": "provision",
            "idempotent": repeat == binding,
            "revision": binding.revision,
            "fingerprint": binding.fingerprint,
        }
    )

    before_index = await backend.probe(data_source)
    steps.append(
        {"step": "probe_before_index", "ready": before_index.ready, "reason": before_index.reason}
    )

    ingested = await backend.ingest(binding, batch)
    steps.append(
        {
            "step": "ingest",
            "state": ingested.state.value,
            "accepted_count": ingested.accepted_count,
            "pending_count": ingested.pending_count,
            "is_complete": ingested.is_complete,
        }
    )

    indexed = await backend.index(binding)
    steps.append(
        {
            "step": "index",
            "state": indexed.state.value,
            "indexed_count": indexed.indexed_count,
            "is_complete": indexed.is_complete,
        }
    )

    repeat_ingest = await backend.ingest(binding, batch)
    steps.append(
        {
            "step": "ingest_again",
            "accepted_count": repeat_ingest.accepted_count,
            "skipped_count": repeat_ingest.skipped_count,
            "content_digest_stable": repeat_ingest.content_digest == indexed.content_digest,
        }
    )

    ready = await backend.probe(data_source)
    steps.append(
        {
            "step": "probe_ready",
            "ready": ready.ready,
            "reason": ready.reason,
            "document_count": ready.document_count,
        }
    )

    retired = await backend.retire(binding)
    steps.append({"step": "retire", "state": retired.state.value})

    try:
        await backend.retire(binding)
    except RetirementError as error:
        steps.append({"step": "retire_again", "error": error.to_dict()["error"]})
    else:  # pragma: no cover - a conforming backend never reaches this branch
        raise AssertionError("retiring an already retired data source must fail")

    after_retire = await backend.probe(data_source)
    steps.append(
        {
            "step": "probe_after_retire",
            "ready": after_retire.ready,
            "reason": after_retire.reason,
        }
    )

    try:
        await backend.ingest(binding, batch)
    except IngestionError as error:
        steps.append({"step": "ingest_after_retire", "error": error.to_dict()["error"]})
    else:  # pragma: no cover - a conforming backend never reaches this branch
        raise AssertionError("ingesting into a retired data source must fail")

    return tuple(steps)


async def assert_data_source_lifecycle_conformance(
    backend: DataSourceBackend,
    *,
    data_source: str = _CONFORMANCE_DATA_SOURCE,
) -> tuple[dict[str, Any], ...]:
    """Assert that a backend honours the pinned lifecycle contract.

    Args:
        backend: Adapter implementing provisioning, ingestion, and readiness.
        data_source: Logical name used for the conformance corpus.

    Returns:
        The recorded conformance steps, for callers that also pin them.

    Raises:
        AssertionError: If any step deviates from the lifecycle contract.
    """
    steps = await run_data_source_lifecycle_conformance(backend, data_source=data_source)
    recorded = {step["step"]: step for step in steps}
    batch_size = len(conformance_content_batch())

    assert recorded["describe_absent"]["state"] == DataSourceState.ABSENT.value
    assert recorded["provision"]["idempotent"] is True
    assert recorded["probe_before_index"]["ready"] is False
    assert recorded["ingest"]["state"] == IndexingState.INDEXING.value
    assert recorded["ingest"]["accepted_count"] == batch_size
    assert recorded["ingest"]["is_complete"] is False
    assert recorded["index"]["state"] == IndexingState.INDEXED.value
    assert recorded["index"]["indexed_count"] == batch_size
    assert recorded["index"]["is_complete"] is True
    assert recorded["ingest_again"]["accepted_count"] == 0
    assert recorded["ingest_again"]["skipped_count"] == batch_size
    assert recorded["ingest_again"]["content_digest_stable"] is True
    assert recorded["probe_ready"]["ready"] is True
    assert recorded["probe_ready"]["document_count"] == batch_size
    assert recorded["retire"]["state"] == DataSourceState.RETIRED.value
    assert recorded["retire_again"]["error"] == "RetirementError"
    assert recorded["probe_after_retire"]["ready"] is False
    assert recorded["ingest_after_retire"]["error"] == "IngestionError"
    return steps
