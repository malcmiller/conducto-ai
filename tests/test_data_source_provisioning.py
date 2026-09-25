"""Tests for deployment-owned data-source provisioning and retirement."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from conducto.core.run_context import CancellationState
from conducto.resources import (
    DataSourceLifecycle,
    DataSourceState,
    IndexingState,
    LifecycleBudget,
    LifecycleCancelledError,
    LifecycleTimeoutError,
    ProvisioningConfig,
    RetirementError,
)
from conducto.resources.adapters import InMemoryDataSourceBackend
from conducto.testing import (
    ManualClock,
    assert_data_source_lifecycle_conformance,
    conformance_content_batch,
    conformance_provisioning_config,
)


def _config(data_source: str = "policy_corpus", **parameters: object) -> ProvisioningConfig:
    """Return a provisioning configuration for one declared data source."""
    return ProvisioningConfig(
        data_source=data_source,
        backend_kind="in_memory",
        parameters=parameters or {"dimension": 3},
    )


def test_a_data_source_is_provisioned_populated_and_retired_without_an_agent() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(backend=backend)

    async def exercise() -> None:
        binding = await lifecycle.provision(_config())
        await lifecycle.ingest(binding, conformance_content_batch())
        populated = await lifecycle.describe("policy_corpus")
        assert populated.indexing is IndexingState.INDEXING
        assert populated.pending_count == 3
        assert not populated.is_queryable

        indexed = await lifecycle.index(binding)
        assert indexed.is_complete
        ready = await lifecycle.describe("policy_corpus")
        assert ready.is_queryable
        assert ready.document_count == 3

        retired = await lifecycle.retire(binding)
        assert retired.state is DataSourceState.RETIRED
        assert not (await lifecycle.describe("policy_corpus")).is_queryable

    asyncio.run(exercise())


def test_provisioning_is_idempotent_for_an_identical_configuration() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        first = await backend.provision(_config())
        second = await backend.provision(_config())
        assert second == first
        assert second.revision == 1

        await backend.ingest(first, conformance_content_batch())
        await backend.index(first)

        changed = await backend.provision(_config(dimension=8))
        assert changed.revision == 2
        assert changed.binding_id != first.binding_id
        reset = await backend.describe("policy_corpus")
        assert reset.indexing is IndexingState.EMPTY
        assert reset.document_count == 0

    asyncio.run(exercise())


def test_a_binding_is_opaque_and_configuration_identity_is_deterministic() -> None:
    config = _config(dimension=3, similarity="cosine")
    other = ProvisioningConfig(
        data_source="policy_corpus",
        backend_kind="in_memory",
        parameters={"similarity": "cosine", "dimension": 3},
    )

    assert config.fingerprint == other.fingerprint
    assert config.fingerprint != _config(dimension=4).fingerprint

    backend = InMemoryDataSourceBackend()
    binding = asyncio.run(backend.provision(config))
    assert binding.binding_id.startswith("bind-")
    assert "://" not in binding.binding_id
    assert set(binding.to_dict()) == {"data_source", "binding_id", "revision", "fingerprint"}


def test_retirement_failures_are_typed_and_never_suppressed() -> None:
    backend = InMemoryDataSourceBackend(fail_retire_for=("policy_corpus",))

    async def exercise() -> None:
        binding = await backend.provision(_config())
        with pytest.raises(RetirementError) as failure:
            await backend.retire(binding)
        assert failure.value.reason == "retire_failed"
        assert failure.value.data_source == "policy_corpus"
        assert failure.value.to_dict() == {
            "error": "RetirementError",
            "data_source": "policy_corpus",
            "reason": "retire_failed",
        }
        still_provisioned = await backend.describe("policy_corpus")
        assert still_provisioned.state is DataSourceState.PROVISIONED

    asyncio.run(exercise())


def test_retiring_an_unknown_or_stale_binding_is_an_explicit_failure() -> None:
    backend = InMemoryDataSourceBackend()

    async def exercise() -> None:
        binding = await backend.provision(_config())
        await backend.retire(binding)
        with pytest.raises(RetirementError) as absent:
            await backend.retire(binding)
        assert absent.value.reason == "not_provisioned"

        current = await backend.provision(_config())
        stale = await backend.provision(_config(dimension=9))
        assert stale.revision > current.revision
        with pytest.raises(RetirementError) as mismatch:
            await backend.retire(current)
        assert mismatch.value.reason == "stale_binding"

    asyncio.run(exercise())


def test_provisioning_respects_deadlines_and_cooperative_cancellation() -> None:
    backend = InMemoryDataSourceBackend()
    clock = ManualClock()

    async def exercise() -> None:
        expired = LifecycleBudget(timeout_seconds=5.0, clock=clock)
        clock.advance(5.0)
        with pytest.raises(LifecycleTimeoutError) as timed_out:
            await backend.provision(_config(), budget=expired)
        assert timed_out.value.reason == "provision_timeout"

        cancellation = CancellationState()
        cancellation.cancel()
        with pytest.raises(LifecycleCancelledError) as cancelled:
            await backend.provision(_config(), budget=LifecycleBudget(cancellation=cancellation))
        assert cancelled.value.reason == "provision_cancelled"

        absent = await backend.describe("policy_corpus")
        assert absent.state is DataSourceState.ABSENT

    asyncio.run(exercise())


def test_an_unbounded_budget_allows_work_and_reports_no_remaining_time() -> None:
    budget = LifecycleBudget()
    assert budget.remaining_seconds() is None
    assert budget.timeout_seconds is None
    assert not budget.cancelled
    budget.check(data_source="policy_corpus", operation="provision")

    with pytest.raises(ValueError, match="positive and finite"):
        LifecycleBudget(timeout_seconds=0)


def test_the_in_memory_reference_backend_passes_lifecycle_conformance() -> None:
    backend = InMemoryDataSourceBackend()
    steps = asyncio.run(assert_data_source_lifecycle_conformance(backend))
    assert [step["step"] for step in steps][:3] == [
        "describe_absent",
        "provision",
        "probe_before_index",
    ]
    assert conformance_provisioning_config().fingerprint == (
        conformance_provisioning_config().fingerprint
    )


def test_lifecycle_contracts_import_without_any_backend_package() -> None:
    """Core lifecycle contracts carry no backend-specific runtime dependency."""
    code = """
import builtins
blocked = {'httpx', 'azure', 'openai', 'ollama', 'starlette', 'fastapi', 'uvicorn', 'numpy'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError('optional backend package unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import asyncio
from conducto.resources import ContentBatch, ContentItem, ProvisioningConfig
from conducto.resources.adapters import InMemoryDataSourceBackend

backend = InMemoryDataSourceBackend()
config = ProvisioningConfig(data_source='corpus', backend_kind='in_memory')
batch = ContentBatch([ContentItem(content_id='doc-1', text='hello')])

async def main():
    binding = await backend.provision(config)
    await backend.ingest(binding, batch)
    progress = await backend.index(binding)
    assert progress.is_complete

asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
