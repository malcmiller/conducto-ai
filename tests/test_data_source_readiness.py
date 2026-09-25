"""Tests for the declarative readiness check policy and its bounded cache."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import pytest

from conducto.a2a.errors import A2AStartupError
from conducto.a2a.hardening import A2AHostSecurityConfig
from conducto.a2a.lifecycle import A2AHostLifecycle
from conducto.core.retrieval import RetrievalQuery
from conducto.core.run_context import CancellationState, RunContext, use_run_context
from conducto.resources import (
    ContentBatch,
    ContentItem,
    DataSourceLifecycle,
    DataSourceNotReadyError,
    LifecycleBudget,
    ProvisionedBinding,
    ProvisioningConfig,
    ReadinessCheck,
    ReadinessCheckedRetriever,
    ReadinessPolicy,
    ReadinessProbeError,
    ReadinessVerdict,
    resolve_readiness_policy,
)
from conducto.resources.adapters import InMemoryDataSourceBackend
from conducto.testing import ManualClock
from conducto.transport.tasks import InMemoryTaskRepository

_DATA_SOURCE = "policy_corpus"
_CONFIG = ProvisioningConfig(
    data_source=_DATA_SOURCE,
    backend_kind="in_memory",
    parameters={"dimension": 3},
)


def _batch(text: str = "Leave policy grants twenty days.") -> ContentBatch:
    """Return a single-item batch with deterministic identity."""
    return ContentBatch((ContentItem(content_id="doc-1", text=text),))


async def _populate(backend: InMemoryDataSourceBackend) -> ProvisionedBinding:
    """Provision, ingest, and index the corpus used by readiness tests."""
    binding = await backend.provision(_CONFIG)
    await backend.ingest(binding, _batch())
    await backend.index(binding)
    return binding


def test_readiness_checks_combine_and_default_to_on_start() -> None:
    combined = ReadinessPolicy(checks=ReadinessCheck.ON_START | ReadinessCheck.ON_INVOKE)
    assert combined.includes(ReadinessCheck.ON_START)
    assert combined.includes(ReadinessCheck.ON_INVOKE)
    assert combined.to_dict()["checks"] == ["ON_INVOKE", "ON_START"]

    default = resolve_readiness_policy(None)
    assert default.checks is ReadinessCheck.ON_START
    assert not default.includes(ReadinessCheck.ON_INVOKE)
    assert default.ttl_seconds == 0.0
    assert ReadinessCheck.NONE not in (ReadinessCheck.ON_START, ReadinessCheck.ON_INVOKE)

    with pytest.raises(ValueError, match="non-negative"):
        ReadinessPolicy(readiness_ttl=timedelta(seconds=-1))


def test_both_check_moments_run_when_the_policy_combines_them() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(checks=ReadinessCheck.ON_START | ReadinessCheck.ON_INVOKE),
    )

    async def exercise() -> None:
        await _populate(backend)
        start = await lifecycle.verify_on_start(_DATA_SOURCE)
        invoke = await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert start is not None and start.ready
        assert invoke is not None and invoke.ready
        assert backend.probe_count == 2

    asyncio.run(exercise())


def test_readiness_check_none_must_be_selected_explicitly_and_never_probes() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(backend=backend, policy=ReadinessPolicy(ReadinessCheck.NONE))

    async def exercise() -> None:
        assert await lifecycle.verify_on_start(_DATA_SOURCE) is None
        assert await lifecycle.verify_on_invoke(_DATA_SOURCE) is None
        assert backend.probe_count == 0

    asyncio.run(exercise())


def test_an_on_start_failure_fails_host_startup_and_holds_it_not_ready() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(backend=backend)
    host = A2AHostLifecycle(
        config=A2AHostSecurityConfig(),
        task_repository=InMemoryTaskRepository(),
        on_startup=lifecycle.startup_check([_DATA_SOURCE]),
    )

    async def exercise() -> None:
        with pytest.raises(A2AStartupError):
            await host.startup()
        assert not host.is_ready()

        await _populate(backend)
        ready_host = A2AHostLifecycle(
            config=A2AHostSecurityConfig(),
            task_repository=InMemoryTaskRepository(),
            on_startup=lifecycle.startup_check([_DATA_SOURCE]),
        )
        await ready_host.startup()
        assert ready_host.is_ready()

    asyncio.run(exercise())


def test_an_on_invoke_failure_refuses_the_invocation_instead_of_returning_documents() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend, policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE)
    )
    guarded = ReadinessCheckedRetriever(
        retriever=backend.retriever(_DATA_SOURCE),
        lifecycle=lifecycle,
        data_source=_DATA_SOURCE,
    )

    async def exercise() -> None:
        with pytest.raises(DataSourceNotReadyError) as absent:
            await guarded.retrieve(RetrievalQuery(query="leave"))
        assert absent.value.reason == "data_source_absent"

        binding = await _populate(backend)
        served = await guarded.retrieve(RetrievalQuery(query="leave"))
        assert [document.text for document in served.documents] == [
            "Leave policy grants twenty days."
        ]

        await backend.retire(binding)
        with pytest.raises(DataSourceNotReadyError) as retired:
            await guarded.retrieve(RetrievalQuery(query="leave"))
        assert retired.value.reason == "data_source_retired"

    asyncio.run(exercise())


def test_an_affirmative_verdict_is_reused_within_the_ttl_and_reprobed_after_expiry() -> None:
    backend = InMemoryDataSourceBackend()
    clock = ManualClock()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(
            checks=ReadinessCheck.ON_INVOKE, readiness_ttl=timedelta(seconds=30)
        ),
        clock=clock,
    )

    async def exercise() -> None:
        await _populate(backend)
        first = await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert first is not None and not first.from_cache
        assert backend.probe_count == 1

        await lifecycle.verify_on_invoke(_DATA_SOURCE)
        clock.advance(29.0)
        cached = await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 1
        assert cached is not None
        assert cached.from_cache
        assert cached.evaluated_at == first.evaluated_at

        clock.advance(1.0)
        refreshed = await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 2
        assert refreshed is not None and not refreshed.from_cache

    asyncio.run(exercise())


def test_a_zero_ttl_probes_on_every_invocation() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE, readiness_ttl=timedelta(0)),
    )

    async def exercise() -> None:
        await _populate(backend)
        for _ in range(3):
            await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 3

    asyncio.run(exercise())


def test_a_negative_verdict_is_never_cached() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(
            checks=ReadinessCheck.ON_INVOKE, readiness_ttl=timedelta(seconds=300)
        ),
    )

    async def exercise() -> None:
        for _ in range(2):
            with pytest.raises(DataSourceNotReadyError):
                await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 2

        await _populate(backend)
        verdict = await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert verdict is not None and verdict.ready
        assert backend.probe_count == 3

    asyncio.run(exercise())


def test_reprovisioning_or_reingestion_invalidates_a_cached_verdict() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(
            checks=ReadinessCheck.ON_INVOKE, readiness_ttl=timedelta(seconds=300)
        ),
    )

    async def exercise() -> None:
        binding = await lifecycle.provision(_CONFIG)
        await lifecycle.ingest(binding, _batch())
        await lifecycle.index(binding)
        await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 1

        await lifecycle.ingest(binding, _batch("Leave policy grants thirty days."))
        with pytest.raises(DataSourceNotReadyError):
            await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 2

        await lifecycle.index(binding)
        await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert backend.probe_count == 3

        replacement = await lifecycle.provision(
            ProvisioningConfig(
                data_source=_DATA_SOURCE,
                backend_kind="in_memory",
                parameters={"dimension": 16},
            )
        )
        assert replacement.revision == 2
        with pytest.raises(DataSourceNotReadyError) as stale:
            await lifecycle.verify_on_invoke(_DATA_SOURCE)
        assert stale.value.reason == "index_incomplete"
        assert backend.probe_count == 4

    asyncio.run(exercise())


def test_a_probe_that_exceeds_its_budget_or_is_cancelled_is_a_readiness_failure() -> None:
    backend = InMemoryDataSourceBackend()
    clock = ManualClock()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE),
    )

    async def exercise() -> None:
        await _populate(backend)
        expired = LifecycleBudget(timeout_seconds=2.0, clock=clock)
        clock.advance(2.0)
        with pytest.raises(ReadinessProbeError) as timed_out:
            await lifecycle.verify_on_invoke(_DATA_SOURCE, budget=expired)
        assert timed_out.value.reason == "readiness_timeout"

        cancellation = CancellationState()
        cancellation.cancel()
        with pytest.raises(ReadinessProbeError) as cancelled:
            await lifecycle.verify_on_invoke(
                _DATA_SOURCE, budget=LifecycleBudget(cancellation=cancellation)
            )
        assert cancelled.value.reason == "readiness_cancelled"

    asyncio.run(exercise())


def test_an_unexpected_backend_failure_is_mapped_to_a_redacted_probe_error() -> None:
    class LeakyBackend(InMemoryDataSourceBackend):
        """Backend whose probe raises a raw client error carrying secrets."""

        async def probe(
            self,
            data_source: str,
            *,
            budget: LifecycleBudget | None = None,
        ) -> ReadinessVerdict:
            """Raise an untyped client failure the way a real SDK would."""
            raise ConnectionError(
                "connect https://search.internal.example/indexes?api-key=SECRET failed"
            )

    backend = LeakyBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend, policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE)
    )

    async def exercise() -> None:
        with pytest.raises(ReadinessProbeError) as failure:
            await lifecycle.verify_on_invoke(_DATA_SOURCE)

        assert failure.value.reason == "readiness_probe_failed"
        assert failure.value.data_source == _DATA_SOURCE
        rendered = str(failure.value)
        payload = repr(failure.value.to_dict())
        for secret in ("https://", "api-key", "SECRET", "search.internal"):
            assert secret not in rendered
            assert secret not in payload
        assert isinstance(failure.value.__cause__, ConnectionError)

    asyncio.run(exercise())


def test_the_readiness_gated_retriever_honours_the_active_run_deadline() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend, policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE)
    )
    guarded = ReadinessCheckedRetriever(
        retriever=backend.retriever(_DATA_SOURCE),
        lifecycle=lifecycle,
        data_source=_DATA_SOURCE,
    )

    def context(*, deadline: float | None, cancellation: CancellationState) -> RunContext:
        return RunContext(
            run_id="run-1",
            correlation_id="corr-1",
            deadline=deadline,
            cancellation=cancellation,
        )

    async def exercise() -> None:
        await _populate(backend)

        expired = context(
            deadline=time.monotonic() - 1.0,
            cancellation=CancellationState(),
        )
        with use_run_context(expired):
            with pytest.raises(ReadinessProbeError) as timed_out:
                await guarded.retrieve(RetrievalQuery(query="leave"))
        assert timed_out.value.reason == "readiness_timeout"
        assert backend.probe_count == 0

        cancellation = CancellationState()
        cancellation.cancel()
        with use_run_context(context(deadline=None, cancellation=cancellation)):
            with pytest.raises(ReadinessProbeError) as cancelled:
                await guarded.retrieve(RetrievalQuery(query="leave"))
        assert cancelled.value.reason == "readiness_cancelled"
        assert backend.probe_count == 0

        live = context(deadline=time.monotonic() + 30.0, cancellation=CancellationState())
        with use_run_context(live):
            served = await guarded.retrieve(RetrievalQuery(query="leave"))
        assert len(served.documents) == 1
        assert backend.probe_count == 1

    asyncio.run(exercise())


def test_readiness_diagnostics_disclose_no_endpoint_or_credential_detail() -> None:
    backend = InMemoryDataSourceBackend()
    lifecycle = DataSourceLifecycle(
        backend=backend,
        policy=ReadinessPolicy(checks=ReadinessCheck.ON_INVOKE),
    )

    async def exercise() -> None:
        await backend.provision(
            ProvisioningConfig(
                data_source=_DATA_SOURCE,
                backend_kind="in_memory",
                parameters={"index_name": "policy-index"},
            )
        )
        with pytest.raises(DataSourceNotReadyError) as failure:
            await lifecycle.verify_on_invoke(_DATA_SOURCE)

        rendered = str(failure.value)
        assert failure.value.to_dict() == {
            "error": "DataSourceNotReadyError",
            "data_source": _DATA_SOURCE,
            "reason": "index_incomplete",
        }
        for secret in ("https://", "http://", "policy-index", "password", "token"):
            assert secret not in rendered

    asyncio.run(exercise())
