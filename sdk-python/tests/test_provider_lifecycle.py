import asyncio
import threading

import pytest

from conducto import (
    FakeModel,
    ModelConfiguration,
    ProviderClientConfig,
    ProviderOwnership,
    ProviderRegistry,
    ProviderShutdownError,
    Runtime,
    RuntimeClosedError,
)
from conducto.adapters import AdapterDependencyError, require_adapter
from conducto.adapters import catalog as adapter_catalog


class _ClosableProvider(FakeModel):
    """Minimal provider fixture that records deterministic cleanup."""

    def __init__(self) -> None:
        super().__init__({})
        self.closes = 0

    def close(self) -> None:
        """Record one synchronous cleanup."""
        self.closes += 1


class _AsyncClosableProvider(_ClosableProvider):
    """Lifecycle fixture exposing the optional asynchronous close protocol."""

    async def aclose(self) -> None:
        """Record one asynchronous cleanup."""
        self.closes += 1


class _FailingClosableProvider(_ClosableProvider):
    """Lifecycle fixture that fails cleanup without disclosing a secret."""

    def close(self) -> None:
        """Raise a local close failure."""
        raise RuntimeError("credential=do-not-report")


class _ThreadRecordingProvider(_ClosableProvider):
    """Synchronous provider that records the thread used for cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.close_thread: int | None = None

    def close(self) -> None:
        """Record the cleanup worker identity."""
        self.close_thread = threading.get_ident()
        super().close()


class _BlockingSyncProvider(_ClosableProvider):
    """Synchronous provider controlled by thread events."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        """Wait for the test to release the blocking close."""
        self.started.set()
        self.release_close.wait()
        super().close()


class _ControlledAsyncProvider(_ClosableProvider):
    """Asynchronous provider controlled without wall-clock sleeps."""

    def __init__(self, *, failure: bool = False) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.finished = asyncio.Event()
        self.failure = failure

    async def aclose(self) -> None:
        """Wait for release, then close or raise the configured failure."""
        self.started.set()
        await self.release_close.wait()
        try:
            if self.failure:
                raise RuntimeError("credential=do-not-report")
            self.closes += 1
        finally:
            self.finished.set()


class _CancelledCleanupProvider(_ClosableProvider):
    """Provider that reports cancellation from its cleanup protocol."""

    async def aclose(self) -> None:
        """Raise cancellation as a provider cleanup outcome."""
        raise asyncio.CancelledError


def _configuration() -> ModelConfiguration:
    """Return safe model metadata shared by lifecycle tests."""
    return ModelConfiguration(provider="fixture", model="fixture")


def test_runtime_closes_transferred_client_and_skips_caller_owned_client() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        owned = _ClosableProvider()
        borrowed = _ClosableProvider()
        registry.register_client(
            "owned",
            owned,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        registry.register_client("borrowed", borrowed, _configuration())

        report = await Runtime(provider_registry=registry).aclose()

        assert owned.closes == 1
        assert borrowed.closes == 0
        assert [item.value for item in report.closed] == ["owned"]
        assert [item.value for item in report.skipped_caller_owned] == ["borrowed"]

    asyncio.run(exercise())


def test_shared_transferred_client_closes_once() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ClosableProvider()
        registry.register_client(
            "first",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        registry.register_client(
            "second",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        report = await Runtime(provider_registry=registry).aclose()

        assert client.closes == 1
        assert [item.value for item in report.closed] == ["first", "second"]

    asyncio.run(exercise())


def test_replaced_client_waits_for_accepted_lease_before_closing() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        retired = _ClosableProvider()
        replacement = _ClosableProvider()
        first = registry.register_client(
            "model",
            retired,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        lease = registry.acquire(first)
        registry.register_client(
            "model",
            replacement,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
            replace=True,
        )

        assert retired.closes == 0
        await lease.release()
        assert retired.closes == 1
        assert replacement.closes == 0

        await Runtime(provider_registry=registry).aclose()
        assert replacement.closes == 1

    asyncio.run(exercise())


def test_runtime_shutdown_is_idempotent_and_rejects_new_use() -> None:
    async def exercise() -> None:
        runtime = Runtime()
        first, second = await asyncio.gather(runtime.aclose(), runtime.aclose())

        assert first == second
        with pytest.raises(RuntimeClosedError):
            runtime.create_run_context(agent_id="agent")
        with pytest.raises(RuntimeClosedError):
            runtime.provider_registry.register_client(
                "model",
                _ClosableProvider(),
                _configuration(),
            )

    asyncio.run(exercise())


def test_missing_adapter_dependency_names_its_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_: str) -> str:
        raise adapter_catalog.metadata.PackageNotFoundError

    monkeypatch.setattr(adapter_catalog.metadata, "version", missing_distribution)

    with pytest.raises(AdapterDependencyError, match=r"conducto-ai\[openai\]"):
        require_adapter("openai")


def test_adapter_dependency_version_must_match_declared_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_catalog.metadata, "version", lambda _: "0.9")

    with pytest.raises(AdapterDependencyError, match=r"openai>=1.0"):
        require_adapter("openai")

    monkeypatch.setattr(adapter_catalog.metadata, "version", lambda _: "1.2.3")
    assert require_adapter("openai").version_specifier == ">=1.0"


def test_async_cleanup_and_failures_are_aggregated_without_exception_text() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        async_client = _AsyncClosableProvider()
        failing_client = _FailingClosableProvider()
        registry.register_client(
            "async",
            async_client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        registry.register_client(
            "failing",
            failing_client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        with pytest.raises(ProviderShutdownError) as raised:
            await Runtime(provider_registry=registry).aclose()

        assert async_client.closes == 1
        assert [
            (item.references[0].value, item.provider, item.cause_type)
            for item in raised.value.report.failures
        ] == [("failing", "fixture", "RuntimeError")]
        assert "credential" not in repr(raised.value.report)

    asyncio.run(exercise())


def test_synchronous_cleanup_runs_off_the_event_loop() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ThreadRecordingProvider()
        event_loop_thread = threading.get_ident()
        registry.register_client(
            "model",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        report = await registry.aclose()

        assert [item.value for item in report.closed] == ["model"]
        assert client.close_thread is not None
        assert client.close_thread != event_loop_thread

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("timeout", "per_client_timeout"),
    ((1.0, 0.01), (0.01, None)),
)
def test_synchronous_cleanup_obeys_client_and_aggregate_deadlines(
    timeout: float,
    per_client_timeout: float | None,
) -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _BlockingSyncProvider()
        registry.register_client(
            "model",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        try:
            with pytest.raises(ProviderShutdownError) as raised:
                await registry.aclose(
                    timeout=timeout,
                    per_client_timeout=per_client_timeout,
                )
        finally:
            client.release_close.set()

        if per_client_timeout is not None:
            assert client.started.is_set()
        assert [(item.cause_type, item.timed_out) for item in raised.value.report.failures] == [
            ("TimeoutError", True)
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ("replace", "deregister"))
def test_unleased_retired_clients_close_immediately(operation: str) -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        retired = _ControlledAsyncProvider()
        registry.register_client(
            "model",
            retired,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        if operation == "replace":
            registry.register_client(
                "model",
                _ClosableProvider(),
                _configuration(),
                ownership=ProviderOwnership.RUNTIME_OWNED,
                replace=True,
            )
        else:
            registry.deregister_model("model")

        await asyncio.wait_for(retired.started.wait(), 1.0)
        retired.release_close.set()
        await asyncio.wait_for(retired.finished.wait(), 1.0)
        assert retired.closes == 1
        await registry.aclose()

    asyncio.run(exercise())


def test_retired_cleanup_revalidates_before_closing_reattached_client() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ControlledAsyncProvider()
        registry.register_client(
            "first",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        registry.deregister_model("first")
        registry.register_client(
            "second",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        await asyncio.sleep(0)

        assert not client.started.is_set()
        registry.deregister_model("second")
        await asyncio.wait_for(client.started.wait(), 1.0)
        client.release_close.set()
        await asyncio.wait_for(client.finished.wait(), 1.0)
        assert client.closes == 1
        await registry.aclose()

    asyncio.run(exercise())


def test_acquire_rejects_a_client_after_cleanup_starts() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ControlledAsyncProvider()
        registration = registry.register_client(
            "model",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        registry.deregister_model("model")
        await asyncio.wait_for(client.started.wait(), 1.0)
        with pytest.raises(RuntimeClosedError):
            registry.acquire(registration)
        client.release_close.set()
        await asyncio.wait_for(client.finished.wait(), 1.0)
        await registry.aclose()

    asyncio.run(exercise())


def test_release_cleanup_failure_is_observed_by_concurrent_shutdown() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ControlledAsyncProvider(failure=True)
        registration = registry.register_client(
            "model",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        lease = registry.acquire(registration)
        registry.deregister_model("model")

        release_task = asyncio.create_task(lease.release())
        await asyncio.wait_for(client.started.wait(), 1.0)
        shutdown_task = asyncio.create_task(registry.aclose())
        client.release_close.set()
        await release_task

        with pytest.raises(ProviderShutdownError) as raised:
            await shutdown_task
        assert [(item.provider, item.cause_type) for item in raised.value.report.failures] == [
            ("fixture", "RuntimeError")
        ]

    asyncio.run(exercise())


def test_provider_cleanup_cancellation_is_reported_without_aborting_others() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        completed = _ClosableProvider()
        registry.register_client(
            "cancelled",
            _CancelledCleanupProvider(),
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )
        registry.register_client(
            "completed",
            completed,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        with pytest.raises(ProviderShutdownError) as raised:
            await registry.aclose()

        assert completed.closes == 1
        assert [item.cause_type for item in raised.value.report.failures] == ["CancelledError"]
        assert [item.value for item in raised.value.report.closed] == ["completed"]

    asyncio.run(exercise())


def test_cancelled_shutdown_waiter_does_not_cancel_shared_cleanup_owner() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ControlledAsyncProvider()
        registry.register_client(
            "model",
            client,
            _configuration(),
            ownership=ProviderOwnership.RUNTIME_OWNED,
        )

        first_waiter = asyncio.create_task(registry.aclose())
        await asyncio.wait_for(client.started.wait(), 1.0)
        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        client.release_close.set()
        report = await registry.aclose()
        assert [item.value for item in report.closed] == ["model"]
        assert client.closes == 1

    asyncio.run(exercise())


def test_factory_client_constructed_during_shutdown_is_cleaned_if_not_published() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        client = _ThreadRecordingProvider()
        construction_started = threading.Event()
        finish_construction = threading.Event()

        class _BlockingFactory:
            def __init__(self, provider: _ThreadRecordingProvider) -> None:
                self.provider = provider

            def create(self, configuration: ProviderClientConfig) -> _ThreadRecordingProvider:
                del configuration
                construction_started.set()
                finish_construction.wait()
                return self.provider

        registry.register_provider_type("fixture", _BlockingFactory(client))
        outcome: list[BaseException] = []

        def construct() -> None:
            try:
                registry.register_provider(
                    "model",
                    provider_type="fixture",
                    configuration=ProviderClientConfig(),
                    model_configuration=_configuration(),
                )
            except BaseException as error:
                outcome.append(error)

        construction_thread = threading.Thread(target=construct)
        construction_thread.start()
        await asyncio.to_thread(construction_started.wait)
        shutdown_task = asyncio.create_task(registry.aclose())
        await asyncio.sleep(0)
        assert registry.closed
        finish_construction.set()
        await asyncio.to_thread(construction_thread.join)

        report = await shutdown_task
        assert len(outcome) == 1
        assert isinstance(outcome[0], RuntimeClosedError)
        assert client.closes == 1
        assert [item.value for item in report.closed] == ["model"]

    asyncio.run(exercise())
