import asyncio

import pytest

from conducto import (
    FakeModel,
    ModelConfiguration,
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
        raise adapter_catalog.PackageNotFoundError

    monkeypatch.setattr(adapter_catalog, "version", missing_distribution)

    with pytest.raises(AdapterDependencyError, match=r"conducto-ai\[openai\]"):
        require_adapter("openai")


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
            (item.references[0].value, item.cause_type) for item in raised.value.report.failures
        ] == [("failing", "RuntimeError")]
        assert "credential" not in str(raised.value.report)

    asyncio.run(exercise())
