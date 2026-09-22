"""Focused contracts for runtime-owned provider registration."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from conducto.core.provider import ModelConfiguration, ProviderResult
from conducto.core.provider_registry import (
    ProviderClientConfig,
    ProviderOwnership,
    ProviderRegistration,
    ProviderRegistry,
)
from conducto.core.runtime_errors import (
    ContradictoryProviderConfigurationError,
    DuplicateModelReferenceError,
    DuplicateProviderTypeError,
    IncompatibleProviderCapabilitiesError,
    ProviderClientValidationError,
    ProviderConstructionError,
    ProviderFactoryValidationError,
    ProviderTypeMismatchError,
    ProviderUnavailableError,
    StaleProviderConstructionError,
    UnknownModelReferenceError,
    UnknownProviderTypeError,
)
from conducto.testing import FakeModel


def test_provider_registry_replaces_only_when_requested() -> None:
    registry = ProviderRegistry()
    first = FakeModel(ProviderResult(structured={}, accepted=True))
    second = FakeModel(ProviderResult(structured={}, accepted=True))
    configuration = ModelConfiguration(provider="fake", model="test")

    registry.register_client("model", first, configuration)
    registration = registry.resolve("model")
    assert isinstance(registration, ProviderRegistration)
    assert registration.client is first
    assert registration.ownership is ProviderOwnership.CALLER_OWNED
    with pytest.raises(DuplicateModelReferenceError, match="already registered"):
        registry.register_client("model", second, configuration)

    registry.register_client("model", second, configuration, replace=True)
    assert registry.resolve("model").client is second


def test_provider_registry_evaluates_availability_on_each_lookup() -> None:
    available = True
    registry = ProviderRegistry()
    registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="test"),
        available=lambda: available,
    )

    assert registry.resolve("model").reference.value == "model"
    available = False
    with pytest.raises(ProviderUnavailableError):
        registry.resolve("model")


def test_registry_package_owns_its_api_without_legacy_registration() -> None:
    from conducto.core import provider_registry
    from conducto.core.provider_registry.configuration import (
        ProviderClientConfig as Configuration,
    )
    from conducto.core.provider_registry.ownership import ProviderOwnership as Ownership
    from conducto.core.provider_registry.registry import ProviderRegistry as Registry

    assert provider_registry.ProviderRegistry is Registry
    assert ProviderClientConfig is Configuration
    assert ProviderOwnership is Ownership
    assert not hasattr(ProviderRegistry, "register")
    assert not hasattr(ProviderRegistry(), "register")
    assert set(provider_registry.__all__) == {
        "DEFAULT_AVAILABILITY_TIMEOUT_SECONDS",
        "ModelBindingSnapshot",
        "ProviderCleanupFailure",
        "ProviderCleanupReport",
        "ProviderClientConfig",
        "ProviderFactory",
        "ProviderLease",
        "ProviderOwnership",
        "ProviderRegistration",
        "ProviderRegistry",
        "ProviderRegistrySnapshot",
        "ProviderTypeRegistration",
    }


def test_deregister_model_prevents_later_resolution_only() -> None:
    registry = ProviderRegistry()
    configuration = ModelConfiguration(provider="fake", model="test")
    registry.register_client(
        "model", FakeModel(ProviderResult(structured={}, accepted=True)), configuration
    )

    accepted = registry.resolve("model")
    registry.deregister_model("model")
    assert accepted.client is not None

    with pytest.raises(UnknownModelReferenceError):
        registry.resolve("model")
    with pytest.raises(UnknownModelReferenceError):
        registry.deregister_model("model")


class _RecordingFactory:
    """A trusted provider-type factory used only for tests."""

    def __init__(self) -> None:
        self.calls: list[ProviderClientConfig] = []

    def create(self, configuration: ProviderClientConfig) -> FakeModel:
        self.calls.append(configuration)
        return FakeModel(ProviderResult(structured={}, accepted=True))


class _FailingFactory:
    def create(self, configuration: ProviderClientConfig) -> FakeModel:
        raise RuntimeError("boom")


def test_register_provider_constructs_once_from_typed_configuration() -> None:
    registry = ProviderRegistry()
    factory = _RecordingFactory()
    registry.register_provider_type("fake", factory)

    registration = registry.register_provider(
        "model",
        provider_type="fake",
        configuration=ProviderClientConfig(endpoint="https://example.test"),
        model_configuration=ModelConfiguration(provider="fake", model="test"),
    )

    assert len(factory.calls) == 1
    assert registration.ownership is ProviderOwnership.RUNTIME_OWNED
    assert registration.provider_type is not None
    assert str(registration.provider_type) == "fake"
    assert registry.resolve("model").client is registration.client


def test_register_provider_type_duplicate_requires_replace() -> None:
    registry = ProviderRegistry()
    registry.register_provider_type("fake", _RecordingFactory())
    with pytest.raises(DuplicateProviderTypeError):
        registry.register_provider_type("fake", _RecordingFactory())
    registry.register_provider_type("fake", _RecordingFactory(), replace=True)


def test_register_provider_type_rejects_non_structural_factory() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ProviderFactoryValidationError):
        registry.register_provider_type("fake", object())  # type: ignore[arg-type]


def test_register_provider_unknown_provider_type_fails() -> None:
    registry = ProviderRegistry()
    with pytest.raises(UnknownProviderTypeError):
        registry.register_provider(
            "model",
            provider_type="missing",
            configuration=ProviderClientConfig(),
            model_configuration=ModelConfiguration(provider="missing", model="test"),
        )


def test_register_provider_mismatched_provider_identifier_fails() -> None:
    registry = ProviderRegistry()
    registry.register_provider_type("fake", _RecordingFactory())
    with pytest.raises(ProviderTypeMismatchError):
        registry.register_provider(
            "model",
            provider_type="fake",
            configuration=ProviderClientConfig(),
            model_configuration=ModelConfiguration(provider="other", model="test"),
        )


def test_register_provider_construction_failure_is_typed_and_does_not_publish() -> None:
    registry = ProviderRegistry()
    registry.register_provider_type("fake", _FailingFactory())
    with pytest.raises(ProviderConstructionError):
        registry.register_provider(
            "model",
            provider_type="fake",
            configuration=ProviderClientConfig(),
            model_configuration=ModelConfiguration(provider="fake", model="test"),
        )
    with pytest.raises(UnknownModelReferenceError):
        registry.resolve("model")


def test_register_provider_validates_required_capabilities() -> None:
    registry = ProviderRegistry()
    registry.register_provider_type("fake", _RecordingFactory())
    with pytest.raises(IncompatibleProviderCapabilitiesError):
        registry.register_provider(
            "model",
            provider_type="fake",
            configuration=ProviderClientConfig(),
            model_configuration=ModelConfiguration(provider="fake", model="test"),
            required_capabilities=frozenset({"cancellation"}),
        )


def test_register_client_rejects_contradictory_connection_config() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ContradictoryProviderConfigurationError):
        registry.register_client(
            "model",
            FakeModel(ProviderResult(structured={}, accepted=True)),
            ModelConfiguration(provider="fake", model="test"),
            connection_config=ProviderClientConfig(endpoint="https://example.test"),
        )
    with pytest.raises(UnknownModelReferenceError):
        registry.resolve("model")


def test_register_client_rejects_structurally_invalid_client() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ProviderClientValidationError):
        registry.register_client(
            "model",
            object(),  # type: ignore[arg-type]
            ModelConfiguration(provider="fake", model="test"),
        )


def test_snapshot_is_immutable_deterministic_and_safe() -> None:
    registry = ProviderRegistry()
    registry.register_provider_type("fake", _RecordingFactory())
    registry.register_client(
        "b-model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="b"),
    )
    registry.register_client(
        "a-model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="a"),
    )

    snapshot = registry.snapshot()
    assert [item.reference.value for item in snapshot.models] == ["a-model", "b-model"]
    assert [str(item) for item in snapshot.provider_types] == ["fake"]
    for item in snapshot.models:
        assert not hasattr(item, "client")
        assert not hasattr(item, "configuration")

    with pytest.raises(AttributeError):
        field_name = "models"
        setattr(snapshot, field_name, ())


def test_concurrent_register_resolve_replace_deregister_stay_isolated() -> None:
    registry = ProviderRegistry()
    configuration = ModelConfiguration(provider="fake", model="test")
    errors: list[Exception] = []

    def worker(index: int) -> None:
        reference = f"model-{index % 4}"
        try:
            registry.register_client(
                reference,
                FakeModel(ProviderResult(structured={}, accepted=True)),
                configuration,
                replace=True,
            )
            registry.resolve(reference)
            if index % 7 == 0:
                registry.deregister_model(reference)
        except UnknownModelReferenceError:
            pass
        except Exception as exc:  # pragma: no cover - failure path surfaces in assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    remaining = registry.snapshot().models
    assert all(item.reference.value.startswith("model-") for item in remaining)


def test_register_client_rejects_contradictory_model_configuration_endpoint() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ContradictoryProviderConfigurationError):
        registry.register_client(
            "model",
            FakeModel(ProviderResult(structured={}, accepted=True)),
            ModelConfiguration(provider="fake", model="test", endpoint="https://example.test"),
        )
    with pytest.raises(UnknownModelReferenceError):
        registry.resolve("model")


def test_register_provider_duplicate_rejected_before_factory_runs() -> None:
    registry = ProviderRegistry()
    factory = _RecordingFactory()
    registry.register_provider_type("fake", factory)
    registry.register_provider(
        "model",
        provider_type="fake",
        configuration=ProviderClientConfig(),
        model_configuration=ModelConfiguration(provider="fake", model="test"),
    )
    assert len(factory.calls) == 1

    with pytest.raises(DuplicateModelReferenceError):
        registry.register_provider(
            "model",
            provider_type="fake",
            configuration=ProviderClientConfig(),
            model_configuration=ModelConfiguration(provider="fake", model="test"),
        )

    # The factory must never be invoked for a registration that cannot be published.
    assert len(factory.calls) == 1


def test_available_predicate_that_hangs_is_bounded_and_treated_as_unavailable() -> None:
    registry = ProviderRegistry()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_health_check() -> bool:
        entered.set()
        try:
            release.wait()
            return True
        finally:
            finished.set()

    registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="test"),
        available=blocked_health_check,
        available_timeout=0.05,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        resolution = executor.submit(registry.resolve, "model")
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(ProviderUnavailableError):
                resolution.result(timeout=2)
        finally:
            release.set()
            assert finished.wait(timeout=2)


def test_available_predicate_that_raises_is_treated_as_unavailable() -> None:
    registry = ProviderRegistry()

    def explodes() -> bool:
        raise RuntimeError("boom")

    registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="test"),
        available=explodes,
    )

    with pytest.raises(ProviderUnavailableError):
        registry.resolve("model")


def test_replace_during_slow_factory_construction_discards_stale_result() -> None:
    registry = ProviderRegistry()
    started = threading.Event()
    release = threading.Event()

    class _SlowFactory:
        @staticmethod
        def create(configuration: ProviderClientConfig) -> FakeModel:
            del configuration
            started.set()
            release.wait(timeout=5)
            return FakeModel(ProviderResult(structured={}, accepted=True))

    registry.register_provider_type("fake", _SlowFactory())
    registry.register_client(
        "model",
        FakeModel(ProviderResult(structured={}, accepted=True)),
        ModelConfiguration(provider="fake", model="test"),
    )

    outcome: list[BaseException] = []

    def slow_register() -> None:
        try:
            registry.register_provider(
                "model",
                provider_type="fake",
                configuration=ProviderClientConfig(),
                model_configuration=ModelConfiguration(provider="fake", model="test"),
                replace=True,
            )
        except BaseException as exc:  # noqa: BLE001 - captured for assertion below
            outcome.append(exc)

    thread = threading.Thread(target=slow_register)
    thread.start()
    started.wait(timeout=5)
    registry.deregister_model("model")
    release.set()
    thread.join(timeout=5)

    assert len(outcome) == 1
    assert isinstance(outcome[0], StaleProviderConstructionError)
    with pytest.raises(UnknownModelReferenceError):
        registry.resolve("model")
