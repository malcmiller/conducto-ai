"""Focused contracts for runtime-owned provider registration."""

import threading

import pytest

from conducto import FakeModel, ModelConfiguration
from conducto.core.provider_registry import (
    ProviderClientConfig,
    ProviderOwnership,
)
from conducto.core.provider_registry import ProviderRegistry as RegistryProviderRegistry
from conducto.core.runtime import (
    ContradictoryProviderConfigurationError,
    DuplicateModelReferenceError,
    DuplicateProviderTypeError,
    IncompatibleProviderCapabilitiesError,
    ProviderClientValidationError,
    ProviderConstructionError,
    ProviderFactoryValidationError,
    ProviderRegistration,
    ProviderRegistry,
    ProviderTypeMismatchError,
    ProviderUnavailableError,
    UnknownModelReferenceError,
    UnknownProviderTypeError,
)


def test_provider_registry_replaces_only_when_requested() -> None:
    registry = ProviderRegistry()
    first = FakeModel({})
    second = FakeModel({})
    configuration = ModelConfiguration(provider="fake", model="test")

    assert ProviderRegistry is RegistryProviderRegistry
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
        FakeModel({}),
        ModelConfiguration(provider="fake", model="test"),
        available=lambda: available,
    )

    assert registry.resolve("model").reference.value == "model"
    available = False
    with pytest.raises(ProviderUnavailableError):
        registry.resolve("model")


def test_legacy_register_delegates_to_register_client_and_warns() -> None:
    registry = ProviderRegistry()
    client = FakeModel({})
    configuration = ModelConfiguration(provider="fake", model="test")

    with pytest.deprecated_call():
        registry.register("model", client, configuration)

    registration = registry.resolve("model")
    assert registration.client is client
    assert registration.ownership is ProviderOwnership.CALLER_OWNED


def test_deregister_model_prevents_later_resolution_only() -> None:
    registry = ProviderRegistry()
    configuration = ModelConfiguration(provider="fake", model="test")
    registry.register_client("model", FakeModel({}), configuration)

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
        return FakeModel({})


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
            FakeModel({}),
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
        "b-model", FakeModel({}), ModelConfiguration(provider="fake", model="b")
    )
    registry.register_client(
        "a-model", FakeModel({}), ModelConfiguration(provider="fake", model="a")
    )

    snapshot = registry.snapshot()
    assert [item.reference.value for item in snapshot.models] == ["a-model", "b-model"]
    assert [str(item) for item in snapshot.provider_types] == ["fake"]
    for item in snapshot.models:
        assert not hasattr(item, "client")
        assert not hasattr(item, "configuration")

    with pytest.raises(AttributeError):
        snapshot.models = ()  # type: ignore[misc]


def test_concurrent_register_resolve_replace_deregister_stay_isolated() -> None:
    registry = ProviderRegistry()
    configuration = ModelConfiguration(provider="fake", model="test")
    errors: list[Exception] = []

    def worker(index: int) -> None:
        reference = f"model-{index % 4}"
        try:
            registry.register_client(reference, FakeModel({}), configuration, replace=True)
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
