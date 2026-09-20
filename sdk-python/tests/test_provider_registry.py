"""Focused contracts for runtime-owned provider registration."""

import pytest

from conducto import FakeModel, ModelConfiguration
from conducto.core.provider_registry import ProviderRegistry as RegistryProviderRegistry
from conducto.core.runtime import (
    ProviderRegistration,
    ProviderRegistry,
    ProviderUnavailableError,
)


def test_provider_registry_replaces_only_when_requested() -> None:
    registry = ProviderRegistry()
    first = FakeModel({})
    second = FakeModel({})
    configuration = ModelConfiguration(provider="fake", model="test")

    assert ProviderRegistry is RegistryProviderRegistry
    registry.register("model", first, configuration)
    registration = registry.resolve("model")
    assert isinstance(registration, ProviderRegistration)
    assert registration.client is first
    with pytest.raises(ValueError, match="already registered"):
        registry.register("model", second, configuration)

    registry.register("model", second, configuration, replace=True)
    assert registry.resolve("model").client is second


def test_provider_registry_evaluates_availability_on_each_lookup() -> None:
    available = True
    registry = ProviderRegistry()
    registry.register(
        "model",
        FakeModel({}),
        ModelConfiguration(provider="fake", model="test"),
        available=lambda: available,
    )

    assert registry.resolve("model").reference.value == "model"
    available = False
    with pytest.raises(ProviderUnavailableError):
        registry.resolve("model")
