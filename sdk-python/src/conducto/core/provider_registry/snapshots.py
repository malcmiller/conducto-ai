"""Credential-free registry inspection captured from a single locked state."""

from dataclasses import dataclass

from ..model_config import ModelReference, ProviderType
from .availability import evaluate_available
from .ownership import ProviderOwnership
from .registration import ProviderRegistration
from .state import _RegistryState


@dataclass(frozen=True, slots=True)
class ModelBindingSnapshot:
    """Safe metadata excluding clients, factories, and connection configuration."""

    reference: ModelReference
    provider: str
    provider_type: ProviderType | None
    ownership: ProviderOwnership
    available: bool


@dataclass(frozen=True, slots=True)
class ProviderRegistrySnapshot:
    """Immutable, deterministically ordered registry contents."""

    models: tuple[ModelBindingSnapshot, ...]
    provider_types: tuple[ProviderType, ...]


def _to_snapshot(registration: ProviderRegistration) -> ModelBindingSnapshot:
    """Project a binding without disclosing its client or connection settings."""
    return ModelBindingSnapshot(
        registration.reference,
        registration.provider,
        registration.provider_type,
        registration.ownership,
        evaluate_available(registration.available, registration.available_timeout),
    )


class _RegistrySnapshots(_RegistryState):
    """Capture state atomically and evaluate health outside the lock."""

    def _collect(self) -> tuple[tuple[ProviderRegistration, ...], tuple[ProviderType, ...]]:
        with self._lock:
            return tuple(self._registrations.values()), tuple(self._provider_types)

    def list_provider_types(self) -> tuple[ProviderType, ...]:
        """Return registered provider types in deterministic sorted order."""
        _, types = self._collect()
        return tuple(sorted(types, key=str))

    def list_models(self) -> tuple[ModelBindingSnapshot, ...]:
        """Return published model bindings as safe, deterministic snapshots."""
        registrations, _ = self._collect()
        return tuple(
            sorted(
                (_to_snapshot(item) for item in registrations),
                key=lambda item: item.reference.value,
            )
        )

    def snapshot(self) -> ProviderRegistrySnapshot:
        """Capture bindings and types together, then evaluate bounded health.

        A concurrent mutation cannot combine bindings and provider types that
        never coexisted. No health predicate executes under the registry lock.
        """
        registrations, types = self._collect()
        models = tuple(
            sorted(
                (_to_snapshot(item) for item in registrations),
                key=lambda item: item.reference.value,
            )
        )
        return ProviderRegistrySnapshot(models, tuple(sorted(types, key=str)))
