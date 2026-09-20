"""Thread-safe provider registration and lookup."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .model_config import ModelReference, normalize_reference
from .provider import ModelConfiguration, ModelProvider
from .runtime_errors import ProviderUnavailableError, UnknownModelReferenceError


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """Runtime-owned model registration."""

    reference: ModelReference
    provider: str
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)
    available: bool | Callable[[], bool] = field(default=True, repr=False, compare=False)


class ProviderRegistry:
    """Thread-safe runtime-owned registry of model references and clients."""

    def __init__(self) -> None:
        self._registrations: dict[ModelReference, ProviderRegistration] = {}
        self._lock = threading.RLock()

    def register(
        self,
        reference: ModelReference | str,
        client: ModelProvider,
        configuration: ModelConfiguration,
        *,
        available: bool | Callable[[], bool] = True,
        replace: bool = False,
    ) -> None:
        """Register a provider client for a model reference.

        Args:
            reference: The model reference to bind to the client.
            client: Provider client that implements the model protocol.
            configuration: Provider-neutral configuration for the model.
            available: Optional availability predicate or fixed state.
            replace: Whether to replace an existing registration.

        Raises:
            ValueError: If the reference already exists and replacement is disabled.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        registration = ProviderRegistration(
            model_reference,
            configuration.provider,
            client,
            configuration,
            available,
        )
        with self._lock:
            if model_reference in self._registrations and not replace:
                raise ValueError(f"Model reference '{model_reference}' is already registered")
            self._registrations[model_reference] = registration

    def resolve(self, reference: ModelReference | str) -> ProviderRegistration:
        """Resolve a registered model reference and validate provider availability.

        Args:
            reference: Model reference to resolve.

        Returns:
            The registered provider binding.

        Raises:
            UnknownModelReferenceError: If the model is not registered.
            ProviderUnavailableError: If the provider is marked unavailable.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        with self._lock:
            registration = self._registrations.get(model_reference)
        if registration is None:
            raise UnknownModelReferenceError(f"Unknown model reference '{model_reference}'")
        available = (
            registration.available() if callable(registration.available) else registration.available
        )
        if not available:
            raise ProviderUnavailableError(
                f"Provider for model reference '{model_reference}' is unavailable"
            )
        return registration
