"""Provider-type registration and atomic reservation of factory construction."""

from ..model_config import ModelReference, ProviderType, normalize_provider_type
from ..runtime_errors import (
    DuplicateModelReferenceError,
    DuplicateProviderTypeError,
    ProviderFactoryValidationError,
    UnknownProviderTypeError,
)
from .configuration import ProviderFactory, ProviderTypeRegistration
from .state import _RegistryState


class _ProviderTypes(_RegistryState):
    """Manage trusted factory types in the registry's synchronization domain."""

    def register_provider_type(
        self,
        provider_type: ProviderType | str,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a trusted factory without constructing a provider.

        Args:
            provider_type: Normalized identifier for the provider family.
            factory: Structural factory implementing ``create(configuration)``.
            replace: Whether an existing factory may be replaced.

        Raises:
            ProviderFactoryValidationError: If ``create`` is not callable.
            DuplicateProviderTypeError: If the type exists and replacement is disabled.
        """
        normalized = normalize_provider_type(provider_type)
        if not callable(getattr(factory, "create", None)):
            raise ProviderFactoryValidationError(
                f"Factory for provider type '{normalized}' must implement create(configuration)"
            )
        registration = ProviderTypeRegistration(normalized, factory)
        with self._lock:
            self._require_open()
            if normalized in self._provider_types and not replace:
                raise DuplicateProviderTypeError(
                    f"Provider type '{normalized}' is already registered"
                )
            self._provider_types[normalized] = registration

    def deregister_provider_type(self, provider_type: ProviderType | str) -> None:
        """Remove a factory without invalidating existing model bindings.

        Raises:
            UnknownProviderTypeError: If no factory is registered for the type.
        """
        normalized = normalize_provider_type(provider_type)
        with self._lock:
            self._require_open()
            if self._provider_types.pop(normalized, None) is None:
                raise UnknownProviderTypeError(f"Unknown provider type '{normalized}'")

    def _begin_construction(
        self,
        reference: ModelReference,
        provider_type: ProviderType,
        *,
        replace: bool,
    ) -> tuple[ProviderFactory, int, int]:
        """Reserve construction and capture its binding mutation generation."""
        with self._lock:
            self._require_open()
            if reference in self._registrations and not replace:
                raise DuplicateModelReferenceError(
                    f"Model reference '{reference}' is already registered"
                )
            registration = self._provider_types.get(provider_type)
            if registration is None:
                raise UnknownProviderTypeError(f"Unknown provider type '{provider_type}'")
            expected_generation = self._generations.get(reference, 0)
            construction_id = self._next_construction_id
            self._next_construction_id += 1
            self._active_constructions[construction_id] = (reference, str(provider_type))
            return registration.factory, expected_generation, construction_id

    def _finish_construction(self, construction_id: int) -> None:
        """Release a construction reservation and wake shutdown waiters."""
        with self._lease_condition:
            self._active_constructions.pop(construction_id, None)
            self._lease_condition.notify_all()
