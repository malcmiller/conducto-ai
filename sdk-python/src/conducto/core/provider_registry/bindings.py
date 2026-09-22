"""Validated model registration, atomic publication, and immutable resolution."""

from __future__ import annotations

from collections.abc import Callable

from ..model_config import (
    ModelReference,
    ProviderType,
    normalize_provider_type,
    normalize_reference,
)
from ..provider import ModelConfiguration, ModelProvider, validate_provider_capabilities
from ..runtime_errors import (
    ContradictoryProviderConfigurationError,
    DuplicateModelReferenceError,
    ProviderClientValidationError,
    ProviderConstructionError,
    ProviderTypeMismatchError,
    ProviderUnavailableError,
    StaleProviderConstructionError,
    UnknownModelReferenceError,
)
from .availability import DEFAULT_AVAILABILITY_TIMEOUT_SECONDS, evaluate_available
from .configuration import ProviderClientConfig
from .factories import _ProviderTypes
from .lifecycle import _ProviderLifecycle
from .ownership import ProviderOwnership, _ProviderClientRecord
from .registration import ProviderRegistration


def _validate_client_structure(client: ModelProvider, provider_label: ProviderType | str) -> None:
    if not hasattr(client, "capabilities") or not callable(getattr(client, "complete", None)):
        raise ProviderClientValidationError(
            f"Provider client for '{provider_label}' does not satisfy the structural "
            "provider protocol (missing capabilities or a callable complete())"
        )


class _ModelBindings(_ProviderTypes, _ProviderLifecycle):
    """Publish validated bindings and coordinate construction with client ownership."""

    def register_provider(
        self,
        reference: ModelReference | str,
        *,
        provider_type: ProviderType | str,
        configuration: ProviderClientConfig,
        model_configuration: ModelConfiguration,
        available: bool | Callable[[], bool] = True,
        available_timeout: float = DEFAULT_AVAILABILITY_TIMEOUT_SECONDS,
        ownership: ProviderOwnership = ProviderOwnership.RUNTIME_OWNED,
        replace: bool = False,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> ProviderRegistration:
        """Construct and publish a binding from one typed configuration source.

        Factories run once, outside the lock. A duplicate is rejected before
        construction. Publication checks the captured mutation generation so
        stale construction cannot replace or resurrect a newer binding.
        Unpublished owned clients are retained for coordinated cleanup.

        Args:
            reference: Credential-free model reference to bind.
            provider_type: Registered type whose factory constructs the client.
            configuration: Typed provider-neutral construction inputs.
            model_configuration: Safe model settings stored on the binding.
            available: Fixed health state or bounded availability predicate.
            available_timeout: Maximum seconds to wait for a health predicate.
            ownership: Party responsible for retirement and shutdown cleanup.
            replace: Whether an existing binding may be replaced.
            required_capabilities: Capabilities the client must advertise.

        Returns:
            The published immutable registration.

        Raises:
            UnknownProviderTypeError: If the provider type has no factory.
            ProviderTypeMismatchError: If model settings name a different type.
            DuplicateModelReferenceError: If replacement is disabled.
            ProviderConstructionError: If the factory raises.
            ProviderClientValidationError: If structural validation fails.
            IncompatibleProviderCapabilitiesError: If capabilities are missing.
            StaleProviderConstructionError: If the reference changed in flight.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        normalized_type = normalize_provider_type(provider_type)
        if model_configuration.provider.strip().lower() != str(normalized_type):
            raise ProviderTypeMismatchError(
                f"Model configuration provider '{model_configuration.provider}' does not "
                f"match provider type '{normalized_type}'"
            )
        factory, expected_generation, construction_id = self._begin_construction(
            model_reference, normalized_type, replace=replace
        )
        client: ModelProvider | None = None
        cleanup_record: _ProviderClientRecord | None = None
        try:
            try:
                constructed_client = factory.create(configuration)
            except Exception as exc:
                raise ProviderConstructionError(
                    f"Provider type '{normalized_type}' failed to construct a client for "
                    f"model reference '{model_reference}'"
                ) from exc
            client = constructed_client
            _validate_client_structure(constructed_client, normalized_type)
            validate_provider_capabilities(
                model_reference, constructed_client.capabilities, required_capabilities
            )
            registration = ProviderRegistration(
                model_reference,
                model_configuration.provider,
                constructed_client,
                model_configuration,
                available,
                ownership,
                normalized_type,
                available_timeout,
            )
            cleanup_record = self._publish(
                model_reference,
                registration,
                replace=replace,
                expected_generation=expected_generation,
            )
            return registration
        except Exception:
            if client is not None:
                cleanup_record = self._retire_unpublished_client(
                    model_reference, model_configuration.provider, client, ownership
                )
            raise
        finally:
            self._finish_construction(construction_id)
            if cleanup_record is not None:
                self._schedule_retired_cleanup(cleanup_record)

    def register_client(
        self,
        reference: ModelReference | str,
        client: ModelProvider,
        configuration: ModelConfiguration,
        *,
        ownership: ProviderOwnership = ProviderOwnership.CALLER_OWNED,
        connection_config: ProviderClientConfig | None = None,
        available: bool | Callable[[], bool] = True,
        available_timeout: float = DEFAULT_AVAILABILITY_TIMEOUT_SECONDS,
        replace: bool = False,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> ProviderRegistration:
        """Publish a preconstructed, structurally validated provider client.

        Args:
            reference: Credential-free model reference to bind.
            client: Client satisfying the structural provider protocol.
            configuration: Model settings; the endpoint must be unset.
            ownership: Caller ownership by default, or explicit runtime ownership.
            connection_config: Must be omitted or contain only default values.
            available: Fixed health state or bounded availability predicate.
            available_timeout: Maximum seconds to wait for a health predicate.
            replace: Whether an existing binding may be replaced.
            required_capabilities: Capabilities the client must advertise.

        Returns:
            The published immutable registration.

        Raises:
            ContradictoryProviderConfigurationError: If connection settings
                accompany an already configured client.
            ProviderClientValidationError: If structural validation fails.
            IncompatibleProviderCapabilitiesError: If capabilities are missing.
            DuplicateModelReferenceError: If replacement is disabled.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        if connection_config is not None and not connection_config.is_default():
            raise ContradictoryProviderConfigurationError(
                "Connection configuration cannot be combined with a preconstructed "
                "provider client; construct the client with that configuration instead"
            )
        if configuration.endpoint is not None:
            raise ContradictoryProviderConfigurationError(
                "Model configuration endpoint cannot be combined with a preconstructed "
                "provider client; construct the client with that endpoint instead"
            )
        _validate_client_structure(client, configuration.provider)
        validate_provider_capabilities(model_reference, client.capabilities, required_capabilities)
        registration = ProviderRegistration(
            model_reference,
            configuration.provider,
            client,
            configuration,
            available,
            ownership,
            None,
            available_timeout,
        )
        retired = self._publish(model_reference, registration, replace=replace)
        if retired is not None:
            self._schedule_retired_cleanup(retired)
        return registration

    def resolve(self, reference: ModelReference | str) -> ProviderRegistration:
        """Select a published binding and evaluate its bounded availability.

        Resolution never constructs a provider.

        Raises:
            UnknownModelReferenceError: If the reference is not registered.
            ProviderUnavailableError: If the provider is unavailable.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        with self._lock:
            self._require_open()
            registration = self._registrations.get(model_reference)
        if registration is None:
            raise UnknownModelReferenceError(f"Unknown model reference '{model_reference}'")
        if not evaluate_available(registration.available, registration.available_timeout):
            raise ProviderUnavailableError(
                f"Provider for model reference '{model_reference}' is unavailable"
            )
        return registration

    def deregister_model(self, reference: ModelReference | str) -> None:
        """Remove a binding without invalidating accepted-use leases.

        Raises:
            UnknownModelReferenceError: If the reference is not registered.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        with self._lock:
            self._require_open()
            registration = self._registrations.pop(model_reference, None)
            if registration is None:
                raise UnknownModelReferenceError(f"Unknown model reference '{model_reference}'")
            self._detach(model_reference, registration)
            self._generations[model_reference] = self._generations.get(model_reference, 0) + 1
            retired = self._eligible_retired_record(registration)
        if retired is not None:
            self._schedule_retired_cleanup(retired)

    def _publish(
        self,
        reference: ModelReference,
        registration: ProviderRegistration,
        *,
        replace: bool,
        expected_generation: int | None = None,
    ) -> _ProviderClientRecord | None:
        """Atomically validate and publish, returning an eligible retired client."""
        with self._lock:
            self._require_open()
            if (
                expected_generation is not None
                and self._generations.get(reference, 0) != expected_generation
            ):
                raise StaleProviderConstructionError(
                    f"Model reference '{reference}' was replaced or deregistered while "
                    "its provider client was under construction"
                )
            if reference in self._registrations and not replace:
                raise DuplicateModelReferenceError(
                    f"Model reference '{reference}' is already registered"
                )
            previous = self._registrations.get(reference)
            record = self._record_for(registration)
            if previous is not None:
                self._detach(reference, previous)
            record.references.add(reference)
            record.retired_references.discard(reference)
            self._registrations[reference] = registration
            self._generations[reference] = self._generations.get(reference, 0) + 1
            if previous is None or previous.client is registration.client:
                return None
            return self._eligible_retired_record(previous)
