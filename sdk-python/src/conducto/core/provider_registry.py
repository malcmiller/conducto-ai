"""Thread-safe provider type, client, and model-reference registration.

This module maps two distinct, credential-free registration operations onto
immutable published bindings:

* Provider type registration describes how trusted application code
  constructs one provider family (:meth:`ProviderRegistry.register_provider_type`).
* Model reference registration binds a credential-free model reference to one
  configured provider client, either built from a typed configuration source
  (:meth:`ProviderRegistry.register_provider`) or supplied preconstructed by the
  caller (:meth:`ProviderRegistry.register_client`).

Model resolution (:meth:`ProviderRegistry.resolve`) only ever selects an
already-published, immutable binding; it never constructs a client.

``ProviderRegistry.register`` remains available as a deprecated compatibility
path that delegates to ``register_client`` with caller-owned semantics.
"""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .model_config import (
    ModelReference,
    ProviderType,
    freeze_metadata,
    normalize_provider_type,
    normalize_reference,
    validate_metadata,
)
from .provider import (
    AsynchronouslyClosableProvider,
    ModelConfiguration,
    ModelProvider,
    SynchronouslyClosableProvider,
    validate_provider_capabilities,
)
from .runtime_errors import (
    ContradictoryProviderConfigurationError,
    DuplicateModelReferenceError,
    DuplicateProviderTypeError,
    ProviderClientValidationError,
    ProviderConstructionError,
    ProviderFactoryValidationError,
    ProviderOwnershipError,
    ProviderShutdownError,
    ProviderTypeMismatchError,
    ProviderUnavailableError,
    RuntimeClosedError,
    StaleProviderConstructionError,
    UnknownModelReferenceError,
    UnknownProviderTypeError,
)

#: Default bound applied to a callable ``available`` predicate. A predicate
#: that has not completed within this many seconds is treated as unavailable
#: rather than blocking resolution, listing, or snapshotting indefinitely.
DEFAULT_AVAILABILITY_TIMEOUT_SECONDS = 2.0

#: Shared, bounded worker pool used to evaluate callable availability
#: predicates off the calling thread so a slow or hung health check cannot
#: block the registry lock or the caller.
_AVAILABILITY_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="conducto-provider-availability"
)


class ProviderOwnership(StrEnum):
    """Declares which party is responsible for closing a provider client.

    Story 6.3 delivers actual shutdown orchestration; this value is
    captured here, so later lifecycle management has a stable, typed signal.
    """

    RUNTIME_OWNED = "runtime_owned"
    CALLER_OWNED = "caller_owned"


class ProviderFactory(Protocol):
    """Structural contract for a trusted, allowlisted provider-type factory.

    Implementations construct one provider family's client from a single
    typed configuration source. Factories are invoked directly by trusted
    application code; the registry never imports modules, scans entry
    points, or executes configuration-supplied code on a caller's behalf.
    """

    def create(self, configuration: ProviderClientConfig) -> ModelProvider:
        """Construct a provider client from a typed configuration source.

        Args:
            configuration: Typed, provider-neutral construction inputs.

        Returns:
            A client satisfying the structural provider protocol.
        """
        ...


@dataclass(frozen=True, slots=True)
class ProviderClientConfig:
    """Typed, provider-neutral construction inputs for a factory.

    This is the single source of truth for connection-affecting settings
    passed to a factory. Concrete provider integrations may define their own
    richer configuration types; this shape covers the fields the registry
    itself understands and validates.

    Attributes:
        endpoint: Provider endpoint URL, or ``None`` to use the factory default.
        credential_ref: Opaque reference to an externally held credential (for
            example, an environment variable name or secret-store key). The
            actual secret value must never be stored here.
        transport: Transport-level options (for example, pool size, keep-alive).
        tls: TLS/certificate options.
        proxy: Proxy options.
        provider_defaults: Additional provider-native construction defaults.
    """

    endpoint: str | None = None
    credential_ref: str | None = None
    transport: Mapping[str, Any] = field(default_factory=dict)
    tls: Mapping[str, Any] = field(default_factory=dict)
    proxy: Mapping[str, Any] = field(default_factory=dict)
    provider_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject embedded secret-bearing keys and freeze nested mappings."""
        for mapping in (self.transport, self.tls, self.proxy, self.provider_defaults):
            validate_metadata(mapping)
        object.__setattr__(self, "transport", freeze_metadata(self.transport))
        object.__setattr__(self, "tls", freeze_metadata(self.tls))
        object.__setattr__(self, "proxy", freeze_metadata(self.proxy))
        object.__setattr__(self, "provider_defaults", freeze_metadata(self.provider_defaults))

    def is_default(self) -> bool:
        """Return ``True`` when no connection-affecting field has been set."""
        return self == ProviderClientConfig()


@dataclass(frozen=True, slots=True)
class ProviderTypeRegistration:
    """Immutable binding between a provider type and its trusted factory."""

    provider_type: ProviderType
    factory: ProviderFactory = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """Runtime-owned or caller-owned model registration."""

    reference: ModelReference
    provider: str
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)
    available: bool | Callable[[], bool] = field(default=True, repr=False, compare=False)
    ownership: ProviderOwnership = ProviderOwnership.CALLER_OWNED
    provider_type: ProviderType | None = field(default=None, compare=False)
    available_timeout: float = field(default=DEFAULT_AVAILABILITY_TIMEOUT_SECONDS, compare=False)


@dataclass(frozen=True, slots=True)
class ProviderCleanupFailure:
    """Safe diagnostic for one provider client that did not close.

    Attributes:
        references: Model references that intentionally shared the client.
        provider: Provider identity declared by the binding.
        cause_type: Local exception class name without exception text.
        timed_out: Whether the configured cleanup bound expired.
    """

    references: tuple[ModelReference, ...]
    provider: str
    cause_type: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCleanupReport:
    """Immutable aggregate result of one provider cleanup pass."""

    closed: tuple[ModelReference, ...] = ()
    skipped_caller_owned: tuple[ModelReference, ...] = ()
    failures: tuple[ProviderCleanupFailure, ...] = ()


@dataclass(slots=True)
class _ProviderClientRecord:
    """Private identity-based lifecycle state for one unique client."""

    client: ModelProvider
    ownership: ProviderOwnership
    references: set[ModelReference] = field(default_factory=set)
    retired_references: set[ModelReference] = field(default_factory=set)
    leases: int = 0
    close_started: bool = False
    closed: bool = False

    def identities(self) -> tuple[ModelReference, ...]:
        """Return every known reference in deterministic order."""
        return tuple(sorted(self.references | self.retired_references, key=lambda item: item.value))


class ProviderLease:
    """One accepted runtime-mediated use of a provider binding.

    Leases are private runtime coordination objects. Agents and run contexts
    borrow bindings; they never receive a lease or own a client.
    """

    def __init__(self, registry: ProviderRegistry, client_id: int) -> None:
        self._registry = registry
        self._client_id = client_id
        self._released = False

    async def release(self) -> None:
        """Release this accepted-use lease and retire eligible clients."""
        if self._released:
            return
        self._released = True
        await self._registry._release_lease(self._client_id)


@dataclass(frozen=True, slots=True)
class ModelBindingSnapshot:
    """Safe, credential-free metadata describing one published model binding.

    Snapshots never carry the client, factory, or provider-specific
    connection configuration; they exist purely for inspection, logging, and
    diagnostics.
    """

    reference: ModelReference
    provider: str
    provider_type: ProviderType | None
    ownership: ProviderOwnership
    available: bool


@dataclass(frozen=True, slots=True)
class ProviderRegistrySnapshot:
    """Immutable, deterministically ordered view of the registry contents."""

    models: tuple[ModelBindingSnapshot, ...]
    provider_types: tuple[ProviderType, ...]


def _validate_client_structure(client: ModelProvider, provider_label: ProviderType | str) -> None:
    """Reject clients that do not satisfy the structural provider protocol."""
    if not hasattr(client, "capabilities") or not callable(getattr(client, "complete", None)):
        raise ProviderClientValidationError(
            f"Provider client for '{provider_label}' does not satisfy the structural "
            "provider protocol (missing capabilities or a callable complete())"
        )


def _evaluate_available(available: bool | Callable[[], bool], timeout: float) -> bool:
    """Evaluate a fixed or bounded callable availability state.

    A callable predicate never runs inline on the calling thread; it is
    submitted to a bounded worker pool and given at most ``timeout`` seconds
    to complete. A predicate that raises, hangs, or exceeds the timeout is
    treated as unavailable rather than propagating or blocking indefinitely,
    so a slow or hung health check can never stall resolution, listing, or
    snapshotting.
    """
    if not callable(available):
        return bool(available)
    future = _AVAILABILITY_EXECUTOR.submit(available)
    try:
        return bool(future.result(timeout=timeout))
    except Exception:  # noqa: BLE001 - bounded predicate, never propagate or block
        return False


def _to_snapshot(registration: ProviderRegistration) -> ModelBindingSnapshot:
    """Project a registration into safe, credential-free snapshot metadata."""
    return ModelBindingSnapshot(
        registration.reference,
        registration.provider,
        registration.provider_type,
        registration.ownership,
        _evaluate_available(registration.available, registration.available_timeout),
    )


def _provider_label(record: _ProviderClientRecord) -> str:
    """Return a safe provider identity from the record's known bindings."""
    return type(record.client).__name__


def _minimum_timeout(first: float | None, second: float | None) -> float | None:
    """Return the stricter non-negative timeout, if either exists."""
    if second is not None and second <= 0:
        raise TimeoutError
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


async def _close_client(client: ModelProvider, timeout: float | None) -> None:
    """Close a client through its optional structural cleanup protocol."""
    if isinstance(client, AsynchronouslyClosableProvider):
        close = client.aclose()
    elif isinstance(client, SynchronouslyClosableProvider):
        client.close()
        return
    else:
        return
    if timeout is None:
        await close
        return
    await asyncio.wait_for(close, timeout)


class ProviderRegistry:
    """Thread-safe registry mapping provider types and model references to bindings.

    Registry mutation and resolution hold a lock only for the brief dictionary
    update itself; factory construction and availability evaluation happen
    outside the lock so they never block concurrent resolution or unrelated
    registrations.
    """

    def __init__(self) -> None:
        self._provider_types: dict[ProviderType, ProviderTypeRegistration] = {}
        self._registrations: dict[ModelReference, ProviderRegistration] = {}
        self._generations: dict[ModelReference, int] = {}
        self._lock = threading.RLock()
        self._lease_condition = threading.Condition(self._lock)
        self._clients: dict[int, _ProviderClientRecord] = {}
        self._closing = False
        self._shutdown_report: ProviderCleanupReport | None = None

    @property
    def closed(self) -> bool:
        """Return whether shutdown has begun and new provider use is rejected."""
        with self._lock:
            return self._closing

    def _require_open(self) -> None:
        """Reject mutations and new resolution after shutdown begins."""
        if self._closing:
            raise RuntimeClosedError("Provider registry is shutting down")

    # ---- provider type registration --------------------------------------------------

    def register_provider_type(
        self,
        provider_type: ProviderType | str,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a trusted, allowlisted factory for one provider type.

        Args:
            provider_type: Normalized identifier for the provider family.
            factory: Structural factory implementing ``create(configuration)``.
            replace: Whether to replace an existing factory registration.

        Raises:
            ProviderFactoryValidationError: If the factory lacks a callable ``create``.
            DuplicateProviderTypeError: If already registered and replace is False.
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
        """Remove a provider type factory registration.

        Existing published model bindings continue to resolve normally;
        only later :meth:`register_provider` calls for this provider type
        are affected.

        Args:
            provider_type: Normalized identifier for the provider family.

        Raises:
            UnknownProviderTypeError: If no factory is registered for the type.
        """
        normalized = normalize_provider_type(provider_type)
        with self._lock:
            self._require_open()
            if self._provider_types.pop(normalized, None) is None:
                raise UnknownProviderTypeError(f"Unknown provider type '{normalized}'")

    def _get_factory(self, provider_type: ProviderType) -> ProviderFactory:
        with self._lock:
            self._require_open()
            registration = self._provider_types.get(provider_type)
        if registration is None:
            raise UnknownProviderTypeError(f"Unknown provider type '{provider_type}'")
        return registration.factory

    # ---- configuration-owned construction --------------------------------------------

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
        """Construct and publish a provider binding from one typed configuration source.

        The factory is invoked exactly once, outside any registry lock, so a
        slow or failing construction never blocks concurrent resolution or
        unrelated registrations. The registry is only mutated after
        construction, structural validation, and capability validation all
        succeed, so other runs never observe a partially constructed binding.

        A duplicate reference is rejected before construction starts so a
        factory is never invoked for a registration that cannot be
        published. The reference's mutation generation is also captured
        before construction starts; if another thread replaces or
        deregisters this exact reference while construction is still in
        flight, the now-stale result is discarded with
        :class:`StaleProviderConstructionError` instead of silently
        overwriting the newer binding or resurrecting a deregistered
        reference. A final atomic duplicate check still runs at publish time
        to close the race between the preflight check and construction.

        Args:
            reference: Credential-free model reference to bind.
            provider_type: Registered provider type whose factory constructs the client.
            configuration: Typed, provider-neutral construction inputs (endpoint,
                authentication reference, transport/TLS/proxy options, provider
                defaults).
            model_configuration: Safe, provider-neutral model configuration stored
                on the published binding.
            available: Optional availability predicate or fixed state.
            available_timeout: Bound, in seconds, on how long a callable
                ``available`` predicate may run before being treated as
                unavailable.
            ownership: Lifecycle ownership recorded for Story 6.3 shutdown.
                Defaults to runtime-owned, since the registry constructed the client.
            replace: Whether to replace an existing registration for this reference.
            required_capabilities: Capabilities the constructed client must advertise.

        Returns:
            The published, immutable provider registration.

        Raises:
            UnknownProviderTypeError: If the provider type has no registered factory.
            ProviderTypeMismatchError: If the model configuration provider does not
                match the provider type.
            DuplicateModelReferenceError: If already registered and replace is False.
            ProviderConstructionError: If the factory raises while constructing the client.
            ProviderClientValidationError: If the constructed client fails structural
                validation.
            IncompatibleProviderCapabilitiesError: If a required capability is missing.
            StaleProviderConstructionError: If the reference was replaced or
                deregistered while construction was in flight.
        """
        model_reference = normalize_reference(reference)
        assert model_reference is not None
        normalized_type = normalize_provider_type(provider_type)
        if model_configuration.provider.strip().lower() != str(normalized_type):
            raise ProviderTypeMismatchError(
                f"Model configuration provider '{model_configuration.provider}' does not "
                f"match provider type '{normalized_type}'"
            )
        with self._lock:
            if model_reference in self._registrations and not replace:
                raise DuplicateModelReferenceError(
                    f"Model reference '{model_reference}' is already registered"
                )
            expected_generation = self._generations.get(model_reference, 0)
        factory = self._get_factory(normalized_type)
        try:
            client = factory.create(configuration)
        except Exception as exc:
            raise ProviderConstructionError(
                f"Provider type '{normalized_type}' failed to construct a client for "
                f"model reference '{model_reference}'"
            ) from exc
        _validate_client_structure(client, normalized_type)
        validate_provider_capabilities(model_reference, client.capabilities, required_capabilities)
        registration = ProviderRegistration(
            model_reference,
            model_configuration.provider,
            client,
            model_configuration,
            available,
            ownership,
            normalized_type,
            available_timeout,
        )
        self._publish(
            model_reference, registration, replace=replace, expected_generation=expected_generation
        )
        return registration

    # ---- preconstructed client --------------------------------------------------------

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
        """Register a preconstructed, structurally validated provider client.

        Args:
            reference: Credential-free model reference to bind.
            client: An object satisfying the Story 6.1 structural provider protocol.
            configuration: Safe, provider-neutral model configuration stored on the
                published binding. Its ``endpoint`` must be left unset; the client
                already embeds whatever endpoint it was constructed with.
            ownership: Whether the caller retains ownership or transfers it to the
                runtime for Story 6.3 shutdown. Defaults to caller-owned, since the
                caller constructed the client.
            connection_config: Must be omitted or left at all defaults. Supplying
                connection-affecting values alongside a ready client is rejected
                since the client was already constructed with its own settings.
            available: Optional availability predicate or fixed state.
            available_timeout: Bound, in seconds, on how long a callable
                ``available`` predicate may run before being treated as
                unavailable.
            replace: Whether to replace an existing registration for this reference.
            required_capabilities: Capabilities the client must advertise.

        Returns:
            The published, immutable provider registration.

        Raises:
            ContradictoryProviderConfigurationError: If connection_config carries
                non-default endpoint, credential, proxy, TLS, or transport values,
                or if configuration.endpoint is set.
            ProviderClientValidationError: If the client fails structural validation.
            IncompatibleProviderCapabilitiesError: If a required capability is missing.
            DuplicateModelReferenceError: If already registered and replace is False.
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
        self._publish(model_reference, registration, replace=replace)
        return registration

    # ---- legacy compatibility ----------------------------------------------------------

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

        Deprecated:
            Use :meth:`register_client` for preconstructed clients or
            :meth:`register_provider` for factory-constructed clients instead.
            This method assumes caller ownership and no connection
            configuration; it is kept only as a compatibility path and will be
            removed once Story 6.3 lands.

        Args:
            reference: The model reference to bind to the client.
            client: Provider client that implements the model protocol.
            configuration: Provider-neutral configuration for the model.
            available: Optional availability predicate or fixed state.
            replace: Whether to replace an existing registration.

        Raises:
            DuplicateModelReferenceError: If the reference already exists and
                replacement is disabled.
        """
        warnings.warn(
            "ProviderRegistry.register(...) is deprecated; use register_client(...) for "
            "preconstructed clients or register_provider(...) for factory-constructed "
            "clients instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.register_client(
            reference,
            client,
            configuration,
            ownership=ProviderOwnership.CALLER_OWNED,
            available=available,
            replace=replace,
        )

    # ---- resolution and deregistration ---------------------------------------------------

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
            self._require_open()
            registration = self._registrations.get(model_reference)
        if registration is None:
            raise UnknownModelReferenceError(f"Unknown model reference '{model_reference}'")
        if not _evaluate_available(registration.available, registration.available_timeout):
            raise ProviderUnavailableError(
                f"Provider for model reference '{model_reference}' is unavailable"
            )
        return registration

    def deregister_model(self, reference: ModelReference | str) -> None:
        """Remove a published model binding.

        Deregistration prevents later resolution but does not invalidate a
        binding already returned by a prior, in-flight call to :meth:`resolve`.

        Args:
            reference: Model reference to remove.

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

    def acquire(self, registration: ProviderRegistration) -> ProviderLease:
        """Accept one runtime-mediated use of an already-resolved binding.

        The binding may have been replaced or deregistered after resolution.
        It remains valid for this call because its immutable client record is
        retained until the lease is released.

        Args:
            registration: Immutable registration selected by model resolution.

        Returns:
            A lease that must be released after the model call completes.

        Raises:
            RuntimeClosedError: If shutdown began before this call was accepted.
        """
        client_id = id(registration.client)
        with self._lock:
            self._require_open()
            record = self._clients.get(client_id)
            if record is None or record.client is not registration.client or record.closed:
                raise RuntimeClosedError("Provider binding is no longer usable")
            record.leases += 1
        return ProviderLease(self, client_id)

    async def _release_lease(self, client_id: int) -> None:
        """Release one lease and close a now-retired owned client."""
        with self._lock:
            record = self._clients.get(client_id)
            if record is None:
                return
            record.leases -= 1
            if record.leases < 0:
                raise RuntimeError("Provider client lease released more than once")
            self._lease_condition.notify_all()
        await self._close_records(self._retired_records())

    async def aclose(
        self,
        *,
        timeout: float | None = 30.0,
        per_client_timeout: float | None = 10.0,
        max_concurrency: int = 4,
    ) -> ProviderCleanupReport:
        """Stop resolution, drain accepted calls, and close runtime-owned clients.

        Args:
            timeout: Aggregate grace period for accepted provider calls and cleanup.
            per_client_timeout: Maximum time assigned to one asynchronous close.
            max_concurrency: Maximum number of clients closed simultaneously.

        Returns:
            A safe aggregate cleanup report when every owned client closed.

        Raises:
            ProviderShutdownError: If any owned client remains leased, times out,
                or raises while closing.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when supplied")
        if per_client_timeout is not None and per_client_timeout <= 0:
            raise ValueError("per_client_timeout must be positive when supplied")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        with self._lock:
            if self._shutdown_report is not None:
                if self._shutdown_report.failures:
                    raise ProviderShutdownError(self._shutdown_report)
                return self._shutdown_report
            self._closing = True
        deadline = time.monotonic() + timeout if timeout is not None else None
        await asyncio.to_thread(self._wait_for_leases, deadline)
        report = await self._close_records(
            self._shutdown_records(),
            per_client_timeout=per_client_timeout,
            max_concurrency=max_concurrency,
            deadline=deadline,
        )
        with self._lock:
            self._shutdown_report = report
        if report.failures:
            raise ProviderShutdownError(report)
        return report

    def _wait_for_leases(self, deadline: float | None) -> None:
        """Wait synchronously for accepted calls without blocking an event loop."""
        with self._lease_condition:
            while any(record.leases for record in self._clients.values()):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return
                self._lease_condition.wait(remaining)

    def _record_for(self, registration: ProviderRegistration) -> _ProviderClientRecord:
        """Return the stable identity record and reject ownership changes."""
        client_id = id(registration.client)
        record = self._clients.get(client_id)
        if record is None:
            record = _ProviderClientRecord(registration.client, registration.ownership)
            self._clients[client_id] = record
        elif record.client is not registration.client:
            raise RuntimeError("Provider client identity collision")
        elif record.ownership is not registration.ownership:
            raise ProviderOwnershipError(
                "A provider client cannot be rebound with different lifecycle ownership"
            )
        return record

    def _detach(self, reference: ModelReference, registration: ProviderRegistration) -> None:
        """Mark a binding as retired while preserving any accepted-use leases."""
        record = self._clients[id(registration.client)]
        record.references.discard(reference)
        record.retired_references.add(reference)

    def _retired_records(self) -> tuple[_ProviderClientRecord, ...]:
        """Collect owned records no longer bound and no longer leased."""
        with self._lock:
            return tuple(
                record
                for record in self._clients.values()
                if (
                    record.ownership is ProviderOwnership.RUNTIME_OWNED
                    and not record.references
                    and not record.leases
                    and not record.closed
                    and not record.close_started
                )
            )

    def _shutdown_records(self) -> tuple[_ProviderClientRecord, ...]:
        """Collect every record in deterministic client identity order."""
        with self._lock:
            return tuple(
                sorted(
                    self._clients.values(),
                    key=lambda item: tuple(reference.value for reference in item.identities()),
                )
            )

    async def _close_records(
        self,
        records: tuple[_ProviderClientRecord, ...],
        *,
        per_client_timeout: float | None = 10.0,
        max_concurrency: int = 4,
        deadline: float | None = None,
    ) -> ProviderCleanupReport:
        """Close eligible records, retaining safe failures for the caller."""
        del max_concurrency
        closed: list[ModelReference] = []
        skipped: list[ModelReference] = []
        failures: list[ProviderCleanupFailure] = []
        for record in records:
            identities = record.identities()
            if record.ownership is ProviderOwnership.CALLER_OWNED:
                skipped.extend(identities)
                continue
            with self._lock:
                if record.leases:
                    failures.append(
                        ProviderCleanupFailure(
                            identities, _provider_label(record), "TimeoutError", timed_out=True
                        )
                    )
                    continue
                if record.closed or record.close_started:
                    continue
                record.close_started = True
            try:
                remaining = None if deadline is None else deadline - time.monotonic()
                limit = _minimum_timeout(per_client_timeout, remaining)
                await _close_client(record.client, limit)
            except TimeoutError:
                failures.append(
                    ProviderCleanupFailure(
                        identities, _provider_label(record), "TimeoutError", timed_out=True
                    )
                )
                with self._lock:
                    record.close_started = False
            except Exception as error:
                failures.append(
                    ProviderCleanupFailure(
                        identities, _provider_label(record), type(error).__name__
                    )
                )
                with self._lock:
                    record.close_started = False
            else:
                with self._lock:
                    record.closed = True
                closed.extend(identities)
        return ProviderCleanupReport(
            tuple(sorted(set(closed), key=lambda item: item.value)),
            tuple(sorted(set(skipped), key=lambda item: item.value)),
            tuple(failures),
        )

    def _publish(
        self,
        reference: ModelReference,
        registration: ProviderRegistration,
        *,
        replace: bool,
        expected_generation: int | None = None,
    ) -> None:
        """Atomically insert a registration, honoring duplicate/replace/staleness rules.

        Args:
            reference: Model reference being published.
            registration: Fully constructed and validated registration to publish.
            replace: Whether to replace an existing registration for this reference.
            expected_generation: Mutation generation observed before construction
                started. When provided and it no longer matches the reference's
                current generation, another thread replaced or deregistered this
                reference while construction was in flight, and the stale result
                is rejected instead of being published.
        """
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

    # ---- snapshots -----------------------------------------------------------------------

    def _collect(
        self,
    ) -> tuple[tuple[ProviderRegistration, ...], tuple[ProviderType, ...]]:
        """Capture registrations and provider types from one lock acquisition."""
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
        """Return an immutable, deterministically ordered view of the registry.

        Both halves of the snapshot are captured from a single lock
        acquisition, so a concurrent mutation cannot combine model bindings
        and provider types that never coexisted.
        """
        registrations, types = self._collect()
        models = tuple(
            sorted(
                (_to_snapshot(item) for item in registrations),
                key=lambda item: item.reference.value,
            )
        )
        provider_types = tuple(sorted(types, key=str))
        return ProviderRegistrySnapshot(models, provider_types)
