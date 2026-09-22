"""Intentional public API for provider registration and lifecycle management.

Use ``register_provider`` for factory construction and ``register_client`` for
preconstructed clients. Resolution only selects already-published bindings.
"""

from .availability import DEFAULT_AVAILABILITY_TIMEOUT_SECONDS
from .configuration import ProviderClientConfig, ProviderFactory, ProviderTypeRegistration
from .ownership import (
    ProviderCleanupFailure,
    ProviderCleanupReport,
    ProviderLease,
    ProviderOwnership,
)
from .registration import ProviderRegistration
from .registry import ProviderRegistry
from .snapshots import ModelBindingSnapshot, ProviderRegistrySnapshot

__all__ = [
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
]
