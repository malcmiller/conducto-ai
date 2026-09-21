"""Metadata-only catalog for first-party optional provider adapters.

Importing this package never imports a provider SDK, resolves credentials,
constructs a client, performs network I/O, or mutates the environment.
Concrete adapters are selected only by trusted application code and are
introduced independently of the core runtime.
"""

from .catalog import (
    AdapterDependencyError,
    AdapterSpec,
    DiscoveredExternalAdapter,
    ExternalAdapterSpec,
    available_adapter_specs,
    discover_external_adapters,
    load_external_adapter,
    require_adapter,
)

__all__ = [
    "AdapterDependencyError",
    "AdapterSpec",
    "DiscoveredExternalAdapter",
    "ExternalAdapterSpec",
    "available_adapter_specs",
    "discover_external_adapters",
    "load_external_adapter",
    "require_adapter",
]
