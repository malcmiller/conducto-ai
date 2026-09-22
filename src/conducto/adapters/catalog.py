"""Safe metadata for optional first-party provider adapter dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Compatibility metadata for one first-party adapter family.

    Attributes:
        name: Stable adapter family identifier.
        extra: Conducto package extra that installs its provider SDK.
        distribution: Provider SDK distribution name checked at selection.
        version_specifier: Compatible provider SDK version range.
        protocol_version: Provider protocol/API profile supported by the adapter.
    """

    name: str
    extra: str
    distribution: str
    version_specifier: str
    protocol_version: str


class AdapterDependencyError(RuntimeError):
    """An explicitly selected adapter is unavailable in this environment."""

    def __init__(self, spec: AdapterSpec) -> None:
        """Describe the compatible extra without exposing environment details."""
        super().__init__(
            f"Adapter '{spec.name}' requires {spec.distribution}{spec.version_specifier}; "
            f"install conducto-ai[{spec.extra}]"
        )
        self.adapter = spec.name
        self.extra = spec.extra


@dataclass(frozen=True, slots=True)
class ExternalAdapterSpec:
    """Application allowlist entry for one external provider adapter.

    Attributes:
        name: Stable application-selected entry-point name.
        distribution: Distribution permitted to supply that entry point.
        version: Exact compatible distribution version for the application.
    """

    name: str
    distribution: str
    version: str


@dataclass(frozen=True, slots=True)
class DiscoveredExternalAdapter:
    """Metadata for an allowlisted external adapter that has not been loaded."""

    spec: ExternalAdapterSpec
    entry_point: metadata.EntryPoint


_ADAPTER_SPECS = (
    AdapterSpec(
        "microsoft-foundry",
        "microsoft-foundry",
        "azure-ai-projects",
        ">=1.0.0b11",
        "foundry-v1",
    ),
    AdapterSpec("ollama", "ollama", "ollama", ">=0.5", "ollama-v1"),
    AdapterSpec("openai", "openai", "openai", ">=1.0", "openai-compatible-v1"),
)


def available_adapter_specs() -> tuple[AdapterSpec, ...]:
    """Return supported adapter metadata in deterministic name order."""
    return _ADAPTER_SPECS


def require_adapter(name: str) -> AdapterSpec:
    """Validate an explicitly selected first-party adapter dependency.

    Args:
        name: One stable adapter family identifier from
            :func:`available_adapter_specs`.

    Returns:
        The adapter compatibility metadata. This function does not import the
        SDK or create a provider client.

    Raises:
        KeyError: If ``name`` is not a supported first-party adapter.
        AdapterDependencyError: If the selected adapter's SDK is absent or
            outside its declared compatibility range.
    """
    spec = next((item for item in _ADAPTER_SPECS if item.name == name), None)
    if spec is None:
        raise KeyError(f"Unknown first-party adapter '{name}'")
    try:
        installed_version = Version(metadata.version(spec.distribution))
    except (metadata.PackageNotFoundError, InvalidVersion) as error:
        raise AdapterDependencyError(spec) from error
    if installed_version not in SpecifierSet(spec.version_specifier):
        raise AdapterDependencyError(spec)
    return spec


def discover_external_adapters(
    allowlist: Mapping[str, ExternalAdapterSpec],
) -> tuple[DiscoveredExternalAdapter, ...]:
    """Discover compatible, application-allowlisted provider entry points.

    Discovery reads only installed package metadata. It neither imports an
    adapter nor executes any application-supplied module path. Applications
    must explicitly select one returned result with
    :func:`load_external_adapter`.

    Args:
        allowlist: Trusted application mapping keyed by entry-point name. Each
            value pins the distribution identity and compatible version.

    Returns:
        Compatible adapter metadata in deterministic name order.
    """
    candidates = metadata.entry_points(group="conducto.providers")
    discovered: list[DiscoveredExternalAdapter] = []
    for candidate in sorted(candidates, key=lambda item: item.name):
        expected = allowlist.get(candidate.name)
        distribution = candidate.dist
        if (
            expected is None
            or distribution is None
            or distribution.metadata["Name"] != expected.distribution
            or distribution.version != expected.version
        ):
            continue
        discovered.append(DiscoveredExternalAdapter(expected, candidate))
    return tuple(discovered)


def load_external_adapter(adapter: DiscoveredExternalAdapter) -> Any:
    """Load a previously discovered, allowlisted external adapter on selection.

    Args:
        adapter: Metadata returned by :func:`discover_external_adapters`.

    Returns:
        The object provided by the selected trusted entry point.
    """
    return adapter.entry_point.load()
