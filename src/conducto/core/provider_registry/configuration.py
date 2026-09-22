"""Immutable, credential-free provider construction inputs and factory contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..model_config import ProviderType, freeze_metadata, validate_metadata
from ..provider import ModelProvider


class ProviderFactory(Protocol):
    """Construct a provider family through an explicitly registered factory.

    The registry never imports modules or executes configuration-supplied code.
    """

    def create(self, configuration: ProviderClientConfig) -> ModelProvider:
        """Construct a client from typed, provider-neutral connection inputs."""
        ...


@dataclass(frozen=True, slots=True)
class ProviderClientConfig:
    """Typed, provider-neutral construction inputs for a factory.

    Attributes:
        endpoint: Endpoint URL, or ``None`` to use the factory default.
        credential_ref: Opaque reference to an externally held credential,
            never the credential value.
        transport: Transport-level options.
        tls: TLS and certificate options.
        proxy: Proxy options.
        provider_defaults: Provider-native construction defaults.
    """

    endpoint: str | None = None
    credential_ref: str | None = None
    transport: Mapping[str, Any] = field(default_factory=dict)
    tls: Mapping[str, Any] = field(default_factory=dict)
    proxy: Mapping[str, Any] = field(default_factory=dict)
    provider_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject secret-bearing keys and freeze all nested mappings."""
        for mapping in (self.transport, self.tls, self.proxy, self.provider_defaults):
            validate_metadata(mapping)
        object.__setattr__(self, "transport", freeze_metadata(self.transport))
        object.__setattr__(self, "tls", freeze_metadata(self.tls))
        object.__setattr__(self, "proxy", freeze_metadata(self.proxy))
        object.__setattr__(self, "provider_defaults", freeze_metadata(self.provider_defaults))

    def is_default(self) -> bool:
        """Return whether every connection-affecting field is unset."""
        return self == ProviderClientConfig()


@dataclass(frozen=True, slots=True)
class ProviderTypeRegistration:
    """Immutable binding between a provider type and its trusted factory."""

    provider_type: ProviderType
    factory: ProviderFactory = field(repr=False, compare=False)
