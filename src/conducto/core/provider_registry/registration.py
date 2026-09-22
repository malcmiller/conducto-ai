"""Immutable model-to-client bindings published by a provider registry."""

from collections.abc import Callable
from dataclasses import dataclass, field

from ..model_config import ModelReference, ProviderType
from ..provider import ModelConfiguration, ModelProvider
from .availability import DEFAULT_AVAILABILITY_TIMEOUT_SECONDS
from .ownership import ProviderOwnership


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """An immutable runtime-owned or caller-owned model registration."""

    reference: ModelReference
    provider: str
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)
    available: bool | Callable[[], bool] = field(default=True, repr=False, compare=False)
    ownership: ProviderOwnership = ProviderOwnership.CALLER_OWNED
    provider_type: ProviderType | None = field(default=None, compare=False)
    available_timeout: float = field(default=DEFAULT_AVAILABILITY_TIMEOUT_SECONDS, compare=False)
