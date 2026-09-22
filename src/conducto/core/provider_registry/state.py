"""Single synchronization domain shared by registry implementation components."""

import asyncio
import threading

from ..model_config import ModelReference, ProviderType
from ..runtime_errors import RuntimeClosedError
from .configuration import ProviderTypeRegistration
from .ownership import ProviderCleanupReport, _ProviderClientRecord
from .registration import ProviderRegistration


class _RegistryState:
    """Own the lock and state used by cooperative registry implementation bases.

    Every component operates on this same instance and reentrant lock.
    Construction, health predicates, and client cleanup run outside that lock.
    """

    def __init__(self) -> None:
        self._provider_types: dict[ProviderType, ProviderTypeRegistration] = {}
        self._registrations: dict[ModelReference, ProviderRegistration] = {}
        self._generations: dict[ModelReference, int] = {}
        self._lock = threading.RLock()
        self._lease_condition = threading.Condition(self._lock)
        self._clients: dict[int, _ProviderClientRecord] = {}
        self._active_constructions: dict[int, tuple[ModelReference, str]] = {}
        self._next_construction_id = 0
        self._cleanup_owners: set[asyncio.Task[None]] = set()
        self._background_cleanups: set[asyncio.Task[ProviderCleanupReport]] = set()
        self._closing = False
        self._shutdown_report: ProviderCleanupReport | None = None
        self._shutdown_task: asyncio.Task[ProviderCleanupReport] | None = None

    @property
    def closed(self) -> bool:
        """Return whether shutdown has begun and new provider use is rejected."""
        with self._lock:
            return self._closing

    def _require_open(self) -> None:
        """Reject mutations and new resolution after shutdown begins."""
        if self._closing:
            raise RuntimeClosedError("Provider registry is shutting down")
