"""Immutable commands and outcomes for catalog-owned instance management."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from ._models import CatalogEntry


class CatalogInstanceState(StrEnum):
    """Lifecycle of a managed instance, independent of its logical siblings."""

    ACTIVE = "active"
    DRAINING = "draining"
    REMOVED = "removed"
    REVOKED = "revoked"
    EXPIRED = "expired"


class CatalogManagedCode(StrEnum):
    """Stable, transport-neutral managed-operation outcome codes."""

    OK = "ok"
    NOT_FOUND = "not_found"
    IDENTITY_CONFLICT = "identity_conflict"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STALE_GENERATION = "stale_generation"
    INVALID_LEASE = "invalid_lease"
    INVALID_HANDLE = "invalid_handle"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REPLAY_REJECTED = "replay_rejected"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass(frozen=True, slots=True)
class CatalogManagedCommand:
    """Validated, authenticated intent submitted by the registration service.

    The caller owns authentication, policy, and canonical request fingerprinting.
    The catalog owns admission, handle binding, generations, and replay fencing.
    ``expected_generation=0`` is reserved for a new instance. Revoke targets only
    this instance and accepts a separately authorized administrator principal.
    Renewal preserves any longer existing lease rather than shortening expiry.
    """

    operation: Literal["register", "renew", "drain", "deregister", "status", "revoke"]
    agent_id: str
    instance_id: str
    owner: str
    environment: str
    subject_id: str
    issuer: str
    idempotency_key: str
    fingerprint: str
    expected_generation: int
    lease_seconds: float = 60.0
    lease_handle: str = field(default="", repr=False)
    entry: CatalogEntry | None = field(default=None, repr=False)
    deployment_id: str = ""
    provenance: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class CatalogManagedResult:
    """Immutable outcome; only successful registration receipts contain a handle."""

    code: CatalogManagedCode
    generation: int = 0
    state: CatalogInstanceState | None = None
    lease_expires_at: float | None = None
    lease_handle: str = field(default="", repr=False)
    subject_id: str = ""
    issuer: str = ""
