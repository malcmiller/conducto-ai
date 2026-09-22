"""Immutable deployment registration wire contracts, separate from A2A invocation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter

from conducto.core.catalog import DeploymentType

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Operation = Literal["register", "renew", "drain", "deregister", "status", "revoke"]


class RegistrationCode(StrEnum):
    """Safe, stable non-success reasons; no remote exception text is exposed."""

    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNAUTHENTICATED = "unauthenticated"
    UNAUTHORIZED = "unauthorized"
    IDENTITY_CONFLICT = "identity_conflict"
    ENDPOINT_MISMATCH = "endpoint_mismatch"
    INVALID_CARD = "invalid_card"
    CARD_UNAVAILABLE = "card_unavailable"
    CATALOG_UNAVAILABLE = "catalog_unavailable"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    INVALID_LEASE = "invalid_lease"
    INVALID_HANDLE = "invalid_handle"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    STALE_GENERATION = "stale_generation"
    REPLAY_REJECTED = "replay_rejected"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"
    CAPACITY_EXCEEDED = "capacity_exceeded"


class RequestBase(BaseModel):
    """Bounded request identity and freshness envelope.

    ``issued_at`` is epoch seconds from a synchronized deployment clock. Keys
    identify entire operations, not sessions. Generation zero means new instance.
    Correlation is diagnostic and excluded from idempotency fingerprints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    protocol_version: str = Field(default="1", max_length=16)
    owner: Identifier
    environment: Identifier
    agent_id: Identifier
    instance_id: Identifier
    idempotency_key: Identifier
    issued_at: float = Field(allow_inf_nan=False, ge=0, strict=True)
    expected_generation: int = Field(default=0, ge=0, strict=True)
    correlation_id: Identifier


class RegisterRequest(RequestBase):
    """Request admission using a remotely retrieved, policy-bound Agent Card.

    URLs and signatures are excluded from representation. Provenance is an
    opaque deployment reference, not a caller assertion of verified trust.
    """

    operation: Literal["register"] = "register"
    agent_card_url: str = Field(min_length=1, max_length=2048, repr=False)
    deployment_id: Identifier
    deployment_type: DeploymentType = DeploymentType.REMOTE_CONTAINER
    provenance: Identifier
    lease_seconds: float = Field(default=60, gt=0, allow_inf_nan=False, strict=True)
    signature: SecretStr | None = Field(default=None, repr=False, exclude=True, max_length=8192)


class LeaseRequest(RequestBase):
    """Identity-bound operation requiring a confidential opaque lease handle."""

    lease_handle: SecretStr = Field(repr=False, exclude=True, min_length=1, max_length=256)


class RenewRequest(LeaseRequest):
    """Renew the current active generation without changing identity or metadata."""

    operation: Literal["renew"] = "renew"
    lease_seconds: float = Field(default=60, gt=0, allow_inf_nan=False, strict=True)


class DrainRequest(LeaseRequest):
    """Stop new catalog selection while existing invocations finish."""

    operation: Literal["drain"] = "drain"


class DeregisterRequest(LeaseRequest):
    """Remove one instance permanently; restarting requires a new instance ID."""

    operation: Literal["deregister"] = "deregister"


class StatusRequest(LeaseRequest):
    """Read current state; this operation never extends a lease."""

    operation: Literal["status"] = "status"


class RevokeRequest(RequestBase):
    """Administrator-only terminal revocation of one instance, without its handle."""

    operation: Literal["revoke"] = "revoke"


RegistrationRequest = Annotated[
    RegisterRequest
    | RenewRequest
    | DrainRequest
    | DeregisterRequest
    | StatusRequest
    | RevokeRequest,
    Field(discriminator="operation"),
]
REQUEST_ADAPTER: TypeAdapter[RegistrationRequest] = TypeAdapter(RegistrationRequest)


class RegistrationResult(BaseModel):
    """Safe operation outcome and an explicitly confidential admission grant.

    Normal serialization and repr exclude the handle. Only the authenticated
    transport response may use ``result_document`` to deliver it to its owner.
    ``ready`` describes admission, not application readiness or remote health.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    code: RegistrationCode
    correlation_id: str = ""
    generation: int = Field(default=0, ge=0)
    state: Literal["active", "draining", "removed", "revoked", "expired"] | None = None
    lease_expires_at: float | None = Field(default=None, allow_inf_nan=False)
    lease_handle: SecretStr | None = Field(default=None, repr=False, exclude=True)

    @property
    def ok(self) -> bool:
        """Return whether the requested operation succeeded."""
        return self.code is RegistrationCode.OK

    @property
    def ready(self) -> bool:
        """Return whether this successful response represents active admission."""
        return self.ok and self.state == "active" and self.lease_expires_at is not None

    def is_ready_at(self, now: float) -> bool:
        """Gate local readiness against an epoch clock as well as admission state.

        Hosts must still clear readiness on failed renewal or revocation. This
        snapshot cannot predict a later administrative policy change.
        """
        return self.ready and self.lease_expires_at is not None and now < self.lease_expires_at


def request_document(request: RegistrationRequest) -> dict[str, Any]:
    """Encode confidential transport input; never send this document to telemetry."""
    document = request.model_dump(mode="json")
    if isinstance(request, LeaseRequest):
        document["lease_handle"] = request.lease_handle.get_secret_value()
    if isinstance(request, RegisterRequest) and request.signature is not None:
        document["signature"] = request.signature.get_secret_value()
    return document


def result_document(result: RegistrationResult) -> dict[str, Any]:
    """Encode an authenticated response, including its confidential admission grant."""
    document = result.model_dump(mode="json")
    if result.lease_handle is not None:
        document["lease_handle"] = result.lease_handle.get_secret_value()
    return document
