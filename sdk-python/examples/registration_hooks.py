"""Typed deployment startup and shutdown hooks; hosts own scheduling and readiness.

Install ``conducto-ai[registration]``. Construct ``RegistrationClient`` with an
application-owned authenticated HTTP client and token provider. Do not print
wire requests, access tokens, or admission handles.
"""

from __future__ import annotations

import time
import uuid

from conducto.registration import (
    DeregisterRequest,
    DrainRequest,
    RegisterRequest,
    RegistrationResult,
    RenewRequest,
)


def startup_request(*, instance_id: str, card_url: str) -> RegisterRequest:
    """Create a deployment request once; preserve it verbatim for bounded retries."""
    return RegisterRequest(
        owner="payments",
        environment="production",
        agent_id="payments.invoice",
        instance_id=instance_id,
        agent_card_url=card_url,
        deployment_id="invoice-release-42",
        provenance="build-42",
        idempotency_key=str(uuid.uuid4()),
        correlation_id=str(uuid.uuid4()),
        issued_at=time.time(),
        lease_seconds=60,
    )


def renewal_request(
    initial: RegisterRequest, admission: RegistrationResult, current: RegistrationResult
) -> RenewRequest:
    """Construct the next renewal only after the previous generation is confirmed.

    Keep ``admission`` private: only it contains the lease handle. Keep the returned
    request unchanged until its outcome is known. Never use a new key to retry a
    timeout. Refresh credentials through the client's token provider instead.
    """
    if admission.lease_handle is None or not current.ready:
        raise ValueError("renewal requires confirmed active admission")
    return RenewRequest(
        owner=initial.owner,
        environment=initial.environment,
        agent_id=initial.agent_id,
        instance_id=initial.instance_id,
        lease_handle=admission.lease_handle,
        expected_generation=current.generation,
        lease_seconds=initial.lease_seconds,
        idempotency_key=str(uuid.uuid4()),
        correlation_id=initial.correlation_id,
        issued_at=time.time(),
    )


def drain_request(renewal: RenewRequest) -> DrainRequest:
    """Prepare drain before waiting for in-flight work; also clear local readiness.

    ``renewal`` carries the latest confirmed generation, not an already-submitted
    renewal's old generation. Retain the returned request for ambiguous outcomes;
    a failed drain is not permission to declare success.
    """
    return DrainRequest(
        **renewal.model_dump(
            exclude={"operation", "lease_seconds", "idempotency_key", "issued_at"}
        ),
        lease_handle=renewal.lease_handle,
        idempotency_key=str(uuid.uuid4()),
        issued_at=time.time(),
    )


def shutdown_request(renewal: RenewRequest, drained: RegistrationResult) -> DeregisterRequest:
    """Prepare removal after the host has finished its bounded in-flight drain."""
    if not drained.ok or drained.state != "draining":
        raise ValueError("deregistration requires a confirmed drain")
    return DeregisterRequest(
        **renewal.model_dump(
            exclude={
                "operation",
                "lease_seconds",
                "idempotency_key",
                "expected_generation",
                "issued_at",
            }
        ),
        lease_handle=renewal.lease_handle,
        expected_generation=drained.generation,
        idempotency_key=str(uuid.uuid4()),
        issued_at=time.time(),
    )
