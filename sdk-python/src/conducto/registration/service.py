"""Authenticated deployment control plane backed by the authoritative catalog."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import SecretStr

from conducto.core.catalog import (
    AgentCatalog,
    CatalogEntry,
    CatalogManagedCommand,
    CatalogManagedResult,
    CatalogProviderUnavailableError,
    CatalogValidationError,
)
from conducto.core.logging import emit_event, log_context
from conducto.core.telemetry import (
    SPAN_REGISTRATION_SERVER,
    current_trace_ids,
    extract_trace_context,
    start_span,
)
from conducto.security.audit import (
    AuditCategory,
    AuditDecision,
    AuditDeliveryError,
    AuditEmitter,
    AuditEvent,
    AuditEventName,
    AuditOutcome,
)
from conducto.security.tokens import TokenValidationError, TokenValidator, ValidatedIdentity
from conducto.security.trust import TrustPolicy

from .models import (
    LeaseRequest,
    RegisterRequest,
    RegistrationCode,
    RegistrationRequest,
    RegistrationResult,
    RenewRequest,
    request_document,
)
from .policy import RegistrationGrant

if TYPE_CHECKING:
    import httpx

    from conducto.transport.a2a import DiscoveryPolicy


class RegistrationService:
    """Authenticate, authorize and discover; delegate every mutation to the catalog.

    Args:
        catalog: Shared catalog already consumed by running orchestrators.
        validator: Application-owned deployment bearer-token validator.
        trust_policy: A dedicated registration audience and operation scopes.
        grants: Exact application-owned identity and endpoint allowlist.
        discovery_policy: Story 4 network, protocol, response-size and timeout policy.
        audit: Required attributable evidence sink; admission fails closed.
        http_client: Optional application-owned card retrieval client.
        clock: Epoch clock shared with the catalog and token validator.
        max_request_age: Bounded request replay window, in seconds.

    Notes:
        No implicit retries occur here. The in-memory catalog is the transaction
        and receipt owner; multiple services sharing it share idempotency.
        OAuth is mandatory. TLS, including optional mTLS, belongs to the server;
        certificate-bearing trust policies are rejected rather than bypassed.
    """

    def __init__(
        self,
        *,
        catalog: AgentCatalog,
        validator: TokenValidator,
        trust_policy: TrustPolicy,
        grants: Sequence[RegistrationGrant],
        discovery_policy: DiscoveryPolicy,
        audit: AuditEmitter,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
        max_request_age: float = 300,
    ) -> None:
        if not math.isfinite(max_request_age) or not 0 < max_request_age <= 3600:
            raise ValueError("max_request_age must be finite and between zero and 3600")
        if trust_policy.certificate is not None:
            raise ValueError("registration certificate validation belongs to the TLS server")
        if audit.policy.max_buffered_events:
            raise ValueError("registration evidence requires synchronous sink acknowledgement")
        grant_keys = {
            (
                grant.issuer,
                grant.subject_id,
                grant.owner,
                grant.environment,
                grant.agent_id,
                grant.agent_card_url,
            )
            for grant in grants
        }
        if len(grant_keys) != len(grants):
            raise ValueError("duplicate registration identity/card grants are ambiguous")
        self.catalog = catalog
        self._validator = validator
        self._trust_policy = trust_policy
        self._grants = tuple(grants)
        self._discovery_policy = discovery_policy
        self._audit = audit
        self._http_client = http_client
        self._clock = clock
        self._max_request_age = max_request_age
        self._expiry_lock = asyncio.Lock()
        self._pending_expiry: deque[tuple[str, str, CatalogManagedResult]] = deque()

    async def handle(
        self,
        request: RegistrationRequest,
        *,
        authorization_header: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> RegistrationResult:
        """Perform one bounded operation with safe typed failures and trace attribution.

        Cancellation propagates. Unexpected programming errors are not disguised
        as successful or retryable responses. After a lost response, callers must
        retry the identical request or read status, never invent a new generation.
        """
        extracted = extract_trace_context(trace_headers or {})
        with (
            log_context(correlation_id=request.correlation_id),
            start_span(
                SPAN_REGISTRATION_SERVER,
                kind="server",
                remote_context=extracted.context,
                attributes={
                    "conducto.correlation_id": request.correlation_id,
                    "conducto.registration.operation": request.operation,
                    "conducto.invalid_remote_context": extracted.invalid_remote_context,
                },
            ) as span,
        ):
            try:
                identity = self._authenticate(authorization_header)
                if identity is None:
                    result = self._failure(request, RegistrationCode.UNAUTHENTICATED)
                else:
                    result = await self._execute(request, identity)
                await self._evidence(request, result, identity)
            except AuditDeliveryError:
                result = self._failure(request, RegistrationCode.AUDIT_UNAVAILABLE)
            emit_event(
                "conducto.registration.completed.v1",
                outcome="success" if result.ok else "failure",
                error_category=result.code.value,
                agent_id=request.agent_id,
                instance_id=request.instance_id,
            )
            span.set_outcome("success" if result.ok else "denied", reason=result.code.value)
            return result

    def _authenticate(self, header: str | None) -> ValidatedIdentity | None:
        if not header or len(header) > 16_384 or not header.startswith("Bearer "):
            return None
        try:
            return self._validator.validate(header[7:], policy=self._trust_policy)
        except TokenValidationError:
            return None

    def _freshness(
        self, request: RegistrationRequest, identity: ValidatedIdentity
    ) -> RegistrationCode | None:
        now = self._clock()
        if identity.expires_at <= now or identity.not_before > now:
            return RegistrationCode.UNAUTHENTICATED
        if not 0 <= now - request.issued_at <= self._max_request_age:
            return RegistrationCode.REPLAY_REJECTED
        return None

    async def _execute(
        self, request: RegistrationRequest, identity: ValidatedIdentity
    ) -> RegistrationResult:
        if request.protocol_version != "1":
            return self._failure(request, RegistrationCode.UNSUPPORTED_VERSION)
        invalid = self._freshness(request, identity)
        if invalid is not None:
            return self._failure(request, invalid)
        grants = tuple(grant for grant in self._grants if grant.permits(identity, request))
        if not grants:
            return self._failure(request, RegistrationCode.UNAUTHORIZED)
        if isinstance(request, RegisterRequest):
            grants = tuple(
                grant for grant in grants if grant.agent_card_url == request.agent_card_url
            )
            if not grants:
                return self._failure(request, RegistrationCode.ENDPOINT_MISMATCH)
        grant = grants[0]
        if isinstance(request, RegisterRequest | RenewRequest):
            if request.lease_seconds > min(item.max_lease_seconds for item in grants):
                return self._failure(request, RegistrationCode.INVALID_LEASE)
        if (
            isinstance(request, RegisterRequest)
            and request.deployment_type not in grant.deployment_types
        ):
            return self._failure(request, RegistrationCode.UNAUTHORIZED)
        command = self._command(request, identity)
        await self._evidence(request, None, identity)
        try:
            cached = self.catalog.lookup_managed_request(command)
            if cached is not None:
                return self._result(request, cached)
            if isinstance(request, RegisterRequest):
                entry = await self._retrieve(request, grant)
                if isinstance(entry, RegistrationCode):
                    return self._failure(request, entry)
                command = replace(command, entry=entry)
            # Discovery and audit may have consumed the token/request validity window.
            invalid = self._freshness(request, identity)
            if invalid is not None:
                return self._failure(request, invalid)
            if isinstance(request, RegisterRequest | RenewRequest):
                if self._clock() + request.lease_seconds > identity.expires_at:
                    return self._failure(request, RegistrationCode.INVALID_LEASE)
            return self._result(request, self.catalog.manage_instance(command))
        except CatalogProviderUnavailableError:
            return self._failure(request, RegistrationCode.CATALOG_UNAVAILABLE)
        except CatalogValidationError:
            return self._failure(request, RegistrationCode.INVALID_CARD)

    async def _retrieve(
        self, request: RegisterRequest, grant: RegistrationGrant
    ) -> CatalogEntry | RegistrationCode:
        from conducto.transport.a2a import discover_agent
        from conducto.transport.errors import (
            CompatibilityError,
            DiscoveryError,
            LimitExceededError,
            ProtocolError,
        )

        try:
            async with asyncio.timeout(self._discovery_policy.request_timeout):
                descriptor = await discover_agent(
                    request.agent_card_url,
                    policy=self._discovery_policy,
                    http_client=self._http_client,
                    correlation_id=request.correlation_id,
                )
        except CompatibilityError:
            return RegistrationCode.UNSUPPORTED_VERSION
        except (ProtocolError, LimitExceededError, ValueError):
            return RegistrationCode.INVALID_CARD
        except (DiscoveryError, TimeoutError):
            return RegistrationCode.CARD_UNAVAILABLE
        if descriptor.endpoint_url != grant.endpoint_url:
            return RegistrationCode.ENDPOINT_MISMATCH
        if descriptor.name != grant.card_name:
            return RegistrationCode.IDENTITY_CONFLICT
        if (
            not descriptor.name
            or not descriptor.version
            or not descriptor.capabilities
            or any(not skill for skill in descriptor.capabilities)
            or len(set(descriptor.capabilities)) != len(descriptor.capabilities)
        ):
            return RegistrationCode.INVALID_CARD
        return CatalogEntry(
            agent_id=request.agent_id,
            instance_id=request.instance_id,
            owner=request.owner,
            deployment_type=request.deployment_type,
            agent_card_url=request.agent_card_url,
            agent_card=descriptor.card,
            provenance=request.provenance,
            trust_policy_ref=grant.trust_policy_ref,
            signature=request.signature.get_secret_value() if request.signature else None,
            supported_versions=frozenset({"1.0"}),
            transports=frozenset({"JSONRPC"}),
            lease_seconds=request.lease_seconds,
        )

    @staticmethod
    def _command(
        request: RegistrationRequest, identity: ValidatedIdentity
    ) -> CatalogManagedCommand:
        document = request_document(request)
        document.pop("correlation_id")
        digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return CatalogManagedCommand(
            operation=request.operation,
            agent_id=request.agent_id,
            instance_id=request.instance_id,
            owner=request.owner,
            environment=request.environment,
            subject_id=identity.subject,
            issuer=identity.issuer,
            idempotency_key=request.idempotency_key,
            fingerprint=digest,
            expected_generation=request.expected_generation,
            lease_seconds=request.lease_seconds
            if isinstance(request, RegisterRequest | RenewRequest)
            else 60,
            lease_handle=request.lease_handle.get_secret_value()
            if isinstance(request, LeaseRequest)
            else "",
            deployment_id=request.deployment_id if isinstance(request, RegisterRequest) else "",
            provenance=request.provenance if isinstance(request, RegisterRequest) else "",
        )

    @staticmethod
    def _result(request: RegistrationRequest, result: CatalogManagedResult) -> RegistrationResult:
        return RegistrationResult.model_validate(
            {
                "code": RegistrationCode(result.code.value),
                "correlation_id": request.correlation_id,
                "generation": result.generation,
                "state": result.state.value if result.state is not None else None,
                "lease_expires_at": result.lease_expires_at,
                "lease_handle": SecretStr(result.lease_handle) if result.lease_handle else None,
            }
        )

    @staticmethod
    def _failure(request: RegistrationRequest, code: RegistrationCode) -> RegistrationResult:
        return RegistrationResult(code=code, correlation_id=request.correlation_id)

    async def _evidence(
        self,
        request: RegistrationRequest,
        result: RegistrationResult | None,
        identity: ValidatedIdentity | None,
    ) -> None:
        trace = current_trace_ids()
        await self._audit.emit(
            AuditEvent(
                event_name=AuditEventName.REGISTRATION_REQUESTED
                if result is None
                else AuditEventName.REGISTRATION_COMPLETED,
                category=AuditCategory.CATALOG,
                decision=AuditDecision.ALLOW if result is None or result.ok else AuditDecision.DENY,
                outcome=AuditOutcome.PENDING
                if result is None
                else AuditOutcome.SUCCESS
                if result.ok
                else AuditOutcome.REJECTED,
                reason_code=result.code.value if result else "authorized",
                subject_id=identity.subject if identity else "",
                issuer=identity.issuer if identity else "",
                agent_id=request.agent_id,
                resource=request.instance_id,
                correlation_id=request.correlation_id,
                policy_version=self._trust_policy.version,
                trace_id=trace.trace_id if trace else "",
                span_id=trace.span_id if trace else "",
                extensions={
                    "operation": request.operation,
                    "generation": result.generation if result else 0,
                },
            ),
            required=True,
        )

    async def expire(self) -> int:
        """Sweep fake-clock-compatible leases and emit original-principal attribution.

        Snapshots already exclude expired instances even if this scheduler is down.
        Audit/storage failures propagate; failed evidence stays pending for the
        next sweep on this service object with a stable sink-deduplication ID.
        Scheduling and supervision belong to the host.
        """
        async with self._expiry_lock:
            self._pending_expiry.extend(self.catalog.expire_managed_instances())
            delivered = 0
            while self._pending_expiry:
                agent_id, instance_id, result = self._pending_expiry[0]
                event_id = hashlib.sha256(
                    json.dumps([agent_id, instance_id, result.generation, "expired"]).encode()
                ).hexdigest()
                await self._audit.emit(
                    AuditEvent(
                        event_name=AuditEventName.REGISTRATION_EXPIRED,
                        category=AuditCategory.CATALOG,
                        decision=AuditDecision.NOT_APPLICABLE,
                        outcome=AuditOutcome.SUCCESS,
                        reason_code="lease_expired",
                        subject_id=result.subject_id,
                        issuer=result.issuer,
                        agent_id=agent_id,
                        resource=instance_id,
                        event_id=event_id,
                        extensions={"generation": result.generation},
                    ),
                    required=True,
                )
                self._pending_expiry.popleft()
                delivered += 1
            return delivered
