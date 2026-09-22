"""In-process deployment control-plane acceptance without real network or clocks."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import replace
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256
from pydantic import ValidationError

from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.core.catalog import AgentCatalog, CatalogProviderUnavailableError
from conducto.core.telemetry import configure_in_memory_tracing
from conducto.registration import (
    DeregisterRequest,
    DrainRequest,
    RegisterRequest,
    RegistrationCode,
    RegistrationGrant,
    RegistrationService,
    RenewRequest,
    RevokeRequest,
    StatusRequest,
)
from conducto.registration.asgi import RegistrationASGI
from conducto.registration.client import RegistrationClient
from conducto.registration.models import REQUEST_ADAPTER, Operation, request_document
from conducto.security import (
    AudiencePolicy,
    ClockSkewPolicy,
    IssuerPolicy,
    JWTBearerTokenValidator,
    ScopePolicy,
    StaticJWKSResolver,
    TrustPolicy,
)
from conducto.security.audit import (
    AuditDeliveryError,
    AuditEmitter,
    FailingAuditSink,
    InMemoryAuditSink,
)
from conducto.transport.a2a import DiscoveryPolicy

pytestmark = pytest.mark.acceptance
CARD_URL = "http://127.0.0.1:8081/.well-known/agent-card.json"
ENDPOINT = "http://127.0.0.1:8081/a2a"
ISSUER = "test-issuer"
OPERATIONS = ("register", "renew", "drain", "deregister", "status", "revoke")


@a2a_agent(name="DeployedAgent", version="1.0.0", description="Deployment acceptance agent.")
class DeployedAgent(BaseAgent):
    """Loopback agent whose card is available before capability readiness."""

    @a2a_capability(name="echo", description="Echo a deployment test value.")
    def echo(self, value: str) -> str:
        """Return a deterministic value without model access."""
        return value


class Clock:
    """Shared catalog, replay-window, and JWT fake clock."""

    value = 1500.0

    def __call__(self) -> float:
        return self.value

    def now(self) -> float:
        """Return fake epoch seconds for the token validator."""
        return self.value


class Harness:
    """Own generated identities and two entirely in-process HTTP services."""

    def __init__(self) -> None:
        self.clock = Clock()
        self.catalog = AgentCatalog(clock=self.clock)
        self.sink = InMemoryAuditSink()
        self.card = DeployedAgent().get_agent_card(ENDPOINT)
        self.card_status = 200
        self.card_requests: list[dict[str, str]] = []
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.token_expiry = 3000
        self.validator = JWTBearerTokenValidator(
            StaticJWKSResolver({(ISSUER, "test-key"): self.key.public_key()}), clock=self.clock
        )
        self.policy = TrustPolicy(
            version="test-v1",
            issuer=IssuerPolicy(issuer=ISSUER, allowed_algorithms=frozenset({"ES256"})),
            audience=AudiencePolicy(frozenset({"registration"})),
            scopes=ScopePolicy(frozenset(f"registration:{op}" for op in OPERATIONS)),
            clock_skew=ClockSkewPolicy(0),
        )
        self.grant = RegistrationGrant(
            issuer=ISSUER,
            subject_id="deployment",
            owner="team",
            environment="test",
            agent_id="team.echo",
            card_name="DeployedAgent",
            agent_card_url=CARD_URL,
            endpoint_url=ENDPOINT,
        )
        self.discovery_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.agent_app))
        self.service = self.make_service()
        self.control_http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=RegistrationASGI(self.service))
        )
        self.client = RegistrationClient(
            endpoint="http://127.0.0.1:8080",
            http_client=self.control_http,
            token_provider=self.token_provider,
            allow_insecure_loopback=True,
        )

    def make_service(self, *, audit: AuditEmitter | None = None) -> RegistrationService:
        """Construct another frontend sharing the same authoritative catalog."""
        return RegistrationService(
            catalog=self.catalog,
            validator=self.validator,
            trust_policy=self.policy,
            grants=(
                self.grant,
                replace(
                    self.grant, subject_id="admin", operations=frozenset[Operation]({"revoke"})
                ),
            ),
            discovery_policy=DiscoveryPolicy(
                allowed_schemes=frozenset({"http"}),
                allowed_ports=frozenset({8081}),
                allow_loopback=True,
                allow_private_networks=True,
            ),
            audit=audit or AuditEmitter(self.sink),
            http_client=self.discovery_http,
            clock=self.clock,
        )

    async def agent_app(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Serve the agent's actual generated card while invocation stays non-ready."""
        del receive
        self.card_requests.append({key.decode(): value.decode() for key, value in scope["headers"]})
        status = self.card_status if scope["path"] == "/.well-known/agent-card.json" else 503
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps(self.card).encode()})

    def token(self, *, subject: str = "deployment", audience: str = "registration") -> str:
        """Sign an ephemeral test JWT; no generated key leaves this process."""
        header = {"alg": "ES256", "kid": "test-key"}
        claims = {
            "iss": ISSUER,
            "sub": subject,
            "aud": audience,
            "nbf": 1000,
            "iat": 1000,
            "exp": self.token_expiry,
            "scope": " ".join(f"registration:{op}" for op in OPERATIONS),
        }

        def encode(value: bytes) -> str:
            return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

        data = f"{encode(json.dumps(header).encode())}.{encode(json.dumps(claims).encode())}"
        r, s = decode_dss_signature(self.key.sign(data.encode(), ec.ECDSA(SHA256())))
        return f"{data}.{encode(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"

    async def token_provider(self) -> str:
        """Supply the deployment token to the typed client."""
        return self.token()

    def request(self, **changes: Any) -> RegisterRequest:
        """Build a new validated request with deterministic identity and timestamps."""
        values = {
            "owner": "team",
            "environment": "test",
            "agent_id": "team.echo",
            "instance_id": "pod-1",
            "idempotency_key": "create-1",
            "issued_at": self.clock(),
            "correlation_id": "deploy-1",
            "agent_card_url": CARD_URL,
            "deployment_id": "release-1",
            "provenance": "build-1",
            "lease_seconds": 60,
        }
        return RegisterRequest.model_validate({**values, **changes})

    async def close(self) -> None:
        """Close only the clients owned by this test harness."""
        await self.control_http.aclose()
        await self.discovery_http.aclose()


def test_loopback_registration_renewal_drain_shutdown_and_live_consumer() -> None:
    async def run() -> None:
        harness = Harness()
        try:
            consumer = harness.catalog.capability_providers
            skill_id = harness.card["skills"][0]["id"]
            assert consumer(skill_id) == ()
            request = harness.request()
            first, duplicate = await asyncio.gather(
                harness.client.send(request), harness.client.send(request)
            )
            assert first.ok and first.ready and duplicate == first
            assert first.lease_handle is not None
            (record,) = consumer(skill_id)
            assert len(record.instances) == 1
            assert record.capabilities[0].input_schema["properties"]["value"]["type"] == "string"
            harness.card_status = 503
            # Receipts belong to the catalog, not the service process or discovery cache.
            replay = await harness.make_service().handle(
                request, authorization_header=f"Bearer {harness.token()}"
            )
            assert replay == first
            common = {
                "owner": request.owner,
                "environment": request.environment,
                "agent_id": request.agent_id,
                "instance_id": request.instance_id,
                "correlation_id": request.correlation_id,
                "issued_at": harness.clock(),
                "lease_handle": first.lease_handle,
            }
            renewal = RenewRequest.model_validate(
                {**common, "idempotency_key": "renew-1", "expected_generation": first.generation}
            )
            competing = renewal.model_copy(update={"idempotency_key": "renew-2"})
            outcomes = await asyncio.gather(
                harness.client.send(renewal), harness.client.send(competing)
            )
            assert sorted(item.code for item in outcomes) == ["ok", "stale_generation"]
            renewed = next(item for item in outcomes if item.ok)
            draining = await harness.client.send(
                DrainRequest.model_validate(
                    {
                        **common,
                        "idempotency_key": "drain-1",
                        "expected_generation": renewed.generation,
                    }
                )
            )
            assert draining.ok and not draining.ready and draining.state == "draining"
            assert consumer(skill_id) == ()
            removed = await harness.client.send(
                DeregisterRequest.model_validate(
                    {
                        **common,
                        "idempotency_key": "remove-1",
                        "expected_generation": draining.generation,
                    }
                )
            )
            assert removed.ok and removed.state == "removed"
            assert not (await harness.client.send(request)).ok
            assert all(event.subject_id == "deployment" for event in harness.sink.events)
        finally:
            await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"owner": "other"}, RegistrationCode.UNAUTHORIZED),
        ({"environment": "prod"}, RegistrationCode.UNAUTHORIZED),
        ({"agent_id": "other"}, RegistrationCode.UNAUTHORIZED),
        (
            {"agent_card_url": "http://127.0.0.1:8081/substitute"},
            RegistrationCode.ENDPOINT_MISMATCH,
        ),
        ({"lease_seconds": 301}, RegistrationCode.INVALID_LEASE),
        ({"issued_at": 1000}, RegistrationCode.REPLAY_REJECTED),
        ({"issued_at": 1501}, RegistrationCode.REPLAY_REJECTED),
        ({"protocol_version": "2"}, RegistrationCode.UNSUPPORTED_VERSION),
    ],
)
def test_untrusted_claims_fail_before_discovery(
    changes: dict[str, Any], code: RegistrationCode
) -> None:
    async def run() -> None:
        harness = Harness()
        try:
            result = await harness.client.send(harness.request(**changes))
            assert result.code == code
            assert harness.card_requests == []
            assert harness.catalog.snapshot().agents == ()
            assert harness.sink.events[-1].reason_code == code.value
        finally:
            await harness.close()

    asyncio.run(run())


def test_unauthorized_identity_wrong_audience_and_lifecycle_scope() -> None:
    async def run() -> None:
        harness = Harness()
        try:
            request = harness.request()
            for header in (None, "Bearer invalid", f"Bearer {harness.token(audience='a2a')}"):
                result = await harness.service.handle(request, authorization_header=header)
                assert result.code == RegistrationCode.UNAUTHENTICATED
            result = await harness.service.handle(
                request, authorization_header=f"Bearer {harness.token(subject='intruder')}"
            )
            assert result.code == RegistrationCode.UNAUTHORIZED
            admitted = await harness.client.send(request)
            revoke = RevokeRequest(
                owner="team",
                environment="test",
                agent_id="team.echo",
                instance_id="pod-1",
                idempotency_key="revoke-1",
                issued_at=harness.clock(),
                correlation_id="revoke",
                expected_generation=admitted.generation,
            )
            assert (await harness.client.send(revoke)).code == RegistrationCode.UNAUTHORIZED
            revoked = await harness.service.handle(
                revoke, authorization_header=f"Bearer {harness.token(subject='admin')}"
            )
            assert revoked.ok and revoked.state == "revoked"
            assert harness.catalog.snapshot().agents == ()
            assert harness.sink.events[-1].subject_id == "admin"
            assert not (await harness.client.send(request)).ok
        finally:
            await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["identity", "endpoint", "version", "invalid", "outage"])
def test_card_not_caller_metadata_is_authoritative(mode: str) -> None:
    async def run() -> None:
        harness = Harness()
        try:
            if mode == "identity":
                harness.card["name"] = "Impostor"
            elif mode == "endpoint":
                harness.card["supportedInterfaces"][0]["url"] = ENDPOINT + "-other"
            elif mode == "version":
                harness.card["supportedInterfaces"][0]["protocolVersion"] = "0.3"
            elif mode == "invalid":
                harness.card = {"not": "a card"}
            else:
                harness.card_status = 503
            result = await harness.client.send(harness.request())
            assert not result.ok
            assert harness.catalog.snapshot().agents == ()
            with pytest.raises(ValidationError):
                REQUEST_ADAPTER.validate_python(
                    {**request_document(harness.request()), "agent_card": harness.card}
                )
        finally:
            await harness.close()

    asyncio.run(run())


def test_expiry_status_handle_binding_and_conflicting_idempotency() -> None:
    async def run() -> None:
        harness = Harness()
        try:
            request = harness.request()
            admitted = await harness.client.send(request)
            conflict = await harness.client.send(harness.request(lease_seconds=30))
            assert conflict.code == RegistrationCode.IDEMPOTENCY_CONFLICT
            assert admitted.lease_handle is not None
            assert admitted.is_ready_at(harness.clock())
            common = {
                "owner": "team",
                "environment": "test",
                "agent_id": "team.echo",
                "instance_id": "pod-1",
                "issued_at": harness.clock(),
                "correlation_id": "status",
                "expected_generation": admitted.generation,
                "idempotency_key": "status-1",
            }
            wrong = await harness.client.send(
                StatusRequest.model_validate({**common, "lease_handle": "wrong"})
            )
            assert wrong.code == RegistrationCode.INVALID_HANDLE
            status = await harness.client.send(
                StatusRequest.model_validate({**common, "lease_handle": admitted.lease_handle})
            )
            assert status.ready and status.lease_handle is None
            harness.clock.value = 1560
            assert not admitted.is_ready_at(harness.clock())
            assert harness.catalog.snapshot().agents == ()
            assert await harness.service.expire() == 1
            assert await harness.service.expire() == 0
            assert harness.sink.events[-1].reason_code == "lease_expired"
            assert harness.sink.events[-1].subject_id == "deployment"
            assert not (await harness.client.send(request)).ok
        finally:
            await harness.close()

    asyncio.run(run())


def test_trace_propagation_and_sensitive_data_absence(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        harness = Harness()
        tracing = configure_in_memory_tracing()
        tracing.exporter.clear()
        try:
            request = harness.request()
            token = harness.token()
            trace_id = "12345678901234567890123456789012"
            result = await harness.service.handle(
                request,
                authorization_header=f"Bearer {token}",
                trace_headers={"traceparent": f"00-{trace_id}-1234567890123456-01"},
            )
            assert result.ok and result.lease_handle is not None
            assert trace_id in harness.card_requests[0]["traceparent"]
            assert harness.card_requests[0]["x-correlation-id"] == request.correlation_id
            assert "authorization" not in harness.card_requests[0]
            assert all(event.trace_id == trace_id for event in harness.sink.events)
            spans = tracing.exporter.get_finished_spans()
            rendered = (
                repr([(span.name, dict(span.attributes), span.events) for span in spans])
                + repr(harness.sink.events)
                + caplog.text
                + result.model_dump_json()
                + repr(result)
            )
            for secret in (token, result.lease_handle.get_secret_value(), CARD_URL, ENDPOINT):
                assert secret not in rendered
        finally:
            await harness.close()

    caplog.set_level("INFO", logger="conducto")
    asyncio.run(run())


def test_catalog_and_audit_outages_never_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        harness = Harness()
        try:

            def unavailable(*_args: Any, **_kwargs: Any) -> None:
                raise CatalogProviderUnavailableError("sensitive-storage-endpoint")

            monkeypatch.setattr(harness.catalog, "lookup_managed_request", unavailable)
            result = await harness.client.send(harness.request())
            assert result.code == RegistrationCode.CATALOG_UNAVAILABLE
            assert harness.catalog.snapshot().agents == ()
            service = harness.make_service(audit=AuditEmitter(FailingAuditSink()))
            result = await service.handle(
                harness.request(), authorization_header=f"Bearer {harness.token()}"
            )
            assert result.code == RegistrationCode.AUDIT_UNAVAILABLE
            assert harness.catalog.snapshot().agents == ()
        finally:
            await harness.close()

    asyncio.run(run())


def test_lost_admission_response_can_be_recovered_near_token_expiry() -> None:
    async def run() -> None:
        harness = Harness()
        harness.token_expiry = 1561
        try:
            request = harness.request()
            admitted = await harness.client.send(request)
            assert admitted.ready
            harness.clock.value = 1502
            recovered = await harness.client.send(request)
            assert recovered == admitted
            assert recovered.lease_expires_at == 1560
        finally:
            await harness.close()

    asyncio.run(run())


def test_expiration_evidence_is_retried_after_sink_failure() -> None:
    async def run() -> None:
        harness = Harness()
        try:
            assert (await harness.client.send(harness.request())).ready
            harness.sink.max_events = len(harness.sink.events)
            harness.clock.value = 1560
            with pytest.raises(AuditDeliveryError):
                await harness.service.expire()
            assert harness.catalog.snapshot().agents == ()
            harness.sink.max_events += 10
            assert await harness.service.expire() == 1
            assert harness.sink.events[-1].reason_code == "lease_expired"
            assert harness.sink.events[-1].subject_id == "deployment"
            assert await harness.service.expire() == 0
        finally:
            await harness.close()

    asyncio.run(run())
