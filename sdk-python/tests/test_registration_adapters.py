"""Deterministic registration HTTP boundary and confidential-wire coverage."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, MutableMapping
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from conducto.registration.asgi import RegistrationASGI
from conducto.registration.client import RegistrationClient
from conducto.registration.models import (
    RegisterRequest,
    RegistrationCode,
    RegistrationRequest,
    RegistrationResult,
    RenewRequest,
    StatusRequest,
    request_document,
    result_document,
)

_TRACEPARENT = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
_TOKEN = "private-token"
_HANDLE = "private-handle"


def _request() -> RegisterRequest:
    return RegisterRequest(
        owner="owner",
        environment="test",
        agent_id="agent",
        instance_id="instance",
        idempotency_key="operation-1",
        issued_at=100.0,
        correlation_id="correlation-1",
        agent_card_url="https://worker.example/card",
        deployment_id="deployment",
        provenance="revision",
    )


def _success(request: RegistrationRequest) -> RegistrationResult:
    return RegistrationResult(
        code=RegistrationCode.OK,
        correlation_id=request.correlation_id,
        generation=request.expected_generation + 1,
        state="active",
        lease_expires_at=200.0,
        lease_handle=SecretStr(_HANDLE),
    )


async def _token() -> str:
    return _TOKEN


class _Service:
    def __init__(self, code: RegistrationCode = RegistrationCode.OK) -> None:
        self.calls: list[tuple[RegistrationRequest, str | None, Mapping[str, str] | None]] = []
        self.code = code

    async def handle(
        self,
        request: RegistrationRequest,
        *,
        authorization_header: str | None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> RegistrationResult:
        self.calls.append((request, authorization_header, trace_headers))
        if self.code is RegistrationCode.OK:
            return _success(request)
        return RegistrationResult(code=self.code, correlation_id=request.correlation_id)


def _app(service: _Service, **kwargs: Any) -> RegistrationASGI:
    return RegistrationASGI(service, **kwargs)


def _client(http: httpx.AsyncClient, **kwargs: Any) -> RegistrationClient:
    return RegistrationClient(
        endpoint="https://control.example", http_client=http, token_provider=_token, **kwargs
    )


async def _call_app(
    app: RegistrationASGI,
    messages: list[dict[str, Any]],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    path: str = "/registration/v1/register",
    method: str = "POST",
    scope_type: str = "http",
) -> list[dict[str, Any]]:
    outgoing: list[dict[str, Any]] = []
    incoming = iter(messages)

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(message: MutableMapping[str, Any]) -> None:
        outgoing.append(dict(message))

    await app(
        {
            "type": scope_type,
            "method": method,
            "path": path,
            "headers": headers if headers is not None else [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
    return outgoing


def test_asgi_client_round_trip_confidential_grant_and_resource_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "conducto.registration.client.inject_trace_context",
        lambda headers: {**headers, "traceparent": _TRACEPARENT, "tracestate": "vendor=value"},
    )

    async def run() -> None:
        service = _Service()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(service)),
            auth=httpx.BasicAuth("unrelated", "credentials"),
        ) as http:
            result = await _client(http).send(_request())
            assert result.ok
            assert result.lease_handle == SecretStr(_HANDLE)
            assert _HANDLE not in repr(result)
            assert _HANDLE not in result.model_dump_json()
            assert not http.is_closed
        request, authorization, traces = service.calls[0]
        assert request == _request()
        assert authorization == f"Bearer {_TOKEN}"
        assert traces == {"traceparent": _TRACEPARENT, "tracestate": "vendor=value"}

    asyncio.run(run())


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (RegistrationCode.OK, 200),
        (RegistrationCode.INVALID_REQUEST, 400),
        (RegistrationCode.UNAUTHENTICATED, 401),
        (RegistrationCode.UNAUTHORIZED, 403),
        (RegistrationCode.INVALID_HANDLE, 403),
        (RegistrationCode.STALE_GENERATION, 409),
        (RegistrationCode.SERVICE_UNAVAILABLE, 503),
        (RegistrationCode.CATALOG_UNAVAILABLE, 503),
    ],
)
def test_asgi_maps_typed_status_and_disables_caching(code: RegistrationCode, status: int) -> None:
    async def run() -> None:
        service = _Service(code)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(service))) as http:
            response = await http.post(
                "https://control.example/registration/v1/register",
                json=request_document(_request()),
            )
        assert response.status_code == status
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "application/json"
        assert response.headers["x-correlation-id"] == "correlation-1"
        assert response.json()["code"] == code
        assert service.calls[0][1] is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "headers",
    [
        [(b"authorization", b"Bearer secret"), (b"Authorization", b"Bearer secret")],
        [(b"x-correlation-id", b"correlation-1"), (b"X-Correlation-ID", b"correlation-1")],
        [(b"traceparent", _TRACEPARENT.encode()), (b"Traceparent", _TRACEPARENT.encode())],
        [(b"tracestate", b"vendor=value"), (b"tracestate", b"vendor=value")],
        [(b"x-correlation-id", b"forged")],
        [(b"authorization", b"Bearer secret\r\nforged: value")],
        [(b"authorization", b"\xff")],
        [(b"traceparent", b"malformed")],
        [(b"tracestate", b"vendor=value")],
        [(b"traceparent", _TRACEPARENT.encode()), (b"tracestate", b"vendor=one,vendor=two")],
        [(b"content-length", b"not-a-number")],
        [(b"content-length", b"1")],
        [(b"content-type", b"text/plain")],
    ],
)
def test_asgi_rejects_ambiguous_or_malformed_headers(headers: list[tuple[bytes, bytes]]) -> None:
    service = _Service()
    outgoing = asyncio.run(
        _call_app(
            _app(service),
            [{"type": "http.request", "body": json.dumps(request_document(_request())).encode()}],
            headers=[(b"content-type", b"application/json"), *headers],
        )
    )
    assert outgoing[0]["status"] == 400
    assert b"x-correlation-id" not in dict(outgoing[0]["headers"])
    assert json.loads(outgoing[1]["body"])["code"] == "invalid_request"
    assert not service.calls
    assert b"secret" not in outgoing[1]["body"]


@pytest.mark.parametrize(
    "body",
    [
        b'{"operation":"register","lease_handle":"private-handle"}',
        b'{"operation":"register","operation":"renew"}',
        b'{"operation":"unknown","correlation_id":"evil\\r\\nheader"}',
        b"\xff",
        b"null",
        b"[]",
        b"{",
        b"[" * 2000,
        b'{"issued_at":' + b"9" * 5000 + b"}",
    ],
)
def test_asgi_rejects_malformed_body_without_echo_or_details(body: bytes) -> None:
    service = _Service()
    outgoing = asyncio.run(_call_app(_app(service), [{"type": "http.request", "body": body}]))
    assert outgoing[0]["status"] == 400
    assert b"x-correlation-id" not in dict(outgoing[0]["headers"])
    result = RegistrationResult.model_validate_json(outgoing[1]["body"])
    assert result.code is RegistrationCode.INVALID_REQUEST
    assert result.correlation_id == ""
    assert b"private-handle" not in outgoing[1]["body"]
    assert not service.calls


@pytest.mark.parametrize(
    ("path", "method", "status"),
    [
        ("/registration/v1/renew", "POST", 400),
        ("/registration/v1/register", "GET", 405),
        ("/registration/v1/unknown", "POST", 404),
        ("/elsewhere/register", "POST", 404),
    ],
)
def test_asgi_requires_matching_operation_path(path: str, method: str, status: int) -> None:
    service = _Service()
    outgoing = asyncio.run(
        _call_app(
            _app(service),
            [{"type": "http.request", "body": json.dumps(request_document(_request())).encode()}],
            path=path,
            method=method,
        )
    )
    assert outgoing[0]["status"] == status
    assert not service.calls


def test_asgi_cumulative_body_bound_disconnect_and_lifespan() -> None:
    async def run() -> None:
        service = _Service()
        app = _app(service, max_request_bytes=4)
        outgoing = await _call_app(
            app,
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"45"},
            ],
        )
        assert outgoing[0]["status"] == 413
        assert json.loads(outgoing[1]["body"])["code"] == "invalid_request"
        outgoing = await _call_app(app, [{"type": "http.disconnect"}])
        assert outgoing == []
        outgoing = await _call_app(
            app,
            [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}],
            scope_type="lifespan",
        )
        assert outgoing == [
            {"type": "lifespan.startup.complete"},
            {"type": "lifespan.shutdown.complete"},
        ]
        assert not service.calls

    asyncio.run(run())


def test_asgi_total_timeout_is_unavailable_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    original_timeout = asyncio.timeout
    monkeypatch.setattr("asyncio.timeout", lambda _: original_timeout(0))

    async def run() -> None:
        service = _Service()
        outgoing: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

        async def send(message: MutableMapping[str, Any]) -> None:
            outgoing.append(dict(message))

        await _app(service)(
            {
                "type": "http",
                "path": "/registration/v1/register",
                "method": "POST",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
        assert outgoing[0]["status"] == 504
        assert json.loads(outgoing[1]["body"])["code"] == "service_unavailable"
        assert not service.calls

    asyncio.run(run())


def test_asgi_service_timeout_preserves_safe_correlation() -> None:
    class TimeoutService(_Service):
        async def handle(
            self,
            request: RegistrationRequest,
            *,
            authorization_header: str | None,
            trace_headers: Mapping[str, str] | None = None,
        ) -> RegistrationResult:
            raise TimeoutError("private-token")

    outgoing = asyncio.run(
        _call_app(
            _app(TimeoutService()),
            [{"type": "http.request", "body": json.dumps(request_document(_request())).encode()}],
        )
    )
    assert outgoing[0]["status"] == 504
    result = RegistrationResult.model_validate_json(outgoing[1]["body"])
    assert result.correlation_id == "correlation-1"
    assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
    assert b"private-token" not in outgoing[1]["body"]


@pytest.mark.parametrize("failure", ["transport", "unavailable", "stale"])
def test_client_bounded_retries_preserve_exact_request_and_key(failure: str) -> None:
    async def run() -> None:
        received: list[bytes] = []

        def handle(incoming: httpx.Request) -> httpx.Response:
            received.append(incoming.content)
            if len(received) < 3:
                if failure == "transport":
                    raise httpx.ConnectError("private-token", request=incoming)
                code = (
                    RegistrationCode.STALE_GENERATION
                    if failure == "stale"
                    else RegistrationCode.SERVICE_UNAVAILABLE
                )
                return httpx.Response(
                    503 if failure == "unavailable" else 409,
                    json=result_document(
                        RegistrationResult(code=code, correlation_id="correlation-1")
                    ),
                )
            return httpx.Response(200, json=result_document(_success(_request())))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            request = _request()
            result = await _client(http, max_attempts=3).send(request)
        if failure == "stale":
            assert result.code is RegistrationCode.STALE_GENERATION
            assert len(received) == 1
        else:
            assert result.ok
            assert len(received) == 3
        assert all(body == received[0] for body in received)
        assert json.loads(received[0])["idempotency_key"] == request.idempotency_key
        assert request.expected_generation == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        {"correlation_id": "forged"},
        {"lease_handle": None},
        {"lease_handle": ""},
        {"lease_expires_at": None},
        {"lease_expires_at": 50},
        {"generation": 0},
        {"state": "draining"},
        {"unknown": "private-token"},
    ],
)
def test_client_rejects_forged_or_incomplete_admission(mutation: dict[str, Any]) -> None:
    async def run() -> None:
        calls = 0

        def handle(_incoming: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={**result_document(_success(_request())), **mutation})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            result = await _client(http, max_attempts=3).send(_request())
        assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
        assert result.correlation_id == "correlation-1"
        assert result.lease_handle is None
        assert calls == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "response_kind",
    [
        "http_failure",
        "non_json",
        "malformed",
        "large",
        "redirect",
        "header",
        "integer_limit",
        "duplicate",
        "utf8",
    ],
)
def test_client_rejects_untrusted_responses_without_following_redirects(response_kind: str) -> None:
    async def run() -> None:
        calls = 0

        def handle(_incoming: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if response_kind == "non_json":
                return httpx.Response(200, text="private-token")
            invalid_bodies = {
                "malformed": b"{",
                "large": b" " * 16385,
                "integer_limit": b'{"generation":' + b"9" * 5000 + b"}",
                "duplicate": b'{"code":"service_unavailable","code":"ok"}',
                "utf8": b"\xff",
            }
            if response_kind in invalid_bodies:
                return httpx.Response(
                    200,
                    content=invalid_bodies[response_kind],
                    headers={"content-type": "application/json"},
                )
            if response_kind == "redirect":
                return httpx.Response(307, headers={"location": "https://untrusted.example/"})
            return httpx.Response(
                500 if response_kind == "http_failure" else 200,
                json=result_document(_success(_request())),
                headers={"x-correlation-id": "forged"} if response_kind == "header" else None,
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), follow_redirects=True
        ) as http:
            result = await _client(http, max_attempts=3).send(_request())
        assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
        assert result.lease_handle is None
        assert calls == 1
        assert _TOKEN not in repr(result)

    asyncio.run(run())


def test_client_rejects_compression_before_consuming_or_decoding_response() -> None:
    class UnreadStream(httpx.AsyncByteStream):
        consumed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            self.consumed = True
            yield b"untrusted compressed response"

    async def run() -> None:
        stream = UnreadStream()

        def handle(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                stream=stream,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            result = await _client(http).send(_request())
        assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
        assert not stream.consumed

    asyncio.run(run())


def test_client_renewal_preserves_confidential_handle_and_rejects_rotation() -> None:
    async def run() -> None:
        base = _request()
        request = RenewRequest(
            **base.model_dump(
                exclude={
                    "operation",
                    "agent_card_url",
                    "deployment_id",
                    "deployment_type",
                    "provenance",
                    "lease_seconds",
                    "expected_generation",
                }
            ),
            expected_generation=1,
            lease_handle=SecretStr(_HANDLE),
        )

        def handle(incoming: httpx.Request) -> httpx.Response:
            assert json.loads(incoming.content)["lease_handle"] == _HANDLE
            return httpx.Response(
                200,
                json={**result_document(_success(request)), "lease_handle": "rotated-secret"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            result = await _client(http).send(request)
        assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
        assert _HANDLE not in repr(request)

    asyncio.run(run())


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://control.example",
        "https://user:secret@control.example",
        "https://control.example/path",
        "https://control.example?secret=value",
        "https://control.example#fragment",
        "https://control.example?",
        "https://control.example#",
        "https://control.example:bad",
        "https://control.example:0",
        "https://control.example\\evil",
        "https://control.example\n",
    ],
)
def test_client_rejects_unsafe_origins(endpoint: str) -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as http:
            with pytest.raises(ValueError, match="registration endpoint"):
                RegistrationClient(endpoint=endpoint, http_client=http, token_provider=_token)

    asyncio.run(run())


@pytest.mark.parametrize(
    "endpoint", ["http://localhost", "http://127.0.0.1:8000", "http://[::1]:8000"]
)
def test_client_loopback_http_requires_explicit_opt_in(endpoint: str) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(_Service()))) as http:
            with pytest.raises(ValueError, match="HTTPS"):
                RegistrationClient(endpoint=endpoint, http_client=http, token_provider=_token)
            client = RegistrationClient(
                endpoint=endpoint,
                http_client=http,
                token_provider=_token,
                allow_insecure_loopback=True,
            )
            assert (await client.send(_request())).ok

    asyncio.run(run())


def test_client_timeout_and_cancellation_do_not_expose_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_timeout = asyncio.timeout
    monkeypatch.setattr("asyncio.timeout", lambda _: original_timeout(0))

    async def run() -> None:
        attempts = 0

        async def pending_token() -> str:
            nonlocal attempts
            attempts += 1
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

        async with httpx.AsyncClient() as http:
            client = RegistrationClient(
                endpoint="https://control.example",
                http_client=http,
                token_provider=pending_token,
                max_attempts=2,
            )
            result = await client.send(_request())
            assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
            assert attempts == 2

            async def cancelled_token() -> str:
                raise asyncio.CancelledError

            client = RegistrationClient(
                endpoint="https://control.example", http_client=http, token_provider=cancelled_token
            )
            with pytest.raises(asyncio.CancelledError):
                await client.send(_request())

    asyncio.run(run())


def test_client_bounds_chunked_response_and_closes_stream() -> None:
    class ResponseStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.chunks = 0

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(3):
                self.chunks += 1
                yield b" " * 9000

        async def aclose(self) -> None:
            self.closed = True

    async def run() -> None:
        stream = ResponseStream()

        def handle(_incoming: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=stream, headers={"content-type": "application/json"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            result = await _client(http).send(_request())
        assert result.code is RegistrationCode.SERVICE_UNAVAILABLE
        assert stream.chunks == 2
        assert stream.closed

    asyncio.run(run())


@pytest.mark.parametrize(("request_type", "generation"), [(StatusRequest, 1), (RenewRequest, 2)])
def test_client_status_and_renewal_accept_active_results_without_grant(
    request_type: type[StatusRequest] | type[RenewRequest], generation: int
) -> None:
    async def run() -> None:
        request = request_type(
            owner="owner",
            environment="test",
            agent_id="agent",
            instance_id="instance",
            idempotency_key="operation-2",
            issued_at=100.0,
            expected_generation=1,
            correlation_id="correlation-2",
            lease_handle=SecretStr(_HANDLE),
        )
        result = RegistrationResult(
            code=RegistrationCode.OK,
            correlation_id=request.correlation_id,
            generation=generation,
            state="active",
            lease_expires_at=200.0,
        )

        def handle(_incoming: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=result_document(result))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            response = await _client(http).send(request)
        assert response == result
        assert response.ok
        assert response.lease_handle is None

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["valid", "missing_expiry", "stale_generation", "inactive"])
def test_client_validates_renewal_shape(kind: str) -> None:
    async def run() -> None:
        base = _request()
        request = RenewRequest(
            owner=base.owner,
            environment=base.environment,
            agent_id=base.agent_id,
            instance_id=base.instance_id,
            idempotency_key=base.idempotency_key,
            issued_at=base.issued_at,
            correlation_id=base.correlation_id,
            expected_generation=1,
            lease_handle=SecretStr(_HANDLE),
        )

        def handle(_incoming: httpx.Request) -> httpx.Response:
            response = result_document(_success(request))
            response.pop("lease_handle")
            if kind == "missing_expiry":
                response["lease_expires_at"] = None
            elif kind == "stale_generation":
                response["generation"] = request.expected_generation
            elif kind == "inactive":
                response["state"] = "draining"
            return httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            result = await _client(http).send(request)
        assert result.ok is (kind == "valid")
        assert result.lease_handle is None

    asyncio.run(run())


@pytest.mark.parametrize("token", ["", "secret\r\nforged: value", "secret value", "\u0100"])
def test_client_rejects_unsafe_tokens_without_transmitting(token: str) -> None:
    async def run() -> None:
        async def provide() -> str:
            return token

        def handle(_incoming: httpx.Request) -> httpx.Response:
            raise AssertionError("unsafe credentials must not reach HTTP")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = RegistrationClient(
                endpoint="https://control.example", http_client=http, token_provider=provide
            )
            result = await client.send(_request())
        assert result.code is RegistrationCode.UNAUTHENTICATED

    asyncio.run(run())


@pytest.mark.parametrize("attempts", [0, 4, True, 1.5])
def test_client_rejects_invalid_attempt_budgets(attempts: Any) -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as http:
            with pytest.raises(ValueError, match="max_attempts"):
                _client(http, max_attempts=attempts)

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_adapters_reject_invalid_timeout_budgets(timeout: float) -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="request_timeout"):
            _app(_Service(), request_timeout=timeout)
        async with httpx.AsyncClient() as http:
            with pytest.raises(ValueError, match="request_timeout"):
                _client(http, request_timeout=timeout)

    asyncio.run(run())
