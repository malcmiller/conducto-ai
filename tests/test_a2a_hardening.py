"""Deterministic tests for the A2A ASGI hardening and operational lifecycle."""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ParamSpec

import httpx
import pytest
from a2a.types.a2a_pb2 import Message, Part, Role, Task, TaskState

from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.a2a import (
    A2AASGI,
    A2AConcurrencyLimiter,
    A2AHostConfigurationError,
    A2AHostLifecycle,
    A2AHostSecurityConfig,
    A2AHostState,
    A2APayloadLimitError,
    A2ARequestContext,
    A2ARequestRejectedError,
    A2AShutdownError,
    A2AStartupError,
)
from conducto.a2a.hardening import enforce_metadata_limit, evaluate_request, sanitize_scope
from conducto.core.invocation_results import InvocationResult, InvocationSuccess
from conducto.transport import InMemoryTaskRepository

ENDPOINT_URL = "https://agent.example/a2a"
RPC_PATH = "/a2a"
CARD_PATH = "/.well-known/agent-card.json"
_V1_HEADERS = {"A2A-Version": "1.0"}

_P = ParamSpec("_P")


def _sync(test: Callable[_P, Awaitable[None]]) -> Callable[_P, None]:
    """Run an async test body on a fresh event loop, matching repository style."""

    @functools.wraps(test)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


@a2a_agent(name="Echo Agent", version="1.0", description="Echoes one value.")
class _EchoAgent(BaseAgent):
    """Minimal reflected agent used only to build a valid Agent Card."""

    @a2a_capability(name="echo", description="Echo one value.")
    def echo(self, value: str) -> dict[str, str]:
        """Echo the supplied value; never invoked by the adapter directly."""
        return {"value": value}


class _Handler:
    """Deterministic handler seam whose completion is event-driven."""

    def __init__(self, *, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancel_calls: list[str] = []
        self.in_flight = 0
        self.peak = 0

    async def handle_message(
        self,
        message: Message,
        *,
        task_id: str,
        context_id: str,
        request_context: A2ARequestContext,
    ) -> InvocationResult:
        """Record the call, optionally wait on an injected gate, then succeed."""
        del message, context_id, request_context
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.entered.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            return InvocationSuccess(correlation_id=task_id, value={"echo": True})
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.in_flight -= 1

    async def cancel(self, task_id: str) -> None:
        """Record an explicit cancellation request for the given task."""
        self.cancel_calls.append(task_id)


def _message(text: str = "hello", *, parts: int = 1, task_id: str = "") -> Message:
    """Build a minimal text message with a controllable part count."""
    message = Message(message_id="msg-1", role=Role.ROLE_USER)
    for index in range(parts):
        message.parts.append(Part(text=f"{text}-{index}"))
    if task_id:
        message.task_id = task_id
    return message


def _send_payload(message: Message, request_id: str = "1") -> dict[str, Any]:
    """Build a JSON-RPC SendMessage envelope for the given message."""
    parts = [{"text": part.text} for part in message.parts]
    payload: dict[str, Any] = {
        "messageId": message.message_id,
        "role": "ROLE_USER",
        "parts": parts,
    }
    if message.task_id:
        payload["taskId"] = message.task_id
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "SendMessage",
        "params": {"message": payload},
    }


def _build_app(
    *,
    config: A2AHostSecurityConfig | None = None,
    handler: _Handler | None = None,
    repository: InMemoryTaskRepository | None = None,
    **kwargs: Any,
) -> tuple[A2AASGI, _Handler, InMemoryTaskRepository]:
    """Build one isolated hardened application plus its injected collaborators."""
    seam = handler or _Handler()
    store = repository or InMemoryTaskRepository()
    app = A2AASGI(
        agent=_EchoAgent(),
        endpoint_url=ENDPOINT_URL,
        task_repository=store,
        request_handler=seam,
        security_config=config,
        **kwargs,
    )
    return app, seam, store


def _client(app: A2AASGI) -> httpx.AsyncClient:
    """Return an in-process HTTP client bound to the hardened application."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://agent.example"
    )


def _scope(
    *,
    method: str = "POST",
    path: str = RPC_PATH,
    scheme: str = "https",
    headers: Mapping[str, str] | None = None,
    client: tuple[str, int] | None = ("203.0.113.9", 4321),
) -> dict[str, Any]:
    """Build a raw ASGI HTTP scope for direct hardening-boundary tests."""
    merged = {"host": "agent.example", "content-type": "application/json"}
    merged.update({key.lower(): value for key, value in (headers or {}).items()})
    return {
        "type": "http",
        "method": method,
        "path": path,
        "scheme": scheme,
        "client": client,
        "headers": [
            (key.encode("latin-1"), value.encode("latin-1")) for key, value in merged.items()
        ],
    }


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_schemes": frozenset({"ftp"})},
        {"allowed_methods": frozenset({"P0ST"})},
        {"allowed_content_types": frozenset({"not-a-media-type"})},
        {"allowed_paths": frozenset({"relative"})},
        {"max_header_count": 0},
        {"max_request_body_bytes": -1},
        {"max_page_size": 0},
        {"max_accepted_concurrency": 0},
        {"request_deadline_seconds": 0.0},
        {"drain_deadline_seconds": float("inf")},
        {"liveness_path": "healthz"},
        {"trusted_proxies": frozenset({"not-an-address"})},
        {"allowed_ports": frozenset({0})},
        {"allowed_hosts": frozenset({"bad host"})},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    """Every invalid configuration value fails fast with a typed error."""
    with pytest.raises(A2AHostConfigurationError):
        A2AHostSecurityConfig(**kwargs)


def test_configuration_is_immutable_and_isolated() -> None:
    """Configuration is frozen and never shares mutable defaults between hosts."""
    first = A2AHostSecurityConfig()
    second = A2AHostSecurityConfig(max_page_size=7)
    with pytest.raises(AttributeError):
        first.max_page_size = 5  # type: ignore[misc]
    assert first.max_page_size != second.max_page_size
    assert first.allowed_paths == frozenset()


def test_explicit_paths_must_include_mounted_routes() -> None:
    """A host refuses to start when its policy hides a mounted route."""
    config = A2AHostSecurityConfig(allowed_paths=frozenset({"/other"}))
    with pytest.raises(A2AHostConfigurationError):
        _build_app(config=config)


def test_default_paths_are_derived_from_mounted_routes() -> None:
    """Omitted paths are derived from the mounted JSON-RPC, card, and probes."""
    app, _, _ = _build_app(
        config=A2AHostSecurityConfig(liveness_path="/livez", readiness_path="/readyz")
    )
    assert app.security_config.allowed_paths == frozenset(
        {RPC_PATH, CARD_PATH, "/livez", "/readyz"}
    )


# --------------------------------------------------------------------------- #
# Scope validation: host, scheme, authority, path, method, content
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("scope_kwargs", "config_kwargs", "reason", "status"),
    [
        ({"method": "DELETE"}, {}, "method_not_allowed", 405),
        ({"headers": {"host": ""}}, {}, "missing_host", 400),
        ({"headers": {"host": "bad host"}}, {}, "invalid_host", 400),
        ({"headers": {"host": "agent.example:0"}}, {}, "invalid_host", 400),
        ({"headers": {"host": "agent.example:70000"}}, {}, "invalid_host", 400),
        ({"headers": {"host": "agent.example:abc"}}, {}, "invalid_host", 400),
        (
            {"headers": {"host": "evil.example"}},
            {"allowed_hosts": frozenset({"agent.example"})},
            "host_not_allowed",
            400,
        ),
        (
            {"headers": {"host": "agent.example:8443"}},
            {"allowed_ports": frozenset({443})},
            "port_not_allowed",
            400,
        ),
        (
            {"headers": {"host": "agent.example:8443"}},
            {"allowed_authorities": frozenset({"agent.example:443"})},
            "authority_not_allowed",
            400,
        ),
        ({"scheme": "http"}, {"allowed_schemes": frozenset({"https"})}, "scheme_not_allowed", 400),
        ({"path": "/nope"}, {"allowed_paths": frozenset({RPC_PATH})}, "path_not_allowed", 404),
        ({"path": "/a b"}, {}, "invalid_path", 400),
        (
            {"headers": {"content-type": "text/plain"}},
            {},
            "unsupported_media_type",
            415,
        ),
        (
            {"headers": {"content-type": "application/json; charset=utf-16"}},
            {},
            "unsupported_charset",
            415,
        ),
        ({"headers": {"content-encoding": "gzip"}}, {}, "unsupported_content_encoding", 415),
        ({"headers": {"content-length": "abc"}}, {}, "invalid_content_length", 400),
    ],
)
def test_scope_validation_rejects_malformed_requests(
    scope_kwargs: dict[str, Any],
    config_kwargs: dict[str, Any],
    reason: str,
    status: int,
) -> None:
    """Malformed transport framing is rejected before any body byte is read."""
    config = A2AHostSecurityConfig(**config_kwargs)
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(_scope(**scope_kwargs), config=config)
    assert error.value.reason == reason
    assert error.value.status_code == status


def test_missing_content_type_on_post_is_rejected() -> None:
    """A JSON-RPC POST without a media type never reaches the adapter."""
    scope = _scope()
    scope["headers"] = [(b"host", b"agent.example")]
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(scope, config=A2AHostSecurityConfig())
    assert error.value.reason == "missing_content_type"


def test_repeated_host_header_is_ambiguous() -> None:
    """Duplicate single-valued headers are rejected instead of guessed."""
    scope = _scope()
    scope["headers"].append((b"host", b"other.example"))
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(scope, config=A2AHostSecurityConfig())
    assert error.value.reason == "ambiguous_header"


@pytest.mark.parametrize(
    ("authority", "allowed", "accepted"),
    [
        ("agent.example", frozenset({"agent.example"}), True),
        ("agent.example:443", frozenset({"agent.example:443"}), True),
        ("agent.example:8443", frozenset({"agent.example:443"}), False),
        ("192.0.2.10", frozenset({"192.0.2.10"}), True),
        ("[2001:db8::1]", frozenset({"[2001:db8::1]"}), True),
    ],
)
def test_authority_allow_list_boundaries(
    authority: str, allowed: frozenset[str], accepted: bool
) -> None:
    """Authorities are matched exactly, including IP literals and ports."""
    config = A2AHostSecurityConfig(allowed_authorities=allowed)
    scope = _scope(headers={"host": authority})
    if accepted:
        assert evaluate_request(scope, config=config).authority == authority
        return
    with pytest.raises(A2ARequestRejectedError):
        evaluate_request(scope, config=config)


# --------------------------------------------------------------------------- #
# Header limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_header_count_boundaries(delta: int) -> None:
    """Header counts below and at the limit pass; above the limit is rejected."""
    config = A2AHostSecurityConfig(max_header_count=8, max_total_header_bytes=65_536)
    scope = _scope()
    base = len(scope["headers"])
    for index in range(config.max_header_count + delta - base):
        scope["headers"].append((f"x-pad-{index}".encode("latin-1"), b"v"))
    if delta > 0:
        with pytest.raises(A2ARequestRejectedError) as error:
            evaluate_request(scope, config=config)
        assert error.value.reason == "header_count_exceeded"
        return
    assert evaluate_request(scope, config=config).host == "agent.example"


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_aggregate_header_byte_boundaries(delta: int) -> None:
    """Aggregate header bytes are bounded at, below, and above the limit."""
    scope = _scope()
    used = sum(len(key) + len(value) + 2 for key, value in scope["headers"])
    padding = 64
    limit = used + padding + delta
    scope["headers"].append((b"x-pad", b"a" * (padding - len(b"x-pad") - 2)))
    config = A2AHostSecurityConfig(max_total_header_bytes=limit)
    if delta < 0:
        with pytest.raises(A2ARequestRejectedError) as error:
            evaluate_request(scope, config=config)
        assert error.value.reason == "header_bytes_exceeded"
        return
    assert evaluate_request(scope, config=config).has_bearer is False


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_authorization_header_byte_boundaries(delta: int) -> None:
    """Bearer credentials are bounded and their value is never retained."""
    config = A2AHostSecurityConfig(max_authorization_header_bytes=32)
    token = "a" * (config.max_authorization_header_bytes - len("Bearer ") + delta)
    scope = _scope(headers={"authorization": f"Bearer {token}"})
    if delta > 0:
        with pytest.raises(A2ARequestRejectedError) as error:
            evaluate_request(scope, config=config)
        assert error.value.reason == "authorization_header_too_large"
        return
    facts = evaluate_request(scope, config=config)
    assert facts.has_bearer is True
    assert token not in repr(facts)


@pytest.mark.parametrize(
    "value",
    ["Basic abc", "Bearer", "Bearer  ", "Bearer not a token", "bearer ###"],
)
def test_malformed_bearer_identity_is_rejected(value: str) -> None:
    """Malformed authorization headers are rejected without echoing the value."""
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(_scope(headers={"authorization": value}), config=A2AHostSecurityConfig())
    assert error.value.reason == "invalid_authorization"
    assert value not in str(error.value)


def test_bearer_credential_is_not_exposed_by_facts_or_scope() -> None:
    """A valid credential never appears in derived facts or the sanitized scope."""
    secret = "supersecrettoken"
    scope = _scope(headers={"authorization": f"Bearer {secret}"})
    facts = evaluate_request(scope, config=A2AHostSecurityConfig())
    assert secret not in repr(facts)
    sanitized = sanitize_scope(scope, facts)
    assert any(key == b"authorization" for key, _ in sanitized["headers"])


# --------------------------------------------------------------------------- #
# Trace and correlation handling
# --------------------------------------------------------------------------- #


def test_valid_trace_headers_are_propagated() -> None:
    """Well-formed W3C trace context is parsed and propagated unchanged."""
    traceparent = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    facts = evaluate_request(
        _scope(headers={"traceparent": traceparent, "tracestate": "vendor=1"}),
        config=A2AHostSecurityConfig(),
    )
    assert facts.traceparent == traceparent
    assert facts.tracestate == "vendor=1"


def test_malformed_trace_headers_are_dropped_not_trusted() -> None:
    """Malformed trace context is dropped and never forwarded to the adapter."""
    scope = _scope(headers={"traceparent": "not-a-traceparent", "tracestate": "vendor=1"})
    facts = evaluate_request(scope, config=A2AHostSecurityConfig())
    assert facts.traceparent == ""
    assert "traceparent" in facts.dropped_headers
    sanitized = sanitize_scope(scope, facts)
    assert not any(key == b"traceparent" for key, _ in sanitized["headers"])


def test_oversized_trace_header_is_rejected() -> None:
    """Trace headers above the configured bound are rejected outright."""
    config = A2AHostSecurityConfig(max_trace_header_bytes=16)
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(_scope(headers={"tracestate": "v" * 64}), config=config)
    assert error.value.reason == "trace_header_too_large"


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_correlation_header_boundaries(delta: int) -> None:
    """Correlation identifiers are bounded at, below, and above the limit."""
    config = A2AHostSecurityConfig(max_correlation_header_bytes=16)
    value = "c" * (config.max_correlation_header_bytes + delta)
    scope = _scope(headers={"x-correlation-id": value})
    if delta > 0:
        with pytest.raises(A2ARequestRejectedError) as error:
            evaluate_request(scope, config=config)
        assert error.value.reason == "correlation_header_too_large"
        return
    assert evaluate_request(scope, config=config).correlation_id == value


def test_malformed_correlation_header_is_rejected() -> None:
    """Correlation identifiers with control characters are rejected."""
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(
            _scope(headers={"x-correlation-id": "bad value"}), config=A2AHostSecurityConfig()
        )
    assert error.value.reason == "invalid_correlation_header"


# --------------------------------------------------------------------------- #
# Proxy trust and transport identity
# --------------------------------------------------------------------------- #


def test_untrusted_forwarded_headers_are_rejected() -> None:
    """Forwarded headers from an untrusted peer are rejected, never honored."""
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(
            _scope(headers={"x-forwarded-host": "evil.example"}), config=A2AHostSecurityConfig()
        )
    assert error.value.reason == "untrusted_forwarded_header"


def test_untrusted_forwarded_headers_can_be_stripped_instead() -> None:
    """When configured to strip, untrusted forwarded values never reach the app."""
    config = A2AHostSecurityConfig(reject_untrusted_forwarded_headers=False)
    scope = _scope(headers={"x-forwarded-host": "evil.example"})
    facts = evaluate_request(scope, config=config)
    assert facts.host == "agent.example"
    assert not any(key == b"x-forwarded-host" for key, _ in sanitize_scope(scope, facts)["headers"])


def test_trusted_proxy_forwarded_values_are_honored() -> None:
    """Forwarded host, proto, and client are honored only from trusted peers."""
    config = A2AHostSecurityConfig(
        trusted_proxies=frozenset({"10.0.0.5"}),
        trust_forwarded_host=True,
        trust_forwarded_proto=True,
        trust_forwarded_for=True,
        allowed_hosts=frozenset({"public.example"}),
        allowed_schemes=frozenset({"https"}),
    )
    facts = evaluate_request(
        _scope(
            scheme="http",
            client=("10.0.0.5", 9000),
            headers={
                "x-forwarded-host": "public.example",
                "x-forwarded-proto": "https",
                "x-forwarded-for": "198.51.100.7, 10.0.0.5",
            },
        ),
        config=config,
    )
    assert facts.via_trusted_proxy is True
    assert (facts.host, facts.scheme, facts.client) == ("public.example", "https", "198.51.100.7")


def test_forwarded_values_from_untrusted_peer_do_not_change_authority() -> None:
    """A direct client cannot promote its scheme or authority via headers."""
    config = A2AHostSecurityConfig(
        trusted_proxies=frozenset({"10.0.0.5"}),
        trust_forwarded_host=True,
        trust_forwarded_proto=True,
        reject_untrusted_forwarded_headers=False,
        allowed_hosts=frozenset({"agent.example"}),
    )
    facts = evaluate_request(
        _scope(
            scheme="http",
            client=("203.0.113.9", 5000),
            headers={"x-forwarded-host": "public.example", "x-forwarded-proto": "https"},
        ),
        config=config,
    )
    assert (facts.host, facts.scheme, facts.via_trusted_proxy) == ("agent.example", "http", False)


def test_mtls_identity_header_from_client_is_never_trusted() -> None:
    """A client-supplied peer-subject header is stripped and never honored."""
    scope = _scope(headers={"x-conducto-mtls-subject": "CN=attacker"})
    facts = evaluate_request(scope, config=A2AHostSecurityConfig())
    assert facts.mtls_subject is None
    sanitized = sanitize_scope(scope, facts)
    assert not any(key == b"x-conducto-mtls-subject" for key, _ in sanitized["headers"])


def test_mtls_identity_is_accepted_only_through_the_server_seam() -> None:
    """Verified peer identity flows in through the explicit integration seam."""
    scope = _scope(headers={"x-conducto-mtls-subject": "CN=attacker"})
    facts = evaluate_request(
        scope,
        config=A2AHostSecurityConfig(),
        mtls_identity_extractor=lambda _scope: "CN=verified",
    )
    assert facts.mtls_subject == "CN=verified"
    sanitized = sanitize_scope(scope, facts)
    injected = [value for key, value in sanitized["headers"] if key == b"x-conducto-mtls-subject"]
    assert injected == [b"CN=verified"]


@pytest.mark.parametrize("subject", ["", "CN=" + "x" * 512, "CN=\x01bad"])
def test_malformed_peer_identity_from_the_seam_is_rejected(subject: str) -> None:
    """An integration seam returning a malformed subject fails the request."""
    with pytest.raises(A2ARequestRejectedError) as error:
        evaluate_request(
            _scope(),
            config=A2AHostSecurityConfig(),
            mtls_identity_extractor=lambda _scope: subject,
        )
    assert error.value.reason == "invalid_peer_identity"


def test_sanitize_scope_does_not_mutate_the_caller_scope() -> None:
    """Scope sanitization is isolating, so layered instances stay independent."""
    scope = _scope(headers={"x-conducto-client-address": "1.2.3.4"})
    original = list(scope["headers"])
    facts = evaluate_request(scope, config=A2AHostSecurityConfig())
    sanitized = sanitize_scope(scope, facts)
    assert scope["headers"] == original
    addresses = [
        value for key, value in sanitized["headers"] if key == b"x-conducto-client-address"
    ]
    assert addresses == [b"203.0.113.9"]


# --------------------------------------------------------------------------- #
# Payload limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_metadata_byte_boundaries(delta: int) -> None:
    """Metadata is bounded at, below, and above the configured byte budget."""
    config = A2AHostSecurityConfig(max_metadata_bytes=64)
    filler = "x" * max(0, config.max_metadata_bytes - len('{"k":""}') + delta)
    metadata = {"k": filler}
    encoded = len(json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    if encoded > config.max_metadata_bytes:
        with pytest.raises(A2APayloadLimitError):
            enforce_metadata_limit(metadata, config)
        return
    enforce_metadata_limit(metadata, config)


@pytest.mark.parametrize("delta", [-1, 0, 1])
@_sync
async def test_message_part_boundaries(delta: int) -> None:
    """Message part counts are accepted up to, and rejected above, the limit."""
    config = A2AHostSecurityConfig(max_message_parts=3)
    app, _, _ = _build_app(config=config)
    async with _client(app) as client:
        response = await client.post(
            RPC_PATH,
            json=_send_payload(_message(parts=max(1, config.max_message_parts + delta))),
            headers=_V1_HEADERS,
        )
    body = response.json()
    if delta > 0:
        assert "error" in body
        return
    assert "result" in body


@_sync
async def test_page_size_above_the_limit_is_rejected() -> None:
    """List page sizes above the configured maximum are rejected."""
    app, _, _ = _build_app(config=A2AHostSecurityConfig(max_page_size=5))
    async with _client(app) as client:
        response = await client.post(
            RPC_PATH,
            json={"jsonrpc": "2.0", "id": "1", "method": "ListTasks", "params": {"pageSize": 6}},
            headers=_V1_HEADERS,
        )
        accepted = await client.post(
            RPC_PATH,
            json={"jsonrpc": "2.0", "id": "2", "method": "ListTasks", "params": {"pageSize": 5}},
            headers=_V1_HEADERS,
        )
    assert "error" in response.json()
    assert "result" in accepted.json()


@_sync
async def test_history_above_the_limit_rejects_continuation() -> None:
    """A task whose history is at the bound cannot accept further messages."""
    store = InMemoryTaskRepository()
    task = Task(id="task-1", context_id="ctx-1")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.history.append(_message("first"))
    task.history.append(_message("second"))
    await store.create(task)
    app, _, _ = _build_app(config=A2AHostSecurityConfig(max_history_messages=1), repository=store)
    async with _client(app) as client:
        response = await client.post(
            RPC_PATH,
            json=_send_payload(_message(task_id="task-1")),
            headers=_V1_HEADERS,
        )
    assert "error" in response.json()


@_sync
async def test_request_body_above_the_limit_is_rejected_before_execution() -> None:
    """Oversized bodies are rejected without invoking the capability seam."""
    app, handler, _ = _build_app(config=A2AHostSecurityConfig(max_request_body_bytes=128))
    async with _client(app) as client:
        response = await client.post(
            RPC_PATH,
            json=_send_payload(_message("x" * 4096)),
            headers=_V1_HEADERS,
        )
    assert response.status_code == 413
    assert response.json() == {"reason": "request_body_too_large"}
    assert handler.entered.is_set() is False


@_sync
async def test_response_above_the_limit_returns_an_explicit_safe_failure() -> None:
    """A response that cannot be emitted within limits fails explicitly and safely."""
    app, _, _ = _build_app(config=A2AHostSecurityConfig(max_response_body_bytes=8))
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert response.status_code == 500
    assert response.json() == {"reason": "response_too_large"}


# --------------------------------------------------------------------------- #
# Concurrency admission
# --------------------------------------------------------------------------- #


def test_concurrency_limiter_boundaries() -> None:
    """The admission gate never queues and rejects deterministically at capacity."""
    limiter = A2AConcurrencyLimiter(limit=2, per_key_limit=1)
    assert limiter.try_acquire("a") is True
    assert limiter.try_acquire("a") is False
    assert limiter.try_acquire("b") is True
    assert limiter.try_acquire("c") is False
    limiter.release("a")
    assert limiter.try_acquire("c") is True
    assert limiter.in_flight == 2


@_sync
async def test_queue_saturation_rejects_without_partial_execution() -> None:
    """Over-capacity requests are rejected rather than queued or partly executed."""
    gate = asyncio.Event()
    handler = _Handler(gate=gate)
    app, _, _ = _build_app(
        config=A2AHostSecurityConfig(max_accepted_concurrency=1), handler=handler
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post(RPC_PATH, json=_send_payload(_message(), "1"), headers=_V1_HEADERS)
        )
        await handler.entered.wait()
        rejected = await client.post(
            RPC_PATH, json=_send_payload(_message(), "2"), headers=_V1_HEADERS
        )
        gate.set()
        accepted = await first
    assert rejected.status_code == 429
    assert rejected.json() == {"reason": "concurrency_limit_reached"}
    assert "result" in accepted.json()
    assert handler.peak == 1


@_sync
async def test_capability_concurrency_is_bounded_independently() -> None:
    """The in-flight capability gate bounds execution beyond request admission."""
    gate = asyncio.Event()
    handler = _Handler(gate=gate)
    app, _, _ = _build_app(
        config=A2AHostSecurityConfig(max_capability_concurrency=1), handler=handler
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post(RPC_PATH, json=_send_payload(_message(), "1"), headers=_V1_HEADERS)
        )
        await handler.entered.wait()
        second = await client.post(
            RPC_PATH, json=_send_payload(_message(), "2"), headers=_V1_HEADERS
        )
        gate.set()
        await first
    assert "error" in second.json()
    assert handler.peak == 1


def test_per_caller_isolation_keeps_distinct_principals_independent() -> None:
    """Per-caller bounds isolate principals so one cannot starve another."""
    limiter = A2AConcurrencyLimiter(limit=4, per_key_limit=1)
    assert limiter.try_acquire("mtls:CN=a") is True
    assert limiter.try_acquire("mtls:CN=a") is False
    assert limiter.try_acquire("mtls:CN=b") is True


# --------------------------------------------------------------------------- #
# Disconnect, cancellation, timeout, deadline, and drain
# --------------------------------------------------------------------------- #


@_sync
async def test_request_deadline_cancels_work_and_leaves_terminal_state() -> None:
    """A server deadline cancels work and drives the task out of an active state."""
    handler = _Handler(gate=asyncio.Event())
    app, _, store = _build_app(
        config=A2AHostSecurityConfig(
            request_deadline_seconds=0.05, cancellation_deadline_seconds=1.0
        ),
        handler=handler,
    )
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert response.status_code == 504
    assert response.json() == {"reason": "request_deadline_exceeded"}
    assert handler.cancelled.is_set()
    tasks, _ = await store.list(page_size=10)
    assert [task.status.state for task in tasks] == [TaskState.TASK_STATE_CANCELED]


@_sync
async def test_capability_timeout_is_distinct_and_terminal() -> None:
    """A capability deadline is distinct from a request deadline and fails the task."""
    handler = _Handler(gate=asyncio.Event())
    app, _, store = _build_app(
        config=A2AHostSecurityConfig(
            capability_deadline_seconds=0.05, request_deadline_seconds=10.0
        ),
        handler=handler,
    )
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert response.status_code == 200
    assert "error" in response.json()
    assert handler.cancel_calls
    tasks, _ = await store.list(page_size=10)
    assert [task.status.state for task in tasks] == [TaskState.TASK_STATE_FAILED]


@_sync
async def test_disconnect_after_acceptance_cancels_and_terminalizes_the_task() -> None:
    """A client disconnect after acceptance cancels work and leaves no active task."""
    handler = _Handler(gate=asyncio.Event())
    app, _, store = _build_app(handler=handler)
    body = json.dumps(_send_payload(_message())).encode("utf-8")
    disconnect = asyncio.Event()
    sent: list[Mapping[str, Any]] = []

    async def receive() -> Mapping[str, Any]:
        if not sent:
            sent.append({"sent": True})
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, Any]) -> None:
        raise AssertionError(f"no response expected after disconnect: {message['type']}")

    scope = _scope(headers={"content-length": str(len(body)), "a2a-version": "1.0"})
    served = asyncio.create_task(app(scope, receive, send))
    await handler.entered.wait()
    disconnect.set()
    await served
    assert handler.cancelled.is_set()
    tasks, _ = await store.list(page_size=10)
    assert [task.status.state for task in tasks] == [TaskState.TASK_STATE_CANCELED]


@_sync
async def test_disconnect_before_acceptance_never_executes_a_capability() -> None:
    """A disconnect before admission leaves no task and never runs a capability."""
    handler = _Handler()
    app, _, store = _build_app(handler=handler)

    async def receive() -> Mapping[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, Any]) -> None:
        del message

    await app(_scope(), receive, send)
    await asyncio.sleep(0)
    assert handler.entered.is_set() is False
    tasks, _ = await store.list(page_size=10)
    assert tasks == ()


@_sync
async def test_caller_cancellation_is_distinct_and_leaves_terminal_state() -> None:
    """Cancelling the serving task propagates cancellation and terminalizes work."""
    handler = _Handler(gate=asyncio.Event())
    app, _, store = _build_app(handler=handler)
    async with _client(app) as client:
        request = asyncio.create_task(
            client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
        )
        await handler.entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    await asyncio.sleep(0)
    assert handler.cancelled.is_set()
    tasks, _ = await store.list(page_size=10)
    assert [task.status.state for task in tasks] == [TaskState.TASK_STATE_CANCELED]


@_sync
async def test_explicit_cancellation_remains_owned_by_the_protocol_adapter() -> None:
    """An explicit CancelTask call still reaches the Story 4.4 dispatch path."""
    store = InMemoryTaskRepository()
    task = Task(id="task-9", context_id="ctx-9")
    task.status.state = TaskState.TASK_STATE_WORKING
    await store.create(task)
    app, _, _ = _build_app(repository=store)
    async with _client(app) as client:
        response = await client.post(
            RPC_PATH,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "CancelTask",
                "params": {"id": "task-9"},
            },
            headers=_V1_HEADERS,
        )
    assert response.status_code == 200
    stored = await store.get("task-9")
    assert stored is not None
    assert stored.status.state == TaskState.TASK_STATE_CANCELED


@_sync
async def test_drain_rejects_new_work_after_becoming_unready() -> None:
    """Drain marks the host unready first, then rejects newly arriving work."""
    app, _, _ = _build_app(config=A2AHostSecurityConfig(drain_deadline_seconds=0.05))
    assert app.is_ready() is True
    await app.drain()
    assert app.is_ready() is False
    assert app.is_alive() is True
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert response.status_code == 503
    assert response.json() == {"reason": "service_unavailable"}


@_sync
async def test_drain_grace_expiry_cancels_inflight_and_clears_active_tasks() -> None:
    """Work outstanding at grace expiry is cancelled and left non-ambiguous."""
    handler = _Handler(gate=asyncio.Event())
    app, _, store = _build_app(
        config=A2AHostSecurityConfig(drain_deadline_seconds=0.05, request_deadline_seconds=10.0),
        handler=handler,
    )
    async with _client(app) as client:
        request = asyncio.create_task(
            client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
        )
        await handler.entered.wait()
        await app.drain()
        await asyncio.gather(request, return_exceptions=True)
    assert app.lifecycle_state is A2AHostState.DRAINING
    tasks, _ = await store.list(page_size=10)
    assert all(
        task.status.state not in {TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING}
        for task in tasks
    )


# --------------------------------------------------------------------------- #
# Lifespan, readiness, and shutdown
# --------------------------------------------------------------------------- #


class _Lifespan:
    """Minimal deterministic ASGI lifespan driver."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self.outbox: list[Mapping[str, Any]] = []

    async def receive(self) -> Mapping[str, Any]:
        """Return the next queued lifespan message."""
        return await self.inbox.get()

    async def send(self, message: Mapping[str, Any]) -> None:
        """Record one lifespan message emitted by the application."""
        self.outbox.append(message)


@_sync
async def test_lifespan_startup_and_shutdown_complete() -> None:
    """A host with a dependency check becomes ready only after lifespan startup."""
    started = asyncio.Event()

    async def on_startup() -> None:
        started.set()

    app, _, _ = _build_app(on_startup=on_startup)
    assert app.is_ready() is False
    driver = _Lifespan()
    await driver.inbox.put({"type": "lifespan.startup"})
    await driver.inbox.put({"type": "lifespan.shutdown"})
    await app({"type": "lifespan"}, driver.receive, driver.send)
    assert started.is_set()
    assert [message["type"] for message in driver.outbox] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
    assert app.is_alive() is False


@_sync
async def test_lifespan_startup_failure_keeps_the_host_unready() -> None:
    """A failed dependency check reports startup failure and never becomes ready."""

    async def on_startup() -> None:
        raise RuntimeError("database credentials rejected at db.internal:5432")

    app, _, _ = _build_app(on_startup=on_startup)
    driver = _Lifespan()
    await driver.inbox.put({"type": "lifespan.startup"})
    await app({"type": "lifespan"}, driver.receive, driver.send)
    assert driver.outbox[0]["type"] == "lifespan.startup.failed"
    assert driver.outbox[0]["message"] == "startup_failed"
    assert app.is_ready() is False
    assert app.lifecycle_state is A2AHostState.FAILED


@_sync
async def test_startup_deadline_failure_is_reported_without_detail() -> None:
    """A startup check that never completes fails with a stable reason code."""

    async def on_startup() -> None:
        await asyncio.Event().wait()

    app, _, _ = _build_app(
        config=A2AHostSecurityConfig(startup_deadline_seconds=0.05), on_startup=on_startup
    )
    with pytest.raises(A2AStartupError) as error:
        await app.startup()
    assert error.value.reason == "startup_timeout"
    assert app.is_ready() is False


@_sync
async def test_probe_endpoints_expose_no_dependency_detail() -> None:
    """Liveness and readiness expose only coarse status, never dependency detail."""
    app, _, _ = _build_app(
        config=A2AHostSecurityConfig(liveness_path="/livez", readiness_path="/readyz")
    )
    async with _client(app) as client:
        alive = await client.get("/livez")
        ready = await client.get("/readyz")
        await app.drain()
        unready = await client.get("/readyz")
        still_alive = await client.get("/livez")
    assert alive.json() == {"status": "alive"}
    assert ready.json() == {"status": "ready"}
    assert unready.status_code == 503
    assert unready.json() == {"status": "unready"}
    assert still_alive.status_code == 200


@_sync
async def test_aclose_is_idempotent() -> None:
    """Closing twice performs no further work and raises nothing."""
    app, _, _ = _build_app()
    await app.aclose()
    await app.aclose()
    assert app.is_alive() is False
    assert app.lifecycle_state is A2AHostState.CLOSED


@_sync
async def test_cleanup_failure_is_reported_not_hidden() -> None:
    """A failing resource closer surfaces a stable reason instead of being hidden."""

    async def failing() -> None:
        raise RuntimeError("token cache at https://secrets.internal failed")

    app, _, _ = _build_app(resource_closers={"token_cache": failing})
    with pytest.raises(A2AShutdownError) as error:
        await app.aclose()
    assert error.value.reasons == ("token_cache_failed",)
    assert "secrets.internal" not in str(error.value)


@_sync
async def test_lifespan_shutdown_failure_reports_reason_codes() -> None:
    """Lifespan shutdown reports cleanup failures through stable reason codes."""

    async def failing() -> None:
        raise RuntimeError("boom")

    app, _, _ = _build_app(resource_closers={"audit": failing})
    driver = _Lifespan()
    await driver.inbox.put({"type": "lifespan.shutdown"})
    await app({"type": "lifespan"}, driver.receive, driver.send)
    assert driver.outbox[-1]["type"] == "lifespan.shutdown.failed"
    assert driver.outbox[-1]["message"] == "audit_failed"


@_sync
async def test_close_sweeps_ambiguous_active_tasks() -> None:
    """Accepted tasks never remain indefinitely in an ambiguous active state."""
    store = InMemoryTaskRepository()
    stuck = Task(id="stuck", context_id="ctx")
    stuck.status.state = TaskState.TASK_STATE_WORKING
    await store.create(stuck)
    app, _, _ = _build_app(repository=store)
    await app.aclose()
    remaining = await store.get("stuck")
    assert remaining is not None
    assert remaining.status.state == TaskState.TASK_STATE_FAILED


@_sync
async def test_requests_are_rejected_after_close() -> None:
    """A closed host accepts no further work."""
    app, _, _ = _build_app()
    await app.aclose()
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert response.status_code == 503


@_sync
async def test_lifecycle_uses_the_injected_clock() -> None:
    """Bounded shutdown phases read time only from the injected clock."""
    reads: list[float] = []

    def clock() -> float:
        reads.append(len(reads))
        return float(len(reads))

    lifecycle = A2AHostLifecycle(
        config=A2AHostSecurityConfig(),
        task_repository=InMemoryTaskRepository(),
        clock=clock,
    )
    lifecycle.mark_ready()
    await lifecycle.aclose()
    assert reads


# --------------------------------------------------------------------------- #
# Instance isolation and diagnostic hygiene
# --------------------------------------------------------------------------- #


@_sync
async def test_instances_have_independent_configuration_and_lifecycle() -> None:
    """Two hosts never share configuration, readiness, concurrency, or tasks."""
    first, _, first_store = _build_app(config=A2AHostSecurityConfig(max_page_size=5))
    second, _, second_store = _build_app(config=A2AHostSecurityConfig(max_page_size=9))
    await first.aclose()
    assert first.is_alive() is False
    assert second.is_alive() is True
    assert second.is_ready() is True
    assert first.security_config.max_page_size != second.security_config.max_page_size
    assert first_store is not second_store


@_sync
async def test_agent_card_and_errors_expose_no_private_detail() -> None:
    """Cards and rejections never leak credentials, endpoints, or exception text."""
    app, _, _ = _build_app()
    async with _client(app) as client:
        card = await client.get(CARD_PATH)
        rejection = await client.post(
            RPC_PATH,
            json=_send_payload(_message()),
            headers={**_V1_HEADERS, "Authorization": "Bearer not valid"},
        )
    assert "Bearer" not in card.text
    assert rejection.status_code == 400
    assert rejection.json() == {"reason": "invalid_authorization"}


@_sync
async def test_handler_exception_text_is_never_returned() -> None:
    """A failing capability seam never surfaces its exception text to a caller."""

    class _Failing:
        async def handle_message(
            self,
            message: Message,
            *,
            task_id: str,
            context_id: str,
            request_context: A2ARequestContext,
        ) -> InvocationResult:
            """Always fail with a message that must never reach the caller."""
            del message, task_id, context_id, request_context
            raise RuntimeError("secret connection string Server=db;Pwd=hunter2")

    app = A2AASGI(
        agent=_EchoAgent(),
        endpoint_url=ENDPOINT_URL,
        task_repository=InMemoryTaskRepository(),
        request_handler=_Failing(),
    )
    async with _client(app) as client:
        response = await client.post(RPC_PATH, json=_send_payload(_message()), headers=_V1_HEADERS)
    assert "hunter2" not in response.text
    assert "error" in response.json()


@_sync
async def test_non_http_scopes_are_closed() -> None:
    """Non-HTTP connections are refused; this profile pins JSON-RPC over HTTP."""
    app, _, _ = _build_app()
    outbox: list[Mapping[str, Any]] = []

    async def receive() -> Mapping[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: Mapping[str, Any]) -> None:
        outbox.append(message)

    await app({"type": "websocket"}, receive, send)
    assert outbox == [{"type": "websocket.close", "code": 1008}]
