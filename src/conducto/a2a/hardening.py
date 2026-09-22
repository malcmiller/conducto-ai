"""Immutable transport hardening configuration for the inbound A2A ASGI host.

This module owns validated host, proxy, header, body, payload, concurrency, and
deadline configuration plus the credential-free transport facts derived from one
ASGI scope. It never imports Starlette, never performs capability execution, and
never retains credentials: bearer tokens and mTLS material are inspected for
shape only and are never copied into facts, logs, traces, or retained state.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from conducto.core.a2a_profile import (
    MAX_ARTIFACT_PARTS,
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_PARTS,
    MAX_METADATA_BYTES,
    MAX_TASK_ARTIFACTS,
)

from .errors import A2AHostConfigurationError, A2APayloadLimitError, A2ARequestRejectedError

AUTHORIZATION_HEADER: Final = "authorization"
CORRELATION_HEADER: Final = "x-correlation-id"
TRACEPARENT_HEADER: Final = "traceparent"
TRACESTATE_HEADER: Final = "tracestate"
MTLS_SUBJECT_HEADER: Final = "x-conducto-mtls-subject"
CLIENT_ADDRESS_HEADER: Final = "x-conducto-client-address"

RESERVED_HEADERS: Final = frozenset({MTLS_SUBJECT_HEADER, CLIENT_ADDRESS_HEADER})
FORWARDED_HEADERS: Final = frozenset(
    {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-port", "x-forwarded-proto"}
)
_DEFAULT_SCHEME_PORTS: Final = {"http": 80, "https": 443}
_TRACEPARENT_PATTERN: Final = re.compile(r"\A[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}\Z")
_TRACESTATE_PATTERN: Final = re.compile(r"\A[\x20-\x7e]{1,512}\Z")
_TOKEN_PATTERN: Final = re.compile(r"\A[A-Za-z0-9\-._~+/]+=*\Z")
_CORRELATION_PATTERN: Final = re.compile(r"\A[\x21-\x7e]{1,256}\Z")
_SUBJECT_PATTERN: Final = re.compile(r"\A[\x20-\x7e]{1,256}\Z")
_HOST_PATTERN: Final = re.compile(
    r"\A[a-z0-9]([a-z0-9\-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]*[a-z0-9])?)*\Z"
)
_MAX_SUBJECT_BYTES: Final = 256


@runtime_checkable
class A2AMTLSIdentityExtractor(Protocol):
    """Explicit server integration seam for verified mTLS peer identity.

    The embedding server, not the remote caller, owns this seam. A verified peer
    subject is never accepted from an ordinary request header, so a client can
    never assert transport identity by setting one.
    """

    def __call__(self, scope: Mapping[str, Any]) -> str | None:
        """Return the verified mTLS peer subject for one ASGI scope.

        Args:
            scope: Raw inbound ASGI connection scope.

        Returns:
            The verified peer subject, or ``None`` when the connection is not
            mutually authenticated.
        """
        ...


@dataclass(frozen=True, slots=True)
class A2AHostSecurityConfig:
    """Immutable, validated hardening policy for one inbound A2A ASGI host.

    Attributes:
        allowed_schemes: Public request schemes accepted at the boundary.
        allowed_hosts: Exact lowercase hosts accepted in the effective authority.
            An empty set accepts any syntactically valid host.
        allowed_ports: Ports accepted in the effective authority. An empty set
            accepts any port.
        allowed_authorities: Exact lowercase ``host`` or ``host:port`` values. When
            non-empty it takes precedence over ``allowed_hosts``/``allowed_ports``.
        allowed_paths: Exact request paths served by this host. An empty set means
            the mounted A2A and Agent Card routes are derived automatically.
        allowed_methods: HTTP methods accepted before routing.
        allowed_content_types: Request media types accepted for bodied methods.
        trusted_proxies: Immediate transport peer addresses permitted to assert
            forwarded host, scheme, and client information.
        trust_forwarded_host: Whether a trusted proxy may override the authority.
        trust_forwarded_proto: Whether a trusted proxy may override the scheme.
        trust_forwarded_for: Whether a trusted proxy may override the client.
        reject_untrusted_forwarded_headers: Whether forwarded headers from an
            untrusted peer are rejected. When ``False`` they are stripped instead.
        max_header_count: Maximum number of inbound headers.
        max_total_header_bytes: Maximum aggregate inbound header bytes.
        max_authorization_header_bytes: Maximum ``Authorization`` header bytes.
        max_correlation_header_bytes: Maximum correlation header bytes.
        max_trace_header_bytes: Maximum W3C trace header bytes.
        max_request_body_bytes: Maximum accepted request body bytes.
        max_request_body_chunks: Maximum accepted request body chunks.
        max_response_body_bytes: Maximum emitted response body bytes.
        max_metadata_bytes: Maximum serialized metadata bytes.
        max_message_parts: Maximum parts in one inbound message.
        max_artifact_parts: Maximum parts in one emitted artifact.
        max_task_artifacts: Maximum artifacts on one task.
        max_history_messages: Maximum retained history messages on one task.
        max_page_size: Maximum accepted task list page size.
        max_retained_tasks: Maximum retained tasks in the default task store.
        max_accepted_concurrency: Maximum concurrently accepted requests.
        max_caller_concurrency: Maximum concurrently accepted requests per caller.
        max_capability_concurrency: Maximum concurrent in-flight capability calls.
        request_deadline_seconds: Server-imposed deadline for one HTTP request.
        capability_deadline_seconds: Maximum capability execution budget.
        cancellation_deadline_seconds: Bounded wait for cooperative cancellation.
        startup_deadline_seconds: Bounded wait for lifespan startup.
        shutdown_deadline_seconds: Bounded wait for lifespan shutdown.
        drain_deadline_seconds: Bounded wait for accepted work during drain.
        task_store_flush_deadline_seconds: Bounded wait for task-store cleanup.
        liveness_path: Optional exact path serving a dependency-free liveness probe.
        readiness_path: Optional exact path serving a readiness probe.

    Raises:
        A2AHostConfigurationError: If any value is malformed, non-positive, or
            internally inconsistent.
    """

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    allowed_hosts: frozenset[str] = frozenset()
    allowed_ports: frozenset[int] = frozenset()
    allowed_authorities: frozenset[str] = frozenset()
    allowed_paths: frozenset[str] = frozenset()
    allowed_methods: frozenset[str] = frozenset({"GET", "POST"})
    allowed_content_types: frozenset[str] = frozenset({"application/json"})
    trusted_proxies: frozenset[str] = frozenset()
    trust_forwarded_host: bool = False
    trust_forwarded_proto: bool = False
    trust_forwarded_for: bool = False
    reject_untrusted_forwarded_headers: bool = True
    max_header_count: int = 64
    max_total_header_bytes: int = 16_384
    max_authorization_header_bytes: int = 4_096
    max_correlation_header_bytes: int = 256
    max_trace_header_bytes: int = 512
    max_request_body_bytes: int = 262_144
    max_request_body_chunks: int = 512
    max_response_body_bytes: int = 1_048_576
    max_metadata_bytes: int = MAX_METADATA_BYTES
    max_message_parts: int = MAX_MESSAGE_PARTS
    max_artifact_parts: int = MAX_ARTIFACT_PARTS
    max_task_artifacts: int = MAX_TASK_ARTIFACTS
    max_history_messages: int = MAX_HISTORY_MESSAGES
    max_page_size: int = 50
    max_retained_tasks: int = 1_000
    max_accepted_concurrency: int = 64
    max_caller_concurrency: int = 16
    max_capability_concurrency: int = 32
    request_deadline_seconds: float = 30.0
    capability_deadline_seconds: float = 300.0
    cancellation_deadline_seconds: float = 5.0
    startup_deadline_seconds: float = 30.0
    shutdown_deadline_seconds: float = 30.0
    drain_deadline_seconds: float = 30.0
    task_store_flush_deadline_seconds: float = 5.0
    liveness_path: str | None = None
    readiness_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate every configured bound before the host serves."""
        object.__setattr__(self, "allowed_schemes", _normalize_schemes(self.allowed_schemes))
        object.__setattr__(self, "allowed_hosts", _normalize_hosts(self.allowed_hosts))
        object.__setattr__(self, "allowed_ports", _normalize_ports(self.allowed_ports))
        object.__setattr__(
            self, "allowed_authorities", _normalize_authorities(self.allowed_authorities)
        )
        object.__setattr__(
            self, "allowed_paths", frozenset(_validate_path(path) for path in self.allowed_paths)
        )
        object.__setattr__(
            self,
            "allowed_methods",
            frozenset(_validate_method(method) for method in self.allowed_methods),
        )
        object.__setattr__(
            self,
            "allowed_content_types",
            frozenset(_validate_media_type(value) for value in self.allowed_content_types),
        )
        object.__setattr__(self, "trusted_proxies", _normalize_proxies(self.trusted_proxies))
        if not self.allowed_methods:
            raise A2AHostConfigurationError("allowed_methods must not be empty")
        if not self.allowed_content_types:
            raise A2AHostConfigurationError("allowed_content_types must not be empty")
        if not self.allowed_schemes:
            raise A2AHostConfigurationError("allowed_schemes must not be empty")
        trusts_forwarded = (
            self.trust_forwarded_host or self.trust_forwarded_proto or self.trust_forwarded_for
        )
        if trusts_forwarded and not self.trusted_proxies:
            raise A2AHostConfigurationError(
                "trusted_proxies is required before forwarded headers can be trusted"
            )
        for name, value in _int_limits(self).items():
            if value <= 0:
                raise A2AHostConfigurationError(f"{name} must be a positive integer")
        for name, seconds in _deadlines(self).items():
            if not math.isfinite(seconds) or seconds <= 0:
                raise A2AHostConfigurationError(f"{name} must be a finite positive number")
        for name, probe in (
            ("liveness_path", self.liveness_path),
            ("readiness_path", self.readiness_path),
        ):
            if probe is not None:
                object.__setattr__(self, name, _validate_path(probe))
        if (
            self.liveness_path is not None
            and self.readiness_path is not None
            and self.liveness_path == self.readiness_path
        ):
            raise A2AHostConfigurationError("liveness_path and readiness_path must differ")


@dataclass(frozen=True, slots=True)
class A2ATransportFacts:
    """Credential-free transport facts derived from one validated ASGI scope.

    Attributes:
        method: Validated uppercase HTTP method.
        path: Validated exact request path.
        scheme: Effective public scheme after trusted-proxy resolution.
        host: Effective lowercase host after trusted-proxy resolution.
        port: Effective port, or ``None`` when the authority omits one.
        peer: Immediate transport peer address, if the server supplied one.
        client: Effective client address after trusted-proxy resolution.
        via_trusted_proxy: Whether the immediate peer is a configured proxy.
        has_bearer: Whether a well-formed bearer credential was presented. The
            credential value itself is never retained here.
        traceparent: Valid inbound W3C ``traceparent``, or ``""`` when absent or
            malformed.
        tracestate: Valid inbound W3C ``tracestate``, or ``""``.
        correlation_id: Validated inbound correlation identifier, or ``""``.
        mtls_subject: Verified peer subject from the server integration seam.
    """

    method: str
    path: str
    scheme: str
    host: str
    port: int | None
    peer: str | None
    client: str | None
    via_trusted_proxy: bool
    has_bearer: bool
    traceparent: str = ""
    tracestate: str = ""
    correlation_id: str = ""
    mtls_subject: str | None = None
    dropped_headers: frozenset[str] = field(default_factory=frozenset)

    @property
    def authority(self) -> str:
        """Return the effective normalized ``host`` or ``host:port`` authority."""
        return self.host if self.port is None else f"{self.host}:{self.port}"

    @property
    def caller_key(self) -> str:
        """Return a stable, credential-free key used for per-caller concurrency."""
        if self.mtls_subject:
            return f"mtls:{self.mtls_subject}"
        return f"peer:{self.client or self.peer or 'unknown'}"


def evaluate_request(
    scope: Mapping[str, Any],
    *,
    config: A2AHostSecurityConfig,
    mtls_identity_extractor: A2AMTLSIdentityExtractor | None = None,
) -> A2ATransportFacts:
    """Validate one inbound HTTP scope before any body is read or work executes.

    Args:
        scope: Raw inbound ASGI HTTP scope.
        config: Immutable hardening policy applied to this host.
        mtls_identity_extractor: Optional server-owned verified-identity seam.

    Returns:
        Credential-free transport facts for the accepted request.

    Raises:
        A2ARequestRejectedError: If the request violates any configured bound,
            asserts untrusted forwarded information, or is otherwise malformed.
    """
    method = str(scope.get("method", "GET")).upper()
    if method not in config.allowed_methods:
        raise A2ARequestRejectedError("method_not_allowed", status_code=405)
    headers = _decode_headers(scope.get("headers", ()), config)
    peer = _peer_address(scope.get("client"))
    via_trusted_proxy = peer is not None and peer in config.trusted_proxies
    dropped = _resolve_forwarded_headers(headers, config, via_trusted_proxy=via_trusted_proxy)
    scheme = _effective_scheme(scope, headers, config, via_trusted_proxy=via_trusted_proxy)
    host, port = _effective_authority(headers, config, via_trusted_proxy=via_trusted_proxy)
    path = _validate_request_path(scope, config)
    _validate_content(headers, config, method=method)
    has_bearer = _validate_authorization(headers, config)
    correlation_id = _validate_correlation(headers, config)
    traceparent, tracestate, trace_dropped = _validate_trace(headers, config)
    subject = _resolve_mtls_subject(scope, mtls_identity_extractor)
    client = _effective_client(headers, peer, config, via_trusted_proxy=via_trusted_proxy)
    return A2ATransportFacts(
        method=method,
        path=path,
        scheme=scheme,
        host=host,
        port=port,
        peer=peer,
        client=client,
        via_trusted_proxy=via_trusted_proxy,
        has_bearer=has_bearer,
        traceparent=traceparent,
        tracestate=tracestate,
        correlation_id=correlation_id,
        mtls_subject=subject,
        dropped_headers=dropped | trace_dropped | RESERVED_HEADERS,
    )


def sanitize_scope(scope: Mapping[str, Any], facts: A2ATransportFacts) -> dict[str, Any]:
    """Return an isolated scope carrying only server-asserted transport identity.

    Args:
        scope: Raw inbound ASGI HTTP scope.
        facts: Validated transport facts for the same request.

    Returns:
        A new scope whose headers exclude reserved, untrusted, and malformed
        values and carry the server-asserted peer subject and client address.

    Notes:
        The caller's scope is never mutated, so independent application instances
        and middleware layers remain isolated.
    """
    sanitized = dict(scope)
    raw: list[tuple[bytes, bytes]] = [
        (key, value)
        for key, value in scope.get("headers", ())
        if key.decode("latin-1").lower() not in facts.dropped_headers
    ]
    raw = [(key, value) for key, value in raw if key.decode("latin-1").lower() != "host"]
    raw.append((b"host", facts.authority.encode("latin-1")))
    if facts.client:
        raw.append((CLIENT_ADDRESS_HEADER.encode("latin-1"), facts.client.encode("latin-1")))
    if facts.mtls_subject:
        raw.append((MTLS_SUBJECT_HEADER.encode("latin-1"), facts.mtls_subject.encode("latin-1")))
    sanitized["headers"] = raw
    sanitized["scheme"] = facts.scheme
    return sanitized


def enforce_metadata_limit(metadata: Any, config: A2AHostSecurityConfig) -> None:
    """Reject metadata whose canonical encoding exceeds the configured budget.

    Args:
        metadata: Candidate metadata mapping, or ``None``.
        config: Immutable hardening policy applied to this host.

    Raises:
        A2APayloadLimitError: If the encoded metadata exceeds the limit.
    """
    if metadata is None:
        return
    encoded = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > config.max_metadata_bytes:
        raise A2APayloadLimitError("metadata_bytes_exceeded")


def _decode_headers(raw: Any, config: A2AHostSecurityConfig) -> dict[str, list[str]]:
    """Decode and bound inbound headers before any body byte is read."""
    items = list(raw)
    if len(items) > config.max_header_count:
        raise A2ARequestRejectedError("header_count_exceeded", status_code=431)
    total = 0
    decoded: dict[str, list[str]] = {}
    for key, value in items:
        total += len(key) + len(value) + 2
        if total > config.max_total_header_bytes:
            raise A2ARequestRejectedError("header_bytes_exceeded", status_code=431)
        name = key.decode("latin-1").lower()
        decoded.setdefault(name, []).append(value.decode("latin-1").strip())
    return decoded


def _single(headers: Mapping[str, list[str]], name: str) -> str | None:
    """Return one header value, rejecting ambiguous repeated occurrences."""
    values = headers.get(name)
    if not values:
        return None
    if len(values) > 1:
        raise A2ARequestRejectedError("ambiguous_header", status_code=400)
    return values[0]


def _peer_address(client: Any) -> str | None:
    """Return the immediate transport peer address supplied by the server."""
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return None


def _resolve_forwarded_headers(
    headers: Mapping[str, list[str]],
    config: A2AHostSecurityConfig,
    *,
    via_trusted_proxy: bool,
) -> frozenset[str]:
    """Reject or strip forwarded headers that an untrusted peer asserted."""
    present = {name for name in FORWARDED_HEADERS if headers.get(name)}
    if not present or via_trusted_proxy:
        return frozenset()
    if config.reject_untrusted_forwarded_headers:
        raise A2ARequestRejectedError("untrusted_forwarded_header", status_code=400)
    return frozenset(present)


def _effective_scheme(
    scope: Mapping[str, Any],
    headers: Mapping[str, list[str]],
    config: A2AHostSecurityConfig,
    *,
    via_trusted_proxy: bool,
) -> str:
    """Resolve the public scheme, trusting forwarded values only from proxies."""
    scheme = str(scope.get("scheme", "http")).lower()
    if via_trusted_proxy and config.trust_forwarded_proto:
        forwarded = _single(headers, "x-forwarded-proto")
        if forwarded:
            scheme = forwarded.split(",")[0].strip().lower()
    if scheme not in config.allowed_schemes:
        raise A2ARequestRejectedError("scheme_not_allowed", status_code=400)
    return scheme


def _effective_authority(
    headers: Mapping[str, list[str]],
    config: A2AHostSecurityConfig,
    *,
    via_trusted_proxy: bool,
) -> tuple[str, int | None]:
    """Resolve and authorize the public authority asserted for this request."""
    authority = _single(headers, "host")
    if via_trusted_proxy and config.trust_forwarded_host:
        forwarded = _single(headers, "x-forwarded-host")
        if forwarded:
            authority = forwarded.split(",")[0].strip()
    if not authority:
        raise A2ARequestRejectedError("missing_host", status_code=400)
    host, port = _split_authority(authority)
    if config.allowed_authorities:
        candidates = {host if port is None else f"{host}:{port}", host}
        if not candidates & config.allowed_authorities:
            raise A2ARequestRejectedError("authority_not_allowed", status_code=400)
        return host, port
    if config.allowed_hosts and host not in config.allowed_hosts:
        raise A2ARequestRejectedError("host_not_allowed", status_code=400)
    if config.allowed_ports and (port or 0) not in config.allowed_ports:
        raise A2ARequestRejectedError("port_not_allowed", status_code=400)
    return host, port


def _split_authority(authority: str) -> tuple[str, int | None]:
    """Split and validate a ``host`` or ``host:port`` authority."""
    value = authority.strip().lower()
    if not value or len(value) > 255:
        raise A2ARequestRejectedError("invalid_host", status_code=400)
    host = value
    port: int | None = None
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            raise A2ARequestRejectedError("invalid_host", status_code=400)
        host = value[: closing + 1]
        remainder = value[closing + 1 :]
        if remainder:
            if not remainder.startswith(":"):
                raise A2ARequestRejectedError("invalid_host", status_code=400)
            port = _parse_port(remainder[1:])
    elif ":" in value:
        host, _, raw_port = value.rpartition(":")
        port = _parse_port(raw_port)
    if not _is_valid_host(host):
        raise A2ARequestRejectedError("invalid_host", status_code=400)
    return host, port


def _parse_port(raw: str) -> int:
    """Parse and range-check an authority port."""
    if not raw.isdigit():
        raise A2ARequestRejectedError("invalid_host", status_code=400)
    port = int(raw)
    if not 1 <= port <= 65535:
        raise A2ARequestRejectedError("invalid_host", status_code=400)
    return port


def _is_valid_host(host: str) -> bool:
    """Return whether a host is a valid DNS name or IP literal."""
    if host.startswith("[") and host.endswith("]"):
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ValueError:
            return False
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return bool(_HOST_PATTERN.match(host)) and len(host) <= 253
    return True


def _validate_request_path(scope: Mapping[str, Any], config: A2AHostSecurityConfig) -> str:
    """Validate the exact request path against the configured route surface."""
    path = str(scope.get("path", "/")) or "/"
    if not path.startswith("/") or any(character in path for character in ("\r", "\n", " ")):
        raise A2ARequestRejectedError("invalid_path", status_code=400)
    if config.allowed_paths and path not in config.allowed_paths:
        raise A2ARequestRejectedError("path_not_allowed", status_code=404)
    return path


def _validate_content(
    headers: Mapping[str, list[str]], config: A2AHostSecurityConfig, *, method: str
) -> None:
    """Validate content framing, encoding, and length before reading the body."""
    encoding = _single(headers, "content-encoding")
    if encoding and encoding.lower() != "identity":
        raise A2ARequestRejectedError("unsupported_content_encoding", status_code=415)
    length = _single(headers, "content-length")
    if length is not None:
        if not length.isdigit():
            raise A2ARequestRejectedError("invalid_content_length", status_code=400)
        if int(length) > config.max_request_body_bytes:
            raise A2ARequestRejectedError("request_body_too_large", status_code=413)
    if method not in {"POST", "PUT", "PATCH"}:
        return
    content_type = _single(headers, "content-type")
    if content_type is None:
        raise A2ARequestRejectedError("missing_content_type", status_code=415)
    media_type, _, parameters = content_type.partition(";")
    if media_type.strip().lower() not in config.allowed_content_types:
        raise A2ARequestRejectedError("unsupported_media_type", status_code=415)
    for parameter in parameters.split(";"):
        name, _, value = parameter.partition("=")
        if name.strip().lower() == "charset" and value.strip().strip('"').lower() not in {
            "utf-8",
            "utf8",
        }:
            raise A2ARequestRejectedError("unsupported_charset", status_code=415)


def _validate_authorization(
    headers: Mapping[str, list[str]], config: A2AHostSecurityConfig
) -> bool:
    """Validate bearer credential shape without retaining or logging the value."""
    header = _single(headers, AUTHORIZATION_HEADER)
    if header is None:
        return False
    if len(header.encode("latin-1")) > config.max_authorization_header_bytes:
        raise A2ARequestRejectedError("authorization_header_too_large", status_code=431)
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not _TOKEN_PATTERN.match(credential.strip()):
        raise A2ARequestRejectedError("invalid_authorization", status_code=400)
    return True


def _validate_correlation(headers: Mapping[str, list[str]], config: A2AHostSecurityConfig) -> str:
    """Validate and bound the inbound correlation identifier."""
    header = _single(headers, CORRELATION_HEADER)
    if header is None:
        return ""
    if len(header.encode("latin-1")) > config.max_correlation_header_bytes:
        raise A2ARequestRejectedError("correlation_header_too_large", status_code=431)
    if not _CORRELATION_PATTERN.match(header):
        raise A2ARequestRejectedError("invalid_correlation_header", status_code=400)
    return header


def _validate_trace(
    headers: Mapping[str, list[str]], config: A2AHostSecurityConfig
) -> tuple[str, str, frozenset[str]]:
    """Validate W3C trace headers, dropping malformed values before propagation."""
    dropped: set[str] = set()
    traceparent = _single(headers, TRACEPARENT_HEADER) or ""
    tracestate = _single(headers, TRACESTATE_HEADER) or ""
    for value in (traceparent, tracestate):
        if value and len(value.encode("latin-1")) > config.max_trace_header_bytes:
            raise A2ARequestRejectedError("trace_header_too_large", status_code=431)
    if traceparent and not _TRACEPARENT_PATTERN.match(traceparent.lower()):
        dropped.add(TRACEPARENT_HEADER)
        dropped.add(TRACESTATE_HEADER)
        traceparent = ""
        tracestate = ""
    if tracestate and not _TRACESTATE_PATTERN.match(tracestate):
        dropped.add(TRACESTATE_HEADER)
        tracestate = ""
    return traceparent.lower(), tracestate, frozenset(dropped)


def _resolve_mtls_subject(
    scope: Mapping[str, Any], extractor: A2AMTLSIdentityExtractor | None
) -> str | None:
    """Resolve verified peer identity only through the server integration seam."""
    if extractor is None:
        return None
    subject = extractor(scope)
    if subject is None:
        return None
    if not isinstance(subject, str) or len(subject.encode("utf-8")) > _MAX_SUBJECT_BYTES:
        raise A2ARequestRejectedError("invalid_peer_identity", status_code=400)
    if not _SUBJECT_PATTERN.match(subject):
        raise A2ARequestRejectedError("invalid_peer_identity", status_code=400)
    return subject


def _effective_client(
    headers: Mapping[str, list[str]],
    peer: str | None,
    config: A2AHostSecurityConfig,
    *,
    via_trusted_proxy: bool,
) -> str | None:
    """Resolve the effective client address, trusting proxies only when configured."""
    if via_trusted_proxy and config.trust_forwarded_for:
        forwarded = _single(headers, "x-forwarded-for")
        if forwarded:
            candidate = forwarded.split(",")[0].strip()
            if candidate:
                return candidate
    return peer


def _normalize_schemes(values: frozenset[str]) -> frozenset[str]:
    """Normalize and validate configured public schemes."""
    normalized = {str(value).strip().lower() for value in values}
    unsupported = normalized - set(_DEFAULT_SCHEME_PORTS)
    if unsupported:
        raise A2AHostConfigurationError(f"unsupported schemes: {sorted(unsupported)}")
    return frozenset(normalized)


def _normalize_hosts(values: frozenset[str]) -> frozenset[str]:
    """Normalize and validate configured public hosts."""
    normalized = {str(value).strip().lower() for value in values}
    for host in normalized:
        if not host or not _is_valid_host(host):
            raise A2AHostConfigurationError(f"invalid allowed host: {host!r}")
    return frozenset(normalized)


def _normalize_ports(values: frozenset[int]) -> frozenset[int]:
    """Validate configured public ports."""
    for port in values:
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise A2AHostConfigurationError(f"invalid allowed port: {port!r}")
    return frozenset(values)


def _normalize_authorities(values: frozenset[str]) -> frozenset[str]:
    """Normalize and validate configured exact authorities."""
    normalized: set[str] = set()
    for value in values:
        candidate = str(value).strip().lower()
        try:
            host, port = _split_authority(candidate)
        except A2ARequestRejectedError as error:
            raise A2AHostConfigurationError(f"invalid allowed authority: {value!r}") from error
        normalized.add(host if port is None else f"{host}:{port}")
    return frozenset(normalized)


def _normalize_proxies(values: frozenset[str]) -> frozenset[str]:
    """Validate configured trusted proxy peer addresses."""
    normalized: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        try:
            ipaddress.ip_address(candidate)
        except ValueError as error:
            raise A2AHostConfigurationError(f"invalid trusted proxy address: {value!r}") from error
        normalized.add(candidate)
    return frozenset(normalized)


def _validate_path(path: str) -> str:
    """Validate one exact configured request path."""
    if not isinstance(path, str) or not path.startswith("/") or path != path.strip():
        raise A2AHostConfigurationError(f"invalid configured path: {path!r}")
    if any(character in path for character in ("?", "#", " ", "\r", "\n")):
        raise A2AHostConfigurationError(f"invalid configured path: {path!r}")
    return path


def _validate_method(method: str) -> str:
    """Validate one configured HTTP method."""
    candidate = str(method).strip().upper()
    if not candidate.isalpha():
        raise A2AHostConfigurationError(f"invalid configured method: {method!r}")
    return candidate


def _validate_media_type(value: str) -> str:
    """Validate one configured request media type."""
    candidate = str(value).strip().lower()
    if "/" not in candidate or ";" in candidate:
        raise A2AHostConfigurationError(f"invalid configured media type: {value!r}")
    return candidate


def _int_limits(config: A2AHostSecurityConfig) -> dict[str, int]:
    """Return every configured integer bound for uniform validation."""
    return {
        "max_header_count": config.max_header_count,
        "max_total_header_bytes": config.max_total_header_bytes,
        "max_authorization_header_bytes": config.max_authorization_header_bytes,
        "max_correlation_header_bytes": config.max_correlation_header_bytes,
        "max_trace_header_bytes": config.max_trace_header_bytes,
        "max_request_body_bytes": config.max_request_body_bytes,
        "max_request_body_chunks": config.max_request_body_chunks,
        "max_response_body_bytes": config.max_response_body_bytes,
        "max_metadata_bytes": config.max_metadata_bytes,
        "max_message_parts": config.max_message_parts,
        "max_artifact_parts": config.max_artifact_parts,
        "max_task_artifacts": config.max_task_artifacts,
        "max_history_messages": config.max_history_messages,
        "max_page_size": config.max_page_size,
        "max_retained_tasks": config.max_retained_tasks,
        "max_accepted_concurrency": config.max_accepted_concurrency,
        "max_caller_concurrency": config.max_caller_concurrency,
        "max_capability_concurrency": config.max_capability_concurrency,
    }


def _deadlines(config: A2AHostSecurityConfig) -> dict[str, float]:
    """Return every configured deadline for uniform validation."""
    return {
        "request_deadline_seconds": config.request_deadline_seconds,
        "capability_deadline_seconds": config.capability_deadline_seconds,
        "cancellation_deadline_seconds": config.cancellation_deadline_seconds,
        "startup_deadline_seconds": config.startup_deadline_seconds,
        "shutdown_deadline_seconds": config.shutdown_deadline_seconds,
        "drain_deadline_seconds": config.drain_deadline_seconds,
        "task_store_flush_deadline_seconds": config.task_store_flush_deadline_seconds,
    }


__all__ = [
    "AUTHORIZATION_HEADER",
    "CLIENT_ADDRESS_HEADER",
    "CORRELATION_HEADER",
    "FORWARDED_HEADERS",
    "MTLS_SUBJECT_HEADER",
    "RESERVED_HEADERS",
    "TRACEPARENT_HEADER",
    "TRACESTATE_HEADER",
    "A2AHostSecurityConfig",
    "A2AMTLSIdentityExtractor",
    "A2ATransportFacts",
    "enforce_metadata_limit",
    "evaluate_request",
    "sanitize_scope",
]
