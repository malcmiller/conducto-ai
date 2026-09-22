"""Optional OpenTelemetry helpers for Conducto-owned instrumentation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

TRACE_SCHEMA_VERSION = "1"
TRACER_NAME = "conducto"
MAX_ATTRIBUTE_VALUE_LENGTH = 256
MAX_ATTRIBUTES = 32
MAX_TRACE_HEADER_BYTES = 512

SPAN_GATEWAY_DISCOVER = "conducto.gateway.discover"
SPAN_GATEWAY_INVOKE = "conducto.gateway.invoke"
SPAN_MCP_SERVER = "conducto.mcp.server"
SPAN_MCP_TOOLS_LIST = "conducto.mcp.tools.list"
SPAN_MCP_TOOLS_CALL = "conducto.mcp.tools.call"
SPAN_A2A_CLIENT = "conducto.a2a.client"
SPAN_A2A_SERVER = "conducto.a2a.server"
SPAN_AUTH_EXCHANGE = "conducto.auth.exchange"
SPAN_SECURITY_AUTHORIZE = "conducto.security.authorize"
SPAN_SECURITY_APPROVAL = "conducto.security.approval"
SPAN_MODEL_COMPLETE = "conducto.model.complete"
SPAN_CAPABILITY_INVOKE = "conducto.capability.invoke"
SPAN_DELEGATION_TURN = "conducto.delegation.turn"

_SPAN_NAMES = frozenset(
    {
        SPAN_GATEWAY_DISCOVER,
        SPAN_GATEWAY_INVOKE,
        SPAN_MCP_SERVER,
        SPAN_MCP_TOOLS_LIST,
        SPAN_MCP_TOOLS_CALL,
        SPAN_A2A_CLIENT,
        SPAN_A2A_SERVER,
        SPAN_AUTH_EXCHANGE,
        SPAN_SECURITY_AUTHORIZE,
        SPAN_SECURITY_APPROVAL,
        SPAN_MODEL_COMPLETE,
        SPAN_CAPABILITY_INVOKE,
        SPAN_DELEGATION_TURN,
    }
)

_SAFE_ATTRIBUTE_PREFIXES = (
    "conducto.",
    "rpc.",
    "server.",
    "network.",
    "http.request.method",
    "url.scheme",
    "error.type",
)
_FORBIDDEN_ATTRIBUTE_PARTS = frozenset(
    {
        "argument",
        "authorization",
        "certificate",
        "credential",
        "exception",
        "output",
        "payload",
        "prompt",
        "result",
        "secret",
        "stack",
        "token",
        "traceback",
    }
)


class OpenTelemetryNotInstalledError(RuntimeError):
    """Raised when an explicit tracing setup helper is used without the extra."""


@dataclass(frozen=True, slots=True)
class TraceIds:
    """Safe active trace identifiers for logs and audit events."""

    trace_id: str
    span_id: str


@dataclass(frozen=True, slots=True)
class ExtractedTraceContext:
    """Incoming W3C trace context accepted at a transport boundary."""

    context: Any | None = None
    invalid_remote_context: bool = False


@dataclass(frozen=True, slots=True)
class InMemoryTracing:
    """Application-enabled in-memory tracing objects for deterministic tests."""

    provider: Any
    exporter: Any
    tracer: Any


class SpanHandle:
    """Best-effort wrapper around an OpenTelemetry span."""

    __slots__ = ("_span",)

    def __init__(self, span: Any | None = None) -> None:
        self._span = span

    def set_attribute(self, key: str, value: str | int | float | bool | None) -> None:
        """Set one sanitized attribute when a recording span exists."""
        if self._span is None or value is None or not _safe_attribute_name(key):
            return
        try:
            self._span.set_attribute(key, _safe_attribute_value(value))
        except Exception:
            return

    def set_outcome(self, outcome: str, *, reason: str = "") -> None:
        """Record an expected outcome without exception details."""
        if self._span is None:
            return
        try:
            self._span.set_attribute("conducto.outcome", _safe_attribute_value(outcome))
            if reason:
                self._span.set_attribute("conducto.reason", _safe_attribute_value(reason))
            status = _status_code()
            if status is not None:
                self._span.set_status(status.OK)
        except Exception:
            return

    def set_error(self, category: str) -> None:
        """Record an unexpected safe error category."""
        if self._span is None:
            return
        try:
            safe_category = _safe_attribute_value(category or "internal_error")
            self._span.set_attribute("conducto.outcome", "failure")
            self._span.set_attribute("conducto.error_category", safe_category)
            self._span.set_attribute("error.type", safe_category)
            status = _status_code()
            if status is not None:
                self._span.set_status(status.ERROR)
        except Exception:
            return


@contextmanager
def start_span(
    name: str,
    *,
    kind: str = "internal",
    attributes: Mapping[str, str | int | float | bool | None] | None = None,
    remote_context: Any | None = None,
) -> Iterator[SpanHandle]:
    """Start a Conducto span or deterministically no-op when tracing is absent."""
    if name not in _SPAN_NAMES:
        yield SpanHandle()
        return
    otel = _optional_otel()
    if otel is None:
        yield SpanHandle()
        return
    trace, _, _ = otel
    tracer = trace.get_tracer(TRACER_NAME)
    safe_attributes = sanitize_attributes(
        {
            "conducto.trace_schema_version": TRACE_SCHEMA_VERSION,
            **dict(attributes or {}),
        }
    )
    span_kind = _span_kind(kind)
    try:
        span_context = tracer.start_as_current_span(
            name,
            kind=span_kind,
            context=remote_context,
            attributes=safe_attributes,
            record_exception=False,
            set_status_on_exception=False,
        )
    except Exception:
        yield SpanHandle()
        return
    with span_context as span:
        yield SpanHandle(span)


def sanitize_attributes(
    attributes: Mapping[str, str | int | float | bool | None],
) -> dict[str, str | int | float | bool]:
    """Return bounded low-cardinality attributes accepted by Conducto tracing."""
    safe: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if len(safe) >= MAX_ATTRIBUTES:
            break
        if value is None or not _safe_attribute_name(key):
            continue
        safe[str(key)] = _safe_attribute_value(value)
    return safe


def current_trace_ids() -> TraceIds | None:
    """Return the active span identifiers when OpenTelemetry is configured."""
    otel = _optional_otel()
    if otel is None:
        return None
    trace, _, _ = otel
    try:
        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return TraceIds(
            trace_id=trace.format_trace_id(context.trace_id),
            span_id=trace.format_span_id(context.span_id),
        )
    except Exception:
        return None


def extract_trace_context(
    headers: Mapping[str, str],
    *,
    raw_header_names: Sequence[str] = (),
    max_header_bytes: int = MAX_TRACE_HEADER_BYTES,
) -> ExtractedTraceContext:
    """Extract W3C trace context while ignoring invalid remote parentage."""
    trace_headers = {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in {"traceparent", "tracestate"}
    }
    if not trace_headers:
        return ExtractedTraceContext()
    raw_names = tuple(name.lower() for name in raw_header_names)
    duplicate = raw_names.count("traceparent") > 1 or raw_names.count("tracestate") > 1
    over_limit = (
        sum(len(key) + len(value) for key, value in trace_headers.items()) > max_header_bytes
    )
    otel = _optional_otel()
    if otel is None or duplicate or over_limit:
        return ExtractedTraceContext(invalid_remote_context=duplicate or over_limit)
    trace, _, propagator = otel
    try:
        context = propagator.extract(trace_headers)
        parent = trace.get_current_span(context).get_span_context()
    except Exception:
        return ExtractedTraceContext(invalid_remote_context=True)
    if not parent.is_valid:
        return ExtractedTraceContext(invalid_remote_context=True)
    return ExtractedTraceContext(context=context)


def inject_trace_context(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Inject W3C trace context into a new outbound header mapping."""
    carrier = dict(headers or {})
    otel = _optional_otel()
    if otel is None:
        return carrier
    _, _, propagator = otel
    try:
        propagator.inject(carrier)
    except Exception:
        return carrier
    return carrier


def configure_in_memory_tracing() -> InMemoryTracing:
    """Enable an application-owned in-memory tracer provider for tests.

    Returns:
        The provider, exporter, and Conducto tracer created by the helper.

    Raises:
        OpenTelemetryNotInstalledError: If ``conducto-ai[opentelemetry]`` is
            not installed in the current environment.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise OpenTelemetryNotInstalledError(
            "OpenTelemetry tracing requires conducto-ai[opentelemetry]"
        ) from error

    exporter = InMemorySpanExporter()
    active_provider = trace.get_tracer_provider()
    provider = active_provider if isinstance(active_provider, TracerProvider) else TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    if provider is not active_provider:
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            pass
    return InMemoryTracing(provider, exporter, trace.get_tracer(TRACER_NAME))


def _optional_otel() -> tuple[Any, Any, Any] | None:
    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        return None
    return trace, None, TraceContextTextMapPropagator()


def _status_code() -> Any | None:
    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        return None
    return StatusCode


def _span_kind(kind: str) -> Any:
    try:
        from opentelemetry.trace import SpanKind
    except ImportError:
        return None
    return {
        "client": SpanKind.CLIENT,
        "server": SpanKind.SERVER,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }.get(kind, SpanKind.INTERNAL)


def _safe_attribute_name(key: str) -> bool:
    normalized = str(key).lower()
    return normalized.startswith(_SAFE_ATTRIBUTE_PREFIXES) and not any(
        part in normalized for part in _FORBIDDEN_ATTRIBUTE_PARTS
    )


def _safe_attribute_value(value: str | int | float | bool) -> str | int | float | bool:
    if isinstance(value, bool | int | float):
        return value
    return str(value)[:MAX_ATTRIBUTE_VALUE_LENGTH]


__all__ = [
    "ExtractedTraceContext",
    "InMemoryTracing",
    "OpenTelemetryNotInstalledError",
    "SPAN_A2A_CLIENT",
    "SPAN_A2A_SERVER",
    "SPAN_AUTH_EXCHANGE",
    "SPAN_CAPABILITY_INVOKE",
    "SPAN_DELEGATION_TURN",
    "SPAN_GATEWAY_DISCOVER",
    "SPAN_GATEWAY_INVOKE",
    "SPAN_MCP_SERVER",
    "SPAN_MCP_TOOLS_CALL",
    "SPAN_MCP_TOOLS_LIST",
    "SPAN_MODEL_COMPLETE",
    "SPAN_SECURITY_APPROVAL",
    "SPAN_SECURITY_AUTHORIZE",
    "TRACE_SCHEMA_VERSION",
    "TraceIds",
    "configure_in_memory_tracing",
    "current_trace_ids",
    "extract_trace_context",
    "inject_trace_context",
    "sanitize_attributes",
    "start_span",
]
