"""Library-friendly structured logging for Conducto local-agent workflows.

The facade never configures the root logger. Applications may either attach
their own handlers to ``conducto`` or opt into: func:`configure_logging`.
Payloads are intentionally excluded from lifecycle events. Logging prompts,
model responses, capability arguments, or results require an explicit
``include_sensitive_data=True`` configuration and per-event opt-in.
"""

from __future__ import annotations

# noinspection PyPackageRequirements
import contextvars
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Final, Literal

LOG_SCHEMA_VERSION: Final = "1"
LOGGER_NAME: Final = "conducto"
_SENSITIVE_LOGGER_NAME: Final = "conducto.sensitive"

AGENT_REGISTERED: Final = "conducto.agent.registered.v1"
AGENT_DISCOVERED: Final = "conducto.agent.discovered.v1"
MODEL_SELECTED: Final = "conducto.model.selected.v1"
MODEL_USAGE_RECORDED: Final = "conducto.model.usage_recorded.v1"
ARGUMENTS_VALIDATED: Final = "conducto.capability.arguments_validated.v1"
INVOCATION_STARTED: Final = "conducto.capability.invocation_started.v1"
INVOCATION_COMPLETED: Final = "conducto.capability.invocation_completed.v1"
INVOCATION_FAILED: Final = "conducto.capability.invocation_failed.v1"
INVOCATION_TIMED_OUT: Final = "conducto.capability.invocation_timed_out.v1"
INVOCATION_CANCELLED: Final = "conducto.capability.invocation_cancelled.v1"

EventOutcome = Literal["success", "failure", "timeout", "cancelled"]
_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "conducto_log_context",
    default=None,
)
_SENSITIVE_NAMES = frozenset(
    {
        "approval",
        "arguments",
        "authorization",
        "credential",
        "credentials",
        "model_response",
        "prompt",
        "result",
        "results",
        "secret",
        "token",
    }
)
_RECORD_FIELDS: Final = (
    "schema_version",
    "event",
    "outcome",
    "correlation_id",
    "run_id",
    "agent_id",
    "capability_id",
    "duration_ms",
    "error_category",
    "provider",
    "model_reference",
    "resolution_source",
    "agent_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
_PACKAGE_LOGGER = logging.getLogger(LOGGER_NAME)
_PACKAGE_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    """Bind non-sensitive context to all nested Conducto events.

    ``contextvars`` keeps values isolated across asyncio tasks and carries
    them into synchronous work executed through ``asyncio.to_thread``.
    """

    invalid = set(fields) & _SENSITIVE_NAMES
    if invalid:
        raise ValueError(f"Sensitive logging context is not allowed: {sorted(invalid)!r}")
    current = _CONTEXT.get() or {}
    token = _CONTEXT.set({**current, **{key: str(value) for key, value in fields.items()}})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def get_log_context() -> Mapping[str, str]:
    """Return an immutable snapshot of the current Conducto log context."""
    return dict(_CONTEXT.get() or {})


def emit_event(
    event: str,
    *,
    level: int = logging.INFO,
    outcome: EventOutcome | None = None,
    duration_ms: float | None = None,
    error_category: str | None = None,
    payload: Mapping[str, Any] | None = None,
    include_sensitive_data: bool = False,
    **fields: Any,
) -> None:
    """Emit one versioned Conducto event without exception text or tracebacks.

    Sensitive payloads are omitted unless both this call and:
    func:`configure_logging` explicitly opt in. Lifecycle instrumentation
    does not pass payloads, so its default records never contain them.
    """

    _validate_event_fields(fields)
    logger = logging.getLogger(f"{LOGGER_NAME}.events")
    if not logger.isEnabledFor(level):
        return
    values: dict[str, Any] = {
        "schema_version": LOG_SCHEMA_VERSION,
        "event": event,
        **(_CONTEXT.get() or {}),
        **fields,
    }
    if outcome is not None:
        values["outcome"] = outcome
    if duration_ms is not None:
        values["duration_ms"] = round(duration_ms, 3)
    if error_category is not None:
        values["error_category"] = error_category
    logger.log(level, event, extra=values)
    if payload is not None and include_sensitive_data:
        _emit_sensitive_payload(event, level=level, values=values, payload=payload)


class JsonFormatter(logging.Formatter):
    """Render the stable Conducto event contract as canonical JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize an event record as canonical JSON.

        Args:
            record: The log record to render.

        Returns:
            A deterministic JSON payload for the event.
        """
        event: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }
        for field in _RECORD_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                event[field] = value
        payload = getattr(record, "payload", None)
        if payload is not None:
            event["payload"] = payload
        return json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


class DevelopmentFormatter(logging.Formatter):
    """Render concise, human-readable development events."""

    def format(self, record: logging.LogRecord) -> str:
        """Render an event record as a compact development log line.

        Args:
            record: The log record to render.

        Returns:
            A readable single-line representation of the event.
        """
        parts = [record.levelname, getattr(record, "event", record.getMessage())]
        for field in ("correlation_id", "agent_id", "capability_id", "outcome", "error_category"):
            value = getattr(record, field, None)
            if value is not None:
                parts.append(f"{field}={value}")
        payload = getattr(record, "payload", None)
        if payload is not None:
            parts.append(f"payload={payload!r}")
        return " ".join(parts)


# noinspection PyShadowingBuiltins
def configure_logging(
    *,
    format: Literal["json", "development"] = "development",
    level: int = logging.INFO,
    stream: Any = None,
    include_sensitive_data: bool = False,
) -> logging.Handler:
    """Opt in to a single handler owned by Conducto, without touching root logging.

    Applications that already configure logging should attach a formatter and
    handler themselves instead. This helper is intended for quick starts and
    replaces only a previous handler created by this function.
    """

    logger = logging.getLogger(LOGGER_NAME)
    sensitive_logger = logging.getLogger(_SENSITIVE_LOGGER_NAME)
    for active_logger in (logger, sensitive_logger):
        for handler in tuple(active_logger.handlers):
            if getattr(handler, "_conducto_owned", False):
                active_logger.removeHandler(handler)
                handler.close()
    handler = _create_handler(
        format=format,
        stream=stream,
        include_sensitive_data=False,
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    sensitive_logger.propagate = False
    if include_sensitive_data:
        sensitive_handler = _create_handler(
            format=format,
            stream=stream,
            include_sensitive_data=True,
        )
        sensitive_logger.addHandler(sensitive_handler)
        sensitive_logger.setLevel(level)
        sensitive_logger.propagate = False
    return handler


# noinspection PyShadowingBuiltins
def _create_handler(
    *,
    format: Literal["json", "development"],
    stream: Any,
    include_sensitive_data: bool,
) -> logging.Handler:
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler._conducto_owned = True  # type: ignore[attr-defined]
    handler._conducto_include_sensitive_data = include_sensitive_data  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter() if format == "json" else DevelopmentFormatter())
    return handler


def _validate_event_fields(fields: Mapping[str, Any]) -> None:
    invalid = set(fields) & _SENSITIVE_NAMES
    if invalid:
        raise ValueError(f"Sensitive logging fields are not allowed: {sorted(invalid)!r}")
    permitted = set(_RECORD_FIELDS) - {
        "schema_version",
        "event",
        "outcome",
        "duration_ms",
        "error_category",
    }
    unsupported = set(fields) - permitted
    if unsupported:
        raise ValueError(f"Unsupported logging fields: {sorted(unsupported)!r}")


def _emit_sensitive_payload(
    event: str,
    *,
    level: int,
    values: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    logger = logging.getLogger(_SENSITIVE_LOGGER_NAME)
    if not logger.isEnabledFor(level) or not _allows_sensitive_data(logger):
        return
    logger.log(level, event, extra={**values, "payload": payload})


def _allows_sensitive_data(logger: logging.Logger) -> bool:
    return any(
        getattr(handler, "_conducto_include_sensitive_data", False) for handler in logger.handlers
    )
