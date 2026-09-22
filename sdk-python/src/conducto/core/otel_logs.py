"""Optional OpenTelemetry Logs bridge for Conducto's diagnostic events.

This module is the single, isolated adapter that imports the underscore
("experimental") OpenTelemetry Python Logs API/SDK surfaces
(``opentelemetry._logs``, ``opentelemetry.sdk._logs``). No other Conducto
module imports those packages directly. The pinned compatibility range is
``opentelemetry-api==1.37.0`` / ``opentelemetry-sdk==1.37.0`` (see the
``opentelemetry`` extra in ``pyproject.toml``); because the Logs SDK may
still evolve upstream, only this file needs to change if that surface moves.

Conducto's versioned standard-library events (see :mod:`conducto.core.logging`)
remain the diagnostic event contract; this module never creates a second
event taxonomy or replaces Python ``logging``. Story 2.3 security audit
events remain a separate, application-owned evidence stream: this bridge
never receives audit events, and accepting a log record here never
satisfies mandatory audit delivery.

Importing this module never installs a global ``LoggerProvider``, handler,
processor, exporter, resource, or environment configuration. Applications
construct and own an OpenTelemetry ``LoggerProvider``; :class:`OpenTelemetryLogBridge`
owns only the logging handler it attaches to the ``conducto`` logger.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Final

from .logging import LOGGER_NAME
from .telemetry import OpenTelemetryNotInstalledError, sanitize_attributes

LOG_MAPPING_SCHEMA_VERSION: Final = "1"

# Fields carried on Conducto log records (see `conducto.core.logging._RECORD_FIELDS`)
# that are safe, bounded, and low-cardinality to forward as OpenTelemetry log
# attributes. `trace_id`/`span_id` are intentionally excluded: OpenTelemetry
# derives trace/span correlation itself from the active context.
_LOG_ATTRIBUTE_FIELDS: Final = (
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
    "loop_id",
    "turn",
    "tool_call_id",
    "snapshot_revision",
)
# Ascending Python `logging` level thresholds mapped to their stable OTel Logs
# Data Model severity text per
# https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-severitynumber.
_SEVERITY_THRESHOLDS: Final = (
    (50, "FATAL"),
    (40, "ERROR"),
    (30, "WARN"),
    (20, "INFO"),
    (10, "DEBUG"),
)
_DIAGNOSTIC_LOGGER_NAME: Final = "conducto.otel_logs"
# `propagate = False` prevents bridge failure diagnostics from reaching the
# `conducto` logger tree (and therefore this same bridge's own handler),
# which would otherwise create a recursive telemetry callback loop.
_diagnostic_logger = logging.getLogger(_DIAGNOSTIC_LOGGER_NAME)
_diagnostic_logger.propagate = False
_diagnostic_logger.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class InMemoryLogs:
    """Application-enabled in-memory log objects for deterministic tests."""

    provider: Any
    exporter: Any
    bridge: OpenTelemetryLogBridge


class OpenTelemetryLogBridge:
    """Attaches an application-owned ``LoggerProvider`` to Conducto events.

    Construction is explicit: nothing about importing Conducto installs a
    provider, handler, processor, exporter, resource, or environment
    configuration. The application supplies and owns the ``LoggerProvider``
    (its exporter, credentials, TLS, headers, resource detectors, and
    shutdown); this bridge owns only the logging handler it attaches to the
    target logger, and never shuts down the caller-owned provider.
    """

    def __init__(
        self,
        logger_provider: Any,
        *,
        level: int = logging.INFO,
        logger_name: str = LOGGER_NAME,
    ) -> None:
        """Attach a bridging handler to ``logger_name`` for a caller-owned provider.

        Args:
            logger_provider: An application-constructed OpenTelemetry
                ``LoggerProvider``. Its lifecycle (resource, processors,
                exporters, shutdown) remains fully owned by the caller.
            level: Minimum Python logging level forwarded to OpenTelemetry.
            logger_name: The Conducto logger to attach to. Defaults to
                ``"conducto"``, which also receives ``conducto.events``
                records through normal propagation.

        Raises:
            OpenTelemetryNotInstalledError: If ``conducto-ai[opentelemetry]``
                is not installed in the current environment.
        """
        try:
            from opentelemetry._logs import SeverityNumber  # noqa: F401
            from opentelemetry.sdk._logs import LogRecord  # noqa: F401
        except ImportError as error:  # pragma: no cover - exercised without the extra
            raise OpenTelemetryNotInstalledError(
                "OpenTelemetry log bridging requires conducto-ai[opentelemetry]"
            ) from error

        self._logger_provider = logger_provider
        self._logger = logging.getLogger(logger_name)
        self._handler = _ConductoLogRecordHandler(level=level, logger_provider=logger_provider)
        self._logger.addHandler(self._handler)
        # Match `configure_logging`'s behavior so `conducto.events` records
        # actually reach this handler instead of being filtered out by the
        # logger's default effective level.
        if self._logger.level == logging.NOTSET or self._logger.level > level:
            self._logger.setLevel(level)
        self._closed = False

    @property
    def dropped_count(self) -> int:
        """Number of records dropped due to a translation or export failure."""
        return self._handler.failure_count

    def close(self, *, flush: bool = False, flush_timeout_millis: int = 5000) -> None:
        """Idempotently detach the bridge without shutting down the provider.

        Args:
            flush: When ``True``, perform at most one bounded
                ``force_flush`` call against the caller-owned provider.
            flush_timeout_millis: Bound passed to ``force_flush``. Ignored
                when ``flush`` is ``False``.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._logger.removeHandler(self._handler)
        except Exception:
            _diagnostic_logger.warning("OpenTelemetry log bridge failed to detach its handler")
        try:
            self._handler.close()
        except Exception:
            _diagnostic_logger.warning("OpenTelemetry log bridge failed to close its handler")
        if not flush:
            return
        force_flush = getattr(self._logger_provider, "force_flush", None)
        if not callable(force_flush):
            return
        try:
            flushed = force_flush(timeout_millis=flush_timeout_millis)
        except Exception:
            _diagnostic_logger.warning("OpenTelemetry log bridge force_flush raised an exception")
            return
        if not flushed:
            _diagnostic_logger.warning("OpenTelemetry log bridge force_flush timed out")


class _ConductoLogRecordHandler(logging.Handler):
    """Bounded, redaction-safe translation from stdlib records to OTel logs.

    Only Conducto's stable, bounded, low-cardinality event fields (see
    :mod:`conducto.core.logging`) ever reach an OpenTelemetry processor or
    exporter. Never forwards code location, exception text, or tracebacks.
    Trace/span correlation is derived by the OpenTelemetry SDK itself from
    the active context, so an absent span simply yields an uncorrelated
    record instead of raising or fabricating identifiers.
    """

    def __init__(self, *, level: int, logger_provider: Any) -> None:
        super().__init__(level=level)
        self._logger_provider = logger_provider
        self.failure_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Translate and forward one record, never raising back into the caller."""
        try:
            self._emit(record)
        except Exception:
            self.failure_count += 1
            _diagnostic_logger.warning("OpenTelemetry log bridge failed to emit a record")

    def _emit(self, record: logging.LogRecord) -> None:
        from opentelemetry.sdk._logs import LogRecord

        severity_number, severity_text = _severity(record.levelno)
        event_name = record.getMessage()
        logger = self._logger_provider.get_logger(record.name)
        logger.emit(
            LogRecord(
                timestamp=int(record.created * 1_000_000_000),
                observed_timestamp=time.time_ns(),
                severity_number=severity_number,
                severity_text=severity_text,
                body=event_name,
                event_name=event_name,
                resource=getattr(logger, "resource", None),
                attributes=_safe_attributes(record),
            )
        )


def _severity(levelno: int) -> tuple[Any, str]:
    from opentelemetry._logs import SeverityNumber

    text_to_severity = {
        "FATAL": SeverityNumber.FATAL,
        "ERROR": SeverityNumber.ERROR,
        "WARN": SeverityNumber.WARN,
        "INFO": SeverityNumber.INFO,
        "DEBUG": SeverityNumber.DEBUG,
    }
    for threshold, text in _SEVERITY_THRESHOLDS:
        if levelno >= threshold:
            return text_to_severity[text], text
    return SeverityNumber.UNSPECIFIED, "UNSPECIFIED"


def _safe_attributes(record: logging.LogRecord) -> dict[str, str | int | float | bool]:
    raw: dict[str, str | int | float | bool | None] = {}
    for field in _LOG_ATTRIBUTE_FIELDS:
        value = getattr(record, field, None)
        if value is not None:
            raw[f"conducto.{field}"] = value
    return sanitize_attributes(raw)


def configure_in_memory_logs(
    *, level: int = logging.INFO, logger_name: str = LOGGER_NAME
) -> InMemoryLogs:
    """Enable an application-owned in-memory ``LoggerProvider`` for tests.

    Returns:
        The provider, exporter, and bridge created by the helper.

    Raises:
        OpenTelemetryNotInstalledError: If ``conducto-ai[opentelemetry]`` is
            not installed in the current environment.
    """
    try:
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise OpenTelemetryNotInstalledError(
            "OpenTelemetry log bridging requires conducto-ai[opentelemetry]"
        ) from error

    # `InMemoryLogExporter.__init__` has no upstream type annotations
    # (opentelemetry-sdk==1.37.0); this is a verified false positive under
    # strict mypy, not a Conducto typing gap.
    exporter = InMemoryLogExporter()  # type: ignore[no-untyped-call]
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    bridge = OpenTelemetryLogBridge(provider, level=level, logger_name=logger_name)
    return InMemoryLogs(provider, exporter, bridge)


__all__ = [
    "LOG_MAPPING_SCHEMA_VERSION",
    "InMemoryLogs",
    "OpenTelemetryLogBridge",
    "configure_in_memory_logs",
]
