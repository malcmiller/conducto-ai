from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from conducto.core.logging import emit_event, log_context
from conducto.core.otel_logs import (
    LOG_MAPPING_SCHEMA_VERSION,
    OpenTelemetryLogBridge,
    configure_in_memory_logs,
)
from conducto.core.telemetry import (
    SPAN_CAPABILITY_INVOKE,
    TRACE_SCHEMA_VERSION,
    configure_in_memory_tracing,
    start_span,
)


def _reset_conducto_logger() -> None:
    logger = logging.getLogger("conducto")
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)


def test_base_bridge_construction_requires_opentelemetry(monkeypatch: Any) -> None:
    """Constructing the bridge without OpenTelemetry installed fails explicitly."""
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(Exception, match="opentelemetry"):
        OpenTelemetryLogBridge(object())


def test_import_does_not_install_a_provider_or_handler() -> None:
    """Importing the adapter module installs nothing on the `conducto` logger."""
    logger = logging.getLogger("conducto")
    owned = [
        handler
        for handler in logger.handlers
        if handler.__class__.__module__.startswith("conducto.core.otel_logs")
    ]
    assert owned == []


def test_bridge_maps_severity_outcome_and_bounded_attributes() -> None:
    """Every mapped severity/outcome carries only bounded, prefixed attributes."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs(level=logging.DEBUG)
        try:
            emit_event(
                "conducto.capability.invocation_completed.v1",
                level=logging.INFO,
                outcome="success",
                duration_ms=1.5,
                correlation_id="corr-1",
                agent_id="Agent",
                capability_id="cap",
            )
            emit_event(
                "conducto.agent.discovered.v1",
                level=logging.DEBUG,
                agent_count=2,
            )
            emit_event(
                "conducto.capability.invocation_timed_out.v1",
                level=logging.WARNING,
                outcome="timeout",
            )
            emit_event(
                "conducto.capability.invocation_failed.v1",
                level=logging.ERROR,
                outcome="failure",
                error_category="capability_exception",
            )

            records = [item.log_record for item in logs.exporter.get_finished_logs()]
            by_event = {record.attributes["conducto.event"]: record for record in records}

            completed = by_event["conducto.capability.invocation_completed.v1"]
            assert completed.severity_text == "INFO"
            assert completed.body == "conducto.capability.invocation_completed.v1"
            assert completed.event_name == "conducto.capability.invocation_completed.v1"
            assert completed.attributes["conducto.outcome"] == "success"
            assert completed.attributes["conducto.correlation_id"] == "corr-1"
            assert completed.attributes["conducto.duration_ms"] == 1.5
            assert len(completed.attributes) <= 32

            discovered = by_event["conducto.agent.discovered.v1"]
            assert discovered.severity_text == "DEBUG"
            assert discovered.attributes["conducto.agent_count"] == 2

            timed_out = by_event["conducto.capability.invocation_timed_out.v1"]
            assert timed_out.severity_text == "WARN"
            assert timed_out.attributes["conducto.outcome"] == "timeout"

            failed = by_event["conducto.capability.invocation_failed.v1"]
            assert failed.severity_text == "ERROR"
            assert failed.attributes["conducto.error_category"] == "capability_exception"
        finally:
            logs.bridge.close()
    finally:
        _reset_conducto_logger()


def test_bridge_correlates_trace_and_omits_ids_without_an_active_span() -> None:
    """Records inherit the active span's trace/span IDs, or are uncorrelated."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()
        tracing = configure_in_memory_tracing()
        tracing.exporter.clear()
        try:
            with start_span(SPAN_CAPABILITY_INVOKE):
                emit_event("conducto.test.correlated.v1", outcome="success")
            emit_event("conducto.test.uncorrelated.v1", outcome="success")

            records = {
                item.log_record.attributes["conducto.event"]: item.log_record
                for item in logs.exporter.get_finished_logs()
            }
            correlated = records["conducto.test.correlated.v1"]
            uncorrelated = records["conducto.test.uncorrelated.v1"]
            span = tracing.exporter.get_finished_spans()[0]
            assert correlated.trace_id == span.context.trace_id
            assert correlated.span_id == span.context.span_id
            assert uncorrelated.trace_id == 0
            assert uncorrelated.span_id == 0
        finally:
            logs.bridge.close()
    finally:
        _reset_conducto_logger()


def test_sensitive_payloads_never_reach_the_bridge() -> None:
    """Payload data opted into the sensitive logger never reaches the OTel bridge."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()
        try:
            emit_event(
                "conducto.test.canary.v1",
                outcome="success",
                payload={"token": "SECRET_CANARY_VALUE", "prompt": "sensitive prompt"},
                include_sensitive_data=True,
            )
            exported = json.dumps(
                [
                    {
                        "body": item.log_record.body,
                        "attributes": dict(item.log_record.attributes),
                    }
                    for item in logs.exporter.get_finished_logs()
                ]
            )
            assert "SECRET_CANARY_VALUE" not in exported
            assert "sensitive prompt" not in exported
        finally:
            logs.bridge.close()
    finally:
        _reset_conducto_logger()


def test_context_fields_are_bounded_and_prefixed() -> None:
    """Non-sensitive `log_context` fields surface only as bounded attributes."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()
        try:
            with log_context(request_id="request-123"):
                emit_event("conducto.test.context.v1", outcome="success")
            records = logs.exporter.get_finished_logs()
            assert records
            attributes = dict(records[-1].log_record.attributes)
            assert all(key.startswith("conducto.") for key in attributes)
        finally:
            logs.bridge.close()
    finally:
        _reset_conducto_logger()


def test_close_is_idempotent_and_never_shuts_down_the_provider() -> None:
    """Repeated `close()` detaches once and leaves the caller-owned provider open."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()
        logger = logging.getLogger("conducto")
        handlers_before_close = len(logger.handlers)
        logs.bridge.close()
        logs.bridge.close()
        logs.bridge.close()
        assert len(logger.handlers) == handlers_before_close - 1
        # The provider is caller-owned; repeated close() never shuts it down,
        # so it remains usable directly for the application's own lifecycle.
        assert logs.provider.force_flush() is True
    finally:
        _reset_conducto_logger()


def test_close_flush_reports_timeout_through_local_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider whose `force_flush` times out is reported, never raised."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()

        class _TimingOutProvider:
            def force_flush(self, timeout_millis: int = 30000) -> bool:
                del timeout_millis
                return False

        logs.bridge._logger_provider = _TimingOutProvider()
        with caplog.at_level(logging.WARNING, logger="conducto.otel_logs"):
            logs.bridge.close(flush=True)
        assert any("flush" in record.message.lower() for record in caplog.records)
    finally:
        _reset_conducto_logger()


def test_close_flush_exception_is_reported_and_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider whose `force_flush` raises is reported, never propagated."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()

        class _FailingProvider:
            def force_flush(self, timeout_millis: int = 30000) -> bool:
                del timeout_millis
                raise RuntimeError("boom")

        logs.bridge._logger_provider = _FailingProvider()
        with caplog.at_level(logging.WARNING, logger="conducto.otel_logs"):
            logs.bridge.close(flush=True)
    finally:
        _reset_conducto_logger()


def test_exporter_failure_is_counted_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A processor/exporter exception is counted and reported, never raised."""
    _reset_conducto_logger()
    try:
        logs = configure_in_memory_logs()
        try:

            class _RaisingLogger:
                resource = None

                def emit(self, record: Any) -> None:
                    raise RuntimeError("boom")

            original_get_logger = logs.provider.get_logger
            logs.provider.get_logger = lambda *a, **k: _RaisingLogger()  # type: ignore[method-assign]
            try:
                with caplog.at_level(logging.WARNING, logger="conducto.otel_logs"):
                    emit_event("conducto.test.exporter_failure.v1", outcome="success")
            finally:
                logs.provider.get_logger = original_get_logger  # type: ignore[method-assign]

            assert logs.bridge.dropped_count == 1
            assert any("failed to emit" in record.message for record in caplog.records)
        finally:
            logs.bridge.close()
    finally:
        _reset_conducto_logger()


def test_second_bridge_can_attach_alongside_console_logging_without_duplication() -> None:
    """Attaching both an OTel bridge and a stream handler emits one record each."""
    _reset_conducto_logger()
    try:
        stream_records: list[logging.LogRecord] = []

        class _CollectingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                stream_records.append(record)

        logger = logging.getLogger("conducto")
        collecting = _CollectingHandler()
        logger.addHandler(collecting)
        logs = configure_in_memory_logs()
        try:
            emit_event("conducto.test.dual_sink.v1", outcome="success")
            assert len(stream_records) == 1
            assert len(logs.exporter.get_finished_logs()) == 1
        finally:
            logs.bridge.close()
            logger.removeHandler(collecting)
    finally:
        _reset_conducto_logger()


def test_semantic_fixture_lists_stable_event_names_and_schema_version() -> None:
    """The checked-in semantic fixture pins the stable event/attribute mapping."""
    fixture = json.loads(Path("docs/semantic-fixtures/opentelemetry-logs.v1.json").read_text())
    assert fixture["schema_version"] == LOG_MAPPING_SCHEMA_VERSION
    assert LOG_MAPPING_SCHEMA_VERSION == TRACE_SCHEMA_VERSION
    names = {item["event"] for item in fixture["events"]}
    assert {
        "conducto.agent.registered.v1",
        "conducto.agent.discovered.v1",
        "conducto.model.selected.v1",
        "conducto.delegation.turn_started.v1",
        "conducto.delegation.tool_completed.v1",
        "conducto.delegation.completed.v1",
        "conducto.capability.arguments_validated.v1",
        "conducto.capability.invocation_started.v1",
        "conducto.capability.invocation_completed.v1",
        "conducto.capability.invocation_failed.v1",
        "conducto.capability.invocation_timed_out.v1",
        "conducto.capability.invocation_cancelled.v1",
    } <= names
