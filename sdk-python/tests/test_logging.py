import asyncio
import io
import logging
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from conducto import (
    BaseAgent,
    FakeModel,
    JsonFormatter,
    ModelConfiguration,
    OrchestratorAgent,
    a2a_agent,
    a2a_capability,
    configure_logging,
    emit_event,
)


def _event_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "conducto.events"]


@contextmanager
def _saved_logging_state() -> Iterator[tuple[logging.Logger, logging.Logger]]:
    logger = logging.getLogger("conducto")
    sensitive_logger = logging.getLogger("conducto.sensitive")
    original_handlers = tuple(logger.handlers)
    original_sensitive_handlers = tuple(sensitive_logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    original_sensitive_level = sensitive_logger.level
    original_sensitive_propagate = sensitive_logger.propagate
    try:
        yield logger, sensitive_logger
    finally:
        for active_logger, handlers in (
            (logger, original_handlers),
            (sensitive_logger, original_sensitive_handlers),
        ):
            for handler in tuple(active_logger.handlers):
                if getattr(handler, "_conducto_owned", False) or handler not in handlers:
                    active_logger.removeHandler(handler)
                    handler.close()
            for handler in handlers:
                if handler not in active_logger.handlers:
                    active_logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        sensitive_logger.setLevel(original_sensitive_level)
        sensitive_logger.propagate = original_sensitive_propagate


def test_import_does_not_configure_root_logging() -> None:
    script = """
import logging
root = logging.getLogger()
before_handlers = len(root.handlers)
before_level = root.level
import conducto
assert len(root.handlers) == before_handlers
assert root.level == before_level
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_json_events_match_golden_fixture() -> None:
    record = logging.LogRecord(
        "conducto.events",
        logging.INFO,
        __file__,
        1,
        "conducto.capability.invocation_completed.v1",
        (),
        None,
    )
    record.created = 1767323045.678
    record.schema_version = "1"
    record.event = "conducto.capability.invocation_completed.v1"
    record.outcome = "success"
    record.correlation_id = "fixture-correlation"
    record.agent_id = "FixtureAgent"
    record.capability_id = "ping"

    rendered = JsonFormatter().format(record)
    fixture = (Path(__file__).parent / "golden" / "fixtures" / "logging_event.json").read_text(
        encoding="utf-8"
    )
    assert rendered == fixture.rstrip("\n")


def test_configure_logging_replaces_only_owned_handler() -> None:
    with _saved_logging_state() as (logger, _sensitive_logger):
        external = logging.NullHandler()
        logger.addHandler(external)
        first = configure_logging(stream=io.StringIO())
        second = configure_logging(stream=io.StringIO(), format="json")
        assert first not in logger.handlers
        assert second in logger.handlers
        assert external in logger.handlers
        assert sum(getattr(item, "_conducto_owned", False) for item in logger.handlers) == 1


def test_default_events_redact_payloads_and_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="conducto"):
        emit_event(
            "conducto.test.v1",
            payload={"token": "secret-token", "prompt": "sensitive prompt"},
        )

    record = _event_records(caplog)[0]
    assert not hasattr(record, "payload")
    assert record.exc_info is None
    assert record.exc_text is None


def test_disabled_logging_skips_extra_field_construction() -> None:
    logger = logging.getLogger("conducto.events")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        emit_event("conducto.test.v1", payload={"token": "secret"})
    finally:
        logger.setLevel(previous_level)


def test_sensitive_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="Sensitive logging fields"):
        emit_event("conducto.test.v1", token="secret")


def test_opted_in_payload_isolated_to_sensitive_handler() -> None:
    normal_stream = io.StringIO()
    sensitive_stream = io.StringIO()
    with _saved_logging_state() as (_logger, sensitive_logger):
        configure_logging(format="json", stream=normal_stream, include_sensitive_data=False)
        sensitive_handler = logging.StreamHandler(sensitive_stream)
        sensitive_handler._conducto_owned = True  # type: ignore[attr-defined]
        sensitive_handler._conducto_include_sensitive_data = True  # type: ignore[attr-defined]
        sensitive_handler.setFormatter(JsonFormatter())
        sensitive_logger.addHandler(sensitive_handler)
        sensitive_logger.setLevel(logging.INFO)
        sensitive_logger.propagate = False

        emit_event(
            "conducto.test.v1",
            payload={"token": "secret-token"},
            include_sensitive_data=True,
        )

        assert "secret-token" not in normal_stream.getvalue()
        assert "secret-token" in sensitive_stream.getvalue()


def test_concurrent_async_and_sync_invocations_keep_context_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @a2a_agent(name="ContextAgent", version="1.0", description="Tests context.")
    class ContextAgent(BaseAgent):
        @a2a_capability(name="sync", description="Returns a supplied value.")
        def sync(self, value: str) -> str:
            return value

        @a2a_capability(name="async", description="Returns a supplied value.")
        async def asynchronous(self, value: str) -> str:
            await asyncio.sleep(0)
            return value

    async def exercise() -> None:
        orchestrator = OrchestratorAgent()
        orchestrator.register_agent(ContextAgent())
        await asyncio.gather(
            orchestrator.invoke("ContextAgent", "sync", {"value": "one"}, correlation_id="one"),
            orchestrator.invoke("ContextAgent", "async", {"value": "two"}, correlation_id="two"),
        )

    with caplog.at_level(logging.DEBUG, logger="conducto"):
        asyncio.run(exercise())

    completed = [
        record
        for record in _event_records(caplog)
        if getattr(record, "event", None) == "conducto.capability.invocation_completed.v1"
    ]
    assert {
        (getattr(record, "correlation_id", None), getattr(record, "capability_id", None))
        for record in completed
    } == {
        ("one", "sync"),
        ("two", "async"),
    }


def test_concurrent_model_overrides_keep_provenance_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @a2a_agent(name="ModelAgent", version="1.0", description="Tests model provenance.")
    class ModelAgent(BaseAgent):
        @a2a_capability(name="ping", description="Replies with pong.")
        def ping(self) -> str:
            return "pong"

    async def exercise() -> None:
        orchestrator = OrchestratorAgent()
        orchestrator.register_agent(ModelAgent())
        await asyncio.gather(
            orchestrator.route(
                "ping",
                model_provider=FakeModel({"agent_id": "ModelAgent", "capability_id": "ping"}),
                model_config=ModelConfiguration(provider="first", model="model-one"),
                correlation_id="first-correlation",
            ),
            orchestrator.route(
                "ping",
                model_provider=FakeModel({"agent_id": "ModelAgent", "capability_id": "ping"}),
                model_config=ModelConfiguration(provider="second", model="model-two"),
                correlation_id="second-correlation",
            ),
        )

    with caplog.at_level(logging.DEBUG, logger="conducto"):
        asyncio.run(exercise())

    models = [
        record
        for record in _event_records(caplog)
        if getattr(record, "event", None) == "conducto.model.selected.v1"
    ]
    assert {
        (
            getattr(record, "correlation_id", None),
            getattr(record, "provider", None),
            getattr(record, "model_reference", None),
            getattr(record, "resolution_source", None),
        )
        for record in models
    } == {
        ("first-correlation", "first", "model-one", "call_override"),
        ("second-correlation", "second", "model-two", "call_override"),
    }

    discovered = [
        record
        for record in _event_records(caplog)
        if getattr(record, "event", None) == "conducto.agent.discovered.v1"
    ]
    assert {getattr(record, "correlation_id", None) for record in discovered} == {
        "first-correlation",
        "second-correlation",
    }
