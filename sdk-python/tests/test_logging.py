import asyncio
import io
import logging
from pathlib import Path

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


def _event_records(catalog: object) -> list[logging.LogRecord]:
    return [
        record
        for record in catalog.records  # type: ignore[attr-defined]
        if record.name == "conducto.events"
    ]


def test_import_does_not_configure_root_logging() -> None:
    root = logging.getLogger()
    before_handlers = tuple(root.handlers)
    before_level = root.level

    __import__("conducto")

    assert tuple(root.handlers) == before_handlers
    assert root.level == before_level


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
    logger = logging.getLogger("conducto")
    original_handlers = tuple(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    external = logging.NullHandler()
    logger.addHandler(external)
    try:
        first = configure_logging(stream=io.StringIO())
        second = configure_logging(stream=io.StringIO(), format="json")
        assert first not in logger.handlers
        assert second in logger.handlers
        assert external in logger.handlers
        assert sum(getattr(item, "_conducto_owned", False) for item in logger.handlers) == 1
    finally:
        for handler in tuple(logger.handlers):
            if getattr(handler, "_conducto_owned", False) or handler is external:
                logger.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_default_events_redact_payloads_and_exceptions(catalog: object) -> None:
    with catalog.at_level(logging.INFO, logger="conducto"):  # type: ignore[attr-defined]
        emit_event(
            "conducto.test.v1",
            payload={"token": "secret-token", "prompt": "sensitive prompt"},
        )

    record = _event_records(catalog)[0]
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


def test_concurrent_async_and_sync_invocations_keep_context_isolated(catalog: object) -> None:
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

    with catalog.at_level(logging.INFO, logger="conducto"):  # type: ignore[attr-defined]
        asyncio.run(exercise())

    completed = [
        record
        for record in _event_records(catalog)
        if getattr(record, "event", None) == "conducto.capability.invocation_completed.v1"
    ]
    assert {
        (getattr(record, "correlation_id", None), getattr(record, "capability_id", None))
        for record in completed
    } == {
        ("one", "sync"),
        ("two", "async"),
    }


def test_concurrent_model_overrides_keep_provenance_isolated(catalog: object) -> None:
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

    with catalog.at_level(logging.INFO, logger="conducto"):  # type: ignore[attr-defined]
        asyncio.run(exercise())

    models = [
        record
        for record in _event_records(catalog)
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
        ("first-correlation", "first", "model-one", "invocation_override"),
        ("second-correlation", "second", "model-two", "invocation_override"),
    }
