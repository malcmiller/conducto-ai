from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
from typing import Any

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.gateway_models import DiscoveryQuery
from conducto.core.run_context import use_run_context
from conducto.core.telemetry import (
    SPAN_CAPABILITY_INVOKE,
    SPAN_MCP_SERVER,
    TRACE_SCHEMA_VERSION,
    configure_in_memory_tracing,
    current_trace_ids,
    extract_trace_context,
    inject_trace_context,
    start_span,
)
from conducto.security import (
    AuditEmitter,
    AuthorizationContext,
    InMemoryAuditSink,
    Principal,
    SecurityPipeline,
    require_scope,
)


@a2a_agent(name="TelemetryAgent", version="1.0.0", description="Telemetry test agent.")
class TelemetryAgent(BaseAgent):
    """Agent used by focused tracing tests."""

    @a2a_capability(name="echo", description="Echo a safe value.")
    @require_scope("telemetry:run")
    def echo(self, value: str) -> str:
        """Return the supplied value through a synchronous worker."""
        assert current_trace_ids() is not None
        return value


def _authorization() -> AuthorizationContext:
    return AuthorizationContext(
        principal=Principal(
            subject_id="subject",
            issuer="issuer",
            audience="conducto",
            scopes=frozenset({"telemetry:run"}),
        ),
        task_id="task-1",
        correlation_id="correlation-1",
    )


def test_base_tracing_helpers_noop_without_opentelemetry_imports(monkeypatch: Any) -> None:
    """OpenTelemetry imports are lazy and no-op when the extra is absent."""
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with start_span(SPAN_CAPABILITY_INVOKE) as span:
        span.set_outcome("success")
    assert current_trace_ids() is None
    assert inject_trace_context({"accept": "application/json"}) == {"accept": "application/json"}


def test_runtime_spans_audit_ids_and_sensitive_values_are_safe() -> None:
    """Runtime invocation emits connected safe spans and audit trace correlation."""
    tracing = configure_in_memory_tracing()
    tracing.exporter.clear()
    sink = InMemoryAuditSink()
    runtime = Runtime(security_pipeline=SecurityPipeline(audit_emitter=AuditEmitter(sink)))

    async def exercise() -> None:
        result = await runtime.invoke(
            TelemetryAgent(),
            "echo",
            {"value": "SECRET_CANARY_SHOULD_NOT_APPEAR"},
            authorization=_authorization(),
            correlation_id="correlation-1",
        )
        assert getattr(result, "value", None) == "SECRET_CANARY_SHOULD_NOT_APPEAR"

    asyncio.run(exercise())

    spans = tracing.exporter.get_finished_spans()
    names = [span.name for span in spans]
    assert "conducto.capability.invoke" in names
    assert "conducto.security.authorize" in names
    capability_span = next(span for span in spans if span.name == "conducto.capability.invoke")
    security_span = next(span for span in spans if span.name == "conducto.security.authorize")
    assert security_span.context.trace_id == capability_span.context.trace_id
    assert security_span.parent is not None
    assert security_span.parent.span_id == capability_span.context.span_id
    assert sink.events
    assert all(event.trace_id and event.span_id for event in sink.events)
    assert {event.trace_id for event in sink.events} == {
        format(capability_span.context.trace_id, "032x")
    }
    exported = json.dumps(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes or {}),
                "events": [event.name for event in span.events],
                "status": span.status.description,
            }
            for span in spans
        ],
        sort_keys=True,
    )
    assert "SECRET_CANARY_SHOULD_NOT_APPEAR" not in exported


def test_w3c_remote_context_is_connected_and_invalid_context_is_marked() -> None:
    """Valid remote context parents server spans; invalid context is ignored safely."""
    tracing = configure_in_memory_tracing()
    tracing.exporter.clear()
    with start_span(SPAN_CAPABILITY_INVOKE):
        headers = inject_trace_context({})
    extracted = extract_trace_context(headers)
    with start_span(SPAN_MCP_SERVER, kind="server", remote_context=extracted.context) as span:
        span.set_outcome("success")
    invalid = extract_trace_context({"traceparent": "not-a-valid-context"})
    with start_span(
        SPAN_MCP_SERVER,
        kind="server",
        remote_context=invalid.context,
        attributes={"conducto.invalid_remote_context": invalid.invalid_remote_context},
    ) as span:
        span.set_outcome("success")

    spans = tracing.exporter.get_finished_spans()
    parent, valid_server, invalid_server = spans[-3:]
    assert valid_server.context.trace_id == parent.context.trace_id
    assert valid_server.parent is not None
    assert valid_server.parent.span_id == parent.context.span_id
    assert invalid_server.context.trace_id != parent.context.trace_id
    assert invalid_server.attributes["conducto.invalid_remote_context"] is True


def test_gateway_discovery_and_invocation_emit_spans() -> None:
    """Local gateway discovery and invocation use stable operation names."""
    tracing = configure_in_memory_tracing()
    tracing.exporter.clear()
    agent = TelemetryAgent()
    registry = AgentRegistry()
    registry.register(agent)
    runtime = Runtime(agent_registry=registry)
    context = runtime.create_run_context(
        agent_id="caller",
        authorization=_authorization(),
        correlation_id="correlation-1",
        allowed_capabilities=frozenset({"echo"}),
    )

    async def exercise() -> None:
        with use_run_context(context):
            discovery = await context.gateway.discover(
                DiscoveryQuery(capability_ids=frozenset({"echo"}))
            )
            assert len(discovery) == 1
            result = await context.gateway.invoke(discovery[0].binding, {"value": "safe"})
            assert getattr(result, "value", None) == "safe"

    asyncio.run(exercise())

    names = [span.name for span in tracing.exporter.get_finished_spans()]
    assert "conducto.gateway.discover" in names
    assert "conducto.gateway.invoke" in names
    assert "conducto.capability.invoke" in names


def test_semantic_fixture_lists_exported_span_names() -> None:
    """The checked-in semantic fixture pins stable span names and required fields."""
    fixture = json.loads(Path("docs/semantic-fixtures/opentelemetry-spans.v1.json").read_text())
    assert fixture["schema_version"] == TRACE_SCHEMA_VERSION
    names = {item["name"] for item in fixture["spans"]}
    assert {
        "conducto.gateway.discover",
        "conducto.gateway.invoke",
        "conducto.mcp.server",
        "conducto.mcp.tools.list",
        "conducto.mcp.tools.call",
        "conducto.a2a.client",
        "conducto.a2a.server",
        "conducto.auth.exchange",
        "conducto.security.authorize",
        "conducto.security.approval",
        "conducto.model.complete",
        "conducto.capability.invoke",
        "conducto.delegation.turn",
        "conducto.registration.server",
    } == names
