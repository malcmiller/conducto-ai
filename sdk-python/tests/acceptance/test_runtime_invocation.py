"""Acceptance coverage for standalone and orchestrated model-backed agents."""

import asyncio
import threading
import time

import pytest
from pydantic import BaseModel

from conducto import (
    BaseAgent,
    ChatMessage,
    FakeModel,
    InvocationSuccess,
    InvocationTimeout,
    ModelConfiguration,
    NoActiveRunContextError,
    OrchestratorAgent,
    ProviderRegistry,
    RunConfig,
    Runtime,
    Usage,
    a2a_agent,
    a2a_capability,
    require_run_context,
)

pytestmark = pytest.mark.acceptance


class AuditReportDraft(BaseModel):
    report: str


@a2a_agent(
    name="AuditAgent",
    version="1.0",
    description="Drafts audit reports with a runtime-managed model.",
    default_model="worker",
    model_required=True,
)
class AuditAgent(BaseAgent):
    @a2a_capability(name="draft", description="Drafts an audit report.")
    async def draft_audit_report(self, subject: str) -> dict[str, str]:
        context = require_run_context()
        draft = await context.models.require().complete_typed(
            messages=(ChatMessage(role="user", content=subject),),
            response_type=AuditReportDraft,
        )
        assert isinstance(draft, AuditReportDraft)
        return {
            "report": draft.report,
            "model": str(context.model_reference),
        }


def _runtime(
        *,
        worker_reference: str = "worker",
        worker_report: str = "approved",
) -> tuple[Runtime, FakeModel, FakeModel]:
    registry = ProviderRegistry()
    router = FakeModel(
        {
            "agent_id": "AuditAgent",
            "capability_id": "draft",
            "arguments": {"subject": "annual controls"},
        },
        usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
    )
    worker = FakeModel(
        {"report": worker_report},
        usage=Usage(input_tokens=8, output_tokens=3, total_tokens=11),
    )
    registry.register(
        "router",
        router,
        ModelConfiguration(provider="routing-provider", model="routing-model"),
    )
    registry.register(
        worker_reference,
        worker,
        ModelConfiguration(provider="worker-provider", model=worker_reference),
    )
    return Runtime(provider_registry=registry), router, worker


def test_model_backed_agent_runs_standalone_and_through_orchestrator() -> None:
    async def exercise() -> None:
        runtime, router, worker = _runtime()
        agent = AuditAgent()

        standalone = await runtime.invoke(
            agent,
            agent.draft_audit_report,
            {"subject": "annual controls"},
            correlation_id="standalone",
        )
        assert isinstance(standalone, InvocationSuccess)
        assert standalone.value == {"report": "approved", "model": "worker"}
        assert standalone.metadata is not None
        assert [call.purpose for call in standalone.metadata.model_calls] == ["capability"]
        assert standalone.metadata.usage == Usage(
            input_tokens=8,
            output_tokens=3,
            total_tokens=11,
        )

        orchestrator = OrchestratorAgent(runtime=runtime, model_reference="router")
        orchestrator.register_agent(agent)
        routed = await orchestrator.route(
            "Draft the audit",
            correlation_id="orchestrated",
        )
        assert isinstance(routed, InvocationSuccess)
        assert routed.value == {"report": "approved", "model": "worker"}
        assert routed.metadata is not None
        assert routed.metadata.model_reference == "worker"
        assert [(call.purpose, call.model_reference) for call in routed.metadata.model_calls] == [
            ("routing", "router"),
            ("capability", "worker"),
        ]
        assert routed.metadata.usage == Usage(
            input_tokens=13,
            output_tokens=5,
            total_tokens=18,
        )
        assert router.calls == 1
        assert worker.calls == 2

    asyncio.run(exercise())


def test_direct_model_backed_capability_call_fails_without_runtime_context() -> None:
    with pytest.raises(NoActiveRunContextError, match="No active Conducto run context"):
        asyncio.run(AuditAgent().draft_audit_report("annual controls"))


def test_concurrent_standalone_and_orchestrated_runs_are_isolated() -> None:
    async def exercise() -> None:
        runtime, _router, _worker = _runtime()
        first = FakeModel({"report": "first"})
        second = FakeModel({"report": "second"})
        runtime.provider_registry.register(
            "first",
            first,
            ModelConfiguration(provider="first-provider", model="first"),
        )
        runtime.provider_registry.register(
            "second",
            second,
            ModelConfiguration(provider="second-provider", model="second"),
        )
        agent = AuditAgent()
        orchestrator = OrchestratorAgent(runtime=runtime, model_reference="router")
        orchestrator.register_agent(agent)

        standalone, routed = await asyncio.gather(
            runtime.invoke(
                agent,
                "draft",
                {"subject": "first"},
                run_config=RunConfig(model="first"),
                correlation_id="first-run",
            ),
            orchestrator.route(
                "Draft second",
                agent_run_config=RunConfig(model="second"),
                correlation_id="second-run",
            ),
        )

        assert isinstance(standalone, InvocationSuccess)
        assert isinstance(routed, InvocationSuccess)
        assert standalone.value["model"] == "first"
        assert routed.value["model"] == "second"
        assert standalone.metadata is not None
        assert routed.metadata is not None
        assert standalone.metadata.model_reference == "first"
        assert routed.metadata.model_reference == "second"
        assert [call.model_reference for call in routed.metadata.model_calls] == [
            "router",
            "second",
        ]

    asyncio.run(exercise())


def test_gateway_cannot_escape_its_invocation_scope() -> None:
    failures: list[type[BaseException]] = []

    @a2a_agent(
        name="BackgroundAgent",
        version="1.0",
        description="Checks invocation-scoped model access.",
        default_model="worker",
        model_required=True,
    )
    class BackgroundAgent(BaseAgent):
        @a2a_capability(name="spawn", description="Spawns delayed model work.")
        async def spawn(self) -> str:
            context = require_run_context()

            async def use_gateway_later() -> None:
                await asyncio.sleep(0)
                try:
                    await context.models.require().complete_typed(
                        messages=(ChatMessage(role="user", content="late"),),
                        response_type=AuditReportDraft,
                    )
                except BaseException as error:
                    failures.append(type(error))

            asyncio.create_task(use_gateway_later())
            return "scheduled"

    async def exercise() -> None:
        runtime, _router, worker = _runtime()
        result = await runtime.invoke(BackgroundAgent(), "spawn", {})
        assert isinstance(result, InvocationSuccess)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert failures == [NoActiveRunContextError]
        assert worker.calls == 0

    asyncio.run(exercise())


def test_gateway_supports_awaited_child_tasks_within_the_invocation() -> None:
    @a2a_agent(
        name="TaskGroupAgent",
        version="1.0",
        description="Uses the gateway from an awaited child task.",
        default_model="worker",
        model_required=True,
    )
    class TaskGroupAgent(BaseAgent):
        @a2a_capability(name="draft", description="Drafts in a child task.")
        async def draft(self) -> str:
            context = require_run_context()

            async def complete() -> AuditReportDraft:
                return await context.models.require().complete_typed(
                    messages=(ChatMessage(role="user", content="child"),),
                    response_type=AuditReportDraft,
                )

            result = await asyncio.create_task(complete())
            return result.report

    async def exercise() -> None:
        runtime, _router, worker = _runtime()
        result = await runtime.invoke(TaskGroupAgent(), "draft", {})
        assert isinstance(result, InvocationSuccess)
        assert result.value == "approved"
        assert worker.calls == 1

    asyncio.run(exercise())


def test_runtime_rejects_capability_bound_to_another_agent_instance() -> None:
    async def exercise() -> None:
        runtime, _router, worker = _runtime()
        requested_agent = AuditAgent()
        other_agent = AuditAgent()

        result = await runtime.invoke(
            requested_agent,
            other_agent.draft_audit_report,
            {"subject": "annual controls"},
        )

        assert not isinstance(result, InvocationSuccess)
        assert worker.calls == 0

    asyncio.run(exercise())


def test_runtime_preserves_sync_capability_lock_after_timeout() -> None:
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    class LockAgent(BaseAgent):
        """Lock test agent."""

        @a2a_capability(name="work", description="Does blocking work.")
        def work(self, delay: float) -> str:
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(delay)
            with guard:
                active -= 1
            return "done"

    async def exercise() -> None:
        runtime = Runtime()
        agent = LockAgent()
        first = await runtime.invoke(agent, "work", {"delay": 0.04}, timeout=0.001)
        second = await runtime.invoke(agent, "work", {"delay": 0}, timeout=0.001)
        assert isinstance(first, InvocationTimeout)
        assert isinstance(second, InvocationTimeout)
        await asyncio.sleep(0.06)
        final = await runtime.invoke(agent, "work", {"delay": 0})
        assert isinstance(final, InvocationSuccess)

    asyncio.run(exercise())
    assert maximum_active == 1
