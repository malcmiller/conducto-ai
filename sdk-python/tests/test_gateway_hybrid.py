"""Tests for the hybrid local-plus-remote capability gateway."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from conducto import AgentRegistry, BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.catalog import AgentCatalog, CatalogEntry, CatalogLifecycleState, DeploymentType
from conducto.core.gateway import GatewaySelectionMode, GatewaySelectionPolicy, RemoteTransportError
from conducto.core.gateway_models import (
    CapabilityBinding,
    DiscoveryQuery,
    GatewayFailureCode,
    SelectionStatus,
)
from conducto.core.invocation_results import (
    InvocationAuthorizationFailure,
    InvocationBudgetExhausted,
    InvocationDelegationFailure,
    InvocationFailure,
    InvocationResult,
    InvocationSuccess,
)
from conducto.core.model_config import RunConfig
from conducto.core.run_context import (
    DelegationBudget,
    require_run_context,
    use_run_context,
)
from conducto.security import AuthorizationContext, Principal, require_scope
from conducto.testing import InMemoryA2ATransport


@a2a_agent(name="LocalLookup", version="1.0.0", description="Local lookup provider.")
class LocalLookup(BaseAgent):
    @a2a_capability(name="lookup", description="Lookup a value locally.")
    def lookup(self, query: str) -> str:
        return f"local:{query}"


@a2a_agent(name="RemoteLookup", version="1.0.0", description="Remote lookup provider.")
class RemoteLookup(BaseAgent):
    @a2a_capability(name="lookup", description="Lookup a value remotely.")
    def lookup(self, query: str) -> str:
        return f"remote:{query}"


@a2a_agent(name="RemoteLookupB", version="1.0.0", description="Second remote provider.")
class RemoteLookupB(BaseAgent):
    @a2a_capability(name="lookup", description="Lookup a value remotely.")
    def lookup(self, query: str) -> str:
        return f"remote-b:{query}"


@a2a_agent(name="ScopedRemoteProbe", version="1.0.0", description="Scoped remote probe.")
class ScopedRemoteProbe(BaseAgent):
    @a2a_capability(name="report", description="Report invocation context.")
    @require_scope("records:read")
    def report(self) -> dict[str, Any]:
        return _context_snapshot()


@a2a_agent(name="LocalProbe", version="1.0.0", description="Local context probe.")
class LocalProbe(BaseAgent):
    @a2a_capability(name="report", description="Report invocation context.")
    def report(self) -> dict[str, Any]:
        return _context_snapshot()


@a2a_agent(name="LocalDelegator", version="1.0.0", description="Delegates to a stored binding.")
class LocalDelegator(BaseAgent):
    binding: CapabilityBinding | None = None

    @a2a_capability(name="delegate", description="Delegate to a stored binding.")
    async def delegate(self) -> dict[str, Any]:
        return await _invoke_stored_binding(type(self).binding)


@a2a_agent(name="RemoteDelegator", version="1.0.0", description="Delegates to a stored binding.")
class RemoteDelegator(BaseAgent):
    binding: CapabilityBinding | None = None

    @a2a_capability(name="delegate", description="Delegate to a stored binding.")
    async def delegate(self) -> dict[str, Any]:
        return await _invoke_stored_binding(type(self).binding)


@a2a_agent(name="RemoteSlow", version="1.0.0", description="Slow remote provider.")
class RemoteSlow(BaseAgent):
    started = threading.Event()
    finish = threading.Event()

    @a2a_capability(name="wait", description="Wait for release.")
    def wait(self) -> str:
        type(self).started.set()
        type(self).finish.wait(timeout=1)
        return "original"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _authorization(*scopes: str, task_id: str = "task-1") -> AuthorizationContext:
    return AuthorizationContext(
        principal=Principal(
            subject_id="caller",
            issuer="tests",
            audience="conducto",
            scopes=frozenset(scopes),
        ),
        task_id=task_id,
        correlation_id="corr",
    )


def _context_snapshot() -> dict[str, Any]:
    context = require_run_context()
    authorization = context.authorization
    remaining_timeout: float | None
    try:
        remaining_timeout = context.remaining_timeout()
    except TimeoutError:
        remaining_timeout = 0.0
    budget = context.remaining_delegation_budget
    scopes = () if authorization is None else tuple(sorted(authorization.principal.scopes))
    task_id = "" if authorization is None else authorization.task_id
    return {
        "agent_id": context.agent_id,
        "correlation_id": context.correlation_id,
        "task_id": task_id,
        "scopes": list(scopes),
        "allowed_capabilities": (
            [] if context.allowed_capabilities is None else sorted(context.allowed_capabilities)
        ),
        "parent_run_id": context.parent_run_id,
        "remaining_calls": budget.calls,
        "remaining_depth": budget.depth,
        "remaining_timeout": remaining_timeout,
    }


async def _invoke_stored_binding(binding: Any) -> dict[str, Any]:
    context = require_run_context()
    assert binding is not None
    result = await context.gateway.invoke(binding, {})
    return _summarize_result(result)


def _summarize_result(result: InvocationResult) -> dict[str, Any]:
    if isinstance(result, InvocationSuccess):
        return {"kind": "success", "value": result.value}
    if isinstance(result, InvocationAuthorizationFailure):
        return {"kind": "authorization", "reason": result.reason_code}
    if isinstance(result, InvocationBudgetExhausted):
        return {"kind": "budget", "budget": result.budget}
    if isinstance(result, InvocationDelegationFailure):
        return {"kind": "delegation", "reason": result.reason_code, "path": result.path}
    if isinstance(result, InvocationFailure):
        return {"kind": "failure", "message": result.message}
    return {"kind": type(result).__name__}


def _entry(
    agent: BaseAgent,
    *,
    instance_id: str,
    deployment: DeploymentType = DeploymentType.REMOTE_CONTAINER,
    transports: frozenset[str] = frozenset({"a2a_jsonrpc"}),
    lease_seconds: float = 30.0,
    scopes: tuple[str, ...] = (),
) -> CatalogEntry:
    card = agent.get_agent_card(
        f"https://example.org/{instance_id}/rpc",
        security_schemes=(
            {
                "oauth": {
                    "type": "oauth2",
                    "flows": {
                        "clientCredentials": {
                            "tokenUrl": "https://example.org/oauth/token",
                            "scopes": {scope: scope for scope in scopes},
                        }
                    },
                }
            }
            if scopes
            else None
        ),
        security_requirements=(({"oauth": list(scopes)},) if scopes else None),
    )
    if scopes:
        for skill in card.get("skills", []):
            skill["securityRequirements"] = [{"schemes": {"oauth": {"list": list(scopes)}}}]
    return CatalogEntry(
        agent_id=agent.agent_metadata.name,
        instance_id=instance_id,
        owner="tests",
        deployment_type=deployment,
        agent_card_url=f"https://example.org/{instance_id}/agent-card.json",
        agent_card=card,
        transports=transports,
        supported_versions=frozenset({agent.agent_metadata.version}),
        lease_seconds=lease_seconds,
    )


def _runtime(
    *,
    registry: AgentRegistry | None = None,
    catalog: AgentCatalog | None = None,
    transport: InMemoryA2ATransport | None = None,
    clock: _Clock | None = None,
    selection_policy: GatewaySelectionPolicy | None = None,
    allowed_deployments: frozenset[DeploymentType] | None = None,
) -> Runtime:
    effective_clock = clock or _Clock()
    effective_catalog = catalog or AgentCatalog(clock=effective_clock)
    effective_transport = transport or InMemoryA2ATransport()
    return Runtime(
        agent_registry=registry or AgentRegistry(),
        agent_catalog=effective_catalog,
        gateway_transport=effective_transport,
        gateway_clock=effective_clock,
        gateway_selection_policy=selection_policy,
        gateway_allowed_deployments=allowed_deployments,
    )


def test_local_only_discovery_and_invocation_still_work_with_hybrid_gateway() -> None:
    registry = AgentRegistry()
    registry.register(LocalLookup())
    runtime = _runtime(registry=registry)

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            selection = await context.gateway.lookup("LocalLookup", "lookup")
            assert selection.status is SelectionStatus.SELECTED
            assert selection.binding is not None
            result = await context.gateway.invoke(selection.binding, {"query": "hello"})
            assert isinstance(result, InvocationSuccess)
            assert result.value == "local:hello"

    asyncio.run(exercise())


def test_new_remote_agent_becomes_discoverable_without_runtime_restart() -> None:
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    remote = RemoteLookup()
    transport = InMemoryA2ATransport()
    runtime = _runtime(catalog=catalog, transport=transport, clock=clock)

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            gateway = context.gateway
            query = DiscoveryQuery(agent_id="RemoteLookup", capability_ids=frozenset({"lookup"}))
            initial = await gateway.discover(query)
            assert initial.failure is not None
            assert initial.failure.code is GatewayFailureCode.NO_MATCH

            catalog.register_instance(_entry(remote, instance_id="remote-1"))
            transport.replace_routes({("RemoteLookup", "remote-1"): remote})

            discovered = await gateway.discover(query)
            assert [candidate.descriptor.agent_id for candidate in discovered] == ["RemoteLookup"]
            selected = await gateway.lookup("RemoteLookup", "lookup")
            assert selected.binding is not None
            invocation = await gateway.invoke(selected.binding, {"query": "hello"})
            assert isinstance(invocation, InvocationSuccess)
            assert invocation.value == "remote:hello"

    asyncio.run(exercise())


def test_remote_discovery_excludes_unauthorized_revoked_expired_incompatible_and_over_budget() -> (
    None
):
    scoped = ScopedRemoteProbe()

    async def discover(
        catalog: AgentCatalog,
        *,
        clock: _Clock,
        authorization: AuthorizationContext | None,
        query: DiscoveryQuery,
        budget: DelegationBudget | None = None,
    ) -> GatewayFailureCode | None:
        transport = InMemoryA2ATransport({("ScopedRemoteProbe", "remote-1"): scoped})
        runtime = _runtime(catalog=catalog, transport=transport, clock=clock)
        context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            authorization=authorization,
            delegation_budget=budget,
        )
        with use_run_context(context):
            result = await context.gateway.discover(query)
        return None if result.failure is None else result.failure.code

    async def exercise() -> None:
        query = DiscoveryQuery(agent_id="ScopedRemoteProbe", capability_ids=frozenset({"report"}))

        unauthorized_clock = _Clock()
        unauthorized_catalog = AgentCatalog(clock=unauthorized_clock)
        unauthorized_catalog.register_instance(
            _entry(scoped, instance_id="remote-1", scopes=("records:read",))
        )
        assert (
            await discover(
                unauthorized_catalog,
                clock=unauthorized_clock,
                authorization=_authorization(),
                query=query,
            )
            is GatewayFailureCode.DISCOVERY_DENIED
        )

        revoked_clock = _Clock()
        revoked_catalog = AgentCatalog(clock=revoked_clock)
        revoked_catalog.register_instance(
            _entry(scoped, instance_id="remote-1", scopes=("records:read",))
        )
        revoked_catalog.set_lifecycle("ScopedRemoteProbe", CatalogLifecycleState.REVOKED)
        assert (
            await discover(
                revoked_catalog,
                clock=revoked_clock,
                authorization=_authorization("records:read"),
                query=query,
            )
            is GatewayFailureCode.NO_MATCH
        )

        expired_clock = _Clock()
        expired_catalog = AgentCatalog(clock=expired_clock)
        expired_catalog.register_instance(
            _entry(
                scoped,
                instance_id="remote-1",
                lease_seconds=5.0,
                scopes=("records:read",),
            )
        )
        expired_clock.now = 6.0
        assert (
            await discover(
                expired_catalog,
                clock=expired_clock,
                authorization=_authorization("records:read"),
                query=query,
            )
            is GatewayFailureCode.NO_MATCH
        )

        incompatible_clock = _Clock()
        incompatible_catalog = AgentCatalog(clock=incompatible_clock)
        incompatible_catalog.register_instance(
            _entry(scoped, instance_id="remote-1", scopes=("records:read",))
        )
        assert (
            await discover(
                incompatible_catalog,
                clock=incompatible_clock,
                authorization=_authorization("records:read"),
                query=DiscoveryQuery(
                    agent_id="ScopedRemoteProbe",
                    capability_ids=frozenset({"report"}),
                    version_constraint=">=2.0.0",
                ),
            )
            is GatewayFailureCode.NO_MATCH
        )
        assert (
            await discover(
                incompatible_catalog,
                clock=incompatible_clock,
                authorization=_authorization("records:read"),
                query=DiscoveryQuery(
                    agent_id="ScopedRemoteProbe",
                    capability_ids=frozenset({"report"}),
                    input_schema={
                        "type": "object",
                        "properties": {"unexpected": {"type": "integer"}},
                    },
                ),
            )
            is GatewayFailureCode.NO_MATCH
        )

        budget_clock = _Clock()
        budget_catalog = AgentCatalog(clock=budget_clock)
        budget_catalog.register_instance(
            _entry(scoped, instance_id="remote-1", scopes=("records:read",))
        )
        assert (
            await discover(
                budget_catalog,
                clock=budget_clock,
                authorization=_authorization("records:read"),
                query=query,
                budget=DelegationBudget(calls=0),
            )
            is GatewayFailureCode.DISCOVERY_DENIED
        )

    asyncio.run(exercise())


def test_no_eligible_instance_reports_explicit_failure() -> None:
    remote = RemoteLookup()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(
        _entry(
            remote,
            instance_id="remote-1",
            deployment=DeploymentType.FOUNDRY,
        )
    )
    runtime = _runtime(
        catalog=catalog,
        transport=InMemoryA2ATransport({("RemoteLookup", "remote-1"): remote}),
        clock=clock,
        allowed_deployments=frozenset({DeploymentType.REMOTE_CONTAINER}),
    )

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            query = DiscoveryQuery(agent_id="RemoteLookup", capability_ids=frozenset({"lookup"}))
            discovered = await context.gateway.discover(query)
            assert discovered.failure is not None
            assert discovered.failure.code is GatewayFailureCode.NO_ELIGIBLE_INSTANCE
            selected = await context.gateway.select(query)
            assert selected.status is SelectionStatus.FAILED
            assert selected.failure is not None
            assert selected.failure.code is GatewayFailureCode.NO_ELIGIBLE_INSTANCE

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_agent"),
    [
        (GatewaySelectionMode.AMBIGUOUS, SelectionStatus.AMBIGUOUS, None),
        (GatewaySelectionMode.LOCAL_PREFERRED, SelectionStatus.SELECTED, "LocalLookup"),
        (GatewaySelectionMode.REMOTE_PREFERRED, SelectionStatus.SELECTED, "RemoteLookup"),
    ],
)
def test_selection_mode_controls_local_vs_remote_preference(
    mode: GatewaySelectionMode,
    expected_status: SelectionStatus,
    expected_agent: str | None,
) -> None:
    registry = AgentRegistry()
    registry.register(LocalLookup())
    remote = RemoteLookup()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(remote, instance_id="remote-1"))
    runtime = _runtime(
        registry=registry,
        catalog=catalog,
        transport=InMemoryA2ATransport({("RemoteLookup", "remote-1"): remote}),
        clock=clock,
        selection_policy=GatewaySelectionPolicy(mode=mode),
    )

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            selected = await context.gateway.select(
                DiscoveryQuery(capability_ids=frozenset({"lookup"}))
            )
            assert selected.status is expected_status
            if expected_agent is not None:
                assert selected.descriptor is not None
                assert selected.descriptor.agent_id == expected_agent
            else:
                assert selected.failure is not None
                assert selected.failure.code is GatewayFailureCode.AMBIGUOUS

    asyncio.run(exercise())


def test_round_robin_and_sticky_selection_are_deterministic() -> None:
    first = RemoteLookup()
    second = RemoteLookupB()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(first, instance_id="remote-1"))
    catalog.register_instance(_entry(second, instance_id="remote-2"))
    routes = {
        ("RemoteLookup", "remote-1"): first,
        ("RemoteLookupB", "remote-2"): second,
    }

    round_robin_runtime = _runtime(
        catalog=catalog,
        transport=InMemoryA2ATransport(routes),
        clock=clock,
        selection_policy=GatewaySelectionPolicy(mode=GatewaySelectionMode.ROUND_ROBIN),
    )
    sticky_runtime = _runtime(
        catalog=catalog,
        transport=InMemoryA2ATransport(routes),
        clock=clock,
        selection_policy=GatewaySelectionPolicy(mode=GatewaySelectionMode.STICKY_TASK),
    )

    async def exercise() -> None:
        round_context = round_robin_runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
        )
        with use_run_context(round_context):
            order = []
            for _ in range(4):
                selected = await round_context.gateway.select(
                    DiscoveryQuery(capability_ids=frozenset({"lookup"}))
                )
                assert selected.descriptor is not None
                order.append(selected.descriptor.agent_id)
            assert order == ["RemoteLookup", "RemoteLookupB", "RemoteLookup", "RemoteLookupB"]

        first_task = sticky_runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            authorization=_authorization(task_id="task-a"),
        )
        second_task = sticky_runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            authorization=_authorization(task_id="task-a"),
        )
        with use_run_context(first_task):
            first_selection = await first_task.gateway.select(
                DiscoveryQuery(capability_ids=frozenset({"lookup"}))
            )
        with use_run_context(second_task):
            second_selection = await second_task.gateway.select(
                DiscoveryQuery(capability_ids=frozenset({"lookup"}))
            )
        assert first_selection.descriptor is not None
        assert second_selection.descriptor is not None
        assert first_selection.descriptor.agent_id == second_selection.descriptor.agent_id

    asyncio.run(exercise())


def test_transport_failures_report_the_selected_runtime_boundary() -> None:
    remote = RemoteLookup()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(remote, instance_id="remote-1"))

    def fail(*_: Any) -> RemoteTransportError:
        return RemoteTransportError(
            "deterministic transport failure",
            boundary="a2a_jsonrpc",
            acceptance_uncertain=True,
        )

    transport = InMemoryA2ATransport(
        {("RemoteLookup", "remote-1"): remote},
        failure_injector=lambda *args: fail(*args),
    )
    runtime = _runtime(catalog=catalog, transport=transport, clock=clock)

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            selected = await context.gateway.lookup("RemoteLookup", "lookup")
            assert selected.binding is not None
            result = await context.gateway.invoke(selected.binding, {"query": "hello"})
            assert isinstance(result, InvocationFailure)
            assert "a2a_jsonrpc" in result.message
            assert "RemoteLookup:lookup" in result.message

    asyncio.run(exercise())


def test_nested_local_and_remote_delegation_preserve_context_without_extending_deadlines() -> None:
    LocalDelegator.binding = None
    RemoteDelegator.binding = None
    registry = AgentRegistry()
    registry.register(LocalProbe())
    registry.register(LocalDelegator())
    remote_probe = ScopedRemoteProbe()
    remote_delegator = RemoteDelegator()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(remote_probe, instance_id="remote-probe"))
    catalog.register_instance(_entry(remote_delegator, instance_id="remote-delegator"))
    transport = InMemoryA2ATransport(
        {
            ("ScopedRemoteProbe", "remote-probe"): remote_probe,
            ("RemoteDelegator", "remote-delegator"): remote_delegator,
        }
    )
    runtime = _runtime(registry=registry, catalog=catalog, transport=transport, clock=clock)

    async def exercise() -> None:
        context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            authorization=_authorization("records:read"),
            run_config=RunConfig(timeout=5.0),
            allowed_capabilities=frozenset(
                {
                    "LocalDelegator:delegate",
                    "LocalProbe:report",
                    "RemoteDelegator:delegate",
                    "ScopedRemoteProbe:report",
                }
            ),
            delegation_budget=DelegationBudget(calls=4),
        )
        with use_run_context(context):
            root_remaining = context.remaining_timeout()

            remote_probe_selection = await context.gateway.lookup("ScopedRemoteProbe", "report")
            local_probe_selection = await context.gateway.lookup("LocalProbe", "report")
            assert remote_probe_selection.binding is not None
            assert local_probe_selection.binding is not None
            LocalDelegator.binding = remote_probe_selection.binding
            RemoteDelegator.binding = local_probe_selection.binding

            local_delegate = await context.gateway.lookup("LocalDelegator", "delegate")
            remote_delegate = await context.gateway.lookup("RemoteDelegator", "delegate")
            assert local_delegate.binding is not None
            assert remote_delegate.binding is not None

            local_result = await context.gateway.invoke(local_delegate.binding, {})
            remote_result = await context.gateway.invoke(remote_delegate.binding, {})
            assert isinstance(local_result, InvocationSuccess)
            assert isinstance(remote_result, InvocationSuccess)

            local_value = local_result.value
            remote_value = remote_result.value
            assert local_value["kind"] == "success"
            assert remote_value["kind"] == "success"

            nested_remote = local_value["value"]
            nested_local = remote_value["value"]
            for nested in (nested_remote, nested_local):
                assert nested["correlation_id"] == "corr"
                assert nested["task_id"] == "task-1"
                assert nested["scopes"] == ["records:read"]
                assert nested["parent_run_id"]
                assert nested["remaining_calls"] <= 3
                assert nested["remaining_timeout"] is not None
                assert nested["remaining_timeout"] <= root_remaining

    asyncio.run(exercise())


def test_nested_delegation_rejects_authority_amplification_budget_escape_cycle_and_depth() -> None:
    LocalDelegator.binding = None
    RemoteDelegator.binding = None
    registry = AgentRegistry()
    registry.register(LocalProbe())
    registry.register(LocalDelegator())
    remote_delegator = RemoteDelegator()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(remote_delegator, instance_id="remote-delegator"))
    transport = InMemoryA2ATransport({("RemoteDelegator", "remote-delegator"): remote_delegator})
    runtime = _runtime(registry=registry, catalog=catalog, transport=transport, clock=clock)

    async def exercise() -> None:
        bootstrap = runtime.create_run_context(agent_id="Bootstrap", correlation_id="corr")
        with use_run_context(bootstrap):
            local_probe = await bootstrap.gateway.lookup("LocalProbe", "report")
            remote_delegate = await bootstrap.gateway.lookup("RemoteDelegator", "delegate")
            local_delegate = await bootstrap.gateway.lookup("LocalDelegator", "delegate")
            assert local_probe.binding is not None
            assert remote_delegate.binding is not None
            assert local_delegate.binding is not None
            local_probe_binding = local_probe.binding
            remote_delegate_binding = remote_delegate.binding
            local_delegate_binding = local_delegate.binding

        authority_context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            allowed_capabilities=frozenset({"RemoteDelegator:delegate"}),
            delegation_budget=DelegationBudget(calls=3),
        )
        with use_run_context(authority_context):
            RemoteDelegator.binding = local_probe_binding
            authority_result = await authority_context.gateway.invoke(remote_delegate_binding, {})
            assert isinstance(authority_result, InvocationSuccess)
            assert authority_result.value == {
                "kind": "authorization",
                "reason": GatewayFailureCode.DISCOVERY_DENIED.value,
            }

        budget_context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            allowed_capabilities=frozenset(
                {"LocalDelegator:delegate", "RemoteDelegator:delegate", "LocalProbe:report"}
            ),
            delegation_budget=DelegationBudget(calls=2),
        )
        with use_run_context(budget_context):
            RemoteDelegator.binding = local_probe_binding
            LocalDelegator.binding = remote_delegate_binding
            budget_result = await budget_context.gateway.invoke(local_delegate_binding, {})
            assert isinstance(budget_result, InvocationSuccess)
            assert budget_result.value == {
                "kind": "success",
                "value": {"kind": "budget", "budget": "calls, tokens, or cost"},
            }

        cycle_context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            allowed_capabilities=frozenset({"LocalDelegator:delegate", "RemoteDelegator:delegate"}),
            delegation_budget=DelegationBudget(calls=4),
        )
        with use_run_context(cycle_context):
            RemoteDelegator.binding = local_delegate_binding
            LocalDelegator.binding = remote_delegate_binding
            cycle_result = await cycle_context.gateway.invoke(local_delegate_binding, {})
            assert isinstance(cycle_result, InvocationSuccess)
            assert cycle_result.value == {
                "kind": "success",
                "value": {
                    "kind": "delegation",
                    "reason": GatewayFailureCode.CYCLE_DETECTED.value,
                    "path": [["LocalDelegator", "delegate"], ["RemoteDelegator", "delegate"]],
                },
            }

        depth_context = runtime.create_run_context(
            agent_id="Caller",
            correlation_id="corr",
            allowed_capabilities=frozenset(
                {"LocalDelegator:delegate", "RemoteDelegator:delegate", "LocalProbe:report"}
            ),
            delegation_budget=DelegationBudget(max_depth=1, calls=4),
        )
        with use_run_context(depth_context):
            RemoteDelegator.binding = local_probe_binding
            LocalDelegator.binding = remote_delegate_binding
            depth_result = await depth_context.gateway.invoke(local_delegate_binding, {})
            assert isinstance(depth_result, InvocationSuccess)
            assert depth_result.value == {
                "kind": "delegation",
                "reason": GatewayFailureCode.DEPTH_EXCEEDED.value,
                "path": [["LocalDelegator", "delegate"]],
            }

    asyncio.run(exercise())


def test_catalog_mutation_during_in_flight_remote_call_does_not_redirect_task() -> None:
    RemoteSlow.started = threading.Event()
    RemoteSlow.finish = threading.Event()
    registry = AgentRegistry()
    slow = RemoteSlow()
    replacement = RemoteLookup()
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(slow, instance_id="remote-1"))
    transport = InMemoryA2ATransport({("RemoteSlow", "remote-1"): slow})
    runtime = _runtime(registry=registry, catalog=catalog, transport=transport, clock=clock)

    async def exercise() -> None:
        context = runtime.create_run_context(agent_id="Caller", correlation_id="corr")
        with use_run_context(context):
            selected = await context.gateway.lookup("RemoteSlow", "wait")
            assert selected.binding is not None
            invocation = asyncio.create_task(context.gateway.invoke(selected.binding, {}))
            await asyncio.to_thread(RemoteSlow.started.wait, 1)
            catalog.register_instance(_entry(replacement, instance_id="remote-2"))
            transport.replace_routes(
                {
                    ("RemoteSlow", "remote-1"): slow,
                    ("RemoteLookup", "remote-2"): replacement,
                }
            )
            RemoteSlow.finish.set()
            result = await invocation
            assert isinstance(result, InvocationSuccess)
            assert result.value == "original"

    asyncio.run(exercise())
