"""Deterministic in-memory A2A transport doubles for gateway tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from conducto.core.agent import BaseAgent
from conducto.core.catalog import (
    AgentInstanceRecord,
    CatalogAgentRecord,
    CatalogCapabilityDescriptor,
)
from conducto.core.gateway import RemoteTransportError
from conducto.core.invocation_results import InvocationResult
from conducto.core.run_context import RunContext
from conducto.core.runtime import Runtime


@dataclass(frozen=True, slots=True)
class _Route:
    agent: BaseAgent
    runtime: Runtime | None = None


class InMemoryA2ATransport:
    """Dispatch remote gateway calls to in-memory agents without network access.

    Args:
        routes: Mapping from ``(agent_id, instance_id)`` to the remote agent, or
            to ``(runtime, agent)`` when the destination uses a different runtime.
        failure_injector: Optional callback that may raise a deterministic
            ``RemoteTransportError`` for a selected instance before dispatch.
        boundary_name: Stable transport-boundary name reported on failures.
        supported_transports: Catalog transport bindings this adapter accepts.
    """

    def __init__(
        self,
        routes: Mapping[
            tuple[str, str],
            BaseAgent | tuple[Runtime, BaseAgent],
        ]
        | None = None,
        *,
        failure_injector: Callable[
            [
                CatalogAgentRecord,
                AgentInstanceRecord,
                CatalogCapabilityDescriptor,
                Mapping[str, Any],
            ],
            RemoteTransportError | None,
        ]
        | None = None,
        boundary_name: str = "a2a_jsonrpc",
        supported_transports: frozenset[str] = frozenset({"a2a_jsonrpc"}),
    ) -> None:
        self._routes = {
            key: (
                _Route(agent=value[1], runtime=value[0])
                if isinstance(value, tuple)
                else _Route(agent=value)
            )
            for key, value in (routes or {}).items()
        }
        self._failure_injector = failure_injector
        self._boundary_name = boundary_name
        self._supported_transports = frozenset(supported_transports)

    @property
    def boundary_name(self) -> str:
        """Return the stable in-memory transport-boundary name."""
        return self._boundary_name

    @property
    def supported_transports(self) -> frozenset[str]:
        """Return the catalog transport bindings this adapter accepts."""
        return self._supported_transports

    def replace_routes(
        self,
        routes: Mapping[
            tuple[str, str],
            BaseAgent | tuple[Runtime, BaseAgent],
        ],
    ) -> None:
        """Atomically replace the complete route table."""
        self._routes = {
            key: (
                _Route(agent=value[1], runtime=value[0])
                if isinstance(value, tuple)
                else _Route(agent=value)
            )
            for key, value in routes.items()
        }

    async def invoke(
        self,
        *,
        runtime: Runtime,
        context: RunContext,
        agent: CatalogAgentRecord,
        instance: AgentInstanceRecord,
        capability: CatalogCapabilityDescriptor,
        arguments: Mapping[str, Any],
        timeout: float | None,
    ) -> InvocationResult:
        """Invoke the selected remote capability through a deterministic local double."""
        if self._failure_injector is not None:
            injected = self._failure_injector(agent, instance, capability, arguments)
            if injected is not None:
                raise injected
        route = self._routes.get((agent.agent_id, instance.instance_id))
        if route is None:
            raise RemoteTransportError(
                f"No in-memory route for {agent.agent_id}:{instance.instance_id}",
                boundary=self._boundary_name,
            )
        destination_runtime = route.runtime or runtime
        return await destination_runtime.invoke(
            route.agent,
            capability.capability_id,
            arguments,
            timeout=timeout,
            correlation_id=context.correlation_id,
            authorization=context.authorization,
            allowed_capabilities=context.allowed_capabilities,
            delegation_budget=context.delegation_budget,
        )
