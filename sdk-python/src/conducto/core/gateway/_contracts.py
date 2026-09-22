"""Model-neutral gateway contract, independent of registry and runtime adapters."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from ..gateway_models import (
    CapabilityBinding,
    CapabilityDescriptor,
    DiscoveryQuery,
    DiscoveryResult,
    SelectionOutcome,
    ToolDiscoveryResult,
)
from ..invocation_results import InvocationResult
from ..run_context import RunContext

GatewayPolicy = Callable[[RunContext, CapabilityDescriptor], bool]


class AgentGateway(Protocol):
    """Async contract consumed by agents for discovery and dispatch."""

    async def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Return authorized compatible candidates from one discovery snapshot."""
        ...

    async def lookup(self, agent_id: str, capability_id: str) -> SelectionOutcome:
        """Resolve an exact capability without a model decision."""
        ...

    async def select(self, query: DiscoveryQuery) -> SelectionOutcome:
        """Select deterministically or report an explicit typed selection failure."""
        ...

    async def discover_tools(self, query: DiscoveryQuery) -> ToolDiscoveryResult:
        """Project authorized candidates into bounded, safe model-facing tools."""
        ...

    async def invoke(
        self,
        binding: CapabilityBinding,
        arguments: Mapping[str, Any],
        *,
        timeout: float | None = None,
        token_cost: int = 0,
        cost: float = 0,
    ) -> InvocationResult:
        """Revalidate opaque authority and dispatch through the shared runtime contract."""
        ...
