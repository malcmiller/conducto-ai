"""Model-neutral gateway contract, independent of registry and runtime adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ..catalog import AgentInstanceRecord, CatalogAgentRecord, CatalogCapabilityDescriptor
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

if TYPE_CHECKING:
    from ..runtime import Runtime

GatewayPolicy = Callable[[RunContext, CapabilityDescriptor], bool]
GatewayRemotePolicy = Callable[
    [RunContext, CatalogAgentRecord, CatalogCapabilityDescriptor],
    bool,
]


class GatewaySelectionMode(StrEnum):
    """Deterministic selection strategy for equally eligible gateway candidates."""

    AMBIGUOUS = "ambiguous"
    LOCAL_PREFERRED = "local_preferred"
    REMOTE_PREFERRED = "remote_preferred"
    ROUND_ROBIN = "round_robin"
    STICKY_TASK = "sticky_task"
    STICKY_SESSION = "sticky_session"


@dataclass(frozen=True, slots=True)
class GatewaySelectionPolicy:
    """Deterministic provider and instance selection hooks for mixed discovery.

    Attributes:
        mode: Selection strategy used when more than one candidate remains after
            authorization and compatibility filtering.
        allow_pre_acceptance_failover: Whether a remote invocation may try a
            different healthy instance after an explicit transport failure that
            proves the destination did not accept the request.
    """

    mode: GatewaySelectionMode = GatewaySelectionMode.AMBIGUOUS
    allow_pre_acceptance_failover: bool = False


class RemoteTransportError(RuntimeError):
    """Typed remote transport failure raised before a gateway result is mapped."""

    def __init__(
        self,
        message: str,
        *,
        boundary: str,
        acceptance_uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.boundary = boundary
        self.acceptance_uncertain = acceptance_uncertain


class RemoteGatewayTransport(Protocol):
    """Runtime-owned adapter that preserves the local invocation contract remotely."""

    @property
    def boundary_name(self) -> str:
        """Return the stable runtime-boundary name reported on transport failures."""
        ...

    @property
    def supported_transports(self) -> frozenset[str]:
        """Return the catalog transport bindings this adapter can dispatch."""
        ...

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
        """Invoke one remote capability through the adapter's transport boundary."""
        ...


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
