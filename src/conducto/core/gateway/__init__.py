"""Policy-governed discovery and invocation through opaque capability bindings."""

from ._contracts import (
    AgentGateway,
    GatewayPolicy,
    GatewayRemotePolicy,
    GatewaySelectionMode,
    GatewaySelectionPolicy,
    RemoteGatewayTransport,
    RemoteTransportError,
)
from ._hybrid import HybridAgentGateway
from ._local import LocalAgentGateway

__all__ = [
    "AgentGateway",
    "GatewayPolicy",
    "GatewayRemotePolicy",
    "GatewaySelectionMode",
    "GatewaySelectionPolicy",
    "HybridAgentGateway",
    "LocalAgentGateway",
    "RemoteGatewayTransport",
    "RemoteTransportError",
]
