"""Policy-governed discovery and invocation through opaque capability bindings."""

from ._contracts import AgentGateway, GatewayPolicy
from ._local import LocalAgentGateway

__all__ = ["AgentGateway", "GatewayPolicy", "LocalAgentGateway"]
