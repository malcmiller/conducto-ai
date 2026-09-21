"""Network transport contracts and optional adapters."""

from .a2a import A2AClient, DiscoveryPolicy, RemoteAgentDescriptor, discover_agent
from .errors import (
    CompatibilityError,
    DiscoveryError,
    LimitExceededError,
    ProtocolError,
    RemoteTaskError,
    TransportError,
)
from .tasks import InMemoryTaskRepository, TaskRepository

__all__ = [
    "A2AClient",
    "CompatibilityError",
    "DiscoveryError",
    "DiscoveryPolicy",
    "InMemoryTaskRepository",
    "LimitExceededError",
    "ProtocolError",
    "RemoteAgentDescriptor",
    "RemoteTaskError",
    "TaskRepository",
    "TransportError",
    "discover_agent",
]
