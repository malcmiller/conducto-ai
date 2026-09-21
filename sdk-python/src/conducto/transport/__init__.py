"""Network transport contracts and optional adapters."""

from .a2a import A2AClient, DiscoveryPolicy, RemoteAgentDescriptor, discover_agent
from .auth import (
    MTLSPeerIdentity,
    authenticate_incoming_request,
    build_authorization_header,
    build_delegated_token_request,
)
from .errors import (
    AuthenticationError,
    CompatibilityError,
    DiscoveryError,
    LimitExceededError,
    ProtocolError,
    RemoteTaskError,
    TLSConfigurationError,
    TransportError,
)
from .tasks import InMemoryTaskRepository, TaskRepository
from .tls import (
    ClientCertificate,
    TLSPolicy,
    build_client_ssl_context,
    build_server_ssl_context,
)

__all__ = [
    "A2AClient",
    "AuthenticationError",
    "CompatibilityError",
    "ClientCertificate",
    "DiscoveryError",
    "DiscoveryPolicy",
    "InMemoryTaskRepository",
    "LimitExceededError",
    "MTLSPeerIdentity",
    "ProtocolError",
    "RemoteAgentDescriptor",
    "RemoteTaskError",
    "TLSConfigurationError",
    "TLSPolicy",
    "TaskRepository",
    "TransportError",
    "authenticate_incoming_request",
    "build_authorization_header",
    "build_client_ssl_context",
    "build_delegated_token_request",
    "build_server_ssl_context",
    "discover_agent",
]
