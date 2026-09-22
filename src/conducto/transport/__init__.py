"""Network transport contracts and lazily loaded optional adapters."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .a2a import A2AClient as A2AClient
    from .a2a import DiscoveryPolicy as DiscoveryPolicy
    from .a2a import RemoteAgentDescriptor as RemoteAgentDescriptor
    from .a2a import discover_agent as discover_agent
    from .auth import MTLSPeerIdentity as MTLSPeerIdentity
    from .auth import authenticate_incoming_request as authenticate_incoming_request
    from .auth import build_authorization_header as build_authorization_header
    from .auth import build_delegated_token_request as build_delegated_token_request
    from .errors import AuthenticationError as AuthenticationError
    from .errors import CompatibilityError as CompatibilityError
    from .errors import DiscoveryError as DiscoveryError
    from .errors import LimitExceededError as LimitExceededError
    from .errors import ProtocolError as ProtocolError
    from .errors import RemoteTaskError as RemoteTaskError
    from .errors import TLSConfigurationError as TLSConfigurationError
    from .errors import TransportError as TransportError
    from .tasks import InMemoryTaskRepository as InMemoryTaskRepository
    from .tasks import TaskRepository as TaskRepository
    from .tls import ClientCertificate as ClientCertificate
    from .tls import TLSPolicy as TLSPolicy
    from .tls import build_client_ssl_context as build_client_ssl_context
    from .tls import build_server_ssl_context as build_server_ssl_context

_EXPORT_MODULES = {
    "A2AClient": ".a2a",
    "AuthenticationError": ".errors",
    "ClientCertificate": ".tls",
    "CompatibilityError": ".errors",
    "DiscoveryError": ".errors",
    "DiscoveryPolicy": ".a2a",
    "InMemoryTaskRepository": ".tasks",
    "LimitExceededError": ".errors",
    "MTLSPeerIdentity": ".auth",
    "ProtocolError": ".errors",
    "RemoteAgentDescriptor": ".a2a",
    "RemoteTaskError": ".errors",
    "TLSConfigurationError": ".errors",
    "TLSPolicy": ".tls",
    "TaskRepository": ".tasks",
    "TransportError": ".errors",
    "authenticate_incoming_request": ".auth",
    "build_authorization_header": ".auth",
    "build_client_ssl_context": ".tls",
    "build_delegated_token_request": ".auth",
    "build_server_ssl_context": ".tls",
    "discover_agent": ".a2a",
}


def __getattr__(name: str) -> Any:
    """Load a public transport symbol without importing unrelated adapters."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return deterministic public names for interactive inspection."""
    return sorted({*globals(), *_EXPORT_MODULES})


__all__ = list(_EXPORT_MODULES)
