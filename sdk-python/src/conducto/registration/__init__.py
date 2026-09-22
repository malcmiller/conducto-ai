"""Deployment registration control-plane contracts, not A2A capability endpoints.

HTTP adapters are opt-in imports from ``.asgi`` and ``.client``. Application
code owns credentials, the catalog, audit delivery, and network policy.
"""

from .models import (
    DeregisterRequest,
    DrainRequest,
    RegisterRequest,
    RegistrationCode,
    RegistrationRequest,
    RegistrationResult,
    RenewRequest,
    RevokeRequest,
    StatusRequest,
)
from .policy import RegistrationGrant
from .service import RegistrationService

__all__ = [
    "DeregisterRequest",
    "DrainRequest",
    "RegisterRequest",
    "RegistrationCode",
    "RegistrationGrant",
    "RegistrationRequest",
    "RegistrationResult",
    "RegistrationService",
    "RenewRequest",
    "RevokeRequest",
    "StatusRequest",
]
