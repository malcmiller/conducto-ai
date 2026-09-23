"""Shared helpers for deterministic container-host deployment applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from conducto import BaseAgent, Runtime
from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    create_a2a_app,
)
from conducto.container import ContainerConfig, build_host_security_config
from conducto.core.model_config import RuntimeConfig
from conducto.security import AuthorizationContext, Principal

if TYPE_CHECKING:
    from starlette.types import ASGIApp


@dataclass(frozen=True, slots=True)
class ApplicationResponse:
    """Deterministic response facts returned by one hosted example capability."""

    role: str
    capability: str
    value: str
    agent_id: str
    agent_version: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON payload published by the hosted capability."""
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "capability": self.capability,
            "role": self.role,
            "value": self.value,
        }


async def resolve_identity(request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
    """Return deterministic local authority for the example deployment apps."""
    return A2AAuthenticatedIdentity(
        AuthorizationContext(
            principal=Principal(
                subject_id="container-host-application",
                issuer="container-host-application",
                audience="container-host-application",
                scopes=frozenset({"invoke"}),
            ),
            task_id=request.task_id,
            correlation_id=request.correlation_id,
        )
    )


def response_payload(
    config: ContainerConfig,
    *,
    role: str,
    capability: str,
    value: str,
) -> dict[str, str]:
    """Return one deterministic capability payload for the configured role."""
    return ApplicationResponse(
        role=role,
        capability=capability,
        value=value,
        agent_id=config.agent_id,
        agent_version=config.agent_version,
    ).to_dict()


def build_application(config: ContainerConfig, *, agent: BaseAgent) -> ASGIApp:
    """Build the shared A2A host for one deterministic deployment agent."""
    return create_a2a_app(
        agent=agent,
        runtime=Runtime(config=RuntimeConfig(default_model=config.model_reference)),
        public_url=config.public_url,
        endpoint_path=config.a2a_endpoint,
        identity_resolver=resolve_identity,
        security_config=build_host_security_config(config),
    )


__all__ = [
    "ApplicationResponse",
    "build_application",
    "resolve_identity",
    "response_payload",
]
