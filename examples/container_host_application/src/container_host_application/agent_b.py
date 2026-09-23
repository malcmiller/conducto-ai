"""Deterministic Agent B application for the container-host example."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.container import ContainerConfig

from .applications import build_application, response_payload

if TYPE_CHECKING:
    from starlette.types import ASGIApp


@a2a_agent(
    name="ContainerHostAgentB",
    version="0.1.0",
    description="Deterministic Agent B role for a shared deployment image.",
)
class AgentB(BaseAgent):
    """Return deterministic Agent B identity without using a model."""

    def __init__(self, config: ContainerConfig) -> None:
        """Initialize the role with the configured image identity."""
        super().__init__(model_reference=config.model_reference)
        self._config = config
        self.agent_metadata = replace(
            self.agent_metadata,
            name=config.agent_id,
            version=config.agent_version,
        )

    @a2a_capability(name="respond", description="Return the configured Agent B role.")
    def respond(self, value: str) -> dict[str, str]:
        """Return a deterministic Agent B capability payload."""
        return response_payload(
            self._config,
            role="agent-b",
            capability="respond",
            value=value,
        )


def build_app(config: ContainerConfig) -> ASGIApp:
    """Build the Agent B ASGI application for ``config``."""
    return build_application(config, agent=AgentB(config))
