"""Deterministic orchestrator application for the container-host example."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.container import ContainerConfig

from .applications import build_application, response_payload

if TYPE_CHECKING:
    from starlette.types import ASGIApp


@a2a_agent(
    name="ContainerHostOrchestrator",
    version="0.1.0",
    description="Deterministic orchestrator role for a shared deployment image.",
)
class OrchestratorAgent(BaseAgent):
    """Return deterministic orchestrator identity without using a model."""

    def __init__(self, config: ContainerConfig) -> None:
        """Initialize the role with the configured image identity."""
        super().__init__(model_reference=config.model_reference)
        self._config = config
        self.agent_metadata = replace(
            self.agent_metadata,
            name=config.agent_id,
            version=config.agent_version,
        )

    @a2a_capability(name="coordinate", description="Return the configured orchestrator role.")
    def coordinate(self, value: str) -> dict[str, str]:
        """Return a deterministic orchestrator capability payload."""
        return response_payload(
            self._config,
            role="orchestrator",
            capability="coordinate",
            value=value,
        )


def build_app(config: ContainerConfig) -> ASGIApp:
    """Build the orchestrator ASGI application for ``config``."""
    return build_application(config, agent=OrchestratorAgent(config))
