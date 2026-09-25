"""Application-facing entry points for the pre-v1 Conducto SDK.

Import specialized contracts from their owning ``conducto.core`` packages,
adapters from ``conducto.providers``, and deterministic fakes from
``conducto.testing``. Importing Conducto does not configure clients or logging.
"""

from .core.agent import BaseAgent
from .core.decorators import (
    a2a_agent,
    a2a_capability,
    budget,
    classification,
    requires_scope,
    side_effect,
    timeout,
    tool,
)
from .core.model_config import (
    AgentModelConfig,
    ModelReference,
    ModelRequirement,
    RunConfig,
    RuntimeConfig,
)
from .core.orchestrator import OrchestratorAgent
from .core.registry import AgentRegistry
from .core.run_context import RunContext, get_run_context, require_run_context
from .core.runtime import Runtime

__all__ = [
    "AgentModelConfig",
    "AgentRegistry",
    "BaseAgent",
    "ModelReference",
    "ModelRequirement",
    "OrchestratorAgent",
    "RunConfig",
    "RunContext",
    "Runtime",
    "RuntimeConfig",
    "a2a_agent",
    "a2a_capability",
    "budget",
    "classification",
    "get_run_context",
    "require_run_context",
    "requires_scope",
    "side_effect",
    "timeout",
    "tool",
]
