"""Deterministic OpenAI-compatible model test double.

This package is a standalone test-double image used by the Conducto
containerized acceptance suite (Story 6.11). It implements only the
``/v1/chat/completions`` and ``/v1/models`` routes exercised by
:class:`conducto.providers.openai_compatible.OpenAICompatibleProvider`. It is
a scripted response server, not a real inference runtime: it never loads
model weights, calls a real provider, or returns non-deterministic output.
"""

from __future__ import annotations

from .scenarios import DEFAULT_SCENARIO_NAME, SCENARIOS, ModelScenario, resolve_scenario
from .server import ModelFixtureConfig, create_app, main

__all__ = [
    "DEFAULT_SCENARIO_NAME",
    "SCENARIOS",
    "ModelFixtureConfig",
    "ModelScenario",
    "create_app",
    "main",
    "resolve_scenario",
]
