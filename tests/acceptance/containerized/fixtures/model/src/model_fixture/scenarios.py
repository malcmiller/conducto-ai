"""Deterministic named scenarios for the OpenAI-compatible model fixture.

Scenario selection controls how ``POST /v1/chat/completions`` responds so
acceptance tests can exercise native tool-call turns, malformed responses,
duplicate tool-call ids, upstream errors, and delayed/hung responses without
any real inference, model weights, or network access. Scenario selection is
fixed for the lifetime of one process and is chosen with the
``MODEL_FIXTURE_SCENARIO`` environment variable, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCENARIO_NAME = "default"
"""Scenario name used when ``MODEL_FIXTURE_SCENARIO`` is unset."""


@dataclass(frozen=True, slots=True)
class ModelScenario:
    """Immutable configuration for one deterministic model-fixture behavior.

    Attributes:
        name: Stable scenario identifier.
        http_status: HTTP status code returned for
            ``/v1/chat/completions``; a non-200 value short-circuits before
            any response body is built.
        delay_seconds: Bounded delay applied before responding; used to
            exercise client-side request timeouts. The fixture's
            ``timeout`` scenario delay is instead controlled by
            ``MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS`` so tests can bound it
            without rebuilding the image.
        malformed_body: When true, the response body is not valid JSON.
        duplicate_tool_call: When true, a tool-call turn returns two native
            tool calls that share the same call id.
    """

    name: str
    http_status: int = 200
    delay_seconds: float = 0.0
    malformed_body: bool = False
    duplicate_tool_call: bool = False


SCENARIOS: dict[str, ModelScenario] = {
    "default": ModelScenario(name="default"),
    "malformed-response": ModelScenario(name="malformed-response", malformed_body=True),
    "timeout": ModelScenario(name="timeout"),
    "duplicate-tool-call": ModelScenario(name="duplicate-tool-call", duplicate_tool_call=True),
    "error-500": ModelScenario(name="error-500", http_status=500),
}
"""Named model-fixture scenarios, keyed by ``MODEL_FIXTURE_SCENARIO`` value."""


def resolve_scenario(name: str) -> ModelScenario:
    """Return the named scenario, failing closed for unknown names.

    Args:
        name: Requested scenario name.

    Returns:
        The immutable scenario configuration.

    Raises:
        ValueError: If ``name`` is not a known scenario.
    """
    scenario = SCENARIOS.get(name)
    if scenario is None:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown model fixture scenario {name!r}; available: {choices}")
    return scenario


__all__ = [
    "DEFAULT_SCENARIO_NAME",
    "SCENARIOS",
    "ModelScenario",
    "resolve_scenario",
]
