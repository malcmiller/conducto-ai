"""Deterministic local OAuth/JWKS identity test double.

This package is a standalone test-double image used by the Conducto
containerized acceptance suite (Story 6.11). It issues scripted bearer
tokens and publishes a matching JWKS document so acceptance tests can
exercise ``conducto.security.tokens`` validation outcomes without any real
identity provider, network call, or credential. It is not a production
identity provider and must never be deployed as one.
"""

from __future__ import annotations

from .scenarios import DEFAULT_SCENARIO_NAME, SCENARIOS, IdentityScenario, resolve_scenario
from .server import IdentityFixtureConfig, create_app, main

__all__ = [
    "DEFAULT_SCENARIO_NAME",
    "SCENARIOS",
    "IdentityFixtureConfig",
    "IdentityScenario",
    "create_app",
    "main",
    "resolve_scenario",
]
