"""Deterministic named scenarios for the identity fixture token endpoint.

Each scenario controls how the fixture issues bearer tokens so acceptance
tests can exercise specific bearer-token validation outcomes (invalid
signature, expiry, audience mismatch, insufficient scope) without any
external identity provider or network call. Scenario selection is fixed for
the lifetime of one process and is chosen with the
``IDENTITY_FIXTURE_SCENARIO`` environment variable, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCENARIO_NAME = "default"
"""Scenario name used when ``IDENTITY_FIXTURE_SCENARIO`` is unset."""

DEFAULT_LIFETIME_SECONDS = 300
"""Baseline token lifetime, before a scenario's expiry offset is applied."""


@dataclass(frozen=True, slots=True)
class IdentityScenario:
    """Immutable configuration for one deterministic identity-fixture behavior.

    Attributes:
        name: Stable scenario identifier.
        issuer_override: When set, the ``iss`` claim used instead of the
            configured fixture issuer.
        audience_override: When set, the ``aud`` claim used instead of the
            caller-requested audience.
        scope_override: When set, the ``scope`` claim used instead of the
            caller-requested scope; an empty string yields no granted scope.
        expires_delta_seconds: Offset in seconds applied to
            :data:`DEFAULT_LIFETIME_SECONDS`; negative values produce an
            already-expired token.
        sign_with_untrusted_key: When true, the token is signed with a key
            that is never published in the JWKS document, while the
            protected header still names the trusted key id, so a caller
            resolves the genuine trusted key and signature verification
            fails.
    """

    name: str
    issuer_override: str | None = None
    audience_override: str | None = None
    scope_override: str | None = None
    expires_delta_seconds: int = 0
    sign_with_untrusted_key: bool = False


SCENARIOS: dict[str, IdentityScenario] = {
    "default": IdentityScenario(name="default"),
    "invalid-signature": IdentityScenario(name="invalid-signature", sign_with_untrusted_key=True),
    "expired-token": IdentityScenario(name="expired-token", expires_delta_seconds=-3600),
    "wrong-audience": IdentityScenario(
        name="wrong-audience", audience_override="https://wrong-audience.fixture.invalid"
    ),
    "insufficient-scope": IdentityScenario(name="insufficient-scope", scope_override=""),
}
"""Named identity-fixture scenarios, keyed by ``IDENTITY_FIXTURE_SCENARIO`` value."""


def resolve_scenario(name: str) -> IdentityScenario:
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
        raise ValueError(f"unknown identity fixture scenario {name!r}; available: {choices}")
    return scenario


__all__ = [
    "DEFAULT_LIFETIME_SECONDS",
    "DEFAULT_SCENARIO_NAME",
    "SCENARIOS",
    "IdentityScenario",
    "resolve_scenario",
]
