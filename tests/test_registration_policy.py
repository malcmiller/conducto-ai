"""Construction-time validation of exact deployment endpoint bindings."""

from dataclasses import replace

import pytest

from conducto.registration import RegistrationGrant


def _grant() -> RegistrationGrant:
    return RegistrationGrant(
        issuer="issuer",
        subject_id="deployment",
        owner="team",
        environment="test",
        agent_id="team.agent",
        card_name="Agent",
        agent_card_url="https://agent.example/card",
        endpoint_url="https://agent.example/a2a",
    )


@pytest.mark.parametrize("field", ["agent_card_url", "endpoint_url"])
@pytest.mark.parametrize(
    "url",
    [
        "https://agent.example:bad/a2a",
        "https://agent.example:0/a2a",
        "https://agent.example:65536/a2a",
        "https://agent.example:-1/a2a",
        "https://[invalid/a2a",
        "https://agent.example/a2a\n",
        "https://agent.exa\tmple/a2a",
        "https://agent.exa\rmple/a2a",
        "\x00https://agent.example/a2a",
        " https://agent.example/a2a",
        "https://agent.example/with space",
        "https://agent.example/\x7f",
        "https://agent.example\\other/a2a",
        "https://agent.example/a2a?",
        "https://agent.example/a2a#",
    ],
)
def test_grants_reject_unsafe_urls_without_echoing_them(field: str, url: str) -> None:
    with pytest.raises(ValueError, match="registration URLs") as error:
        replace(_grant(), **{field: url})
    assert url not in str(error.value)


@pytest.mark.parametrize("field", ["agent_card_url", "endpoint_url"])
@pytest.mark.parametrize(
    "url",
    [
        "https://agent.example/a2a",
        "https://agent.example:443/a2a",
        "https://agent.example:65535/a2a",
        "http://127.0.0.1:8081/a2a",
        "https://[::1]:8443/a2a",
    ],
)
def test_grants_preserve_valid_exact_urls(field: str, url: str) -> None:
    grant = replace(_grant(), **{field: url})
    assert getattr(grant, field) == url
