"""Focused tests for the deterministic identity fixture's HTTP contract.

These tests exercise the fixture's own JWKS and token-exchange endpoints in
isolation. Tokens issued by each scenario are also validated with the real
``conducto.security.tokens`` bearer-token validator so the fixture is proven
against the exact contract a Conducto host applies, without standing up a
full agent or orchestrator.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from identity_fixture.scenarios import SCENARIOS, resolve_scenario
from identity_fixture.server import IdentityFixtureConfig, create_app

from conducto.security.tokens import (
    InvalidAudienceError,
    InvalidScopeError,
    InvalidSignatureTokenError,
    JWTBearerTokenValidator,
    StaticJWKSResolver,
    TokenExpiredError,
)
from conducto.security.trust import AudiencePolicy, IssuerPolicy, ScopePolicy, TrustPolicy

pytestmark = pytest.mark.acceptance

_ISSUER = "https://identity.fixture.invalid"
_AUDIENCE = "conducto-agent-b"
_TOKEN_FORM = {
    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
    "subject_token": "caller-subject",
    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
    "audience": _AUDIENCE,
    "scope": "agent.invoke",
}


def _client_for(scenario_name: str) -> TestClient:
    """Build a TestClient bound to one named identity-fixture scenario."""
    config = IdentityFixtureConfig(
        issuer=_ISSUER,
        bind_host="127.0.0.1",
        bind_port=0,
        scenario=resolve_scenario(scenario_name),
        signing_key_path=None,
    )
    return TestClient(create_app(config))


def _decode_segment(segment: str) -> dict[str, Any]:
    """Decode one base64url JWS segment as a JSON object."""
    padded = segment + "=" * (-len(segment) % 4)
    decoded: dict[str, Any] = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    return decoded


def _public_key_from_jwk(jwk: dict[str, str]) -> ec.EllipticCurvePublicKey:
    """Reconstruct an EC public key from one published JWKS entry."""
    x = int.from_bytes(base64.urlsafe_b64decode(jwk["x"] + "=" * (-len(jwk["x"]) % 4)), "big")
    y = int.from_bytes(base64.urlsafe_b64decode(jwk["y"] + "=" * (-len(jwk["y"]) % 4)), "big")
    return ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()


def _trust_policy(*, required_scopes: frozenset[str] = frozenset()) -> TrustPolicy:
    """Build a trust policy matching the default fixture request shape."""
    return TrustPolicy(
        version="fixture-test-1",
        issuer=IssuerPolicy(issuer=_ISSUER),
        audience=AudiencePolicy(audiences=frozenset({_AUDIENCE})),
        scopes=ScopePolicy(
            allowed_scopes=frozenset({"agent.invoke", "agent.delegate"}),
            required_scopes=required_scopes,
        ),
    )


def _validator_for(client: TestClient) -> JWTBearerTokenValidator:
    """Build a validator trusting exactly the fixture's published JWKS key."""
    jwks = client.get("/.well-known/jwks.json").json()
    entry = jwks["keys"][0]
    resolver = StaticJWKSResolver({(_ISSUER, entry["kid"]): _public_key_from_jwk(entry)})
    return JWTBearerTokenValidator(resolver)


def test_health_endpoints_report_ready() -> None:
    """Liveness and readiness endpoints report a stable ready status."""
    client = _client_for("default")
    assert client.get("/livez").json() == {"status": "live"}
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_jwks_publishes_only_the_trusted_key() -> None:
    """The JWKS document exposes exactly one EC P-256 signing key."""
    client = _client_for("default")
    jwks = client.get("/.well-known/jwks.json").json()
    assert len(jwks["keys"]) == 1
    entry = jwks["keys"][0]
    assert entry["kty"] == "EC"
    assert entry["crv"] == "P-256"
    assert entry["alg"] == "ES256"


def test_default_scenario_issues_a_token_the_real_validator_accepts() -> None:
    """A ``default``-scenario token validates against the published JWKS key."""
    client = _client_for("default")
    response = client.post("/token", data=_TOKEN_FORM)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "Bearer"
    assert body["issued_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
    assert body["scope"] == _TOKEN_FORM["scope"]
    assert body["expires_in"] > 0

    identity = _validator_for(client).validate(
        body["access_token"], policy=_trust_policy(required_scopes=frozenset({"agent.invoke"}))
    )
    assert identity.subject == _TOKEN_FORM["subject_token"]
    assert identity.issuer == _ISSUER
    assert _AUDIENCE in identity.audience
    assert identity.scopes == {"agent.invoke"}


def test_invalid_signature_scenario_fails_real_signature_verification() -> None:
    """A signature-spoofed token is rejected by the real bearer-token validator."""
    client = _client_for("invalid-signature")
    body = client.post("/token", data=_TOKEN_FORM).json()
    header = _decode_segment(body["access_token"].split(".")[0])
    jwks = client.get("/.well-known/jwks.json").json()
    assert header["kid"] == jwks["keys"][0]["kid"]

    with pytest.raises(InvalidSignatureTokenError):
        _validator_for(client).validate(body["access_token"], policy=_trust_policy())


def test_expired_token_scenario_fails_real_expiry_validation() -> None:
    """An intentionally expired token is rejected with ``TokenExpiredError``."""
    client = _client_for("expired-token")
    body = client.post("/token", data=_TOKEN_FORM).json()
    claims = _decode_segment(body["access_token"].split(".")[1])
    assert claims["exp"] < int(time.time())

    with pytest.raises(TokenExpiredError):
        _validator_for(client).validate(body["access_token"], policy=_trust_policy())


def test_wrong_audience_scenario_fails_real_audience_validation() -> None:
    """A token bound to an unrelated audience is rejected with ``InvalidAudienceError``."""
    client = _client_for("wrong-audience")
    body = client.post("/token", data=_TOKEN_FORM).json()
    claims = _decode_segment(body["access_token"].split(".")[1])
    assert claims["aud"] != _AUDIENCE

    with pytest.raises(InvalidAudienceError):
        _validator_for(client).validate(body["access_token"], policy=_trust_policy())


def test_insufficient_scope_scenario_fails_real_scope_validation() -> None:
    """A token missing a required scope is rejected with ``InvalidScopeError``."""
    client = _client_for("insufficient-scope")
    body = client.post("/token", data=_TOKEN_FORM).json()
    assert body["scope"] == ""

    with pytest.raises(InvalidScopeError):
        _validator_for(client).validate(
            body["access_token"], policy=_trust_policy(required_scopes=frozenset({"agent.invoke"}))
        )


def test_all_documented_scenarios_are_covered_by_this_module() -> None:
    """Every scenario the fixture supports has a matching test above."""
    covered = {
        "default",
        "invalid-signature",
        "expired-token",
        "wrong-audience",
        "insufficient-scope",
    }
    assert set(SCENARIOS) == covered


def test_token_endpoint_rejects_unsupported_grant_type() -> None:
    """An unsupported grant type is rejected before any token is issued."""
    client = _client_for("default")
    form = {**_TOKEN_FORM, "grant_type": "authorization_code"}
    response = client.post("/token", data=form)
    assert response.status_code == 400


def test_token_endpoint_rejects_unpaired_actor_token() -> None:
    """Supplying ``actor_token`` without ``actor_token_type`` is rejected."""
    client = _client_for("default")
    form = {**_TOKEN_FORM, "actor_token": "actor"}
    response = client.post("/token", data=form)
    assert response.status_code == 400


_IDENTITY_ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    """Return whether Docker is installed, reachable, and runs Linux containers."""
    binary = shutil.which("docker")
    if binary is None:
        return False
    info_result = subprocess.run([binary, "info"], capture_output=True, text=True, check=False)
    if info_result.returncode != 0:
        return False
    os_result = subprocess.run(
        [binary, "version", "--format", "{{.Server.Os}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return os_result.returncode == 0 and os_result.stdout.strip() == "linux"


def _free_port() -> int:
    """Reserve an ephemeral loopback port until the container binds it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_ready(url: str, *, attempts: int = 80) -> None:
    """Poll the bounded readiness endpoint instead of assuming a startup delay."""
    with httpx.Client(timeout=0.25) as http_client:
        for _ in range(attempts):
            try:
                response = http_client.get(url)
                if response.status_code == 200 and response.json() == {"status": "ready"}:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
    raise AssertionError(f"identity fixture at {url} did not become ready")


@pytest.mark.skipif(not _docker_available(), reason="docker is not available")
def test_identity_fixture_image_builds_and_serves_readyz() -> None:
    """The identity fixture image builds, starts, and reports readiness."""
    tag = f"identity-fixture:test-{os.getpid()}"
    container_name = f"identity-fixture-test-{os.getpid()}"
    host_port = _free_port()
    try:
        subprocess.run(["docker", "build", "--tag", tag, "."], cwd=_IDENTITY_ROOT, check=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--read-only",
                "--tmpfs",
                "/tmp/identity-fixture:rw,noexec,nosuid,size=16m",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "-p",
                f"{host_port}:8080",
                tag,
            ],
            check=True,
        )
        _wait_for_ready(f"http://127.0.0.1:{host_port}/readyz")
    finally:
        subprocess.run(["docker", "rm", "--force", container_name], check=False)
        subprocess.run(["docker", "image", "rm", "--force", tag], check=False)
