"""Deterministic local OAuth/JWKS identity fixture server.

This module implements the minimum deterministic test surface an
acceptance-test caller needs: a JWKS document, an RFC 8693 token-exchange
endpoint, and bounded health endpoints. It never calls a real identity
provider, never persists credentials, and never accepts a signing key baked
into the image; a signing key is either mounted at runtime through
``IDENTITY_FIXTURE_SIGNING_KEY_FILE`` or generated fresh in memory for the
lifetime of the process. This module intentionally has no dependency on
``conducto`` so the fixture ships in its own lightweight image.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)
from fastapi import FastAPI, Form, HTTPException

from .scenarios import (
    DEFAULT_LIFETIME_SECONDS,
    DEFAULT_SCENARIO_NAME,
    IdentityScenario,
    resolve_scenario,
)

_TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_ISSUED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
_MAX_FIELD_LENGTH = 2048
_MAX_SUBJECT_LENGTH = 256

DEFAULT_ISSUER = "https://identity.fixture.invalid"
"""Default token issuer used when ``IDENTITY_FIXTURE_ISSUER`` is unset."""


@dataclass(frozen=True, slots=True)
class IdentityFixtureConfig:
    """Validated runtime configuration for the identity fixture process.

    Attributes:
        issuer: Value used for the ``iss`` claim unless a scenario overrides it.
        bind_host: Interface the fixture's HTTP server listens on.
        bind_port: Port the fixture's HTTP server listens on.
        scenario: Resolved scenario applied to every issued token.
        signing_key_path: Optional path to a mounted PEM P-256 private key;
            when omitted, a fresh key is generated for the process lifetime.
    """

    issuer: str = DEFAULT_ISSUER
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    scenario: IdentityScenario = resolve_scenario(DEFAULT_SCENARIO_NAME)
    signing_key_path: str | None = None

    @classmethod
    def load(cls) -> IdentityFixtureConfig:
        """Build configuration from environment variables.

        Returns:
            A validated immutable configuration.

        Raises:
            ValueError: If an environment variable is malformed or names an
                unknown scenario.
        """
        issuer = os.environ.get("IDENTITY_FIXTURE_ISSUER", DEFAULT_ISSUER)
        if not issuer.strip():
            raise ValueError("IDENTITY_FIXTURE_ISSUER must not be empty")
        bind_host = os.environ.get("IDENTITY_FIXTURE_BIND_HOST", "0.0.0.0")
        bind_port_raw = os.environ.get("IDENTITY_FIXTURE_BIND_PORT", "8080")
        try:
            bind_port = int(bind_port_raw)
        except ValueError as error:
            raise ValueError("IDENTITY_FIXTURE_BIND_PORT must be an integer") from error
        if not 1 <= bind_port <= 65535:
            raise ValueError("IDENTITY_FIXTURE_BIND_PORT must be between 1 and 65535")
        scenario_name = os.environ.get("IDENTITY_FIXTURE_SCENARIO", DEFAULT_SCENARIO_NAME)
        scenario = resolve_scenario(scenario_name)
        signing_key_path = os.environ.get("IDENTITY_FIXTURE_SIGNING_KEY_FILE")
        return cls(
            issuer=issuer,
            bind_host=bind_host,
            bind_port=bind_port,
            scenario=scenario,
            signing_key_path=signing_key_path,
        )


@dataclass(frozen=True, slots=True)
class _IdentityFixtureState:
    """Immutable runtime material bound once per process for token issuance."""

    config: IdentityFixtureConfig
    trusted_key: ec.EllipticCurvePrivateKey
    trusted_key_id: str
    untrusted_key: ec.EllipticCurvePrivateKey


def _b64url(data: bytes) -> str:
    """Return unpadded base64url text for ``data``."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _json_dumps(value: Mapping[str, Any]) -> bytes:
    """Return canonical compact JSON bytes for a JWS segment."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _load_or_generate_signing_key(path: str | None) -> ec.EllipticCurvePrivateKey:
    """Load a mounted P-256 signing key, or generate an ephemeral one.

    Args:
        path: Optional path to a mounted PEM-encoded private key.

    Returns:
        A P-256 (secp256r1) private key.

    Raises:
        ValueError: If the mounted file cannot be read or is not a P-256 key.
    """
    if path is None:
        return ec.generate_private_key(ec.SECP256R1())
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise ValueError("identity fixture signing key file could not be read") from error
    try:
        key = load_pem_private_key(data, password=None)
    except ValueError as error:
        raise ValueError(
            "identity fixture signing key file is not a valid PEM private key"
        ) from error
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError("identity fixture signing key must be a P-256 (secp256r1) private key")
    return key


def _key_id(public_key: ec.EllipticCurvePublicKey) -> str:
    """Return a deterministic key id derived from ``public_key``."""
    raw = public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return hashlib.sha256(raw).hexdigest()[:16]


def _jwk(public_key: ec.EllipticCurvePublicKey, key_id: str) -> dict[str, str]:
    """Return one JWKS entry describing ``public_key``."""
    numbers = public_key.public_numbers()
    size = (public_key.curve.key_size + 7) // 8
    return {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "use": "sig",
        "kid": key_id,
        "x": _b64url(numbers.x.to_bytes(size, "big")),
        "y": _b64url(numbers.y.to_bytes(size, "big")),
    }


def _sign(
    private_key: ec.EllipticCurvePrivateKey,
    header: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> str:
    """Return a compact ES256 JWS for ``claims`` signed by ``private_key``."""
    protected = _b64url(_json_dumps(header))
    payload = _b64url(_json_dumps(claims))
    signing_input = f"{protected}.{payload}".encode("ascii")
    der_signature = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{protected}.{payload}.{_b64url(raw_signature)}"


def _validate_token_request(
    *,
    grant_type: str,
    subject_token: str,
    subject_token_type: str,
    audience: str,
    actor_token: str | None,
    actor_token_type: str | None,
) -> None:
    """Reject a malformed token-exchange request before a token is issued.

    Raises:
        HTTPException: If a required field is missing, oversized, or the
            grant type is unsupported.
    """
    required_fields = (grant_type, subject_token, subject_token_type, audience)
    if any(not field.strip() or len(field) > _MAX_FIELD_LENGTH for field in required_fields):
        raise HTTPException(status_code=400, detail={"error": "invalid_request"})
    if grant_type != _TOKEN_EXCHANGE_GRANT_TYPE:
        raise HTTPException(status_code=400, detail={"error": "unsupported_grant_type"})
    if bool(actor_token) != bool(actor_token_type):
        raise HTTPException(status_code=400, detail={"error": "invalid_request"})


def _issue_token(
    state: _IdentityFixtureState, *, subject_token: str, audience: str, scope: str
) -> dict[str, Any]:
    """Build and sign one scenario-shaped RFC 8693 token-exchange response."""
    scenario = state.config.scenario
    issued_at = int(time.time())
    lifetime = DEFAULT_LIFETIME_SECONDS + scenario.expires_delta_seconds
    expires_at = issued_at + lifetime
    granted_scope = scope.strip() if scenario.scope_override is None else scenario.scope_override
    claims: dict[str, Any] = {
        "iss": scenario.issuer_override or state.config.issuer,
        "sub": subject_token[:_MAX_SUBJECT_LENGTH],
        "aud": scenario.audience_override or audience,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": expires_at,
        "scope": granted_scope,
    }
    header = {"alg": "ES256", "kid": state.trusted_key_id}
    signing_key = state.untrusted_key if scenario.sign_with_untrusted_key else state.trusted_key
    access_token = _sign(signing_key, header, claims)
    return {
        "access_token": access_token,
        "issued_token_type": _JWT_ISSUED_TOKEN_TYPE,
        "token_type": "Bearer",
        "expires_in": expires_at - issued_at,
        "scope": granted_scope,
    }


def create_app(config: IdentityFixtureConfig | None = None) -> FastAPI:
    """Build the identity fixture ASGI application.

    Args:
        config: Optional preloaded configuration. When omitted, configuration
            is loaded from the environment.

    Returns:
        A configured FastAPI application exposing health, JWKS, and
        token-exchange endpoints.

    Raises:
        ValueError: If configuration or a mounted signing key is invalid.
    """
    resolved = config or IdentityFixtureConfig.load()
    trusted_key = _load_or_generate_signing_key(resolved.signing_key_path)
    state = _IdentityFixtureState(
        config=resolved,
        trusted_key=trusted_key,
        trusted_key_id=_key_id(trusted_key.public_key()),
        untrusted_key=ec.generate_private_key(ec.SECP256R1()),
    )

    app = FastAPI(title="Conducto identity fixture", docs_url=None, redoc_url=None)
    app.state.identity_fixture = state

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        """Report process liveness."""
        return {"status": "live"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        """Report readiness once scenario configuration and keys are loaded."""
        return {"status": "ready"}

    @app.get("/.well-known/jwks.json")
    async def jwks() -> dict[str, list[dict[str, str]]]:
        """Publish only the trusted verification key."""
        return {"keys": [_jwk(state.trusted_key.public_key(), state.trusted_key_id)]}

    @app.post("/token")
    async def token(
        grant_type: Annotated[str, Form()],
        subject_token: Annotated[str, Form()],
        subject_token_type: Annotated[str, Form()],
        audience: Annotated[str, Form()],
        scope: Annotated[str, Form()] = "",
        actor_token: Annotated[str | None, Form()] = None,
        actor_token_type: Annotated[str | None, Form()] = None,
        resource: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        """Issue a scenario-shaped bearer token for one exchange request."""
        del resource
        _validate_token_request(
            grant_type=grant_type,
            subject_token=subject_token,
            subject_token_type=subject_token_type,
            audience=audience,
            actor_token=actor_token,
            actor_token_type=actor_token_type,
        )
        return _issue_token(state, subject_token=subject_token, audience=audience, scope=scope)

    return app


def main() -> int:
    """Validate configuration and run the identity fixture host.

    Returns:
        Process exit code: ``0`` on normal shutdown, ``78`` for invalid
        configuration, ``130`` on interrupt.
    """
    try:
        config = IdentityFixtureConfig.load()
        app = create_app(config)
    except ValueError as error:
        print(f"identity fixture configuration invalid: {error}", file=sys.stderr)
        return 78
    try:
        import uvicorn

        uvicorn.run(app, host=config.bind_host, port=config.bind_port, log_config=None)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
