"""Reference process host for the immutable Conducto container image.

The host is intentionally model-neutral. It publishes a deterministic local
capability so image verification never needs a model, provider credential, or
network service. Applications can replace :func:`build_app` while retaining
the same configuration and lifecycle boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2AHostSecurityConfig,
    create_a2a_app,
)
from conducto.security import AuthorizationContext, Principal

_PREFIX = "CONDUCTO_"
_SECRET_NAMES = frozenset({"token", "secret", "api_key", "authorization", "password"})


@dataclass(frozen=True, slots=True)
class ContainerConfig:
    """Validated process configuration loaded from a file and environment.

    Environment variables take precedence over the JSON configuration file.
    Secret values are accepted only through ``*_FILE`` variables and are never
    included in diagnostic output.
    """

    agent_id: str = "conducto-reference-agent"
    agent_version: str = "0.1.0"
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000
    public_url: str = "http://127.0.0.1:8000"
    a2a_endpoint: str = "/a2a"
    provider_type: str = "local"
    model_reference: str = "local"
    provider_endpoint: str | None = None
    request_timeout_seconds: float = 30.0
    shutdown_grace_seconds: float = 30.0
    trust_roots_file: str | None = None
    issuer: str | None = None
    audience: str | None = None
    scopes: tuple[str, ...] = ()
    log_format: str = "json"
    writable_path: str = "/tmp/conducto"
    max_concurrency: int = 64
    max_request_bytes: int = 262_144
    config_file: str | None = None

    @classmethod
    def load(cls) -> ContainerConfig:
        """Load JSON file settings, then apply typed environment overrides.

        Returns:
            A validated immutable configuration.

        Raises:
            ValueError: If a setting is malformed or violates a safety bound.
        """
        config_path = os.environ.get("CONDUCTO_CONFIG_FILE")
        values: dict[str, Any] = {}
        if config_path:
            path = Path(config_path)
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("CONDUCTO_CONFIG_FILE could not be read as JSON") from error
            if not isinstance(loaded, dict):
                raise ValueError("CONDUCTO_CONFIG_FILE must contain a JSON object")
            values.update(loaded)
            values["config_file"] = str(path)

        known = {item.name for item in fields(cls)}
        for name in known:
            environment_name = _PREFIX + name.upper()
            raw = os.environ.get(environment_name)
            file_value = os.environ.get(environment_name + "_FILE")
            if raw is not None and file_value is not None:
                raise ValueError(
                    f"{environment_name} and {environment_name}_FILE are mutually exclusive"
                )
            if file_value is not None:
                try:
                    raw = Path(file_value).read_text(encoding="utf-8").strip()
                except OSError as error:
                    raise ValueError(f"{environment_name}_FILE could not be read") from error
            if raw is not None:
                values[name] = raw
        return cls._from_values(values)

    @classmethod
    def _from_values(cls, values: dict[str, Any]) -> ContainerConfig:
        """Coerce supported scalar settings and validate the complete schema."""
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")
        integer_fields = {"bind_port", "max_concurrency", "max_request_bytes"}
        float_fields = {"request_timeout_seconds", "shutdown_grace_seconds"}
        for name in integer_fields:
            if name in values and isinstance(values[name], str):
                try:
                    values[name] = int(values[name])
                except ValueError as error:
                    raise ValueError(f"{name} must be an integer") from error
        for name in float_fields:
            if name in values and isinstance(values[name], str):
                try:
                    values[name] = float(values[name])
                except ValueError as error:
                    raise ValueError(f"{name} must be a number") from error
        if "scopes" in values and isinstance(values["scopes"], str):
            values["scopes"] = tuple(
                item.strip() for item in values["scopes"].split(",") if item.strip()
            )
        elif "scopes" in values:
            values["scopes"] = tuple(values["scopes"])
        result = cls(**values)
        if not result.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if not 1 <= result.bind_port <= 65535:
            raise ValueError("bind_port must be between 1 and 65535")
        if result.request_timeout_seconds <= 0 or result.shutdown_grace_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if result.max_concurrency <= 0 or result.max_request_bytes <= 0:
            raise ValueError("resource limits must be positive")
        if result.log_format not in {"json", "text"}:
            raise ValueError("log_format must be 'json' or 'text'")
        if not result.a2a_endpoint.startswith("/"):
            raise ValueError("a2a_endpoint must start with '/'")
        if any(name in result.model_reference.lower() for name in _SECRET_NAMES):
            raise ValueError("model_reference must not contain secret-like material")
        return result


@a2a_agent(
    name="Conducto Reference Agent",
    version="0.1.0",
    description="Deterministic capability used for installed-image verification.",
)
class ReferenceAgent(BaseAgent):
    """Expose one deterministic capability for single-container smoke tests."""

    @a2a_capability(name="echo", description="Return a deterministic local value.")
    def echo(self, value: str) -> dict[str, str]:
        """Return the supplied value without contacting a provider."""
        return {"value": value}


async def _resolve_identity(request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
    """Create a bounded local identity without accepting caller credentials."""
    return A2AAuthenticatedIdentity(
        AuthorizationContext(
            principal=Principal(
                subject_id="container-local",
                issuer="conducto-container",
                audience="conducto-reference-agent",
                scopes=frozenset(),
            ),
            task_id=request.task_id,
            correlation_id=request.correlation_id,
        )
    )


def build_app(config: ContainerConfig | None = None) -> Any:
    """Build the reference ASGI application from immutable runtime settings."""
    selected = config or ContainerConfig.load()
    return create_a2a_app(
        agent=ReferenceAgent(),
        runtime=Runtime(),
        public_url=selected.public_url,
        endpoint_path=selected.a2a_endpoint,
        identity_resolver=_resolve_identity,
        security_config=A2AHostSecurityConfig(
            allowed_paths=frozenset({selected.a2a_endpoint, "/.well-known/agent-card.json"}),
            liveness_path="/livez",
            readiness_path="/readyz",
            request_deadline_seconds=selected.request_timeout_seconds,
            shutdown_deadline_seconds=selected.shutdown_grace_seconds,
            drain_deadline_seconds=selected.shutdown_grace_seconds,
            max_accepted_concurrency=selected.max_concurrency,
            max_request_body_bytes=selected.max_request_bytes,
        ),
    )


def _diagnostic(error: ValueError) -> str:
    """Return a stable redacted startup diagnostic."""
    message = str(error)
    for name in _SECRET_NAMES:
        message = message.replace(name, "<redacted>")
    return f"configuration invalid: {message}"


def main(argv: list[str] | None = None) -> int:
    """Validate configuration and run the installed reference host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ContainerConfig.load()
        if args.check_config:
            print(json.dumps({"status": "valid", "agent_id": config.agent_id}, sort_keys=True))
            return 0
        import uvicorn

        uvicorn.run(
            build_app(config),
            host=config.bind_host,
            port=config.bind_port,
            log_config=None,
            timeout_graceful_shutdown=int(config.shutdown_grace_seconds),
        )
    except ValueError as error:
        print(_diagnostic(error), file=sys.stderr)
        return 78
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
