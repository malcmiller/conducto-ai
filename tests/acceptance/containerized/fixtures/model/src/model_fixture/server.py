"""Deterministic OpenAI-compatible model fixture server.

This module implements only the OpenAI-compatible routes exercised by
:class:`conducto.providers.openai_compatible.OpenAICompatibleProvider`:
``POST /v1/chat/completions`` and ``GET /v1/models`` (used by the provider's
readiness check). It never loads model weights, never calls a real
provider or the network, and returns only scripted, deterministic output.
This module intentionally has no dependency on ``conducto`` so the fixture
ships in its own lightweight image.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .scenarios import DEFAULT_SCENARIO_NAME, ModelScenario, resolve_scenario

DEFAULT_MODEL_NAME = "fixture-model"
"""Default model id listed by ``GET /v1/models`` and echoed in completions."""

DEFAULT_TIMEOUT_DELAY_SECONDS = 5.0
"""Default delay applied by the ``timeout`` scenario."""

_FIXTURE_COMPLETION_ID = "chatcmpl-fixture"
_FIXTURE_TOOL_CALL_ID = "fixture-call-1"
_FIXTURE_TERMINAL_CONTENT = '{"result": "ok"}'
_MALFORMED_BODY = '{"id": "chatcmpl-fixture", "object": "chat.completion", "choices": ['


@dataclass(frozen=True, slots=True)
class ModelFixtureConfig:
    """Validated runtime configuration for the model fixture process.

    Attributes:
        bind_host: Interface the fixture's HTTP server listens on.
        bind_port: Port the fixture's HTTP server listens on.
        model_name: Model id listed by ``/v1/models`` and echoed by default
            in completion responses.
        scenario: Resolved scenario applied to every chat-completion request.
        timeout_delay_seconds: Delay applied when ``scenario.name ==
            "timeout"``, overriding the scenario table so tests can bound
            the delay without rebuilding the image.
    """

    bind_host: str = "0.0.0.0"
    bind_port: int = 8081
    model_name: str = DEFAULT_MODEL_NAME
    scenario: ModelScenario = resolve_scenario(DEFAULT_SCENARIO_NAME)
    timeout_delay_seconds: float = DEFAULT_TIMEOUT_DELAY_SECONDS

    @classmethod
    def load(cls) -> ModelFixtureConfig:
        """Build configuration from environment variables.

        Returns:
            A validated immutable configuration.

        Raises:
            ValueError: If an environment variable is malformed or names an
                unknown scenario.
        """
        bind_host = os.environ.get("MODEL_FIXTURE_BIND_HOST", "0.0.0.0")
        bind_port_raw = os.environ.get("MODEL_FIXTURE_BIND_PORT", "8081")
        try:
            bind_port = int(bind_port_raw)
        except ValueError as error:
            raise ValueError("MODEL_FIXTURE_BIND_PORT must be an integer") from error
        if not 1 <= bind_port <= 65535:
            raise ValueError("MODEL_FIXTURE_BIND_PORT must be between 1 and 65535")
        model_name = os.environ.get("MODEL_FIXTURE_MODEL", DEFAULT_MODEL_NAME)
        if not model_name.strip():
            raise ValueError("MODEL_FIXTURE_MODEL must not be empty")
        scenario_name = os.environ.get("MODEL_FIXTURE_SCENARIO", DEFAULT_SCENARIO_NAME)
        scenario = resolve_scenario(scenario_name)
        delay_raw = os.environ.get(
            "MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS", str(DEFAULT_TIMEOUT_DELAY_SECONDS)
        )
        try:
            timeout_delay_seconds = float(delay_raw)
        except ValueError as error:
            raise ValueError("MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS must be a number") from error
        if timeout_delay_seconds < 0:
            raise ValueError("MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS must not be negative")
        return cls(
            bind_host=bind_host,
            bind_port=bind_port,
            model_name=model_name,
            scenario=scenario,
            timeout_delay_seconds=timeout_delay_seconds,
        )


def _wants_tool_call(payload: Mapping[str, Any]) -> bool:
    """Return whether this request should receive a native tool-call turn."""
    tools = payload.get("tools")
    messages = payload.get("messages")
    if not isinstance(tools, list) or not tools or not isinstance(messages, list):
        return False
    return not any(
        isinstance(message, dict) and message.get("role") == "tool" for message in messages
    )


def _tool_definition_name(tools: list[Any]) -> str:
    """Return the first requested tool's function name, or a fallback."""
    first = tools[0]
    if isinstance(first, Mapping):
        function = first.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return name
    return "fixture_tool"


def _build_completion(
    model_name: str, payload: Mapping[str, Any], scenario: ModelScenario
) -> dict[str, Any]:
    """Build one deterministic OpenAI-shaped chat-completion response body."""
    requested_model = payload.get("model")
    response_model = (
        requested_model if isinstance(requested_model, str) and requested_model else model_name
    )
    if _wants_tool_call(payload):
        tools = payload["tools"]
        tool_call: dict[str, Any] = {
            "id": _FIXTURE_TOOL_CALL_ID,
            "type": "function",
            "function": {"name": _tool_definition_name(tools), "arguments": "{}"},
        }
        tool_calls = [tool_call, dict(tool_call)] if scenario.duplicate_tool_call else [tool_call]
        message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": _FIXTURE_TERMINAL_CONTENT}
        finish_reason = "stop"
    return {
        "id": _FIXTURE_COMPLETION_ID,
        "object": "chat.completion",
        "created": 0,
        "model": response_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def create_app(config: ModelFixtureConfig | None = None) -> FastAPI:
    """Build the model fixture ASGI application.

    Args:
        config: Optional preloaded configuration. When omitted, configuration
            is loaded from the environment.

    Returns:
        A configured FastAPI application exposing health, model-listing, and
        chat-completion endpoints.

    Raises:
        ValueError: If configuration is invalid.
    """
    resolved = config or ModelFixtureConfig.load()
    app = FastAPI(title="Conducto OpenAI-compatible model fixture", docs_url=None, redoc_url=None)
    app.state.model_fixture_config = resolved

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        """Report process liveness."""
        return {"status": "live"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        """Report readiness once scenario configuration is loaded."""
        return {"status": "ready"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List the single configured deterministic model id."""
        return {
            "object": "list",
            "data": [
                {"id": resolved.model_name, "object": "model", "owned_by": "conducto-fixture"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """Return one scenario-shaped OpenAI-compatible completion response."""
        scenario = resolved.scenario
        delay = (
            resolved.timeout_delay_seconds if scenario.name == "timeout" else scenario.delay_seconds
        )
        if delay:
            await asyncio.sleep(delay)
        if scenario.http_status != 200:
            return JSONResponse(
                status_code=scenario.http_status,
                content={
                    "error": {
                        "message": f"model fixture scenario {scenario.name!r} induced failure",
                        "type": "fixture_error",
                        "code": scenario.name,
                    }
                },
            )
        if scenario.malformed_body:
            return Response(content=_MALFORMED_BODY, media_type="application/json")
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "request body must be a JSON object",
                        "type": "invalid_request",
                    }
                },
            )
        return JSONResponse(content=_build_completion(resolved.model_name, payload, scenario))

    return app


def main() -> int:
    """Validate configuration and run the model fixture host.

    Returns:
        Process exit code: ``0`` on normal shutdown, ``78`` for invalid
        configuration, ``130`` on interrupt.
    """
    try:
        config = ModelFixtureConfig.load()
        app = create_app(config)
    except ValueError as error:
        print(f"model fixture configuration invalid: {error}", file=sys.stderr)
        return 78
    try:
        import uvicorn

        uvicorn.run(app, host=config.bind_host, port=config.bind_port, log_config=None)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
