"""Focused tests for the deterministic model fixture's HTTP contract.

These tests exercise the fixture's own OpenAI-compatible endpoints in
isolation: they assert response status codes, JSON shapes, and the scripted
behavior of each documented scenario without a real orchestrator, provider,
or model weights.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from model_fixture.scenarios import SCENARIOS, resolve_scenario
from model_fixture.server import DEFAULT_MODEL_NAME, ModelFixtureConfig, create_app

pytestmark = pytest.mark.acceptance

_CHAT_REQUEST: dict[str, object] = {
    "model": DEFAULT_MODEL_NAME,
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.0,
}
_TOOL_CHAT_REQUEST: dict[str, object] = {
    **_CHAT_REQUEST,
    "tools": [
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    ],
}


def _client_for(scenario_name: str, **overrides: object) -> TestClient:
    """Build a TestClient bound to one named model-fixture scenario."""
    config = ModelFixtureConfig(scenario=resolve_scenario(scenario_name), **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(config))


def test_health_endpoints_report_ready() -> None:
    """Liveness and readiness endpoints report a stable ready status."""
    client = _client_for("default")
    assert client.get("/livez").json() == {"status": "live"}
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_models_endpoint_lists_configured_model() -> None:
    """``GET /v1/models`` lists exactly the configured deterministic model id."""
    client = _client_for("default", model_name="probe-model")
    body = client.get("/v1/models").json()
    ids = {entry["id"] for entry in body["data"]}
    assert ids == {"probe-model"}


def test_default_scenario_returns_tool_call_when_tools_present() -> None:
    """The default scenario selects a native tool call on the first turn."""
    client = _client_for("default")
    response = client.post("/v1/chat/completions", json=_TOOL_CHAT_REQUEST)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"


def test_default_scenario_returns_terminal_response_after_tool_result() -> None:
    """The default scenario returns terminal content once a tool result exists."""
    client = _client_for("default")
    request_body = {
        **_TOOL_CHAT_REQUEST,
        "messages": [
            *_CHAT_REQUEST["messages"],  # type: ignore[misc]
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "fixture-call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "fixture-call-1", "content": "{}"},
        ],
    }
    response = client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"]


def test_malformed_response_scenario_returns_invalid_json() -> None:
    """The malformed-response scenario returns a body that fails to parse as JSON."""
    client = _client_for("malformed-response")
    response = client.post("/v1/chat/completions", json=_CHAT_REQUEST)
    assert response.status_code == 200
    with pytest.raises(ValueError):
        response.json()


def test_error_500_scenario_returns_server_error() -> None:
    """The error-500 scenario returns an HTTP 500 with a fixture error body."""
    client = _client_for("error-500")
    response = client.post("/v1/chat/completions", json=_CHAT_REQUEST)
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "fixture_error"


def test_duplicate_tool_call_scenario_repeats_call_id() -> None:
    """The duplicate-tool-call scenario returns two tool calls sharing one id."""
    client = _client_for("duplicate-tool-call")
    response = client.post("/v1/chat/completions", json=_TOOL_CHAT_REQUEST)
    tool_calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] == tool_calls[1]["id"]


def test_timeout_scenario_delays_the_response() -> None:
    """The timeout scenario delays its response by the configured bound."""
    client = _client_for("timeout", timeout_delay_seconds=0.05)
    start = time.monotonic()
    response = client.post("/v1/chat/completions", json=_CHAT_REQUEST)
    elapsed = time.monotonic() - start
    assert response.status_code == 200
    assert elapsed >= 0.05


def test_all_documented_scenarios_are_covered_by_this_module() -> None:
    """Every scenario the fixture supports has a matching test above."""
    covered = {
        "default",
        "malformed-response",
        "timeout",
        "duplicate-tool-call",
        "error-500",
    }
    assert set(SCENARIOS) == covered


_MODEL_ROOT = Path(__file__).resolve().parents[1]


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
    raise AssertionError(f"model fixture at {url} did not become ready")


@pytest.mark.skipif(not _docker_available(), reason="docker is not available")
def test_model_fixture_image_builds_and_serves_readyz() -> None:
    """The model fixture image builds, starts, and reports readiness."""
    tag = f"model-fixture:test-{os.getpid()}"
    container_name = f"model-fixture-test-{os.getpid()}"
    host_port = _free_port()
    try:
        subprocess.run(["docker", "build", "--tag", tag, "."], cwd=_MODEL_ROOT, check=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--read-only",
                "--tmpfs",
                "/tmp/model-fixture:rw,noexec,nosuid,size=16m",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "-p",
                f"{host_port}:8081",
                tag,
            ],
            check=True,
        )
        _wait_for_ready(f"http://127.0.0.1:{host_port}/readyz")
    finally:
        subprocess.run(["docker", "rm", "--force", container_name], check=False)
        subprocess.run(["docker", "image", "rm", "--force", tag], check=False)
