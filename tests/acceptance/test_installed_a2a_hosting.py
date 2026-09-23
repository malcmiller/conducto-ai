"""Installed-wheel acceptance coverage for independent local A2A hosts."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from a2a.types.a2a_pb2 import Message, Part, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict

from conducto.transport import DiscoveryError, DiscoveryPolicy, discover_agent

pytestmark = pytest.mark.acceptance

_ROOT = Path(__file__).parents[2]
_CARD_PATH = "/.well-known/agent-card.json"


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one setup command and include captured output in failures."""
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise AssertionError(
            "Command failed during installed-wheel acceptance setup:\n"
            f"command: {error.cmd}\n"
            f"exit code: {error.returncode}\n"
            f"stdout:\n{error.stdout}\n"
            f"stderr:\n{error.stderr}"
        ) from error


def _venv_python(environment: Path) -> Path:
    """Return the platform-specific interpreter path in a virtual environment."""
    return (
        environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else environment / "bin" / "python"
    )


def _free_port() -> int:
    """Reserve an ephemeral loopback port until the child process binds it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wheel_environment(tmp_path: Path) -> Path:
    """Build and install the wheel plus its server extra in an isolated venv."""
    dist = tmp_path / "dist"
    constraints = tmp_path / "constraints.txt"
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=_ROOT,
    )
    wheel = next(dist.glob("*.whl"))
    _run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--extra",
            "a2a-server",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements.txt",
            "--output-file",
            str(constraints),
        ],
        cwd=_ROOT,
    )
    cache_prime_environment = tmp_path / "cache-prime"
    _run(
        ["uv", "venv", "--python", sys.executable, str(cache_prime_environment)],
        cwd=tmp_path,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(_venv_python(cache_prime_environment)),
            "--constraint",
            str(constraints),
            f"{wheel}[a2a-server]",
        ],
        cwd=tmp_path,
    )
    environment = tmp_path / "installed"
    _run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
    )
    python = _venv_python(environment)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--offline",
            "--constraint",
            str(constraints),
            f"{wheel}[a2a-server]",
        ],
        cwd=tmp_path,
    )
    return python


def _observability_path(temporary_directory: Path, agent: str, port: int) -> Path:
    """Return the deterministic per-process observability file path."""
    return temporary_directory / f"{agent}-{port}-observability.jsonl"


def _observations(temporary_directory: Path, agent: str, port: int) -> list[dict[str, Any]]:
    """Read deterministic observability records emitted by one hosted process."""
    path = _observability_path(temporary_directory, agent, port)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _installed_environment_assertions(python: Path) -> None:
    """Assert server-extra metadata and imports belong to the installed wheel."""
    script = """
import importlib.metadata as metadata
import json
from pathlib import Path
import conducto
import fastapi
import starlette
import uvicorn

extras = metadata.metadata("conducto-ai").get_all("Provides-Extra") or []
print(json.dumps({
    "extras": extras,
    "conducto": str(Path(conducto.__file__).resolve()),
    "fastapi": fastapi.__version__,
    "starlette": starlette.__version__,
    "uvicorn": uvicorn.__version__,
}))
"""
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=python.parent.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert "a2a-server" in payload["extras"]
    assert "site-packages" in payload["conducto"].lower()
    assert payload["fastapi"] and payload["starlette"] and payload["uvicorn"]


def _start_agent(
    python: Path,
    *,
    port: int,
    agent: str,
    temporary_directory: Path,
    downstream_card_url: str | None = None,
    composition: str = "direct",
) -> subprocess.Popen[str]:
    """Start one isolated installed-wheel Uvicorn process on loopback only."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CONDUCTO_DEMO_PORT": str(port),
            "CONDUCTO_DEMO_AGENT": agent,
            "CONDUCTO_DEMO_COMPOSITION": composition,
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment["CONDUCTO_DEMO_OBSERVABILITY_PATH"] = str(
        _observability_path(temporary_directory, agent, port)
    )
    if downstream_card_url is not None:
        environment["CONDUCTO_DEMO_DOWNSTREAM_CARD_URL"] = downstream_card_url
    log = (temporary_directory / f"{agent}-{port}.log").open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "conducto.examples.a2a_hosting:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=temporary_directory,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def _wait_ready(port: int, process: subprocess.Popen[str]) -> None:
    """Poll the bounded readiness endpoint instead of assuming a startup delay."""
    url = f"http://127.0.0.1:{port}/readyz"
    async with httpx.AsyncClient(timeout=0.25) as client:
        for _ in range(80):
            if process.poll() is not None:
                raise AssertionError(f"agent process exited with {process.returncode}")
            try:
                response = await client.get(url)
                if response.status_code == 200 and response.json() == {"status": "ready"}:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.025)
    raise AssertionError(f"agent process at {port} did not become ready")


def _policy(port: int) -> DiscoveryPolicy:
    """Return the explicit loopback-only policy for one test process."""
    return DiscoveryPolicy(
        allowed_schemes=frozenset({"http"}),
        allowed_ports=frozenset({port}),
        allow_loopback=True,
        allow_private_networks=True,
        request_timeout=1.0,
    )


async def _send(
    port: int,
    skill_id: str,
    arguments: dict[str, Any],
    *,
    correlation_id: str,
    scopes: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Send one standards-based JSON-RPC message to a hosted process."""
    message = Message(
        message_id=f"{correlation_id}-message",
        role=Role.ROLE_USER,
        parts=[
            Part(
                text=json.dumps(
                    {"skillId": skill_id, "arguments": arguments},
                    separators=(",", ":"),
                )
            )
        ],
    )
    metadata: dict[str, Any] = {"correlationId": correlation_id}
    if timeout is not None:
        metadata["timeoutSeconds"] = timeout
    message.metadata.update({"x-conducto": metadata})
    request = SendMessageRequest(message=message)
    payload = {
        "jsonrpc": "2.0",
        "id": correlation_id,
        "method": "SendMessage",
        "params": MessageToDict(request),
    }
    headers = {"A2A-Version": "1.0"}
    if scopes is not None:
        headers["x-demo-scopes"] = scopes
    async with httpx.AsyncClient(timeout=4.0) as client:
        response = await client.post(f"http://127.0.0.1:{port}/a2a", json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise AssertionError("A2A response must be an object")
    result = body.get("result")
    if not isinstance(result, dict):
        raise AssertionError("A2A response must contain a result")
    task = result.get("task")
    if not isinstance(task, dict):
        raise AssertionError("A2A result must contain a task")
    return task


def _skill(card: dict[str, Any], name: str) -> str:
    """Find a published skill identifier by its public capability name."""
    skills = card.get("skills")
    if not isinstance(skills, list):
        raise AssertionError("Agent Card must contain skills")
    for skill in skills:
        if isinstance(skill, dict) and skill.get("name") == name:
            skill_id = skill.get("id")
            if isinstance(skill_id, str):
                return skill_id
    raise AssertionError(f"Agent Card did not advertise {name!r}")


def _stop(process: subprocess.Popen[str]) -> None:
    """Request bounded Uvicorn shutdown and fail loudly if it leaves a child alive."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=5)
            raise AssertionError("agent process did not stop within the shutdown bound") from error


def test_installed_wheel_hosts_independent_agents_and_delegates(
    tmp_path: Path,
) -> None:
    """Prove direct and FastAPI-composed hosts across installed-wheel processes."""
    python = _wheel_environment(tmp_path)
    _installed_environment_assertions(python)
    agent_a_port, agent_b_port = _free_port(), _free_port()
    while agent_b_port == agent_a_port:
        agent_b_port = _free_port()
    agent_b = _start_agent(
        python,
        port=agent_b_port,
        agent="agent-b",
        temporary_directory=tmp_path,
        composition="fastapi",
    )
    agent_a: subprocess.Popen[str] | None = None
    try:
        asyncio.run(_wait_ready(agent_b_port, agent_b))
        agent_a = _start_agent(
            python,
            port=agent_a_port,
            agent="agent-a",
            temporary_directory=tmp_path,
            downstream_card_url=f"http://127.0.0.1:{agent_b_port}{_CARD_PATH}",
        )
        asyncio.run(_wait_ready(agent_a_port, agent_a))

        orchestrated = subprocess.run(
            [
                str(python),
                "-m",
                "conducto.examples.a2a_orchestrator",
                "--agent-card-url",
                f"http://127.0.0.1:{agent_a_port}{_CARD_PATH}",
                "--value",
                "receipt-7",
                "--correlation-id",
                "installed-chain-1",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = json.loads(orchestrated.stdout)
        assert value["value"] == "receipt-7"
        assert value["correlation_id"] == "installed-chain-1"
        assert value["scopes"] == ["demo:downstream"]

        async def exercise() -> None:
            card_a = await discover_agent(
                f"http://127.0.0.1:{agent_a_port}{_CARD_PATH}",
                policy=_policy(agent_a_port),
                correlation_id="installed-card-a",
            )
            card_b = await discover_agent(
                f"http://127.0.0.1:{agent_b_port}{_CARD_PATH}",
                policy=_policy(agent_b_port),
                correlation_id="installed-card-b",
            )
            assert card_a.name == "InstalledDemoAgentA"
            assert card_b.name == "InstalledDemoAgentB"
            card_a_payload = dict(card_a.card)
            card = dict(card_b.card)
            denied_delegate = await _send(
                agent_a_port,
                _skill(card_a_payload, "delegate"),
                {"value": "denied"},
                correlation_id="installed-delegate-denied",
                scopes="",
            )
            assert denied_delegate["status"]["state"] == "TASK_STATE_REJECTED"
            assert denied_delegate["metadata"]["reason"] == "authorization_denied"
            denied = await _send(
                agent_b_port,
                _skill(card, "restricted"),
                {},
                correlation_id="installed-denied",
                scopes="demo:invoke",
            )
            assert denied["status"]["state"] == "TASK_STATE_REJECTED"
            assert denied["metadata"]["reason"] == "authorization_denied"
            approval = await _send(
                agent_b_port,
                _skill(card, "approval"),
                {"value": "hold"},
                correlation_id="installed-approval",
            )
            assert approval["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
            timed_out = await _send(
                agent_b_port,
                _skill(card, "wait"),
                {},
                correlation_id="installed-timeout",
                timeout=0.01,
            )
            assert timed_out["status"]["state"] == "TASK_STATE_FAILED"
            assert timed_out["metadata"]["reason"] == "timeout"

        asyncio.run(exercise())
        agent_a_observations = _observations(tmp_path, "agent-a", agent_a_port)
        agent_b_observations = _observations(tmp_path, "agent-b", agent_b_port)
        assert len(agent_a_observations) == 1
        assert len(agent_b_observations) == 1
        assert agent_a_observations[0]["correlation_id"] == "installed-chain-1"
        assert agent_b_observations[0]["correlation_id"] == "installed-chain-1"
        assert agent_a_observations[0]["delegated_scopes"] == ["demo:downstream"]
        assert agent_b_observations[0]["scopes"] == ["demo:downstream"]
        assert agent_a_observations[0]["downstream_task_id"] == agent_b_observations[0]["task_id"]
        assert agent_b_observations[0]["lineage"] == [agent_a_observations[0]["task_id"]]
        assert value["task_id"] == agent_b_observations[0]["task_id"]
        assert value["lineage"] == [agent_a_observations[0]["task_id"]]
        _stop(agent_b)
        with pytest.raises(DiscoveryError):
            asyncio.run(
                discover_agent(
                    f"http://127.0.0.1:{agent_b_port}{_CARD_PATH}",
                    policy=_policy(agent_b_port),
                )
            )
        failed_delegation = asyncio.run(
            _send(
                agent_a_port,
                _skill(
                    dict(
                        asyncio.run(
                            discover_agent(
                                f"http://127.0.0.1:{agent_a_port}{_CARD_PATH}",
                                policy=_policy(agent_a_port),
                            )
                        ).card
                    ),
                    "delegate",
                ),
                {"value": "unavailable"},
                correlation_id="installed-unavailable",
            )
        )
        assert failed_delegation["status"]["state"] == "TASK_STATE_FAILED"
        agent_b = _start_agent(
            python,
            port=agent_b_port,
            agent="agent-b",
            temporary_directory=tmp_path,
            composition="fastapi",
        )
        asyncio.run(_wait_ready(agent_b_port, agent_b))
        restarted = asyncio.run(
            _send(
                agent_a_port,
                _skill(
                    dict(
                        asyncio.run(
                            discover_agent(
                                f"http://127.0.0.1:{agent_a_port}{_CARD_PATH}",
                                policy=_policy(agent_a_port),
                            )
                        ).card
                    ),
                    "delegate",
                ),
                {"value": "after-restart"},
                correlation_id="installed-restart",
            )
        )
        assert restarted["status"]["state"] == "TASK_STATE_COMPLETED"
    finally:
        if agent_a is not None:
            _stop(agent_a)
        _stop(agent_b)
