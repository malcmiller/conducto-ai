"""Unit tests for the example container-host deployment applications."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from conducto.container import ContainerConfig
from conducto.container_app import load_container_applications_manifest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_ROOT = _ROOT / "examples" / "container_host_application"
_EXAMPLE_SRC = _EXAMPLE_ROOT / "src"
if str(_EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_SRC))

agent_a = importlib.import_module("container_host_application.agent_a")
agent_b = importlib.import_module("container_host_application.agent_b")
orchestrator = importlib.import_module("container_host_application.orchestrator")


def test_example_manifest_declares_reference_and_role_applications() -> None:
    """The example manifest declares every role hosted by the shared image."""
    manifest = load_container_applications_manifest(_EXAMPLE_ROOT / "conducto-applications.json")

    assert manifest.manifest_version == "1"
    assert manifest.applications == {
        "reference": "conducto.container:build_reference_app",
        "orchestrator": "container_host_application.orchestrator:build_app",
        "agent-a": "container_host_application.agent_a:build_app",
        "agent-b": "container_host_application.agent_b:build_app",
    }


def test_example_factories_build_deterministic_asgi_apps() -> None:
    """Each example factory builds an offline deterministic ASGI application."""
    orchestrator_config = ContainerConfig(agent_id="orchestrator-id", agent_version="1.0.0")
    agent_a_config = ContainerConfig(agent_id="agent-a-id", agent_version="1.0.1")
    agent_b_config = ContainerConfig(agent_id="agent-b-id", agent_version="1.0.2")

    orchestrator_app = orchestrator.build_app(orchestrator_config)
    agent_a_app = agent_a.build_app(agent_a_config)
    agent_b_app = agent_b.build_app(agent_b_config)

    assert callable(orchestrator_app)
    assert callable(agent_a_app)
    assert callable(agent_b_app)
    assert orchestrator_app.endpoint_url == "http://127.0.0.1:8000/a2a"
    assert agent_a_app.endpoint_url == "http://127.0.0.1:8000/a2a"
    assert agent_b_app.endpoint_url == "http://127.0.0.1:8000/a2a"

    orchestrator_agent = orchestrator.OrchestratorAgent(orchestrator_config)
    agent_a_agent = agent_a.AgentA(agent_a_config)
    agent_b_agent = agent_b.AgentB(agent_b_config)

    assert orchestrator_agent.coordinate("flow") == {
        "agent_id": "orchestrator-id",
        "agent_version": "1.0.0",
        "capability": "coordinate",
        "role": "orchestrator",
        "value": "flow",
    }
    assert agent_a_agent.handle("task-a") == {
        "agent_id": "agent-a-id",
        "agent_version": "1.0.1",
        "capability": "handle",
        "role": "agent-a",
        "value": "task-a",
    }
    assert agent_b_agent.respond("task-b") == {
        "agent_id": "agent-b-id",
        "agent_version": "1.0.2",
        "capability": "respond",
        "role": "agent-b",
        "value": "task-b",
    }
