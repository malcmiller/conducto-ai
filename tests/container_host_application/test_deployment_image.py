"""Optional Docker-based checks for the example deployment image."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    """Return whether Docker is installed, reachable, and runs Linux containers.

    The example deployment image is Linux-only. Windows CI runners typically
    ship Docker configured for Windows containers, which cannot pull the
    ``python:3.12-slim-bookworm`` base image, so this check also confirms the
    daemon's container OS before allowing the build-based test to run.
    """
    binary = shutil.which("docker")
    if binary is None:
        return False
    info_result = subprocess.run(
        [binary, "info"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if info_result.returncode != 0:
        return False
    os_result = subprocess.run(
        [binary, "version", "--format", "{{.Server.Os}}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return os_result.returncode == 0 and os_result.stdout.strip() == "linux"


@pytest.mark.skipif(not _docker_available(), reason="docker is not available")
def test_example_deployment_image_builds_and_selects_agent_a() -> None:
    """The example deployment image can run a manifest-selected application."""
    base_tag = f"conducto-agent:test-{os.getpid()}"
    example_tag = f"conducto-host-application:test-{os.getpid()}"
    try:
        subprocess.run(
            ["docker", "build", "--tag", base_tag, "--file", "Dockerfile", "."],
            cwd=_ROOT,
            check=True,
        )
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                f"BASE_IMAGE={base_tag}",
                "--tag",
                example_tag,
                "--file",
                "Dockerfile",
                ".",
            ],
            cwd=_ROOT / "examples" / "container_host_application",
            check=True,
        )
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--tmpfs",
                "/tmp/conducto:rw,noexec,nosuid,size=64m",
                "--mount",
                "type=tmpfs,destination=/var/lib/conducto",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "-e",
                "CONDUCTO_APPLICATION=agent-a",
                "-e",
                "CONDUCTO_AGENT_ID=deployment-agent-a",
                "-e",
                "CONDUCTO_AGENT_VERSION=1.2.3",
                example_tag,
                "--check-config",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        assert payload["application_key"] == "agent-a"
        assert payload["safe_metadata"]["application"]["agent_id"] == "deployment-agent-a"
        assert payload["safe_metadata"]["application"]["agent_version"] == "1.2.3"
    finally:
        subprocess.run(["docker", "image", "rm", "--force", example_tag], cwd=_ROOT, check=False)
        subprocess.run(["docker", "image", "rm", "--force", base_tag], cwd=_ROOT, check=False)
