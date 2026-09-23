"""Tests for the installed-image reference process contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conducto.container import ContainerConfig, main


def test_container_configuration_uses_file_then_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment values override JSON while preserving typed fields."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"agent_id": "from-file", "bind_port": 8111, "scopes": ["invoke"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONDUCTO_CONFIG_FILE", str(path))
    monkeypatch.setenv("CONDUCTO_BIND_PORT", "8222")

    config = ContainerConfig.load()

    assert config.agent_id == "from-file"
    assert config.bind_port == 8222
    assert config.scopes == ("invoke",)


def test_container_configuration_rejects_secret_like_model_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model references remain opaque identifiers and cannot carry credentials."""
    monkeypatch.setenv("CONDUCTO_MODEL_REFERENCE", "api_token=secret")

    with pytest.raises(ValueError, match="secret-like"):
        ContainerConfig.load()


def test_container_check_config_returns_actionable_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Configuration-only startup is deterministic and does not start a server."""
    monkeypatch.setenv("CONDUCTO_BIND_PORT", "not-a-port")

    assert main(["--check-config"]) == 78
    assert "bind_port must be an integer" in capsys.readouterr().err
