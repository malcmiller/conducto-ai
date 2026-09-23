"""Tests for the installed-image reference process contract."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from conducto.container import ContainerConfig, build_app, main
from conducto.container_app import (
    APPLICATION_ENVIRONMENT_VARIABLE,
    APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE,
    load_container_applications_manifest,
)

_FIXTURES = Path(__file__).parent / "container_host_application" / "fixtures"


class _CallableApp:
    """Minimal callable test double used for manifest-selection tests."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Reject accidental invocation during a manifest-selection unit test."""
        raise AssertionError("manifest-selection test app should not receive traffic")


def _register_factory_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    factory: Any,
) -> None:
    """Register one synthetic importable module for manifest-resolution tests."""
    module = types.ModuleType(module_name)
    module.__dict__["build_app"] = factory
    monkeypatch.setitem(sys.modules, module_name, module)


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


def test_container_default_application_remains_reference() -> None:
    """The default selected application preserves the reference host behavior."""
    app = build_app(ContainerConfig())

    assert app.endpoint_url == "http://127.0.0.1:8000/a2a"
    assert app.agent_card.name == "conducto-reference-agent"
    assert app.conducto_container_application == "reference"
    assert app.conducto_container_metadata["application"]["manifest_version"] == "1"


def test_container_builds_manifest_selected_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The selected manifest factory receives the immutable container config."""
    captured: dict[str, Any] = {}

    def factory(config: ContainerConfig) -> _CallableApp:
        captured["config"] = config
        return _CallableApp()

    _register_factory_module(
        monkeypatch,
        module_name="tests.container_selected_factory",
        factory=factory,
    )
    manifest_path = tmp_path / "applications.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifestVersion": "7",
                "applications": {
                    "custom": "tests.container_selected_factory:build_app",
                    "reference": "conducto.container:build_reference_app",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE, str(manifest_path))
    monkeypatch.setenv(APPLICATION_ENVIRONMENT_VARIABLE, "custom")
    monkeypatch.setenv("CONDUCTO_IMAGE_VERSION", "9.9.9")
    monkeypatch.setenv("CONDUCTO_SOURCE_REVISION", "abcdef0")

    config = ContainerConfig(agent_id="custom-agent", agent_version="2.4.6")
    app = build_app(config)

    assert captured["config"] is config
    assert app.conducto_container_application == "custom"
    assert app.conducto_container_metadata == {
        "application": {
            "agent_id": "custom-agent",
            "agent_version": "2.4.6",
            "key": "custom",
            "manifest_path": str(manifest_path),
            "manifest_source": "file",
            "manifest_version": "7",
            "target": "tests.container_selected_factory:build_app",
        },
        "image": {
            "source_revision": "abcdef0",
            "version": "9.9.9",
        },
    }


def test_container_manifest_rejects_unknown_and_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown or empty selected application keys fail before readiness."""
    monkeypatch.setenv(
        APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE,
        str(_FIXTURES / "valid-applications.json"),
    )
    monkeypatch.setenv(APPLICATION_ENVIRONMENT_VARIABLE, "missing")
    with pytest.raises(ValueError, match="unknown application"):
        build_app(ContainerConfig())

    monkeypatch.setenv(APPLICATION_ENVIRONMENT_VARIABLE, " ")
    with pytest.raises(ValueError, match="non-empty application key"):
        build_app(ContainerConfig())


def test_container_manifest_rejects_malformed_and_duplicate_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed manifests are rejected before the host becomes ready."""
    malformed_path = _FIXTURES / "malformed-applications.json"
    with pytest.raises(ValueError, match="lowercase letters, digits, and hyphens"):
        load_container_applications_manifest(malformed_path)

    duplicate_path = _FIXTURES / "duplicate-applications.json"
    with pytest.raises(ValueError, match="duplicate key"):
        load_container_applications_manifest(duplicate_path)

    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{", encoding="utf-8")
    monkeypatch.setenv(APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE, str(invalid_json_path))
    with pytest.raises(ValueError, match="not valid JSON"):
        build_app(ContainerConfig())


@pytest.mark.parametrize(
    ("target", "pattern"),
    [
        ("missing.module:build_app", "module could not be imported"),
        ("conducto.container:missing_attribute", "attribute could not be resolved"),
    ],
)
def test_container_manifest_rejects_invalid_factory_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    pattern: str,
) -> None:
    """Invalid manifest factory references fail before readiness."""
    manifest_path = tmp_path / "applications.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifestVersion": "1",
                "applications": {
                    "broken": target,
                    "reference": "conducto.container:build_reference_app",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE, str(manifest_path))
    monkeypatch.setenv(APPLICATION_ENVIRONMENT_VARIABLE, "broken")

    with pytest.raises(ValueError, match=pattern):
        build_app(ContainerConfig())


def test_container_check_config_redacts_factory_build_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Factory build failures become redacted configuration diagnostics."""

    def factory(_config: ContainerConfig) -> _CallableApp:
        raise RuntimeError("api_token should never leak")

    _register_factory_module(
        monkeypatch,
        module_name="tests.container_failing_factory",
        factory=factory,
    )
    manifest_path = tmp_path / "applications.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifestVersion": "1",
                "applications": {
                    "broken": "tests.container_failing_factory:build_app",
                    "reference": "conducto.container:build_reference_app",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE, str(manifest_path))
    monkeypatch.setenv(APPLICATION_ENVIRONMENT_VARIABLE, "broken")

    assert main(["--check-config"]) == 78
    diagnostic = capsys.readouterr().err
    assert "<redacted>" in diagnostic
    assert "api_token" not in diagnostic


def test_container_check_config_reports_safe_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configuration-only startup publishes safe application diagnostics."""
    monkeypatch.setenv("CONDUCTO_AGENT_ID", "safe-agent")
    monkeypatch.setenv("CONDUCTO_AGENT_VERSION", "3.1.4")
    monkeypatch.setenv("CONDUCTO_IMAGE_VERSION", "1.0.0")
    monkeypatch.setenv("CONDUCTO_SOURCE_REVISION", "deadbeef")
    monkeypatch.setenv("CONDUCTO_PROVIDER_ENDPOINT", "https://secret.example.test/token")

    assert main(["--check-config"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "valid"
    assert payload["application_key"] == "reference"
    assert payload["safe_metadata"]["application"]["agent_id"] == "safe-agent"
    assert payload["safe_metadata"]["application"]["agent_version"] == "3.1.4"
    assert payload["safe_metadata"]["image"]["version"] == "1.0.0"
    assert payload["safe_metadata"]["image"]["source_revision"] == "deadbeef"
    assert "provider_endpoint" not in json.dumps(payload, sort_keys=True)
