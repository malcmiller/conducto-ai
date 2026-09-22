"""Focused contracts for immutable runtime configuration."""

from dataclasses import FrozenInstanceError
from inspect import signature
from typing import Any, cast

import pytest

from conducto.core import runtime
from conducto.core.model_config import ModelReference, RunConfig, RuntimeConfig


def test_runtime_config_normalizes_model_references() -> None:
    assert RuntimeConfig(" default ").default_model == ModelReference("default")


def test_runtime_exports_only_its_facade() -> None:
    assert runtime.__all__ == ["Runtime"]
    for name in (
        "GenerationOptions",
        "ModelConfiguration",
        "ModelProvider",
        "ProviderResult",
        "Usage",
        "complete_with_retries",
        "ProviderClientConfig",
        "ProviderRegistration",
        "ProviderFactory",
        "get_run_context",
        "use_run_context",
    ):
        assert not hasattr(runtime, name)


def test_runtime_invoke_accepts_only_canonical_authorization_keyword() -> None:
    parameters = signature(runtime.Runtime.invoke).parameters

    assert "authorization" in parameters
    assert "authorization_context" not in parameters


def test_run_config_recursively_freezes_safe_metadata() -> None:
    config = RunConfig(metadata={"labels": ["one"], "nested": {"enabled": True}})

    assert config.metadata["labels"] == ("one",)
    assert config.metadata["nested"]["enabled"] is True
    mutable_metadata = cast(dict[str, object], config.metadata)
    with pytest.raises(TypeError):
        mutable_metadata["changed"] = True
    mutable_config = cast(Any, config)
    with pytest.raises(FrozenInstanceError):
        mutable_config.timeout = 1
    with pytest.raises(ValueError, match="Sensitive values"):
        RunConfig(metadata={"nested": {"token": "secret"}})
