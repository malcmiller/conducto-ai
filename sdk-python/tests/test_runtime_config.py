"""Focused contracts for immutable runtime configuration."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from conducto.core.model_config import ModelReference as ConfigModelReference
from conducto.core.runtime import (
    GenerationOptions,
    ModelConfiguration,
    ModelProvider,
    ModelReference,
    ProviderResult,
    RunConfig,
    RuntimeConfig,
    Usage,
    complete_with_retries,
)


def test_runtime_config_re_exports_preserve_identity_and_normalization() -> None:
    assert ModelReference is ConfigModelReference
    assert RuntimeConfig(" default ").default_model == ModelReference("default")


def test_runtime_retains_historical_provider_re_exports() -> None:
    assert all(
        symbol is not None
        for symbol in (
            GenerationOptions,
            ModelConfiguration,
            ModelProvider,
            ProviderResult,
            Usage,
            complete_with_retries,
        )
    )


def test_run_config_recursively_freezes_safe_metadata() -> None:
    config = RunConfig(metadata={"labels": ["one"], "nested": {"enabled": True}})

    assert config.metadata["labels"] == ("one",)
    assert config.metadata["nested"]["enabled"] is True
    mutable_metadata = cast(dict[str, object], config.metadata)
    with pytest.raises(TypeError):
        mutable_metadata["changed"] = True
    with pytest.raises(FrozenInstanceError):
        object.__setattr__(config, "timeout", 1)
    with pytest.raises(ValueError, match="Sensitive values"):
        RunConfig(metadata={"nested": {"token": "secret"}})
