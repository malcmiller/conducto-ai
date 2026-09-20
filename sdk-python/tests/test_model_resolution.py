"""Focused contracts for model precedence and policy evaluation."""

import pytest

from conducto import FakeModel, ModelConfiguration, ProviderCapabilities
from conducto.core.model_config import (
    AgentModelConfig,
    ModelReference,
    ModelRequirement,
    ModelResolutionSource,
    RunConfig,
    RuntimeConfig,
)
from conducto.core.model_resolution import ModelResolver
from conducto.core.provider_registry import ProviderRegistry
from conducto.core.runtime_errors import (
    IncompatibleProviderCapabilitiesError,
    ModelOverrideDeniedError,
)


def _resolver(*references: str) -> ModelResolver:
    registry = ProviderRegistry()
    for reference in references:
        registry.register(
            reference,
            FakeModel({}),
            ModelConfiguration(provider=f"provider-{reference}", model=reference),
        )
    return ModelResolver(registry)


def test_model_resolver_preserves_precedence_and_policy_facts() -> None:
    resolver = _resolver("runtime", "agent", "run", "call")
    observed: list[tuple[ModelReference, ModelResolutionSource]] = []
    binding = resolver.resolve_binding(
        agent_id="Agent",
        agent_config=AgentModelConfig(
            default_model=ModelReference("agent"),
            requirement=ModelRequirement.REQUIRED,
        ),
        run_config=RunConfig(model="run", caller="caller"),
        runtime_config=RuntimeConfig("runtime"),
        policy=lambda context: observed.append((context.model_reference, context.source)) is None,
        call_override="call",
    )

    assert binding is not None
    assert binding.model.reference == ModelReference("call")
    assert observed == [(ModelReference("call"), ModelResolutionSource.CALL_OVERRIDE)]


def test_model_resolver_denies_policy_and_incompatible_capabilities() -> None:
    registry = ProviderRegistry()
    provider = FakeModel({})
    provider.capabilities = ProviderCapabilities()  # type: ignore[misc]
    registry.register(
        "model",
        provider,
        ModelConfiguration(provider="fake", model="model"),
    )
    resolver = ModelResolver(registry)
    with pytest.raises(ModelOverrideDeniedError):
        resolver.resolve_binding(
            agent_id="Agent",
            agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
            run_config=RunConfig(model="model"),
            runtime_config=RuntimeConfig(),
            policy=lambda _context: False,
        )
    with pytest.raises(IncompatibleProviderCapabilitiesError):
        resolver.resolve_binding(
            agent_id="Agent",
            agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
            run_config=RunConfig(model="model"),
            runtime_config=RuntimeConfig(),
            policy=None,
            required_capabilities=frozenset({"structured_output"}),
        )
