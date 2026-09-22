"""Public Python package ownership is independent of the versioned wire format."""

import importlib

import conducto
import conducto.core


def test_application_facade_has_an_explicit_small_surface() -> None:
    assert set(conducto.__all__) == {
        "AgentModelConfig",
        "AgentRegistry",
        "BaseAgent",
        "ModelReference",
        "ModelRequirement",
        "OrchestratorAgent",
        "RunConfig",
        "RunContext",
        "Runtime",
        "RuntimeConfig",
        "a2a_agent",
        "a2a_capability",
        "get_run_context",
        "require_run_context",
        "tool",
    }
    assert all(getattr(conducto, name) is not None for name in conducto.__all__)
    assert not hasattr(conducto, "FakeModel")
    assert not hasattr(conducto, "ProviderRegistry")
    assert not hasattr(conducto.core, "Runtime")


def test_domain_packages_publish_contracts_without_umbrella_aliases() -> None:
    owners = {
        "conducto.core.provider": ("ModelProvider", "ProviderResult", "ProviderToolDefinition"),
        "conducto.core.provider_registry": ("ProviderRegistry", "ProviderClientConfig"),
        "conducto.core.gateway": ("AgentGateway", "LocalAgentGateway"),
        "conducto.core.catalog": ("AgentCatalog", "CatalogEntry"),
        "conducto.core.delegation": ("DelegationConfig", "run_delegation"),
        "conducto.core.invocation_results": ("InvocationSuccess", "RoutingFailure"),
        "conducto.testing": ("FakeModel", "FakeModelRequest"),
    }
    for module_name, symbols in owners.items():
        module = importlib.import_module(module_name)
        assert all(getattr(module, symbol) is not None for symbol in symbols)
    assert not hasattr(importlib.import_module("conducto.core.provider"), "FakeModel")
    registry = importlib.import_module("conducto.core.provider_registry").ProviderRegistry()
    assert not hasattr(registry, "register")
