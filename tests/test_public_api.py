"""Public Python package ownership is independent of the versioned wire format."""

import importlib
import subprocess
import sys

import conducto
import conducto.core


def test_application_facade_has_an_explicit_small_surface() -> None:
    assert set(conducto.__all__) == {
        "AgentModelConfig",
        "AgentRegistry",
        "BaseAgent",
        "DataSourceMetadata",
        "DataSourceRegistrationError",
        "DataSourceRegistry",
        "DataSourceSnapshot",
        "ModelReference",
        "ModelRequirement",
        "MissingDataSourceError",
        "OrchestratorAgent",
        "RunConfig",
        "RunContext",
        "Runtime",
        "RuntimeConfig",
        "a2a_agent",
        "a2a_capability",
        "budget",
        "classification",
        "data_source",
        "get_run_context",
        "require_run_context",
        "requires_scope",
        "side_effect",
        "timeout",
        "tool",
        "uses_data_source",
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
        "conducto.registration": ("RegistrationService", "RegisterRequest", "RegistrationResult"),
        "conducto.core.delegation": ("DelegationConfig", "run_delegation"),
        "conducto.core.invocation_results": ("InvocationSuccess", "RoutingFailure"),
        "conducto.a2a": (
            "A2AAuthenticatedIdentity",
            "A2ARequestContext",
            "A2ARuntimeHandler",
            "create_a2a_app",
        ),
        "conducto.testing": ("FakeModel", "FakeModelRequest"),
    }
    for module_name, symbols in owners.items():
        module = importlib.import_module(module_name)
        assert all(getattr(module, symbol) is not None for symbol in symbols)
    assert not hasattr(importlib.import_module("conducto.core.provider"), "FakeModel")
    registry = importlib.import_module("conducto.core.provider_registry").ProviderRegistry()
    assert not hasattr(registry, "register")


def test_registration_adapters_do_not_make_core_depend_on_http_frameworks() -> None:
    """Core and registration contracts import with HTTP adapters unavailable."""
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'httpx', 'starlette', 'fastapi', 'uvicorn'}:
        raise ImportError('optional HTTP package unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import conducto
import conducto.registration
import conducto.registration.asgi
assert conducto.registration.RegisterRequest
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
