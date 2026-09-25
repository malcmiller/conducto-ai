"""Deterministic model doubles and provider conformance helpers."""

from .a2a_transport import InMemoryA2ATransport
from .fake_model import FakeModel, FakeModelRequest
from .identity import AllowAllIdentityResolver
from .provider_conformance import (
    MANDATORY_FIXTURES,
    PROVIDER_FIXTURE_VERSION,
    ConformanceFixture,
    ScriptedProvider,
    ScriptedProviderCall,
    assert_provider_conformance,
    assert_provider_tool_call_conformance,
    run_provider_conformance,
)
from .resources import (
    LIFECYCLE_FIXTURE_VERSION,
    ManualClock,
    assert_data_source_lifecycle_conformance,
    conformance_content_batch,
    conformance_provisioning_config,
    run_data_source_lifecycle_conformance,
)

__all__ = [
    "PROVIDER_FIXTURE_VERSION",
    "LIFECYCLE_FIXTURE_VERSION",
    "MANDATORY_FIXTURES",
    "ConformanceFixture",
    "FakeModel",
    "FakeModelRequest",
    "AllowAllIdentityResolver",
    "InMemoryA2ATransport",
    "ManualClock",
    "ScriptedProvider",
    "ScriptedProviderCall",
    "assert_data_source_lifecycle_conformance",
    "assert_provider_conformance",
    "assert_provider_tool_call_conformance",
    "conformance_content_batch",
    "conformance_provisioning_config",
    "run_data_source_lifecycle_conformance",
    "run_provider_conformance",
]
