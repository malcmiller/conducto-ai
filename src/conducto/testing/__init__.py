"""Deterministic model doubles and provider conformance helpers."""

from .a2a_transport import InMemoryA2ATransport
from .fake_model import FakeModel, FakeModelRequest
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

__all__ = [
    "PROVIDER_FIXTURE_VERSION",
    "MANDATORY_FIXTURES",
    "ConformanceFixture",
    "FakeModel",
    "FakeModelRequest",
    "InMemoryA2ATransport",
    "ScriptedProvider",
    "ScriptedProviderCall",
    "assert_provider_conformance",
    "assert_provider_tool_call_conformance",
    "run_provider_conformance",
]
