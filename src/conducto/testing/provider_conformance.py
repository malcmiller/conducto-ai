"""Deterministic, provider-neutral conformance helpers.

Application providers can use :func:`assert_provider_conformance`
with credential-free scripted transport. The helper intentionally uses only public
Conducto contracts and never inspects provider internals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from conducto.core.provider import (
    AcceptanceState,
    ChatMessage,
    GenerationOptions,
    ModelProvider,
    ProviderCallContext,
    ProviderCapabilities,
    ProviderEndpointUnavailableError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
    Usage,
    complete_with_retries,
    parse_model_decision,
)

PROVIDER_FIXTURE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ConformanceFixture:
    """Provider-independent expected result for one deterministic contract case."""

    name: str
    expected_acceptance: AcceptanceState | None = None
    expected_request_id: str | None = None
    expected_usage: Usage | None = None


MANDATORY_FIXTURES = (
    ConformanceFixture("success", AcceptanceState.ACCEPTED, "fixture-success"),
    ConformanceFixture("unknown-usage", AcceptanceState.ACCEPTED, "fixture-unknown"),
    ConformanceFixture("malformed-output", AcceptanceState.ACCEPTED, "fixture-malformed"),
)
"""Stable mandatory fixture names consumed by provider adapters."""


@dataclass(frozen=True, slots=True)
class ScriptedProviderCall:
    """Credential-free observation of a scripted provider request."""

    messages: tuple[ChatMessage, ...]
    options: GenerationOptions
    structured_output: StructuredOutputRequest
    tools: tuple[ProviderToolDefinition, ...]
    tool_results: tuple[ToolResultMessage, ...]
    deadline: float | None
    cancellation: bool


class ScriptedProvider:
    """Network-free provider transport for adapter and contract tests."""

    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calling=True,
        usage_reporting=True,
        cancellation=True,
    )

    def __init__(
        self,
        responses: Sequence[ProviderResult | ProviderError],
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._responses = tuple(responses)
        self.capabilities = capabilities or self.capabilities
        self.calls: list[ScriptedProviderCall] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions,
        structured_output: StructuredOutputRequest,
        tools: Sequence[ProviderToolDefinition] = (),
        tool_results: Sequence[ToolResultMessage] = (),
        effective_deadline: float | None = None,
        call_context: ProviderCallContext | None = None,
    ) -> ProviderResult:
        """Return the next deterministic response without sleeping."""
        self.calls.append(
            ScriptedProviderCall(
                tuple(messages),
                options,
                structured_output,
                tuple(tools),
                tuple(tool_results),
                effective_deadline,
                call_context.cancelled if call_context is not None else False,
            )
        )
        if not self._responses:
            raise ProviderError("script exhausted")
        response = self._responses[0]
        self._responses = self._responses[1:]
        if isinstance(response, ProviderError):
            raise response
        return response


async def assert_provider_conformance(provider: ModelProvider) -> None:
    """Run the mandatory deterministic checks shared by provider adapters.

    The provider is expected to be a fresh fixture-backed instance. Conditional
    capability checks are represented by :data:`MANDATORY_FIXTURES` and can be
    run by adapter-specific suites without network access.
    """
    request = StructuredOutputRequest(
        name="conformance",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    result = await complete_with_retries(
        provider,
        (ChatMessage(role="user", content="conformance"),),
        options=GenerationOptions(model="conformance"),
        structured_output=request,
    )
    assert result.structured == {"answer": "ok"}
    assert result.acceptance is AcceptanceState.ACCEPTED
    assert result.usage.input_tokens is None or result.usage.input_tokens >= 0


async def assert_provider_tool_call_conformance(provider: ModelProvider) -> None:
    """Run the deterministic native-tool decision contract against a provider."""
    request = _request()
    request = StructuredOutputRequest(
        name=request.name,
        schema=request.json_schema,
        required=False,
    )
    tools = (
        ProviderToolDefinition(
            tool_id="tool-1",
            name="lookup_tool_1",
            description="Looks up deterministic fixture data.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    )
    result = await complete_with_retries(
        provider,
        (ChatMessage(role="user", content="tool conformance"),),
        options=GenerationOptions(model="conformance"),
        structured_output=request,
        tools=tools,
    )
    decision = parse_model_decision(result, response_type=_ConformanceResponse, tools=tools)
    assert decision.type == "tool_call"
    assert decision.tool_id == "tool-1"
    assert decision.arguments == {"query": "value"}


async def run_provider_conformance(
    provider_factory: Callable[[], ModelProvider],
) -> tuple[str, ...]:
    """Run deterministic mandatory and conditional provider fixtures."""
    await assert_provider_conformance(provider_factory())

    rejected = provider_factory()
    rejected.capabilities = ProviderCapabilities()
    try:
        await complete_with_retries(
            rejected,
            (ChatMessage(role="user", content="capability"),),
            options=GenerationOptions(model="fixture"),
            structured_output=_request(),
        )
    except ProviderError as error:
        assert error.acceptance is AcceptanceState.NOT_ATTEMPTED
    else:
        raise AssertionError("unsupported capability was not rejected before dispatch")

    retried = provider_factory()
    try:
        await complete_with_retries(
            retried,
            (ChatMessage(role="user", content="retry"),),
            options=GenerationOptions(model="fixture", retries=1),
            structured_output=_request(),
        )
    except (ProviderRateLimitError, ProviderEndpointUnavailableError):
        pass

    diagnostic = ProviderError("secret=should-not-escape").diagnostic.to_dict()
    assert "should-not-escape" not in str(diagnostic)
    return tuple(fixture.name for fixture in MANDATORY_FIXTURES)


def _request() -> StructuredOutputRequest:
    """Build the schema shared by deterministic conformance cases."""
    return StructuredOutputRequest(
        name="conformance",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )


class _ConformanceResponse(BaseModel):
    answer: str
