"""Deterministic, provider-neutral conformance helpers.

Application providers can use :func:`assert_provider_conformance` with a
credential-free scripted transport. The helper intentionally uses only public
Conducto contracts and never inspects provider internals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    ModelProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderResult,
    StructuredOutputRequest,
    complete_with_retries,
)

PROVIDER_FIXTURE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ScriptedProviderCall:
    """Credential-free observation of a scripted provider request."""

    messages: tuple[ChatMessage, ...]
    options: GenerationOptions
    structured_output: StructuredOutputRequest
    deadline: float | None


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
        tools: Sequence[Mapping[str, Any]] = (),
        tool_results: Sequence[Any] = (),
        effective_deadline: float | None = None,
    ) -> ProviderResult:
        """Return the next deterministic response without sleeping."""
        del tools, tool_results
        self.calls.append(
            ScriptedProviderCall(
                tuple(messages),
                options,
                structured_output,
                effective_deadline,
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
    """Run mandatory, deterministic checks shared by all provider adapters."""
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
    assert result.acceptance.value in {"accepted", "attempted_not_accepted"}
    assert result.usage is not None
