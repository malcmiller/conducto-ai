"""Deterministic typed model double with safe request observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    ProviderCallContext,
    ProviderCapabilities,
    ProviderError,
    ProviderResult,
    ProviderToolDefinition,
    StructuredOutputRequest,
    ToolResultMessage,
)


@dataclass(frozen=True, slots=True)
class FakeModelRequest:
    """Credential-free observation of one request sent to a fake model.

    Attributes:
        message_roles: Conversation roles without the message content.
        options: Generation settings observed by the fake provider.
        structured_output: Requested terminal structured-output schema.
        tools: Typed definitions for the exact toolbox snapshot.
        tool_results: Prior bounded tool results for the turn.
        effective_deadline: Effective monotonic deadline applied to the turn.
    """

    message_roles: tuple[str, ...]
    options: GenerationOptions
    structured_output: StructuredOutputRequest
    tools: tuple[ProviderToolDefinition, ...] = ()
    tool_results: tuple[ToolResultMessage, ...] = ()
    effective_deadline: float | None = None


class FakeModel:
    """Repeat terminal content or a typed result, or consume a typed script.

    Supply exactly one of ``result`` or ``script``. A mapping supplied as
    ``result`` is literal terminal structured content with ``accepted=True``;
    no keys are interpreted as synthetic terminal or tool-call envelopes.
    Use ``ProviderResult`` for explicit usage, acceptance, or native tool calls.
    Scripts accept only ``ProviderResult`` and ``ProviderError`` instances.
    """

    capabilities = ProviderCapabilities(
        structured_output=True,
        tool_calling=True,
        context_limit=128_000,
        usage_reporting=True,
    )

    def __init__(
        self,
        result: ProviderResult | ProviderError | Mapping[str, Any] | None = None,
        *,
        script: Sequence[ProviderResult | ProviderError] | None = None,
    ) -> None:
        if (result is None) == (script is None):
            raise ValueError("FakeModel requires exactly one of result or script")
        if isinstance(result, Mapping):
            result = ProviderResult(structured=dict(result), accepted=True)
        entries = (result,) if script is None else tuple(script)
        if any(not isinstance(entry, (ProviderResult, ProviderError)) for entry in entries):
            raise TypeError("FakeModel entries must be ProviderResult or ProviderError instances")
        self._result = result
        self._script = tuple(script) if script is not None else None
        self.calls = 0
        self.requests: list[FakeModelRequest] = []

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
        """Record safe request metadata and return or raise the next entry.

        Raises:
            ProviderError: If the configured entry is a failure or the script
                is exhausted.
            TypeError: If tools contain untyped mapping payloads.
        """
        del call_context
        if any(not isinstance(tool, ProviderToolDefinition) for tool in tools):
            raise TypeError("FakeModel tools must be ProviderToolDefinition instances")
        self.calls += 1
        self.requests.append(
            FakeModelRequest(
                message_roles=tuple(message.role for message in messages),
                options=options,
                structured_output=structured_output,
                tools=tuple(tools),
                tool_results=tuple(tool_results),
                effective_deadline=effective_deadline,
            )
        )
        if self._script is None:
            result = self._result
        elif self.calls <= len(self._script):
            result = self._script[self.calls - 1]
        else:
            raise ProviderError("FakeModel script exhausted")
        if isinstance(result, ProviderError):
            raise result
        if not isinstance(result, ProviderResult):
            raise AssertionError("FakeModel has no configured result")
        return result
