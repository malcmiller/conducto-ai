"""Tests for BaseAgent's governed model-completion facade."""

import asyncio
from dataclasses import replace

import pytest

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.invocation_results import InvocationSuccess
from conducto.core.provider import (
    ChatMessage,
    ModelConfiguration,
    ProviderResult,
    StructuredOutputRequest,
)
from conducto.core.provider_registry import ProviderOwnership, ProviderRegistry
from conducto.core.run_context import CancellationState, use_run_context
from conducto.core.runtime_errors import NoActiveRunContextError
from conducto.testing import FakeModel


class _ClosableFakeModel(FakeModel):
    """Fake provider that records runtime-owned cleanup."""

    def __init__(self, result: ProviderResult) -> None:
        super().__init__(result)
        self.closed = False

    async def aclose(self) -> None:
        """Record that the runtime released this provider."""
        self.closed = True


@a2a_agent(
    name="CompletionAgent",
    version="1.0",
    description="Completes prompts through the active runtime.",
    default_model="model",
    model_required=True,
    instructions="Use concise answers.",
)
class _CompletionAgent(BaseAgent):
    @a2a_capability(
        name="draft",
        description="Drafts a response with the governed completion facade.",
        instructions="Answer the prompt.",
    )
    async def draft(self, prompt: str) -> str:
        """Return a structured draft from the governed model call."""
        result = await self.complete(
            prompt,
            structured_output=StructuredOutputRequest(
                name="draft",
                schema={"type": "object", "properties": {"text": {"type": "string"}}},
            ),
        )
        assert result.result.structured is not None
        return str(result.result.structured["text"])


def _runtime(
    provider: FakeModel | None = None,
) -> tuple[Runtime, ProviderRegistry, FakeModel]:
    """Create a runtime with one runtime-owned deterministic provider."""
    registry = ProviderRegistry()
    configured_provider = provider or FakeModel(
        ProviderResult(structured={"text": "done"}, accepted=True)
    )
    registry.register_client(
        "model",
        configured_provider,
        ModelConfiguration(provider="fake", model="model"),
        ownership=ProviderOwnership.RUNTIME_OWNED,
    )
    return Runtime(provider_registry=registry), registry, configured_provider


def test_agent_completion_uses_runtime_context_and_instruction_chain() -> None:
    async def exercise() -> InvocationSuccess:
        runtime, _registry, provider = _runtime()
        result = await runtime.invoke(_CompletionAgent(), "draft", {"prompt": "Write a draft."})
        assert isinstance(result, InvocationSuccess)
        assert result.value == "done"
        assert provider.requests[0].message_roles == ("system", "user")
        assert provider.requests[0].structured_output.name == "draft"
        return result

    result = asyncio.run(exercise())

    assert result.metadata is not None
    assert result.metadata.instruction_chain == ("Use concise answers.", "Answer the prompt.")
    assert [call.purpose for call in result.metadata.model_calls] == ["capability"]


def test_agent_completion_accepts_message_sequences_and_optional_structured_output() -> None:
    async def exercise() -> None:
        runtime, _registry, provider = _runtime(
            FakeModel(ProviderResult(content="done", accepted=True))
        )
        context = runtime.create_run_context(agent_id="CompletionAgent", call_override="model")
        with use_run_context(context):
            result = await _CompletionAgent().complete(
                (ChatMessage(role="user", content="Write a draft."),)
            )

        assert result.result.content == "done"
        assert provider.requests[0].structured_output.required is False

    asyncio.run(exercise())


def test_agent_completion_requires_an_active_run_context() -> None:
    with pytest.raises(NoActiveRunContextError, match="No active Conducto run context"):
        asyncio.run(_CompletionAgent().complete("Write a draft."))


def test_agent_completion_rejects_empty_message_sequences() -> None:
    with pytest.raises(ValueError, match="requires at least one message"):
        asyncio.run(_CompletionAgent().complete(()))


def test_agent_completion_rejects_blank_prompts() -> None:
    with pytest.raises(ValueError, match="prompt cannot be blank"):
        asyncio.run(_CompletionAgent().complete("   "))


def test_agent_completion_honors_cancellation_and_deadlines() -> None:
    async def exercise() -> None:
        runtime, _registry, provider = _runtime()
        cancellation = CancellationState()
        cancelled_context = runtime.create_run_context(
            agent_id="CompletionAgent",
            call_override="model",
            cancellation=cancellation,
        )
        cancellation.cancel()
        with use_run_context(cancelled_context), pytest.raises(asyncio.CancelledError):
            await _CompletionAgent().complete("Write a draft.")

        expired_context = replace(
            runtime.create_run_context(agent_id="CompletionAgent", call_override="model"),
            deadline=0.0,
        )
        with (
            use_run_context(expired_context),
            pytest.raises(TimeoutError, match="Run deadline exceeded"),
        ):
            await _CompletionAgent().complete("Write a draft.")

        assert provider.calls == 0

    asyncio.run(exercise())


def test_agent_completion_releases_runtime_owned_provider_lease() -> None:
    async def exercise() -> None:
        provider = _ClosableFakeModel(ProviderResult(content="done", accepted=True))
        runtime, _registry, _provider = _runtime(provider)
        context = runtime.create_run_context(agent_id="CompletionAgent", call_override="model")
        with use_run_context(context):
            await _CompletionAgent().complete("Write a draft.")

        await runtime.aclose()
        assert provider.closed is True

    asyncio.run(exercise())
