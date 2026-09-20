"""Focused contracts for model gateway delegation and provenance."""

import asyncio

from pydantic import BaseModel

from conducto import (
    ChatMessage,
    FakeModel,
    ModelConfiguration,
    ProviderRegistry,
    Runtime,
    Usage,
)
from conducto.core.model_gateway import ModelCallResult as GatewayModelCallResult
from conducto.core.runtime import ModelCallResult, use_run_context


class Response(BaseModel):
    value: str


def test_model_gateway_records_typed_completion_provenance() -> None:
    async def exercise() -> None:
        registry = ProviderRegistry()
        registry.register(
            "model",
            FakeModel(
                {"value": "ok"},
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
            ModelConfiguration(provider="fake", model="model"),
        )
        runtime = Runtime(provider_registry=registry)
        context = runtime.create_run_context(
            agent_id="Agent",
            call_override="model",
        )
        with use_run_context(context):
            result = await context.models.require().complete_typed(
                (ChatMessage(role="user", content="respond"),),
                response_type=Response,
            )

        assert result == Response(value="ok")
        metadata = context.invocation_metadata()
        assert [call.purpose for call in metadata.model_calls] == ["capability"]
        assert metadata.usage.total_tokens == 3

    assert ModelCallResult is GatewayModelCallResult
    asyncio.run(exercise())
