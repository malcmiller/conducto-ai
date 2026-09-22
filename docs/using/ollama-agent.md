# Connect an agent to Ollama

Ollama runs the model. Conducto owns the typed agent contract, model binding,
execution limits, and result validation.

## Prerequisites

Install the optional adapter and make sure an Ollama daemon and model already
exist:

```bash
pip install "conducto-ai[ollama]"
ollama pull llama3.1:8b
ollama serve
```

Conducto does not start Ollama or download models.

## Define a structured response

```python
import asyncio

from pydantic import BaseModel

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability
from conducto.core.delegation import DelegationConfig
from conducto.core.invocation_results import InvocationSuccess
from conducto.core.provider import ChatMessage, ModelConfiguration
from conducto.core.provider_registry import ProviderOwnership, ProviderRegistry
from conducto.providers import OllamaProvider


class Answer(BaseModel):
    answer: str
    confidence: float


@a2a_agent(
    name="ResearchAgent",
    version="1.0.0",
    description="Answers research questions with a local model.",
    default_model="local-llama",
    model_required=True,
)
class ResearchAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            delegation_config=DelegationConfig(
                max_model_turns=1,
                max_tool_calls=0,
            )
        )

    @a2a_capability(
        name="research.answer",
        description="Answers one research question.",
    )
    async def answer_question(self, question: str) -> Answer:
        outcome = await self.run_delegation(
            (ChatMessage(role="user", content=question),),
            response_type=Answer,
        )
        if not outcome.ok or outcome.value is None:
            raise RuntimeError(f"Model execution failed: {outcome.code.value}")
        return outcome.value
```

## Bind the model and invoke the agent

```python
async def main() -> None:
    providers = ProviderRegistry()
    providers.register_client(
        "local-llama",
        OllamaProvider(
            endpoint="http://localhost:11434",
            model="llama3.1:8b",
            timeout=60,
        ),
        ModelConfiguration(
            provider="ollama",
            model="llama3.1:8b",
        ),
        ownership=ProviderOwnership.RUNTIME_OWNED,
    )

    async with Runtime(provider_registry=providers) as runtime:
        result = await runtime.invoke(
            ResearchAgent(),
            "research.answer",
            {"question": "Why is a typed capability useful?"},
        )

    if not isinstance(result, InvocationSuccess):
        raise RuntimeError(f"Invocation failed: {type(result).__name__}")

    print(result.value)


asyncio.run(main())
```

## Ownership

- The agent refers to the opaque name `local-llama`.
- The runtime registry owns the provider binding.
- The registration explicitly transfers the preconstructed provider to the
  runtime, which closes it when the runtime scope exits.
- Credentials and transport configuration never belong in the agent class or
  Agent Card.

## Production note

Model output quality is not deterministic. Conducto validates the terminal
shape, but the application must choose and test an appropriate model. See the
[Ollama provider reference](../ollama-provider.md) for supported profiles,
readiness, TLS, proxy, authentication, and tool-call behavior.

Next: [connect two local agents](./two-agent-workflow.md).
