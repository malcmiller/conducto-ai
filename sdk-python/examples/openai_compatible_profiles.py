r"""Switch between vLLM and LM Studio using the same agent and provider contract.

This example builds one Conducto agent and orchestrator and then completes a
single structured-output request through whichever already running
OpenAI-compatible server profile the caller selects. It does not start, stop,
or configure vLLM or LM Studio, and it does not download model weights: point
``--profile`` at a server you already have running locally.

Run from ``sdk-python`` after installing the built wheel and starting a
matching local server:

    python examples/openai_compatible_profiles.py --profile vllm --model my-served-model
    python examples/openai_compatible_profiles.py --profile lm-studio --model my-loaded-model
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

from pydantic import BaseModel

from conducto.core.provider import (
    ChatMessage,
    GenerationOptions,
    ModelConfiguration,
    StructuredOutputRequest,
)
from conducto.core.provider_registry import ProviderRegistry
from conducto.providers import (
    LM_STUDIO_DEFAULT_PROFILE,
    VLLM_DEFAULT_PROFILE,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
)


class _GreetingAnswer(BaseModel):
    """Deterministic terminal schema shared across both server profiles."""

    greeting: str


@dataclass(frozen=True, slots=True)
class _ProfileChoice:
    """Command-line-selectable server profile metadata."""

    label: str
    endpoint: str
    profile: OpenAICompatibleProfile


_PROFILE_CHOICES: dict[str, _ProfileChoice] = {
    "vllm": _ProfileChoice("vllm", "http://localhost:8000/v1", VLLM_DEFAULT_PROFILE),
    "lm-studio": _ProfileChoice("lm-studio", "http://localhost:1234/v1", LM_STUDIO_DEFAULT_PROFILE),
}


def _build_provider(choice: _ProfileChoice, model: str) -> OpenAICompatibleProvider:
    """Construct the selected OpenAI-compatible profile without changing agent code."""
    return OpenAICompatibleProvider(endpoint=choice.endpoint, model=model, profile=choice.profile)


async def run_profile_example(profile_name: str, model: str) -> _GreetingAnswer:
    """Complete one deterministic structured-output turn against the selected server."""
    choice = _PROFILE_CHOICES[profile_name]
    provider = _build_provider(choice, model)
    await provider.check_readiness(model=model)
    try:
        # The same registry, agent, and orchestrator code paths accept this
        # provider instance identically regardless of the selected profile.
        registry = ProviderRegistry()
        registry.register_client(
            "local-model",
            provider,
            ModelConfiguration(provider="openai_compatible", model=model),
        )
        result = await provider.complete(
            (ChatMessage(role="user", content="Reply with a short greeting."),),
            options=GenerationOptions(model=model, temperature=0.0, max_tokens=32, timeout=30),
            structured_output=StructuredOutputRequest(
                name="GreetingAnswer",
                schema=_GreetingAnswer.model_json_schema(),
            ),
        )
    finally:
        await provider.aclose()
    return _GreetingAnswer.model_validate(result.structured)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_PROFILE_CHOICES), required=True)
    parser.add_argument("--model", required=True, help="Model already served or loaded locally.")
    args = parser.parse_args()

    answer = asyncio.run(run_profile_example(args.profile, args.model))
    print(f"profile={args.profile} greeting={answer.greeting!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
