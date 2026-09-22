"""Immutable provider-neutral registration and generation settings."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationOptions(BaseModel):
    """Validated generation settings passed to a provider for one call.

    Attributes:
        model: Provider-native model identifier from the resolved registration.
        temperature: Sampling temperature.
        max_tokens: Optional positive output-token limit.
        timeout: Optional positive timeout in seconds for each attempt.
        retries: Number of safe retry attempts after the initial request.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    retries: int = Field(default=0, ge=0)
    stop: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Reject non-finite sampling values."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be finite and non-negative")
        return value


class ModelConfiguration(BaseModel):
    """Immutable provider-neutral model configuration.

    The configuration identifies a provider and its provider model name, plus
    safe request defaults. Credentials, authorization headers, and other
    sensitive provider settings must remain inside the registered client.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None
    timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    retries: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)
