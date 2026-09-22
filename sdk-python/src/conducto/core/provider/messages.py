"""Provider-neutral conversation content and messages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageContentPart(BaseModel):
    """A provider-neutral content part supported by the current contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text", "json"]
    text: str | None = None
    value: dict[str, Any] | list[Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Ensure a content part has exactly the payload its type requires."""
        if self.type == "text" and (self.text is None or self.value is not None):
            raise ValueError("text content parts require only text")
        if self.type == "json" and (self.value is None or self.text is not None):
            raise ValueError("json content parts require only value")


class ChatMessage(BaseModel):
    """Provider-neutral chat message sent to a model client.

    Attributes:
        role: Conversation role understood by the provider adapter.
        content: Text content for the message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    content: str | tuple[MessageContentPart, ...]

    @field_validator("content")
    @classmethod
    def validate_content(
        cls, value: str | tuple[MessageContentPart, ...] | list[MessageContentPart]
    ) -> str | tuple[MessageContentPart, ...]:
        """Normalize content parts to an immutable tuple."""
        if isinstance(value, list):
            return tuple(value)
        return value
