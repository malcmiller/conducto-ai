"""Private shared normalization for adapter values and native tool calls."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from packaging.version import InvalidVersion, Version

from conducto.core.provider import (
    GenerationOptions,
    MalformedStructuredOutputError,
    MessageContentPart,
    ProviderCallContext,
    ProviderResult,
    ProviderTimeoutError,
    ProviderToolCallRequest,
    ProviderToolDefinition,
    StructuredOutputRequest,
    Usage,
    validate_structured_output,
)


@runtime_checkable
class _ModelDumpable(Protocol):
    """Structural serialization hook supplied by supported SDK response models."""

    def model_dump(self) -> object:
        """Return the model's plain serialization payload."""
        ...


def object_to_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce official SDK response objects or plain dictionaries to mappings."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, _ModelDumpable) and callable(value.model_dump):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return cast(Mapping[str, Any], vars(value))
    return {}


def safe_string(value: Any) -> str | None:
    """Return a non-empty string when available."""
    return value if isinstance(value, str) and value.strip() else None


def optional_non_negative_int(value: Any) -> int | None:
    """Return an explicit non-negative token counter without estimating it."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def content_has_value(content: Any) -> bool:
    """Return whether message content carries terminal data."""
    return isinstance(content, str) and bool(content.strip())


def extract_version(payload: Any) -> str | None:
    """Extract a server version from native or mapping-shaped responses."""
    return safe_string(object_to_mapping(payload).get("version"))


def version_at_least(actual: str, minimum: str) -> bool:
    """Compare versions using packaging semantics, rejecting invalid versions."""
    try:
        return Version(actual) >= Version(minimum)
    except InvalidVersion:
        return False


def request_timeout(
    options: GenerationOptions,
    *,
    effective_deadline: float | None,
    call_context: ProviderCallContext | None,
) -> float | None:
    """Merge all deadlines with the request timeout without extending authority."""
    timeout = options.timeout
    deadline = effective_deadline
    if call_context is not None and call_context.deadline is not None:
        deadline = (
            min(deadline, call_context.deadline) if deadline is not None else call_context.deadline
        )
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError(attempted=False)
        timeout = min(timeout, remaining) if timeout is not None else remaining
    return timeout


def native_tool(tool: ProviderToolDefinition) -> dict[str, Any]:
    """Translate a Conducto descriptor to the shared function-tool wire shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.to_dict()["input_schema"],
        },
    }


@dataclass(frozen=True, slots=True)
class ResponseCodec:
    """Bind equivalent normalization to an adapter's diagnostics and ID namespace."""

    name: str
    source: str
    call_prefix: str

    def json_dumps(self, value: Any, label: str) -> str:
        """Serialize deterministically without custom-object fallback."""
        try:
            return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except TypeError:
            raise MalformedStructuredOutputError(
                f"{self.name} {label} is not JSON serializable"
            ) from None

    def message_content_to_text(self, content: str | tuple[MessageContentPart, ...]) -> str:
        """Translate supported text and JSON message parts to text content."""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for part in content:
            if part.type == "text" and part.text is not None:
                parts.append(part.text)
            elif part.type == "json" and part.value is not None:
                parts.append(self.json_dumps(part.value, "message content"))
        return "\n".join(parts)

    def extract_tool_calls(self, message: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Return tool-call mappings, rejecting malformed collections or entries."""
        raw = message.get("tool_calls", ())
        if raw is None:
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise MalformedStructuredOutputError(f"{self.source} returned malformed tool calls")
        calls: list[Mapping[str, Any]] = []
        for item in raw:
            mapped = object_to_mapping(item)
            if not mapped:
                raise MalformedStructuredOutputError(f"{self.source} returned malformed tool call")
            calls.append(mapped)
        return tuple(calls)

    def coerce_arguments(self, value: Any) -> Mapping[str, Any]:
        """Decode tool-call arguments and require an object."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise MalformedStructuredOutputError(
                    f"{self.source} returned malformed tool arguments"
                ) from None
        if not isinstance(value, Mapping):
            raise MalformedStructuredOutputError(
                f"{self.source} returned non-object tool arguments"
            )
        return dict(value)

    def synthesize_call_id(
        self, *, request_id: str | None, tool_id: str, name: str, arguments: Mapping[str, Any]
    ) -> str:
        """Create a stable, adapter-namespaced call ID when the server omits one."""
        argument_text = json.dumps(
            arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        seed = f"{request_id or ''}\0{tool_id}\0{name}\0{argument_text}"
        return f"{self.call_prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    async def normalize_result(
        self,
        message: Mapping[str, Any],
        *,
        structured_output: StructuredOutputRequest,
        usage: Usage,
        request_id: str | None,
        finish_reason: str,
        normalize_tool: Callable[[Mapping[str, Any]], Awaitable[ProviderToolCallRequest]],
    ) -> ProviderResult:
        """Validate a single terminal result or tool call with shared result metadata."""
        content = message.get("content")
        tool_calls = self.extract_tool_calls(message)
        if tool_calls and content_has_value(content):
            raise MalformedStructuredOutputError(
                f"{self.source} returned mixed terminal content and tool calls",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        if len(tool_calls) > 1:
            raise MalformedStructuredOutputError(
                f"{self.source} returned multiple tool calls",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        if tool_calls:
            call = await normalize_tool(tool_calls[0])
            return ProviderResult(
                tool_calls=(call,),
                usage=usage,
                accepted=True,
                request_id=request_id,
                finish_reason=finish_reason,
            )
        if not isinstance(content, str):
            raise MalformedStructuredOutputError(
                f"{self.source} returned no terminal structured content",
                accepted=True,
                usage=usage,
                request_id=request_id,
            )
        try:
            structured = json.loads(content)
        except json.JSONDecodeError:
            raise MalformedStructuredOutputError(
                f"{self.source} returned invalid terminal JSON",
                accepted=True,
                usage=usage,
                request_id=request_id,
            ) from None
        validate_structured_output(structured, structured_output)
        return ProviderResult(
            structured=structured,
            usage=usage,
            accepted=True,
            request_id=request_id,
            finish_reason=finish_reason,
        )

    def resolve_tool(
        self,
        function: Mapping[str, Any],
        tools: tuple[ProviderToolDefinition, ...],
    ) -> tuple[ProviderToolDefinition, str, Mapping[str, Any]]:
        """Resolve exactly one declared tool and validate its argument object."""
        name = safe_string(function.get("name"))
        if name is None:
            raise MalformedStructuredOutputError(
                f"{self.source} returned a tool call without a name"
            )
        matches = [tool for tool in tools if tool.name == name]
        if not matches:
            raise MalformedStructuredOutputError(f"{self.source} returned an unknown tool name")
        if len(matches) != 1:
            raise MalformedStructuredOutputError(f"{self.source} returned an ambiguous tool call")
        arguments = self.coerce_arguments(function.get("arguments"))
        tool = matches[0]
        validate_structured_output(
            arguments,
            StructuredOutputRequest(name=f"{tool.name}_arguments", schema=tool.input_schema),
        )
        return tool, name, arguments
