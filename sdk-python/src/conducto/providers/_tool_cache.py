"""Private bounded, concurrency-safe native tool turn state."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CachedToolCall:
    """Wire-specific assistant message needed to round-trip a tool result."""

    call_id: str
    name: str
    assistant_message: Mapping[str, Any]


class ToolCallCache:
    """Per-provider LRU state; never shared across clients or credentials."""

    def __init__(self, limit: int = 1024) -> None:
        """Create a bounded cache and its asynchronous access lock."""
        self._limit = limit
        self._calls: OrderedDict[str, CachedToolCall] = OrderedDict()
        self._lock = asyncio.Lock()

    async def remember(self, call_id: str, name: str, assistant_message: Mapping[str, Any]) -> None:
        """Remember a call and evict the least recently used entries."""
        async with self._lock:
            self._calls[call_id] = CachedToolCall(call_id, name, assistant_message)
            self._calls.move_to_end(call_id)
            while len(self._calls) > self._limit:
                self._calls.popitem(last=False)

    async def lookup(self, call_id: str) -> CachedToolCall | None:
        """Look up a call while refreshing its recency."""
        async with self._lock:
            cached = self._calls.get(call_id)
            if cached is not None:
                self._calls.move_to_end(call_id)
            return cached
