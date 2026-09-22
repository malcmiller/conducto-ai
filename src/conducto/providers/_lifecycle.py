"""Private asynchronous ownership and shutdown for provider clients."""

from __future__ import annotations

import asyncio

from conducto.core.provider import AsynchronouslyClosableProvider, ProviderError


class AsyncClientLifecycle:
    """Serialize owned-client shutdown without claiming failed closes succeeded.

    A missing client represents borrowed resources, whose lifetime remains
    entirely with their caller. Only successful owned shutdown marks the
    lifecycle closed; exceptions and cancellation propagate and allow retry.
    """

    def __init__(self, client: object | None, *, configuration_error: type[ProviderError]) -> None:
        """Require asynchronous cleanup for owned clients and retain no borrowed client."""
        if client is not None and not isinstance(client, AsynchronouslyClosableProvider):
            raise configuration_error("Owned provider clients must support asynchronous shutdown")
        self._client = client
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        """Return whether the owned client's shutdown completed successfully."""
        return self._closed

    async def aclose(self) -> None:
        """Close owned resources once, waiting for any concurrent shutdown attempt."""
        if self._client is None:
            return
        async with self._lock:
            if self._closed:
                return
            await self._client.aclose()
            self._closed = True
