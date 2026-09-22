"""Cancellation-safe, bounded cleanup of identity-owned provider clients."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future

from ..model_config import ModelReference
from ..provider import (
    AsynchronouslyClosableProvider,
    ModelProvider,
    SynchronouslyClosableProvider,
)
from .ownership import (
    ProviderCleanupFailure,
    ProviderCleanupReport,
    ProviderOwnership,
    _ProviderClientRecord,
)
from .state import _RegistryState


def _minimum_timeout(first: float | None, second: float | None) -> float | None:
    if second is not None and second <= 0:
        raise TimeoutError
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


async def _close_client(client: ModelProvider, timeout: float | None) -> None:
    """Close a client through its optional structural cleanup protocol."""

    async def close() -> None:
        if isinstance(client, AsynchronouslyClosableProvider):
            await client.aclose()
        elif isinstance(client, SynchronouslyClosableProvider):
            await asyncio.to_thread(client.close)

    if timeout is None:
        await close()
    else:
        await asyncio.wait_for(close(), timeout)


class _ProviderCleanup(_RegistryState):
    """Own cleanup tasks independently of cancellation of cleanup waiters."""

    def _schedule_retired_cleanup(self, record: _ProviderClientRecord) -> None:
        """Start cleanup on the running loop or a temporary owner loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._close_records((record,), retired_only=True))
            return
        task = loop.create_task(self._close_records((record,), retired_only=True))
        with self._lock:
            self._background_cleanups.add(task)
        task.add_done_callback(self._discard_background_cleanup)

    def _discard_background_cleanup(self, task: asyncio.Task[ProviderCleanupReport]) -> None:
        with self._lock:
            self._background_cleanups.discard(task)

    def _prepare_cleanup(
        self,
        record: _ProviderClientRecord,
        *,
        retired_only: bool,
    ) -> tuple[str, tuple[ModelReference, ...]]:
        """Atomically revalidate and claim one record for cleanup."""
        with self._lock:
            current = self._clients.get(id(record.client))
            identities = record.identities()
            if current is not record:
                return "ineligible", identities
            if retired_only and (record.references or record.leases):
                return "ineligible", identities
            if record.ownership is ProviderOwnership.CALLER_OWNED:
                return ("ineligible" if retired_only else "skipped"), identities
            if record.leases:
                return "leased", identities
            if record.closed:
                return "closed", identities
            if record.close_started:
                return "waiting", identities
            record.close_started = True
            record.close_failure = None
            record.close_completion = Future()
            return "start", identities

    def _track_cleanup_owner(self, task: asyncio.Task[None]) -> None:
        with self._lock:
            self._cleanup_owners.add(task)
        task.add_done_callback(self._discard_cleanup_owner)

    def _discard_cleanup_owner(self, task: asyncio.Task[None]) -> None:
        with self._lock:
            self._cleanup_owners.discard(task)

    async def _run_record_cleanup(
        self,
        record: _ProviderClientRecord,
        *,
        per_client_timeout: float | None,
        deadline: float | None,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Own one client close to completion despite waiter cancellation."""
        failure: ProviderCleanupFailure | None = None
        identities = record.identities()
        try:
            async with semaphore:
                remaining = None if deadline is None else deadline - time.monotonic()
                limit = _minimum_timeout(per_client_timeout, remaining)
                await _close_client(record.client, limit)
        except asyncio.CancelledError:
            failure = ProviderCleanupFailure(identities, record.provider, "CancelledError")
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
        except TimeoutError:
            failure = ProviderCleanupFailure(
                identities, record.provider, "TimeoutError", timed_out=True
            )
        except Exception as error:
            failure = ProviderCleanupFailure(identities, record.provider, type(error).__name__)
        finally:
            with self._lock:
                record.close_failure = failure
                record.closed = failure is None
                if not record.close_completion.done():
                    record.close_completion.set_result(None)

    async def _wait_for_record_cleanup(
        self,
        record: _ProviderClientRecord,
        identities: tuple[ModelReference, ...],
        deadline: float | None,
    ) -> tuple[
        tuple[ModelReference, ...],
        tuple[ModelReference, ...],
        ProviderCleanupFailure | None,
    ]:
        """Join a shared close owner without transferring cancellation to it."""
        with self._lock:
            completion = record.close_completion
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return (
                (),
                (),
                ProviderCleanupFailure(identities, record.provider, "TimeoutError", timed_out=True),
            )
        try:
            wrapped = asyncio.wrap_future(completion)
            if remaining is None:
                await asyncio.shield(wrapped)
            else:
                await asyncio.wait_for(asyncio.shield(wrapped), remaining)
        except TimeoutError:
            return (
                (),
                (),
                ProviderCleanupFailure(identities, record.provider, "TimeoutError", timed_out=True),
            )
        with self._lock:
            if record.close_failure is not None:
                return (), (), record.close_failure
            if record.closed:
                return identities, (), None
        return (), (), ProviderCleanupFailure(identities, record.provider, "RuntimeError")

    async def _close_record(
        self,
        record: _ProviderClientRecord,
        *,
        retired_only: bool,
        per_client_timeout: float | None,
        deadline: float | None,
        semaphore: asyncio.Semaphore,
    ) -> tuple[
        tuple[ModelReference, ...],
        tuple[ModelReference, ...],
        ProviderCleanupFailure | None,
    ]:
        """Join or start one atomically eligible record cleanup."""
        state, identities = self._prepare_cleanup(record, retired_only=retired_only)
        if state == "ineligible":
            return (), (), None
        if state == "skipped":
            return (), identities, None
        if state == "leased":
            return (
                (),
                (),
                ProviderCleanupFailure(identities, record.provider, "TimeoutError", timed_out=True),
            )
        if state == "closed":
            return identities, (), None
        if state == "start":
            owner = asyncio.create_task(
                self._run_record_cleanup(
                    record,
                    per_client_timeout=per_client_timeout,
                    deadline=deadline,
                    semaphore=semaphore,
                )
            )
            self._track_cleanup_owner(owner)
        return await self._wait_for_record_cleanup(record, identities, deadline)

    async def _close_records(
        self,
        records: tuple[_ProviderClientRecord, ...],
        *,
        per_client_timeout: float | None = 10.0,
        max_concurrency: int = 4,
        deadline: float | None = None,
        retired_only: bool = False,
    ) -> ProviderCleanupReport:
        """Close or join eligible records, retaining safe aggregate outcomes."""
        semaphore = asyncio.Semaphore(max_concurrency)
        outcomes = await asyncio.gather(
            *(
                self._close_record(
                    record,
                    retired_only=retired_only,
                    per_client_timeout=per_client_timeout,
                    deadline=deadline,
                    semaphore=semaphore,
                )
                for record in records
            )
        )
        closed: list[ModelReference] = []
        skipped: list[ModelReference] = []
        failures: list[ProviderCleanupFailure] = []
        for outcome_closed, outcome_skipped, failure in outcomes:
            closed.extend(outcome_closed)
            skipped.extend(outcome_skipped)
            if failure is not None:
                failures.append(failure)
        return ProviderCleanupReport(
            tuple(sorted(set(closed), key=lambda item: item.value)),
            tuple(sorted(set(skipped), key=lambda item: item.value)),
            tuple(
                sorted(
                    failures,
                    key=lambda item: (
                        tuple(reference.value for reference in item.references),
                        item.provider,
                    ),
                )
            ),
        )
