"""Client ownership, accepted-use leases, retirement, and registry shutdown."""

from __future__ import annotations

import asyncio
import time

from ..model_config import ModelReference
from ..provider import ModelProvider
from ..runtime_errors import ProviderOwnershipError, ProviderShutdownError, RuntimeClosedError
from .cleanup import _ProviderCleanup
from .ownership import (
    ProviderCleanupFailure,
    ProviderCleanupReport,
    ProviderLease,
    ProviderOwnership,
    _ProviderClientRecord,
)
from .registration import ProviderRegistration


class _ProviderLifecycle(_ProviderCleanup):
    """Coordinate accepted uses and exactly-once cleanup in shared registry state."""

    def _retire_unpublished_client(
        self,
        reference: ModelReference,
        provider: str,
        client: ModelProvider,
        ownership: ProviderOwnership,
    ) -> _ProviderClientRecord | None:
        """Retain a unique unpublished owned client for coordinated cleanup."""
        if ownership is not ProviderOwnership.RUNTIME_OWNED:
            return None
        with self._lock:
            client_id = id(client)
            existing = self._clients.get(client_id)
            if existing is not None:
                if existing.client is not client:
                    raise RuntimeError("Provider client identity collision")
                return None
            record = _ProviderClientRecord(client, ownership, provider)
            record.retired_references.add(reference)
            self._clients[client_id] = record
            return record

    def acquire(self, registration: ProviderRegistration) -> ProviderLease:
        """Accept one runtime-mediated use of an already-resolved binding.

        A retired binding remains usable only until cleanup atomically starts.
        Accepted uses retain the client record until the lease is released.

        Raises:
            RuntimeClosedError: If shutdown or binding cleanup already began.
        """
        client_id = id(registration.client)
        with self._lock:
            self._require_open()
            record = self._clients.get(client_id)
            if (
                record is None
                or record.client is not registration.client
                or record.closed
                or record.close_started
            ):
                raise RuntimeClosedError("Provider binding is no longer usable")
            record.leases += 1
        return ProviderLease(self, client_id)

    async def release_lease(self, client_id: int) -> None:
        """Release one lease and close a now-retired owned client."""
        with self._lock:
            record = self._clients.get(client_id)
            if record is None:
                return
            record.leases -= 1
            if record.leases < 0:
                raise RuntimeError("Provider client lease released more than once")
            self._lease_condition.notify_all()
            retired = self._eligible_retired_record_by_record(record)
        if retired is not None:
            await self._close_records((retired,), retired_only=True)

    async def aclose(
        self,
        *,
        timeout: float | None = 30.0,
        per_client_timeout: float | None = 10.0,
        max_concurrency: int = 4,
    ) -> ProviderCleanupReport:
        """Stop resolution, drain accepted calls, and close runtime-owned clients.

        Args:
            timeout: Aggregate grace period for accepted calls and cleanup.
            per_client_timeout: Maximum time assigned to one client close.
            max_concurrency: Maximum number of clients closed simultaneously.

        Returns:
            A shared, safe aggregate report when every owned client closed.

        Raises:
            ProviderShutdownError: If an owned client remains leased, times out,
                or raises while closing.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when supplied")
        if per_client_timeout is not None and per_client_timeout <= 0:
            raise ValueError("per_client_timeout must be positive when supplied")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        with self._lock:
            if self._shutdown_report is not None:
                if self._shutdown_report.failures:
                    raise ProviderShutdownError(self._shutdown_report)
                return self._shutdown_report
            if self._shutdown_task is None:
                self._closing = True
                deadline = time.monotonic() + timeout if timeout is not None else None
                self._shutdown_task = asyncio.create_task(
                    self._run_shutdown(
                        deadline=deadline,
                        per_client_timeout=per_client_timeout,
                        max_concurrency=max_concurrency,
                    )
                )
            shutdown_task = self._shutdown_task
        assert shutdown_task is not None
        return await asyncio.shield(shutdown_task)

    async def _run_shutdown(
        self,
        *,
        deadline: float | None,
        per_client_timeout: float | None,
        max_concurrency: int,
    ) -> ProviderCleanupReport:
        """Own the shared shutdown pass independently of awaiting callers."""
        await asyncio.to_thread(self._wait_for_quiescence, deadline)
        records, pending_constructions = self._shutdown_state()
        report = await self._close_records(
            records,
            per_client_timeout=per_client_timeout,
            max_concurrency=max_concurrency,
            deadline=deadline,
        )
        if pending_constructions:
            failures = [
                *report.failures,
                *(
                    ProviderCleanupFailure((reference,), provider, "TimeoutError", timed_out=True)
                    for reference, provider in pending_constructions
                ),
            ]
            report = ProviderCleanupReport(
                report.closed,
                report.skipped_caller_owned,
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
        with self._lock:
            self._shutdown_report = report
        if report.failures:
            raise ProviderShutdownError(report)
        return report

    def _wait_for_quiescence(self, deadline: float | None) -> None:
        """Wait off-loop for accepted calls and factory constructions to settle."""
        with self._lease_condition:
            while self._active_constructions or any(
                record.leases for record in self._clients.values()
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return
                self._lease_condition.wait(remaining)

    def _record_for(self, registration: ProviderRegistration) -> _ProviderClientRecord:
        """Return the stable identity record and reject ownership changes."""
        client_id = id(registration.client)
        record = self._clients.get(client_id)
        if record is None:
            record = _ProviderClientRecord(
                registration.client, registration.ownership, registration.provider
            )
            self._clients[client_id] = record
        elif record.client is not registration.client:
            raise RuntimeError("Provider client identity collision")
        elif record.ownership is not registration.ownership:
            raise ProviderOwnershipError(
                "A provider client cannot be rebound with different lifecycle ownership"
            )
        elif record.close_started or record.closed:
            raise RuntimeClosedError("Provider client cleanup has already begun")
        return record

    def _detach(self, reference: ModelReference, registration: ProviderRegistration) -> None:
        """Retire a binding while preserving any accepted-use leases."""
        record = self._clients[id(registration.client)]
        record.references.discard(reference)
        record.retired_references.add(reference)

    def _eligible_retired_record(
        self, registration: ProviderRegistration
    ) -> _ProviderClientRecord | None:
        record = self._clients.get(id(registration.client))
        if record is None or record.client is not registration.client:
            return None
        return self._eligible_retired_record_by_record(record)

    @staticmethod
    def _eligible_retired_record_by_record(
        record: _ProviderClientRecord,
    ) -> _ProviderClientRecord | None:
        if (
            record.ownership is ProviderOwnership.RUNTIME_OWNED
            and not record.references
            and not record.leases
            and not record.closed
            and not record.close_started
        ):
            return record
        return None

    def _shutdown_state(
        self,
    ) -> tuple[tuple[_ProviderClientRecord, ...], tuple[tuple[ModelReference, str], ...]]:
        """Capture clients and constructions atomically after the drain period."""
        with self._lock:
            records = tuple(
                sorted(
                    self._clients.values(),
                    key=lambda item: tuple(reference.value for reference in item.identities()),
                )
            )
            recorded_references = {
                reference for record in records for reference in record.identities()
            }
            pending = tuple(
                sorted(
                    (
                        item
                        for item in self._active_constructions.values()
                        if item[0] not in recorded_references
                    ),
                    key=lambda item: (item[0].value, item[1]),
                )
            )
            return records, pending
