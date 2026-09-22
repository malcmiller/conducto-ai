"""Operational lifecycle, readiness, concurrency, and bounded shutdown for A2A hosts.

This module owns liveness, readiness, drain, and idempotent close semantics for
one inbound A2A ASGI application plus the non-queuing concurrency gates that keep
overload deterministic. It never performs protocol adaptation or capability
execution; those remain with the Story 4.4 protocol adapter and the runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import Any, Final

from a2a.types.a2a_pb2 import Task, TaskState

from conducto.core.logging import emit_event
from conducto.transport.errors import LimitExceededError, RemoteTaskError
from conducto.transport.tasks import TaskRepository

from .errors import A2AShutdownError, A2AStartupError
from .hardening import A2AHostSecurityConfig

_ACTIVE_TASK_STATES: Final = frozenset(
    {TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING}
)


class A2AHostState(Enum):
    """Observable operational state of one inbound A2A ASGI application."""

    CREATED = "created"
    READY = "ready"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


class A2AConcurrencyLimiter:
    """Non-queuing bounded concurrency gate with an optional per-key bound.

    Args:
        limit: Maximum number of simultaneously held slots.
        per_key_limit: Optional maximum simultaneously held slots per key.

    Notes:
        Acquisition never blocks and never queues. Over-capacity callers are
        rejected immediately so overload is deterministic and partial execution
        cannot begin.
    """

    __slots__ = ("_held", "_limit", "_per_key", "_per_key_limit")

    def __init__(self, *, limit: int, per_key_limit: int | None = None) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if per_key_limit is not None and per_key_limit <= 0:
            raise ValueError("per_key_limit must be positive")
        self._limit = limit
        self._per_key_limit = per_key_limit
        self._held = 0
        self._per_key: dict[str, int] = {}

    @property
    def in_flight(self) -> int:
        """Return the number of currently held slots."""
        return self._held

    def try_acquire(self, key: str = "") -> bool:
        """Acquire one slot without blocking.

        Args:
            key: Caller partition used for the per-key bound.

        Returns:
            ``True`` when a slot was acquired, ``False`` when at capacity.
        """
        if self._held >= self._limit:
            return False
        if self._per_key_limit is not None and self._per_key.get(key, 0) >= self._per_key_limit:
            return False
        self._held += 1
        if self._per_key_limit is not None:
            self._per_key[key] = self._per_key.get(key, 0) + 1
        return True

    def release(self, key: str = "") -> None:
        """Release one previously acquired slot.

        Args:
            key: Caller partition supplied to the matching :meth:`try_acquire`.
        """
        if self._held > 0:
            self._held -= 1
        if self._per_key_limit is None:
            return
        remaining = self._per_key.get(key, 0) - 1
        if remaining > 0:
            self._per_key[key] = remaining
        else:
            self._per_key.pop(key, None)


class A2AHostLifecycle:
    """Startup, readiness, drain, and idempotent close for one A2A ASGI host.

    Args:
        config: Immutable hardening policy that owns every lifecycle deadline.
        task_repository: Task persistence boundary swept during bounded shutdown.
        clock: Monotonic timestamp source used to budget bounded shutdown phases.
        on_startup: Optional application-owned dependency check run at startup.
        resource_closers: Named cleanup callables invoked during bounded close.

    Notes:
        Liveness never reveals dependency details. Readiness becomes ``False``
        before drain rejects new work, and :meth:`aclose` is idempotent: a repeat
        call performs no further work and raises nothing.
    """

    def __init__(
        self,
        *,
        config: A2AHostSecurityConfig,
        task_repository: TaskRepository,
        clock: Callable[[], float] = time.monotonic,
        on_startup: Callable[[], Awaitable[None] | None] | None = None,
        resource_closers: Mapping[str, Callable[[], Awaitable[None] | None]] | None = None,
    ) -> None:
        self._config = config
        self._task_repository = task_repository
        self._clock = clock
        self._on_startup = on_startup
        self._resource_closers = dict(resource_closers or {})
        self._state = A2AHostState.CREATED
        self._inflight: set[asyncio.Task[None]] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def mark_ready(self) -> None:
        """Mark a host with no startup dependency check ready without lifespan.

        Notes:
            Hosts constructed without an ``on_startup`` check have nothing to
            verify, so they may serve immediately when the embedding server does
            not run the ASGI lifespan protocol. A closed or failed host is never
            silently revived.
        """
        if self._state is A2AHostState.CREATED:
            self._state = A2AHostState.READY

    @property
    def state(self) -> A2AHostState:
        """Return the current observable lifecycle state."""
        return self._state

    @property
    def in_flight(self) -> int:
        """Return the number of accepted requests still executing."""
        return len(self._inflight)

    def is_alive(self) -> bool:
        """Return whether the process-level host object can still serve.

        Returns:
            ``True`` until the host is closed. No dependency, endpoint, or
            credential detail is exposed by this signal.
        """
        return self._state is not A2AHostState.CLOSED

    def is_ready(self) -> bool:
        """Return whether the host can accept new work right now."""
        return self._state is A2AHostState.READY

    async def startup(self) -> None:
        """Run the bounded application startup check and become ready.

        Raises:
            A2AStartupError: If the host is closed, draining, or failed, or if
                the startup check fails or exceeds its deadline. The host stays
                unready so no work is accepted; a draining or failed host is
                never silently revived to ready.
        """
        if self._state in (A2AHostState.CLOSED, A2AHostState.DRAINING, A2AHostState.FAILED):
            raise A2AStartupError(f"A2A host is {self._state.value}", reason=self._state.value)
        if self._on_startup is not None:
            try:
                await asyncio.wait_for(
                    _maybe_await(self._on_startup()),
                    timeout=self._config.startup_deadline_seconds,
                )
            except TimeoutError as error:
                self._fail_startup("startup_timeout")
                raise A2AStartupError(
                    "A2A host startup did not complete", reason="startup_timeout"
                ) from error
            except Exception as error:  # noqa: BLE001 - mapped to a typed startup failure
                self._fail_startup("startup_failed")
                raise A2AStartupError(
                    "A2A host startup did not complete", reason="startup_failed"
                ) from error
        self._state = A2AHostState.READY
        emit_event("a2a.host.startup", outcome="success")

    def _fail_startup(self, reason: str) -> None:
        """Record a failed startup so the host never becomes ready."""
        self._state = A2AHostState.FAILED
        emit_event("a2a.host.startup", outcome="failure", error_category=reason)

    def track(self, execution: asyncio.Task[None]) -> None:
        """Register one accepted request execution for drain accounting."""
        self._inflight.add(execution)
        self._idle.clear()

    def untrack(self, execution: asyncio.Task[None]) -> None:
        """Release one accepted request execution from drain accounting."""
        self._inflight.discard(execution)
        if not self._inflight:
            self._idle.set()

    async def drain(self) -> None:
        """Become unready, reject new work, and bound in-flight completion.

        Notes:
            Readiness flips to ``False`` before the wait begins, so a load
            balancer stops sending work before any accepted request is disturbed.
            Work still running after the drain grace period is cancelled and
            bounded by ``cancellation_deadline_seconds`` (not a second full
            ``drain_deadline_seconds`` wait), and every accepted task is driven
            out of an ambiguous active state.
        """
        if self._state is A2AHostState.CLOSED:
            return
        self._state = A2AHostState.DRAINING
        expired = not await self._await_idle(self._config.drain_deadline_seconds)
        if expired:
            await self._cancel_inflight(self._config.cancellation_deadline_seconds)
            await self._sweep_active_tasks()
        emit_event(
            "a2a.host.drain",
            outcome="timeout" if expired else "success",
            error_category="drain_grace_expired" if expired else None,
        )

    async def aclose(self) -> None:
        """Close the host once, bounding every cleanup step.

        Raises:
            A2AShutdownError: If one or more bounded cleanup steps failed. Every
                failing step is reported by a stable reason code; no exception
                text, credential, or endpoint detail is exposed.
        """
        if self._state is A2AHostState.CLOSED:
            return
        self._state = A2AHostState.CLOSED
        reasons: list[str] = []
        if not await self._cancel_inflight(self._config.shutdown_deadline_seconds):
            reasons.append("inflight_cancellation_timeout")
        if not await self._sweep_active_tasks():
            reasons.append("task_store_flush_failed")
        for name, closer in self._resource_closers.items():
            try:
                await asyncio.wait_for(
                    _maybe_await(closer()),
                    timeout=self._config.shutdown_deadline_seconds,
                )
            except TimeoutError:
                reasons.append(f"{name}_timeout")
            except Exception:  # noqa: BLE001 - reported as an explicit reason code
                reasons.append(f"{name}_failed")
        emit_event(
            "a2a.host.close",
            outcome="failure" if reasons else "success",
            error_category=reasons[0] if reasons else None,
        )
        if reasons:
            raise A2AShutdownError("A2A host cleanup did not fully succeed", reasons=tuple(reasons))

    async def _await_idle(self, timeout: float) -> bool:
        """Wait until no accepted request remains, bounded by ``timeout``."""
        if not self._inflight:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def _cancel_inflight(self, timeout: float) -> bool:
        """Cancel every remaining accepted request within a bounded budget."""
        pending = {execution for execution in self._inflight if not execution.done()}
        if not pending:
            return True
        for execution in pending:
            execution.cancel()
        _, unfinished = await asyncio.wait(pending, timeout=timeout)
        return not unfinished

    async def _sweep_active_tasks(self) -> bool:
        """Drive every retained non-terminal task to a documented terminal state."""
        deadline = self._clock() + self._config.task_store_flush_deadline_seconds
        try:
            await asyncio.wait_for(
                self._sweep(deadline),
                timeout=self._config.task_store_flush_deadline_seconds,
            )
        except TimeoutError:
            return False
        except (LimitExceededError, RemoteTaskError):
            return False
        return True

    async def _sweep(self, deadline: float) -> None:
        """Scan retained tasks once and fail every ambiguous active task."""
        page_token = ""
        scanned = 0
        while scanned < self._config.max_retained_tasks and self._clock() < deadline:
            tasks, page_token = await self._task_repository.list(
                page_size=self._config.max_page_size,
                page_token=page_token,
            )
            scanned += len(tasks)
            for task in tasks:
                await self._fail_if_active(task)
            if not page_token:
                break

    async def _fail_if_active(self, task: Task) -> None:
        """Transition one ambiguous active task to the terminal failed state."""
        if task.status.state not in _ACTIVE_TASK_STATES:
            return
        with contextlib.suppress(RemoteTaskError):
            await self._task_repository.compare_and_transition(
                task.id, task.status.state, TaskState.TASK_STATE_FAILED
            )


async def _maybe_await(result: Awaitable[Any] | Any) -> None:
    """Await a callable result only when it is awaitable.

    Notes:
        Detects any object implementing ``__await__`` (coroutines, futures, and
        custom awaitables returned by an injected ``on_startup`` check or
        resource closer), not only coroutine objects and :class:`asyncio.Future`
        instances, so a custom awaitable is never treated as already complete.
    """
    if hasattr(result, "__await__"):
        await result


__all__ = ["A2AConcurrencyLimiter", "A2AHostLifecycle", "A2AHostState"]
