"""Atomic, deterministic task persistence contracts for transport adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from a2a.types.a2a_pb2 import Task, TaskState

from conducto.core.a2a_profile import validate_task_transition

from .errors import LimitExceededError, RemoteTaskError

_TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


class TaskRepository(Protocol):
    """Persistence boundary for A2A task lifecycle operations."""

    async def create(self, task: Task) -> Task:
        """Persist a new task and return an immutable-by-copy snapshot."""

    async def get(self, task_id: str) -> Task | None:
        """Return a task snapshot, if it exists."""

    async def list(
        self, *, context_id: str = "", page_size: int = 50, page_token: str = ""
    ) -> tuple[tuple[Task, ...], str]:
        """Return deterministic task snapshots and an opaque next page token."""

    async def compare_and_transition(
        self, task_id: str, expected_state: TaskState, next_state: TaskState
    ) -> Task:
        """Atomically transition a task only when its current state matches."""

    async def cancel(self, task_id: str) -> Task:
        """Atomically cancel a non-terminal task."""


def _copy(task: Task) -> Task:
    """Return an isolated protobuf copy."""
    snapshot = Task()
    snapshot.CopyFrom(task)
    return snapshot


class InMemoryTaskRepository:
    """Reference task repository with atomic transitions and bounded retention.

    Args:
        max_tasks: Maximum number of retained tasks.
        on_evict: Optional callback invoked with an evicted task snapshot.
    """

    def __init__(
        self,
        *,
        max_tasks: int = 1_000,
        on_evict: Callable[[Task], Awaitable[None] | None] | None = None,
    ) -> None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self._max_tasks = max_tasks
        self._on_evict = on_evict
        self._tasks: dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def create(self, task: Task) -> Task:
        """Persist a task once, evicting the oldest retained task when necessary."""
        if not task.id:
            raise RemoteTaskError("A2A task id is required")
        evicted: Task | None = None
        async with self._lock:
            if task.id in self._tasks:
                raise RemoteTaskError(f"A2A task already exists: {task.id}")
            if len(self._tasks) >= self._max_tasks:
                oldest_id = next(iter(self._tasks))
                evicted = self._tasks.pop(oldest_id)
            self._tasks[task.id] = _copy(task)
            created = _copy(task)
        if evicted is not None and self._on_evict is not None:
            result = self._on_evict(_copy(evicted))
            if result is not None:
                await result
        return created

    async def get(self, task_id: str) -> Task | None:
        """Return an isolated snapshot of a retained task."""
        async with self._lock:
            task = self._tasks.get(task_id)
            return _copy(task) if task is not None else None

    async def list(
        self, *, context_id: str = "", page_size: int = 50, page_token: str = ""
    ) -> tuple[tuple[Task, ...], str]:
        """List tasks in deterministic identifier order."""
        if page_size <= 0:
            raise LimitExceededError("task page_size must be positive")
        async with self._lock:
            tasks = sorted(
                (
                    _copy(task)
                    for task in self._tasks.values()
                    if not context_id or task.context_id == context_id
                ),
                key=lambda task: task.id,
            )
        start = 0
        if page_token:
            try:
                start = next(index + 1 for index, task in enumerate(tasks) if task.id == page_token)
            except StopIteration as exc:
                raise RemoteTaskError("invalid task page token") from exc
        page = tuple(tasks[start : start + page_size])
        next_token = page[-1].id if start + page_size < len(tasks) else ""
        return page, next_token

    async def compare_and_transition(
        self, task_id: str, expected_state: TaskState, next_state: TaskState
    ) -> Task:
        """Compare state then atomically transition, rejecting stale or terminal updates."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise RemoteTaskError(f"A2A task not found: {task_id}")
            if task.status.state != expected_state:
                raise RemoteTaskError("stale A2A task transition")
            current_name = TaskState.Name(expected_state)
            next_name = TaskState.Name(next_state)
            try:
                validate_task_transition(current_name, next_name)
            except ValueError as exc:
                raise RemoteTaskError(str(exc)) from exc
            if expected_state == next_state:
                raise RemoteTaskError("duplicate A2A task transition")
            task.status.state = next_state
            return _copy(task)

    async def cancel(self, task_id: str) -> Task:
        """Cancel a task unless it has reached a terminal state."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise RemoteTaskError(f"A2A task not found: {task_id}")
            if task.status.state in _TERMINAL_STATES:
                raise RemoteTaskError("terminal A2A tasks cannot be canceled")
            task.status.state = TaskState.TASK_STATE_CANCELED
            return _copy(task)
