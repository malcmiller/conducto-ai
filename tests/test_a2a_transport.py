"""Focused tests for A2A result mapping and task persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from a2a.types.a2a_pb2 import Task, TaskState

from conducto.a2a import invocation_result_to_task
from conducto.core.capability_errors import OutputInvariantError
from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationFailure,
    InvocationSuccess,
)
from conducto.security import ApprovalChallenge
from conducto.transport import InMemoryTaskRepository, RemoteTaskError


def test_invocation_results_map_to_sanitized_a2a_task_outcomes() -> None:
    """Success and approval outcomes have deterministic task representations."""
    success = invocation_result_to_task(
        InvocationSuccess("correlation", {"answer": 42}),
        task_id="task-1",
        context_id="context-1",
    )
    approval = invocation_result_to_task(
        InvocationApprovalRequired(
            "correlation",
            ApprovalChallenge(
                "approval-1",
                "agent",
                "capability",
                "task",
                "correlation",
                "reason",
                "role",
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=1),
            ),
        ),
        task_id="task-2",
        context_id="context-1",
    )

    assert success.status.state == TaskState.TASK_STATE_COMPLETED
    assert success.artifacts[0].parts[0].text == '{"answer": 42}'
    assert approval.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    assert approval.metadata.fields["reason"].string_value == "approval_required"


def test_a2a_failure_status_uses_typed_summary_not_exception_detail() -> None:
    """A2A status text excludes caller-supplied failure details."""
    failure = OutputInvariantError("calculate", "provider endpoint must not escape")
    task = invocation_result_to_task(
        InvocationFailure(
            "correlation",
            str(failure),
            failure,
            classification=failure.code.value,
        ),
        task_id="task-failure",
        context_id="context-1",
    )

    assert task.status.message.parts[0].text == (
        "Capability 'calculate' failed during output stage."
    )
    assert "provider endpoint" not in task.status.message.parts[0].text
    assert task.metadata.fields["reason"].string_value == "unsatisfied_output_invariant"


def test_task_repository_transitions_are_atomic_and_terminal_tasks_are_immutable() -> None:
    """A stale racing transition and terminal mutation are rejected."""

    async def run() -> None:
        repository = InMemoryTaskRepository()
        await repository.create(
            Task(
                id="task",
                context_id="context",
                status={"state": TaskState.TASK_STATE_SUBMITTED},
            )
        )
        first, second = await asyncio.gather(
            repository.compare_and_transition(
                "task",
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_WORKING,
            ),
            repository.compare_and_transition(
                "task",
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_WORKING,
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(result, RemoteTaskError) for result in (first, second)) == 1
        await repository.compare_and_transition(
            "task",
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_COMPLETED,
        )
        with pytest.raises(RemoteTaskError, match="Terminal A2A tasks are immutable"):
            await repository.compare_and_transition(
                "task",
                TaskState.TASK_STATE_COMPLETED,
                TaskState.TASK_STATE_WORKING,
            )

    asyncio.run(run())
