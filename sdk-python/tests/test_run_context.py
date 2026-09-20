"""Focused contracts for task-local run contexts."""

import asyncio

from conducto import Runtime, get_run_context
from conducto.core.run_context import RunContext as ContextRunContext
from conducto.core.runtime import RunContext, use_run_context


def test_nested_run_contexts_restore_the_parent_context() -> None:
    parent = RunContext(run_id="parent", correlation_id="parent")
    child = RunContext(run_id="child", correlation_id="child")

    assert RunContext is ContextRunContext
    with use_run_context(parent):
        assert get_run_context() is parent
        with use_run_context(child):
            assert get_run_context() is child
        assert get_run_context() is parent
    assert get_run_context() is None


def test_runtime_activation_is_isolated_between_asyncio_tasks() -> None:
    async def observe(label: str) -> str:
        context = Runtime().create_run_context(agent_id=label, correlation_id=label)
        token = Runtime.activate(context)
        try:
            await asyncio.sleep(0)
            active = get_run_context()
            assert active is not None
            return active.correlation_id
        finally:
            Runtime.deactivate(token)

    async def exercise() -> None:
        assert await asyncio.gather(observe("first"), observe("second")) == [
            "first",
            "second",
        ]

    asyncio.run(exercise())
