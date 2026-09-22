"""High-level construction for runtime-backed inbound A2A applications."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from conducto.core.agent import BaseAgent
from conducto.core.runtime import Runtime
from conducto.transport.tasks import InMemoryTaskRepository, TaskRepository

from .runtime import A2AIdentityResolver, A2ARuntimeHandler

if TYPE_CHECKING:
    from .asgi import A2AASGI

_A2A_RPC_PATH = "/a2a"


def create_a2a_app(
    *,
    agent: BaseAgent,
    runtime: Runtime,
    public_url: str,
    identity_resolver: A2AIdentityResolver,
    task_repository: TaskRepository | None = None,
    clock: Callable[[], float] = time.time,
    max_replay_records: int = 1_000,
    max_timeout: float = 300.0,
    max_delegation_depth: int = 8,
    max_delegation_calls: int = 32,
) -> A2AASGI:
    """Build the recommended runtime-backed inbound A2A ASGI application.

    Args:
        agent: Reflected Conducto agent to publish and invoke.
        runtime: Canonical runtime that owns governed capability execution.
        public_url: Absolute HTTP(S) origin visible to remote A2A clients.
        identity_resolver: Required application-owned authentication boundary.
        task_repository: Optional task persistence implementation. An isolated
            in-memory repository is created when omitted.
        clock: UTC timestamp source used for inbound deadline conversion.
        max_replay_records: Maximum retained runtime replay records.
        max_timeout: Maximum transport-requested invocation timeout.
        max_delegation_depth: Maximum transport-requested delegation depth.
        max_delegation_calls: Maximum transport-requested delegation calls.

    Returns:
        An ASGI application suitable for an application-owned server process.

    Raises:
        ValueError: If ``public_url`` is not an unambiguous HTTP(S) origin.
        A2ADependencyError: If the optional A2A server dependencies are absent.

    Notes:
        Construction does not start a listener, event loop, thread, or
        subprocess. Every accepted capability invocation enters ``runtime``
        through :class:`A2ARuntimeHandler`.
    """
    endpoint_url = _derive_endpoint_url(public_url)
    from .asgi import A2AASGI

    handler = A2ARuntimeHandler(
        runtime=runtime,
        agent=agent,
        identity_resolver=identity_resolver,
        clock=clock,
        max_replay_records=max_replay_records,
        max_timeout=max_timeout,
        max_delegation_depth=max_delegation_depth,
        max_delegation_calls=max_delegation_calls,
    )
    repository = task_repository if task_repository is not None else InMemoryTaskRepository()

    return A2AASGI(
        agent=agent,
        endpoint_url=endpoint_url,
        task_repository=repository,
        request_handler=handler,
    )


def _derive_endpoint_url(public_url: str) -> str:
    """Derive the canonical JSON-RPC endpoint from an absolute public origin."""
    if not isinstance(public_url, str) or not public_url or public_url != public_url.strip():
        raise ValueError("public_url must be a non-empty absolute HTTP(S) origin")
    parsed = urlsplit(public_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("public_url must be an absolute HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("public_url must not contain user information")
    if parsed.hostname is None:
        raise ValueError("public_url must include a host")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("public_url contains an invalid port") from error
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("public_url must not contain a path, query, or fragment")
    return f"{parsed.scheme.lower()}://{parsed.netloc}{_A2A_RPC_PATH}"


__all__ = ["create_a2a_app"]
