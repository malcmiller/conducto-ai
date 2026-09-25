"""High-level construction for runtime-backed inbound A2A applications."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from conducto.core.agent import BaseAgent
from conducto.core.data_sources import DataSourceRegistry
from conducto.core.runtime import Runtime
from conducto.transport.tasks import InMemoryTaskRepository, TaskRepository

from .hardening import A2AHostSecurityConfig, A2AMTLSIdentityExtractor
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
    data_sources: DataSourceRegistry | None = None,
    endpoint_path: str = _A2A_RPC_PATH,
    task_repository: TaskRepository | None = None,
    clock: Callable[[], float] = time.time,
    max_timeout: float = 300.0,
    max_delegation_depth: int = 8,
    max_delegation_calls: int = 32,
    security_config: A2AHostSecurityConfig | None = None,
    mtls_identity_extractor: A2AMTLSIdentityExtractor | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    on_startup: Callable[[], Awaitable[None] | None] | None = None,
    resource_closers: Mapping[str, Callable[[], Awaitable[None] | None]] | None = None,
) -> A2AASGI:
    """Build the recommended runtime-backed inbound A2A ASGI application.

    Args:
        agent: Reflected Conducto agent to publish and invoke.
        runtime: Canonical runtime that owns governed capability execution.
        public_url: Absolute HTTP(S) origin visible to remote A2A clients.
        endpoint_path: Exact path used for the JSON-RPC endpoint.
        identity_resolver: Required application-owned authentication boundary.
        data_sources: Optional configured source metadata used to register the
            hosted agent's declared dependencies.
        task_repository: Optional task persistence implementation. An isolated
            in-memory repository is created when omitted.
        clock: UTC timestamp source used for inbound deadline conversion.
        max_timeout: Maximum transport-requested invocation timeout.
        max_delegation_depth: Maximum transport-requested delegation depth.
        max_delegation_calls: Maximum transport-requested delegation calls.
        security_config: Immutable transport hardening policy. A bounded,
            permissive-authority default is used when omitted.
        mtls_identity_extractor: Optional server-owned seam supplying verified
            mTLS peer identity; peer identity is never read from a header.
        monotonic_clock: Monotonic source budgeting bounded shutdown phases.
        on_startup: Optional application-owned dependency check. When supplied,
            the host stays unready until ASGI lifespan startup succeeds.
        resource_closers: Named cleanup callables invoked during bounded close.

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
    endpoint_url = _derive_endpoint_url(public_url, endpoint_path)
    from .asgi import A2AASGI

    handler = A2ARuntimeHandler(
        runtime=runtime,
        agent=agent,
        identity_resolver=identity_resolver,
        data_sources=data_sources,
        clock=clock,
        max_timeout=max_timeout,
        max_delegation_depth=max_delegation_depth,
        max_delegation_calls=max_delegation_calls,
    )
    config = security_config or A2AHostSecurityConfig()
    repository = (
        task_repository
        if task_repository is not None
        else InMemoryTaskRepository(max_tasks=config.max_retained_tasks)
    )

    return A2AASGI(
        agent=agent,
        endpoint_url=endpoint_url,
        task_repository=repository,
        request_handler=handler,
        security_config=config,
        mtls_identity_extractor=mtls_identity_extractor,
        clock=monotonic_clock,
        on_startup=on_startup,
        resource_closers=resource_closers,
    )


def _derive_endpoint_url(public_url: str, endpoint_path: str) -> str:
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
    if (
        not isinstance(endpoint_path, str)
        or not endpoint_path.startswith("/")
        or endpoint_path != endpoint_path.strip()
        or "?" in endpoint_path
        or "#" in endpoint_path
        or endpoint_path == "/"
    ):
        raise ValueError("endpoint_path must be a non-root absolute path")
    return f"{parsed.scheme.lower()}://{parsed.netloc}{endpoint_path}"


__all__ = ["create_a2a_app"]
