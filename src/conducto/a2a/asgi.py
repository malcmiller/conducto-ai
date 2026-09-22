"""Optional ASGI adapter hosting one Conducto agent's inbound A2A 1.0 surface.

This module owns HTTP/ASGI protocol adaptation, exact route matching, Agent Card
and mounted endpoint consistency, official SDK JSON-RPC dispatch, protocol parsing,
protocol errors, task repository wiring, and the hardened ASGI lifespan, admission,
and drain boundary. It does not own Conducto authorization, approvals, audit, model
resolution, capability invocation, authentication, or process hosting; those remain
with the runtime, the application-owned identity resolver, and the server process
that mounts this adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from google.protobuf.json_format import MessageToDict

from conducto.core.a2a_profile import (
    TERMINAL_TASK_STATES,
    A2AProtocolError,
    parse_agent_card,
    parse_message,
)
from conducto.core.agent import BaseAgent
from conducto.core.invocation_results import InvocationResult
from conducto.transport.errors import LimitExceededError, RemoteTaskError
from conducto.transport.tasks import TaskRepository

from . import invocation_result_to_task
from .errors import (
    A2ADependencyError,
    A2AHostConfigurationError,
    A2APayloadLimitError,
    A2AShutdownError,
    A2AStartupError,
)
from .handler import (
    A2ACancellableRequestHandler,
    A2AContextRequestHandler,
    A2ARequestContext,
    A2ARequestHandler,
)
from .hardening import (
    A2AHostSecurityConfig,
    A2AMTLSIdentityExtractor,
    enforce_metadata_limit,
)
from .lifecycle import A2AConcurrencyLimiter, A2AHostLifecycle, A2AHostState
from .profile import A2A_SERVER_EXTRA, require_a2a_server_dependency

try:
    from a2a.server.context import ServerCallContext
    from a2a.server.events.event_queue import Event
    from a2a.server.id_generator import IDGenerator, IDGeneratorContext, UUIDGenerator
    from a2a.server.request_handlers.request_handler import (
        RequestHandler,
        validate_request_params,
    )
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.types.a2a_pb2 import (
        AgentCard,
        CancelTaskRequest,
        DeleteTaskPushNotificationConfigRequest,
        GetExtendedAgentCardRequest,
        GetTaskPushNotificationConfigRequest,
        GetTaskRequest,
        ListTaskPushNotificationConfigsRequest,
        ListTaskPushNotificationConfigsResponse,
        ListTasksRequest,
        ListTasksResponse,
        Message,
        SendMessageRequest,
        SubscribeToTaskRequest,
        Task,
        TaskPushNotificationConfig,
        TaskState,
    )
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
    from a2a.utils.errors import (
        ContentTypeNotSupportedError,
        ExtensionSupportRequiredError,
        InternalError,
        InvalidParamsError,
        InvalidRequestError,
        TaskNotCancelableError,
        TaskNotFoundError,
        UnsupportedOperationError,
    )
    from starlette.applications import Starlette
    from starlette.types import Receive, Scope, Send

    from .guard import A2ARequestGuard
except ImportError as error:  # pragma: no cover - exercised without the extra installed
    raise A2ADependencyError(
        f"The A2A ASGI server adapter requires Starlette; install conducto-ai[{A2A_SERVER_EXTRA}]",
        extra=A2A_SERVER_EXTRA,
    ) from error

require_a2a_server_dependency()

_DEFAULT_PAGE_SIZE = 50


class _ConductoRequestHandler(RequestHandler):
    """Official-SDK request handler bound to Conducto's repository and seam.

    This class implements the official A2A SDK's abstract dispatch interface. It
    never invokes a reflected agent's capability methods directly; every accepted
    message is delegated to the injected :class:`A2ARequestHandler` seam.
    """

    def __init__(
        self,
        *,
        agent_card: AgentCard,
        task_repository: TaskRepository,
        request_handler: A2ARequestHandler | A2AContextRequestHandler,
        id_generator: IDGenerator,
        config: A2AHostSecurityConfig,
        capability_limiter: A2AConcurrencyLimiter,
    ) -> None:
        self._agent_card = agent_card
        self._task_repository = task_repository
        self._request_handler = request_handler
        self._accepts_request_context = (
            "request_context" in inspect.signature(request_handler.handle_message).parameters
        )
        self._id_generator = id_generator
        self._config = config
        self._capability_limiter = capability_limiter
        self._supported_extensions = frozenset(
            extension.uri for extension in agent_card.capabilities.extensions
        )

    def _reject_unsupported_extensions(self, context: ServerCallContext) -> None:
        """Reject requests that require an extension this profile does not support."""
        unsupported = context.requested_extensions - self._supported_extensions
        if unsupported:
            raise ExtensionSupportRequiredError(
                f"Unsupported required A2A extensions: {sorted(unsupported)}"
            )

    @validate_request_params
    async def on_get_task(self, params: GetTaskRequest, context: ServerCallContext) -> Task | None:
        """Return a persisted task snapshot, or ``None`` when it does not exist."""
        self._reject_unsupported_extensions(context)
        return await self._task_repository.get(params.id)

    @validate_request_params
    async def on_list_tasks(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        """Return one deterministic page of tasks from the injected repository."""
        self._reject_unsupported_extensions(context)
        if params.page_size > self._config.max_page_size:
            raise InvalidParamsError(
                f"page_size exceeds the configured limit {self._config.max_page_size}"
            )
        page_size = params.page_size if params.page_size > 0 else self._config.max_page_size
        try:
            tasks, next_token = await self._task_repository.list(
                context_id=params.context_id,
                page_size=page_size,
                page_token=params.page_token,
            )
        except (LimitExceededError, RemoteTaskError) as exc:
            raise InvalidParamsError(str(exc)) from exc
        response = ListTasksResponse(next_page_token=next_token, page_size=len(tasks))
        response.tasks.extend(tasks)
        return response

    @validate_request_params
    async def on_cancel_task(
        self, params: CancelTaskRequest, context: ServerCallContext
    ) -> Task | None:
        """Cancel a non-terminal task through the injected repository."""
        self._reject_unsupported_extensions(context)
        try:
            if isinstance(self._request_handler, A2ACancellableRequestHandler):
                await self._request_handler.cancel(params.id)
            return await self._task_repository.cancel(params.id)
        except RemoteTaskError as exc:
            text = str(exc)
            if "not found" in text:
                return None
            raise TaskNotCancelableError(text) from exc

    @validate_request_params
    async def on_message_send(self, params: SendMessageRequest, context: ServerCallContext) -> Task:
        """Create or continue one task and invoke the injected request-handler seam."""
        if not self._capability_limiter.try_acquire():
            raise InternalError("A2A capability concurrency limit reached")
        try:
            return await self._send(params, context)
        finally:
            self._capability_limiter.release()

    async def _send(self, params: SendMessageRequest, context: ServerCallContext) -> Task:
        """Execute one admitted message send under the configured payload limits."""
        self._reject_unsupported_extensions(context)
        if context.state.get("request_id") is None:
            raise InvalidRequestError("SendMessage requires a JSON-RPC request identifier")
        message = params.message
        payload = MessageToDict(message, preserving_proto_field_name=False)
        try:
            parse_message(payload)
            self._enforce_message_limits(message, payload)
        except A2AProtocolError as exc:
            text = str(exc)
            if "media type" in text.lower():
                raise ContentTypeNotSupportedError(text) from exc
            raise InvalidParamsError(text) from exc
        except A2APayloadLimitError as exc:
            raise InvalidParamsError(exc.reason) from exc

        if message.task_id:
            task_id = message.task_id
            existing = await self._task_repository.get(task_id)
            if existing is None:
                raise TaskNotFoundError(f"A2A task not found: {task_id}")
            if TaskState.Name(existing.status.state) in TERMINAL_TASK_STATES:
                raise UnsupportedOperationError(
                    f"Task {task_id} is in terminal state {TaskState.Name(existing.status.state)}"
                )
            if len(existing.history) > self._config.max_history_messages:
                raise InvalidRequestError("history_messages_exceeded")
            if message.context_id and message.context_id != existing.context_id:
                raise InvalidParamsError(
                    f"context_id {message.context_id!r} does not match the stored "
                    f"context {existing.context_id!r} for task {task_id}"
                )
            context_id = existing.context_id
            current_state = existing.status.state
        else:
            context_id = message.context_id or self._id_generator.generate(IDGeneratorContext())
            task_id = self._id_generator.generate(IDGeneratorContext(context_id=context_id))
            new_task = Task(id=task_id, context_id=context_id)
            new_task.status.state = TaskState.TASK_STATE_SUBMITTED
            new_task.history.append(message)
            try:
                await self._task_repository.create(new_task)
            except RemoteTaskError as exc:
                raise InvalidRequestError(str(exc)) from exc
            current_state = TaskState.TASK_STATE_SUBMITTED

        # Atomically claim the task before invoking the handler seam so two
        # concurrent continuations of the same task can never both observe the
        # same non-terminal state and both invoke the handler: the loser of the
        # race below fails fast with a typed error instead of racing to persist.
        try:
            claimed = await self._task_repository.compare_and_transition(
                task_id, current_state, TaskState.TASK_STATE_WORKING
            )
        except RemoteTaskError as exc:
            text = str(exc)
            if "not found" in text:
                raise TaskNotFoundError(text) from exc
            raise InvalidRequestError(text) from exc
        current_state = TaskState.TASK_STATE_WORKING

        try:
            result = await asyncio.wait_for(
                self._dispatch(message, task_id=task_id, context_id=context_id, context=context),
                timeout=self._config.capability_deadline_seconds,
            )
        except asyncio.CancelledError:
            if isinstance(self._request_handler, A2ACancellableRequestHandler):
                await self._request_handler.cancel(task_id)
            with contextlib.suppress(RemoteTaskError):
                await self._task_repository.compare_and_transition(
                    task_id, current_state, TaskState.TASK_STATE_CANCELED
                )
            raise
        except TimeoutError as exc:
            if isinstance(self._request_handler, A2ACancellableRequestHandler):
                await self._request_handler.cancel(task_id)
            with contextlib.suppress(RemoteTaskError):
                await self._task_repository.compare_and_transition(
                    task_id, current_state, TaskState.TASK_STATE_FAILED
                )
            raise InternalError("A2A capability deadline exceeded") from exc
        except Exception as exc:  # noqa: BLE001 - mapped to a typed A2A internal error
            with contextlib.suppress(RemoteTaskError):
                await self._task_repository.compare_and_transition(
                    task_id, current_state, TaskState.TASK_STATE_FAILED
                )
            raise InternalError("A2A request handler failed") from exc

        outcome = invocation_result_to_task(result, task_id=task_id, context_id=context_id)
        final_task = Task()
        final_task.CopyFrom(claimed)
        final_task.status.CopyFrom(outcome.status)
        final_task.artifacts.extend(outcome.artifacts)
        if len(outcome.metadata):
            final_task.metadata.update(dict(outcome.metadata.items()))
        try:
            self._enforce_task_limits(final_task)
        except A2APayloadLimitError as exc:
            with contextlib.suppress(RemoteTaskError):
                await self._task_repository.compare_and_transition(
                    task_id, current_state, TaskState.TASK_STATE_FAILED
                )
            raise InternalError(exc.reason) from exc
        try:
            return await self._task_repository.compare_and_update(
                task_id, current_state, final_task
            )
        except RemoteTaskError as exc:
            latest = await self._task_repository.get(task_id)
            if latest is not None and latest.status.state == TaskState.TASK_STATE_CANCELED:
                return latest
            text = str(exc)
            if "not found" in text:
                raise TaskNotFoundError(text) from exc
            raise InternalError(text) from exc

    async def _dispatch(
        self,
        message: Message,
        *,
        task_id: str,
        context_id: str,
        context: ServerCallContext,
    ) -> InvocationResult:
        """Invoke the injected handler seam with or without transport context."""
        if self._accepts_request_context:
            context_handler = self._request_handler
            assert isinstance(context_handler, A2AContextRequestHandler)
            return await context_handler.handle_message(
                message,
                task_id=task_id,
                context_id=context_id,
                request_context=A2ARequestContext(
                    request_id=str(context.state["request_id"]),
                    headers=context.state.get("headers", {}),
                    method=str(context.state.get("method", "SendMessage")),
                ),
            )
        legacy_handler = self._request_handler
        assert isinstance(legacy_handler, A2ARequestHandler)
        return await legacy_handler.handle_message(
            message,
            task_id=task_id,
            context_id=context_id,
        )

    def _enforce_message_limits(self, message: Message, payload: Mapping[str, Any]) -> None:
        """Reject an inbound message that exceeds this host's configured bounds.

        Args:
            message: Parsed inbound A2A message.
            payload: JSON representation used for metadata size accounting.

        Raises:
            A2APayloadLimitError: If a configured message bound is exceeded.
        """
        if len(message.parts) > self._config.max_message_parts:
            raise A2APayloadLimitError("message_parts_exceeded")
        enforce_metadata_limit(payload.get("metadata"), self._config)

    def _enforce_task_limits(self, task: Task) -> None:
        """Reject an outbound task snapshot that exceeds this host's bounds.

        Args:
            task: Task snapshot about to be persisted and returned.

        Raises:
            A2APayloadLimitError: If a configured task bound is exceeded.
        """
        if len(task.artifacts) > self._config.max_task_artifacts:
            raise A2APayloadLimitError("task_artifacts_exceeded")
        if len(task.history) > self._config.max_history_messages:
            raise A2APayloadLimitError("history_messages_exceeded")
        for artifact in task.artifacts:
            if len(artifact.parts) > self._config.max_artifact_parts:
                raise A2APayloadLimitError("artifact_parts_exceeded")

    async def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        """Reject streaming message send; this profile pins non-streaming operations."""
        raise UnsupportedOperationError("Streaming message send is not supported")
        yield  # pragma: no cover - required so this stays an async generator function

    async def on_create_task_push_notification_config(
        self, params: TaskPushNotificationConfig, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        """Reject push notification configuration; this profile omits push notifications."""
        raise UnsupportedOperationError("Push notification configuration is not supported")

    async def on_get_task_push_notification_config(
        self, params: GetTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> TaskPushNotificationConfig:
        """Reject push notification configuration; this profile omits push notifications."""
        raise UnsupportedOperationError("Push notification configuration is not supported")

    async def on_subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        """Reject task subscription; this profile pins non-streaming operations."""
        raise UnsupportedOperationError("Task subscription is not supported")
        yield  # pragma: no cover - required so this stays an async generator function

    async def on_list_task_push_notification_configs(
        self, params: ListTaskPushNotificationConfigsRequest, context: ServerCallContext
    ) -> ListTaskPushNotificationConfigsResponse:
        """Reject push notification configuration; this profile omits push notifications."""
        raise UnsupportedOperationError("Push notification configuration is not supported")

    async def on_delete_task_push_notification_config(
        self, params: DeleteTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> None:
        """Reject push notification configuration; this profile omits push notifications."""
        raise UnsupportedOperationError("Push notification configuration is not supported")

    async def on_get_extended_agent_card(
        self, params: GetExtendedAgentCardRequest, context: ServerCallContext
    ) -> AgentCard:
        """Reject extended agent cards; this profile publishes only the public card."""
        raise UnsupportedOperationError("Extended Agent Cards are not supported")


class A2AASGI:
    """Hardened ASGI callable hosting one Conducto agent's inbound A2A 1.0 surface.

    Args:
        agent: The reflected Conducto agent to publish.
        endpoint_url: Absolute URL advertised as the agent's JSON-RPC endpoint.
        task_repository: Injected task persistence boundary.
        request_handler: Typed seam invoked once per accepted inbound message.
        id_generator: Optional generator for new task and context identifiers.
        agent_card_kwargs: Optional keyword arguments forwarded to ``get_agent_card``.
        security_config: Immutable transport hardening policy. A bounded,
            permissive-authority default is used when omitted.
        mtls_identity_extractor: Optional server-owned seam supplying verified
            mTLS peer identity. Peer identity is never read from a request header.
        clock: Monotonic timestamp source used to budget bounded shutdown phases.
        on_startup: Optional application-owned dependency check. When supplied,
            the host stays unready until ASGI lifespan startup succeeds.
        resource_closers: Named cleanup callables invoked during bounded close.

    Attributes:
        agent_card: Parsed A2A 1.0 Agent Card published by this host.
        endpoint_url: Absolute JSON-RPC endpoint advertised by the card.
        security_config: Effective hardening policy, including derived paths.

    Raises:
        ValueError: If the reflected Agent Card is not a valid A2A 1.0 profile.
        A2AHostConfigurationError: If the configured paths exclude the mounted
            JSON-RPC endpoint or Agent Card route.

    Notes:
        Constructing this adapter never starts a listener, event loop, thread, or
        subprocess; call the instance as an ASGI application from an existing
        server process. Two instances never share task, concurrency, or lifecycle
        state, and no mutable configuration is shared through module globals.
    """

    def __init__(
        self,
        *,
        agent: BaseAgent,
        endpoint_url: str,
        task_repository: TaskRepository,
        request_handler: A2ARequestHandler | A2AContextRequestHandler,
        id_generator: IDGenerator | None = None,
        agent_card_kwargs: Mapping[str, Any] | None = None,
        security_config: A2AHostSecurityConfig | None = None,
        mtls_identity_extractor: A2AMTLSIdentityExtractor | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_startup: Callable[[], Awaitable[None] | None] | None = None,
        resource_closers: Mapping[str, Callable[[], Awaitable[None] | None]] | None = None,
    ) -> None:
        card_payload = agent.get_agent_card(endpoint_url, **dict(agent_card_kwargs or {}))
        try:
            card = parse_agent_card(card_payload)
        except A2AProtocolError as exc:
            raise ValueError(f"Agent Card is not a valid A2A 1.0 profile: {exc}") from exc
        rpc_path = urlparse(endpoint_url).path or "/"
        config = _resolve_paths(security_config or A2AHostSecurityConfig(), rpc_path=rpc_path)
        capability_limiter = A2AConcurrencyLimiter(limit=config.max_capability_concurrency)
        dispatch_handler = _ConductoRequestHandler(
            agent_card=card,
            task_repository=task_repository,
            request_handler=request_handler,
            id_generator=id_generator or UUIDGenerator(),
            config=config,
            capability_limiter=capability_limiter,
        )
        routes = [
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(dispatch_handler, rpc_url=rpc_path),
        ]
        self._lifecycle = A2AHostLifecycle(
            config=config,
            task_repository=task_repository,
            clock=clock,
            on_startup=on_startup,
            resource_closers=resource_closers,
        )
        if on_startup is None:
            self._lifecycle.mark_ready()
        self._app = Starlette(routes=routes)
        self._guard = A2ARequestGuard(
            app=self._app,
            config=config,
            lifecycle=self._lifecycle,
            limiter=A2AConcurrencyLimiter(
                limit=config.max_accepted_concurrency,
                per_key_limit=config.max_caller_concurrency,
            ),
            mtls_identity_extractor=mtls_identity_extractor,
        )
        self.agent_card = card
        self.endpoint_url = endpoint_url
        self.security_config = config

    @property
    def lifecycle_state(self) -> A2AHostState:
        """Return the current observable operational state of this host."""
        return self._lifecycle.state

    def is_alive(self) -> bool:
        """Return whether this host object can still serve, revealing no details."""
        return self._lifecycle.is_alive()

    def is_ready(self) -> bool:
        """Return whether this host can accept new work right now."""
        return self._lifecycle.is_ready()

    async def startup(self) -> None:
        """Run the bounded startup check and become ready.

        Raises:
            A2AStartupError: If the startup check fails or exceeds its deadline.
        """
        await self._lifecycle.startup()

    async def drain(self) -> None:
        """Become unready, reject new work, and bound accepted-work completion."""
        await self._lifecycle.drain()

    async def aclose(self) -> None:
        """Close this host idempotently with bounded cleanup.

        Raises:
            A2AShutdownError: If one or more bounded cleanup steps failed.
        """
        await self._lifecycle.aclose()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one ASGI scope through the hardening boundary or lifespan protocol."""
        kind = scope.get("type")
        if kind == "lifespan":
            await self._lifespan(receive, send)
            return
        if kind != "http":
            await self._reject_unsupported_scope(receive, send)
            return
        await self._guard(scope, receive, send)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        """Implement ASGI lifespan startup and bounded drain-then-close shutdown."""
        while True:
            message = await receive()
            kind = message.get("type")
            if kind == "lifespan.startup":
                try:
                    await self._lifecycle.startup()
                except A2AStartupError as error:
                    await send({"type": "lifespan.startup.failed", "message": error.reason})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif kind == "lifespan.shutdown":
                try:
                    await self._lifecycle.drain()
                    await self._lifecycle.aclose()
                except A2AShutdownError as error:
                    await send(
                        {"type": "lifespan.shutdown.failed", "message": ",".join(error.reasons)}
                    )
                    return
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _reject_unsupported_scope(self, receive: Receive, send: Send) -> None:
        """Close any non-HTTP connection; this profile pins JSON-RPC over HTTP."""
        message = await receive()
        if message.get("type") == "websocket.connect":
            await send({"type": "websocket.close", "code": 1008})


def _resolve_paths(config: A2AHostSecurityConfig, *, rpc_path: str) -> A2AHostSecurityConfig:
    """Derive or validate the exact route surface this host is allowed to serve.

    Args:
        config: Caller-supplied hardening policy.
        rpc_path: Path of the mounted JSON-RPC endpoint.

    Returns:
        A policy whose ``allowed_paths`` covers the mounted routes and probes.

    Raises:
        A2AHostConfigurationError: If explicit paths omit a mounted route.
    """
    mounted = {rpc_path, AGENT_CARD_WELL_KNOWN_PATH}
    probes = {path for path in (config.liveness_path, config.readiness_path) if path is not None}
    if not config.allowed_paths:
        return replace(config, allowed_paths=frozenset(mounted | probes))
    missing = mounted - config.allowed_paths
    if missing:
        raise A2AHostConfigurationError(f"allowed_paths must include {sorted(missing)}")
    return replace(config, allowed_paths=frozenset(config.allowed_paths | probes))


__all__ = ["A2AASGI"]
