"""Optional ASGI adapter hosting one Conducto agent's inbound A2A 1.0 surface.

This module owns HTTP/ASGI protocol adaptation, exact route matching, Agent Card
and mounted endpoint consistency, official SDK JSON-RPC dispatch, protocol parsing,
protocol errors, task repository wiring, and basic ASGI lifespan pass-through. It
does not own Conducto authorization, approvals, audit, model resolution, capability
invocation, authentication, production hardening, or process hosting; those remain
with the runtime, later stories, and the application that mounts this adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Mapping
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
from conducto.transport.errors import LimitExceededError, RemoteTaskError
from conducto.transport.tasks import TaskRepository

from . import invocation_result_to_task
from .errors import A2ADependencyError
from .handler import A2ACancellableRequestHandler, A2ARequestContext, A2ARequestHandler
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
        SendMessageRequest,
        SubscribeToTaskRequest,
        Task,
        TaskPushNotificationConfig,
        TaskState,
    )
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
        request_handler: A2ARequestHandler,
        id_generator: IDGenerator,
    ) -> None:
        self._agent_card = agent_card
        self._task_repository = task_repository
        self._request_handler = request_handler
        self._id_generator = id_generator
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
        page_size = params.page_size if params.page_size > 0 else _DEFAULT_PAGE_SIZE
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
        self._reject_unsupported_extensions(context)
        if context.state.get("request_id") is None:
            raise InvalidRequestError("SendMessage requires a JSON-RPC request identifier")
        message = params.message
        try:
            parse_message(MessageToDict(message, preserving_proto_field_name=False))
        except A2AProtocolError as exc:
            text = str(exc)
            if "media type" in text.lower():
                raise ContentTypeNotSupportedError(text) from exc
            raise InvalidParamsError(text) from exc

        if message.task_id:
            task_id = message.task_id
            existing = await self._task_repository.get(task_id)
            if existing is None:
                raise TaskNotFoundError(f"A2A task not found: {task_id}")
            if TaskState.Name(existing.status.state) in TERMINAL_TASK_STATES:
                raise UnsupportedOperationError(
                    f"Task {task_id} is in terminal state {TaskState.Name(existing.status.state)}"
                )
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
            result = await self._request_handler.handle_message(
                message,
                task_id=task_id,
                context_id=context_id,
                request_context=A2ARequestContext(
                    request_id=(
                        str(context.state["request_id"])
                        if context.state.get("request_id") is not None
                        else ""
                    ),
                    headers=context.state.get("headers", {}),
                    method=str(context.state.get("method", "SendMessage")),
                ),
            )
        except asyncio.CancelledError:
            if isinstance(self._request_handler, A2ACancellableRequestHandler):
                await self._request_handler.cancel(task_id)
            with contextlib.suppress(RemoteTaskError):
                await self._task_repository.compare_and_transition(
                    task_id, current_state, TaskState.TASK_STATE_CANCELED
                )
            raise
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
    """ASGI callable hosting one Conducto agent's inbound A2A 1.0 surface.

    Args:
        agent: The reflected Conducto agent to publish.
        endpoint_url: Absolute URL advertised as the agent's JSON-RPC endpoint.
        task_repository: Injected task persistence boundary.
        request_handler: Typed seam invoked once per accepted inbound message.
        id_generator: Optional generator for new task and context identifiers.
        agent_card_kwargs: Optional keyword arguments forwarded to ``get_agent_card``.

    Notes:
        Constructing this adapter never starts a listener, event loop, thread, or
        subprocess; call the instance as an ASGI application from an existing
        server process. Two instances never share task or lifecycle state.
    """

    def __init__(
        self,
        *,
        agent: BaseAgent,
        endpoint_url: str,
        task_repository: TaskRepository,
        request_handler: A2ARequestHandler,
        id_generator: IDGenerator | None = None,
        agent_card_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        card_payload = agent.get_agent_card(endpoint_url, **dict(agent_card_kwargs or {}))
        try:
            card = parse_agent_card(card_payload)
        except A2AProtocolError as exc:
            raise ValueError(f"Agent Card is not a valid A2A 1.0 profile: {exc}") from exc
        dispatch_handler = _ConductoRequestHandler(
            agent_card=card,
            task_repository=task_repository,
            request_handler=request_handler,
            id_generator=id_generator or UUIDGenerator(),
        )
        rpc_path = urlparse(endpoint_url).path or "/"
        routes = [
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(dispatch_handler, rpc_url=rpc_path),
        ]
        self._app = Starlette(routes=routes)
        self.agent_card = card
        self.endpoint_url = endpoint_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one ASGI scope by delegating to the wrapped Starlette application."""
        await self._app(scope, receive, send)


__all__ = ["A2AASGI"]
