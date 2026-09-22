"""Focused tests for the optional A2A ASGI protocol host adapter."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from importlib import metadata
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.server.id_generator import IDGenerator, IDGeneratorContext
from a2a.types.a2a_pb2 import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskState,
)
from google.protobuf.json_format import MessageToDict

from conducto import BaseAgent, a2a_agent, a2a_capability
from conducto.a2a import (
    A2A_SERVER_EXTRA,
    A2AASGI,
    A2ADependencyError,
    require_a2a_server_dependency,
)
from conducto.core.a2a_profile import parse_agent_card
from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationResult,
    InvocationSuccess,
)
from conducto.security import ApprovalChallenge
from conducto.transport import A2AClient, DiscoveryPolicy, InMemoryTaskRepository

ENDPOINT_URL = "https://agent.example/a2a"
_ORIGIN = "https://agent.example"
RPC_PATH = urlparse(ENDPOINT_URL).path
_V1_HEADERS = {"A2A-Version": "1.0"}


@a2a_agent(name="Echo Agent", version="1.0", description="Echoes one value.")
class _EchoAgent(BaseAgent):
    """Minimal reflected agent used only to build a valid Agent Card."""

    @a2a_capability(name="echo", description="Echo one value.")
    def echo(self, value: str) -> dict[str, str]:
        """Echo the supplied value; never invoked by the adapter directly."""
        return {"value": value}


class _RecordingHandler:
    """Deterministic fake request-handler seam recording every invocation."""

    def __init__(self, results: list[InvocationResult] | None = None) -> None:
        self.calls: list[tuple[Message, str, str]] = []
        self._results = list(results) if results else None

    async def handle_message(
        self, message: Message, *, task_id: str, context_id: str
    ) -> InvocationResult:
        """Record the call and return the next configured (or default) result."""
        self.calls.append((message, task_id, context_id))
        if self._results is not None:
            return self._results.pop(0)
        return InvocationSuccess(correlation_id=task_id, value={"echo": True})


class _FixedIDGenerator(IDGenerator):
    """Deterministic identifier generator used to force id collisions in tests."""

    def __init__(self, value: str) -> None:
        self._value = value

    def generate(self, context: IDGeneratorContext) -> str:
        """Always return the configured identifier, ignoring context."""
        return self._value


def _message(text: str = "hello", *, task_id: str = "", media_type: str = "") -> Message:
    """Build a minimal one-part text message for tests."""
    message = Message(message_id="msg-1", role=Role.ROLE_USER)
    part = Part(text=text)
    if media_type:
        part.media_type = media_type
    message.parts.append(part)
    if task_id:
        message.task_id = task_id
    return message


def _build_app(
    *,
    handler: _RecordingHandler | None = None,
    repository: InMemoryTaskRepository | None = None,
    id_generator: IDGenerator | None = None,
    endpoint_url: str = ENDPOINT_URL,
) -> tuple[A2AASGI, _RecordingHandler, InMemoryTaskRepository]:
    """Build one adapter instance and return it with its handler and repository."""
    handler = handler or _RecordingHandler()
    repository = repository or InMemoryTaskRepository()
    app = A2AASGI(
        agent=_EchoAgent(),
        endpoint_url=endpoint_url,
        task_repository=repository,
        request_handler=handler,
        id_generator=id_generator,
    )
    return app, handler, repository


def _client(app: A2AASGI, *, base_url: str = _ORIGIN) -> httpx.AsyncClient:
    """Build an in-process HTTP client bound to one adapter instance."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


def _envelope(method: str, params: Any, request_id: int = 1) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 envelope from a proto request message."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": MessageToDict(params, preserving_proto_field_name=False),
    }


async def _post(app: A2AASGI, method: str, params: Any, **kwargs: Any) -> httpx.Response:
    """POST one JSON-RPC request to the adapter's advertised endpoint path."""
    async with _client(app) as client:
        headers = {**_V1_HEADERS, **kwargs.pop("headers", {})}
        return await client.post(
            RPC_PATH, json=_envelope(method, params), headers=headers, **kwargs
        )


def test_agent_card_route_serves_the_reflected_agent_card() -> None:
    """The public card route exposes the reflected agent's name and interface."""
    app, _, _ = _build_app()

    async def run() -> httpx.Response:
        async with _client(app) as client:
            return await client.get("/.well-known/agent-card.json")

    response = asyncio.run(run())
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == "Echo Agent"
    assert body["description"] == "Echoes one value."
    assert len(body["supportedInterfaces"]) == 1
    assert body["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
    assert body["supportedInterfaces"][0]["protocolVersion"] == "1.0"


def test_advertised_jsonrpc_url_equals_mounted_endpoint_configuration() -> None:
    """The card's advertised endpoint URL matches the constructed endpoint exactly."""
    app, _, _ = _build_app()

    assert app.agent_card.supported_interfaces[0].url == ENDPOINT_URL
    assert app.endpoint_url == ENDPOINT_URL

    response = asyncio.run(_post(app, "GetTask", GetTaskRequest(id="missing")))

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32001  # routed and dispatched, not 404


def test_message_send_creates_one_task_and_invokes_handler_once() -> None:
    """Sending a new message creates exactly one task and one handler invocation."""
    app, handler, repository = _build_app()

    response = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message())))
    body = response.json()

    assert response.status_code == 200
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(handler.calls) == 1
    _, task_id, context_id = handler.calls[0]
    assert task_id == task["id"]
    assert context_id == task["contextId"]

    async def fetch() -> Task | None:
        return await repository.get(task_id)

    persisted = asyncio.run(fetch())
    assert persisted is not None
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED


def test_get_list_and_cancel_use_the_injected_repository() -> None:
    """Get/list/pagination/cancel operations reflect the injected repository."""
    results: list[InvocationResult] = [
        InvocationApprovalRequired(
            correlation_id="c",
            challenge=ApprovalChallenge(
                "approval-1",
                "agent",
                "capability",
                "task",
                "correlation",
                "reason",
                "role",
                datetime.now(UTC),
                datetime.now(UTC) + timedelta(minutes=1),
            ),
        )
    ]
    app, handler, repository = _build_app(handler=_RecordingHandler(results))

    send_response = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message())))
    task_id = send_response.json()["result"]["task"]["id"]
    assert send_response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    get_response = asyncio.run(_post(app, "GetTask", GetTaskRequest(id=task_id)))
    assert get_response.json()["result"]["id"] == task_id

    list_response = asyncio.run(_post(app, "ListTasks", ListTasksRequest()))
    assert [task["id"] for task in list_response.json()["result"]["tasks"]] == [task_id]

    cancel_response = asyncio.run(_post(app, "CancelTask", CancelTaskRequest(id=task_id)))
    assert cancel_response.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"

    async def fetch() -> Task | None:
        return await repository.get(task_id)

    persisted = asyncio.run(fetch())
    assert persisted is not None
    assert persisted.status.state == TaskState.TASK_STATE_CANCELED


def test_list_tasks_pagination_is_deterministic() -> None:
    """List pagination returns stable ordering and a usable next page token."""
    repository = InMemoryTaskRepository()

    async def seed() -> None:
        for index in range(3):
            task = Task(id=f"task-{index}", context_id="ctx")
            task.status.state = TaskState.TASK_STATE_SUBMITTED
            await repository.create(task)

    asyncio.run(seed())
    app, _, _ = _build_app(repository=repository)

    first = asyncio.run(_post(app, "ListTasks", ListTasksRequest(page_size=2)))
    first_body = first.json()["result"]
    assert [task["id"] for task in first_body["tasks"]] == ["task-0", "task-1"]
    assert first_body["nextPageToken"] == "task-1"

    second = asyncio.run(
        _post(app, "ListTasks", ListTasksRequest(page_size=2, page_token="task-1"))
    )
    assert [task["id"] for task in second.json()["result"]["tasks"]] == ["task-2"]


def test_duplicate_task_creation_is_rejected() -> None:
    """A colliding generated task id surfaces as an invalid request, not a crash."""
    app, _, repository = _build_app(id_generator=_FixedIDGenerator("collide"))

    first = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message("one"))))
    assert first.json()["result"]["task"]["id"] == "collide"

    second = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message("two"))))
    assert second.json()["error"]["code"] == -32600  # INVALID_REQUEST

    async def count() -> int:
        tasks, _ = await repository.list()
        return len(tasks)

    assert asyncio.run(count()) == 1


def test_stale_and_missing_task_continuations_are_rejected() -> None:
    """Continuing a missing task fails, and a terminal task cannot be continued."""
    app, _, _ = _build_app()

    missing = asyncio.run(
        _post(
            app,
            "SendMessage",
            SendMessageRequest(message=_message(task_id="does-not-exist")),
        )
    )
    assert missing.json()["error"]["code"] == -32001  # TASK_NOT_FOUND

    created = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message())))
    task_id = created.json()["result"]["task"]["id"]
    assert created.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"

    continued = asyncio.run(
        _post(
            app,
            "SendMessage",
            SendMessageRequest(message=_message("again", task_id=task_id)),
        )
    )
    assert continued.json()["error"]["code"] == -32004  # UNSUPPORTED_OPERATION


def test_terminal_task_cannot_be_canceled() -> None:
    """A terminal task rejects cancellation with a typed error."""
    app, _, _ = _build_app()

    created = asyncio.run(_post(app, "SendMessage", SendMessageRequest(message=_message())))
    task_id = created.json()["result"]["task"]["id"]

    response = asyncio.run(_post(app, "CancelTask", CancelTaskRequest(id=task_id)))

    assert response.json()["error"]["code"] == -32002  # TASK_NOT_CANCELABLE


@pytest.mark.parametrize(
    ("method", "params_factory"),
    [
        ("SendStreamingMessage", lambda: SendMessageRequest(message=_message())),
        ("SubscribeToTask", lambda: SubscribeToTaskRequest(id="any")),
    ],
)
def test_streaming_operations_are_unsupported(method: str, params_factory: Any) -> None:
    """Streaming message send and task subscription are pinned unsupported operations."""
    app, _, _ = _build_app()

    response = asyncio.run(_post(app, method, params_factory()))

    assert response.json()["error"]["code"] == -32004  # UNSUPPORTED_OPERATION


def test_unsupported_media_type_is_rejected() -> None:
    """A message part with an unsupported media type produces a pinned error."""
    app, handler, _ = _build_app()

    response = asyncio.run(
        _post(
            app,
            "SendMessage",
            SendMessageRequest(message=_message(media_type="application/pdf")),
        )
    )

    assert response.json()["error"]["code"] == -32005  # CONTENT_TYPE_NOT_SUPPORTED
    assert handler.calls == []


def test_unsupported_required_extension_is_rejected() -> None:
    """A required extension the profile does not support is rejected."""
    app, _, _ = _build_app()

    response = asyncio.run(
        _post(
            app,
            "GetTask",
            GetTaskRequest(id="any"),
            headers={"A2A-Extensions": "https://example.com/unsupported"},
        )
    )

    assert response.json()["error"]["code"] == -32008  # EXTENSION_SUPPORT_REQUIRED


def test_unknown_method_is_rejected() -> None:
    """An unknown JSON-RPC method produces the standard method-not-found error."""
    app, _, _ = _build_app()

    async def run() -> httpx.Response:
        async with _client(app) as client:
            return await client.post(
                RPC_PATH,
                json={"jsonrpc": "2.0", "id": 1, "method": "Bogus", "params": {}},
                headers=_V1_HEADERS,
            )

    response = asyncio.run(run())

    assert response.json()["error"]["code"] == -32601


def test_malformed_json_body_is_rejected() -> None:
    """A malformed JSON body produces a JSON parse error, not an unhandled exception."""
    app, _, _ = _build_app()

    async def run() -> httpx.Response:
        async with _client(app) as client:
            return await client.post(
                RPC_PATH,
                content=b"{not json",
                headers={**_V1_HEADERS, "content-type": "application/json"},
            )

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32700


def test_missing_or_mismatched_protocol_version_is_rejected() -> None:
    """A missing or unsupported A2A-Version header is rejected as unsupported."""
    app, _, _ = _build_app()

    async def run() -> tuple[httpx.Response, httpx.Response]:
        async with _client(app) as client:
            no_header = await client.post(
                RPC_PATH, json=_envelope("GetTask", GetTaskRequest(id="x"))
            )
            wrong_version = await client.post(
                RPC_PATH,
                json=_envelope("GetTask", GetTaskRequest(id="x")),
                headers={"A2A-Version": "0.3"},
            )
            return no_header, wrong_version

    no_header, wrong_version = asyncio.run(run())

    assert no_header.json()["error"]["code"] == -32009  # VERSION_NOT_SUPPORTED
    assert wrong_version.json()["error"]["code"] == -32009


def test_multiple_adapter_instances_do_not_share_task_or_lifecycle_state() -> None:
    """Two adapter instances never share tasks, calls, or lifecycle state."""
    app_one, handler_one, repository_one = _build_app()
    app_two, handler_two, repository_two = _build_app()

    asyncio.run(_post(app_one, "SendMessage", SendMessageRequest(message=_message())))

    assert len(handler_one.calls) == 1
    assert len(handler_two.calls) == 0

    async def counts() -> tuple[int, int]:
        tasks_one, _ = await repository_one.list()
        tasks_two, _ = await repository_two.list()
        return len(tasks_one), len(tasks_two)

    assert asyncio.run(counts()) == (1, 0)


def test_discover_agent_validates_the_card_through_an_in_process_transport() -> None:
    """``discover_agent`` validates the card served over an in-process ASGI transport."""
    from conducto.transport import discover_agent

    app, _, _ = _build_app(endpoint_url="https://localhost/a2a")
    policy = DiscoveryPolicy(allow_loopback=True, allow_private_networks=True)

    async def run() -> Any:
        async with _client(app, base_url="https://localhost") as http_client:
            return await discover_agent(
                "https://localhost/.well-known/agent-card.json",
                policy=policy,
                http_client=http_client,
            )

    descriptor = asyncio.run(run())

    assert descriptor.name == "Echo Agent"
    assert descriptor.endpoint_url == "https://localhost/a2a"


def test_a2a_client_exercises_message_and_task_operations() -> None:
    """The official-SDK-backed ``A2AClient`` sends a message and reads the task back."""
    app, handler, _ = _build_app(endpoint_url="https://localhost/a2a")
    policy = DiscoveryPolicy(allow_loopback=True, allow_private_networks=True)

    async def run() -> tuple[Task, Task]:
        from conducto.transport import discover_agent

        async with _client(app, base_url="https://localhost") as http_client:
            descriptor = await discover_agent(
                "https://localhost/.well-known/agent-card.json",
                policy=policy,
                http_client=http_client,
            )
            card = parse_agent_card(dict(descriptor.card))
            sdk_client = ClientFactory(ClientConfig(httpx_client=http_client)).create(card)
            client = A2AClient(descriptor, sdk_client)

            stream = await client.send_message(SendMessageRequest(message=_message()))
            events = [event async for event in stream]
            sent_task = events[0].task
            fetched_task = await client.get_task(GetTaskRequest(id=sent_task.id))
            return sent_task, fetched_task

    sent_task, fetched_task = asyncio.run(run())

    assert sent_task.status.state == TaskState.TASK_STATE_COMPLETED
    assert fetched_task.id == sent_task.id
    assert len(handler.calls) == 1


def test_missing_official_sdk_reports_actionable_installation_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Starlette install surfaces actionable guidance, not a bare ImportError."""

    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError("starlette")

    monkeypatch.setattr(metadata, "version", missing)

    with pytest.raises(A2ADependencyError, match=r"install conducto-ai\[a2a-server\]") as error:
        require_a2a_server_dependency()

    assert error.value.extra == A2A_SERVER_EXTRA


def test_importing_conducto_does_not_eagerly_import_the_asgi_adapter() -> None:
    """Importing ``conducto``/``conducto.a2a`` never loads the optional ASGI module."""
    script = (
        "import sys; import conducto; import conducto.transport; import conducto.a2a; "
        "print('conducto.a2a.asgi' in sys.modules); "
        "conducto.a2a.A2AASGI; "
        "print('conducto.a2a.asgi' in sys.modules)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )

    before, after = completed.stdout.strip().splitlines()
    assert before == "False"
    assert after == "True"


def test_server_dependencies_are_isolated_from_core_and_client_import_paths() -> None:
    """Blocking Starlette and the SDK's route factories does not break base imports."""
    script = (
        "import builtins\n"
        "_original_import = builtins.__import__\n"
        "_blocked = {'starlette.applications'}\n"
        "\n"
        "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name in _blocked or name.startswith('a2a.server.routes'):\n"
        "        raise ImportError(f'blocked: {name}')\n"
        "    return _original_import(name, globals, locals, fromlist, level)\n"
        "\n"
        "builtins.__import__ = _guarded_import\n"
        "import conducto\n"
        "import conducto.transport\n"
        "import conducto.a2a\n"
        "print('base-import-ok')\n"
        "try:\n"
        "    conducto.a2a.A2AASGI\n"
        "    print('unexpected-success')\n"
        "except conducto.a2a.A2ADependencyError as exc:\n"
        "    print('dependency-error:' + str(exc))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()

    assert lines[0] == "base-import-ok"
    assert lines[1].startswith("dependency-error:")
    assert "conducto-ai[a2a-server]" in lines[1]
