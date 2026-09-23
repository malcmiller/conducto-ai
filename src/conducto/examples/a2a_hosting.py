"""Host deterministic A2A agents from an installed Conducto wheel.

Run a standalone server with an application-owned Uvicorn process:

    CONDUCTO_DEMO_AGENT=agent-b CONDUCTO_DEMO_PORT=8002 \
        python -m uvicorn conducto.examples.a2a_hosting:app --host 127.0.0.1 --port 8002

Set ``CONDUCTO_DEMO_COMPOSITION=fastapi`` to mount the same Conducto ASGI
application in a FastAPI application. The example is local-development only:
its identity resolver is deterministic and must be replaced by real
application authentication in production.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from a2a.client import ClientCallContext
from a2a.types.a2a_pb2 import Message, Part, Role, SendMessageRequest, TaskState

from conducto import BaseAgent, Runtime, a2a_agent, a2a_capability, get_run_context
from conducto.a2a import (
    A2AAuthenticatedIdentity,
    A2AAuthenticationRequest,
    A2AHostSecurityConfig,
    create_a2a_app,
)
from conducto.core.agent_card import stable_skill_id
from conducto.security import AuthorizationContext, Principal, require_approval, require_scope
from conducto.transport import A2AClient, DiscoveryPolicy, discover_agent

_LOOPBACK_HOST = "127.0.0.1"
_DEFAULT_SCOPES = frozenset({"demo:invoke", "demo:downstream"})
_DELEGATED_SCOPES = frozenset({"demo:downstream"})


def _environment(name: str, default: str = "") -> str:
    """Return one required demo setting while rejecting whitespace ambiguity."""
    value = os.environ.get(name, default)
    if not value or value != value.strip():
        raise RuntimeError(f"{name} must be a non-empty, whitespace-free value")
    return value


def _port() -> int:
    """Read and validate the loopback port advertised by this process."""
    value = _environment("CONDUCTO_DEMO_PORT", "8001")
    try:
        port = int(value)
    except ValueError as error:
        raise RuntimeError("CONDUCTO_DEMO_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise RuntimeError("CONDUCTO_DEMO_PORT must be between 1 and 65535")
    return port


def _record_observation(agent: str, capability: str, payload: dict[str, Any]) -> None:
    """Append deterministic demo observability when the acceptance test requests it."""
    path = os.environ.get("CONDUCTO_DEMO_OBSERVABILITY_PATH")
    if not path:
        return
    record = {"agent": agent, "capability": capability, **payload}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


async def resolve_demo_identity(request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
    """Return deterministic local authority without accepting credentials.

    ``x-demo-scopes`` is intentionally a test-only, comma-separated input used
    to demonstrate the hosted capability's normal scope enforcement. Production
    applications must derive scopes from verified application authentication.
    """
    requested = request.headers.get("x-demo-scopes")
    scopes = (
        frozenset(scope for scope in requested.split(",") if scope)
        if requested is not None
        else _DEFAULT_SCOPES
    )
    return A2AAuthenticatedIdentity(
        AuthorizationContext(
            principal=Principal(
                subject_id="installed-wheel-demo",
                issuer="conducto-example",
                audience="conducto-example",
                scopes=scopes,
            ),
            task_id=request.task_id,
            correlation_id=request.correlation_id,
        )
    )


@a2a_agent(
    name="InstalledDemoAgentB",
    version="1.0.0",
    description="Deterministic downstream agent for installed-package A2A hosting.",
)
class DownstreamAgent(BaseAgent):
    """Provide deterministic scoped, approval, and cancellation demonstrations."""

    def __init__(self) -> None:
        """Initialize an isolated cancellation gate for this process."""
        self._never_released = asyncio.Event()
        super().__init__()

    @require_scope("demo:downstream")
    @a2a_capability(name="process", description="Process one deterministic value.")
    def process(self, value: str) -> dict[str, Any]:
        """Return a stable downstream result."""
        context = get_run_context()
        assert context is not None
        authorization = context.authorization
        assert authorization is not None
        metadata = context.to_dict()["metadata"]
        assert isinstance(metadata, dict)
        a2a_metadata = metadata.get("a2a")
        assert isinstance(a2a_metadata, dict)
        result = {
            "value": value,
            "correlation_id": context.correlation_id,
            "task_id": authorization.task_id,
            "lineage": a2a_metadata.get("lineage", []),
            "scopes": sorted(authorization.principal.scopes),
        }
        _record_observation("agent-b", "process", result)
        return result

    @require_scope("demo:required")
    @a2a_capability(name="restricted", description="Require a scope not granted by default.")
    def restricted(self) -> str:
        """Return only when the inbound identity carries the required scope."""
        return "authorized"

    @require_approval("reviewer")
    @a2a_capability(name="approval", description="Require approval before execution.")
    def approval(self, value: str) -> str:
        """Return after a separately authenticated approval continuation."""
        return value

    @a2a_capability(name="wait", description="Wait until the runtime cancels this request.")
    async def wait(self) -> str:
        """Block without wall-clock sleeping so timeout cancellation is deterministic."""
        await self._never_released.wait()
        return "unreachable"


@a2a_agent(
    name="InstalledDemoAgentA",
    version="1.0.0",
    description="Deterministic delegator for installed-package A2A hosting.",
)
class DelegatingAgent(BaseAgent):
    """Delegate one governed call to the independently hosted downstream agent."""

    def __init__(self, downstream_card_url: str) -> None:
        """Store the public downstream card URL without sharing downstream state."""
        self._downstream_card_url = downstream_card_url
        super().__init__()

    @require_scope("demo:invoke")
    @a2a_capability(name="delegate", description="Delegate one value to Agent B.")
    async def delegate(self, value: str) -> dict[str, Any]:
        """Discover Agent B and invoke its published process skill through A2A."""
        context = get_run_context()
        assert context is not None
        authorization = context.authorization
        assert authorization is not None
        parsed = urlparse(self._downstream_card_url)
        assert parsed.port is not None
        descriptor = await discover_agent(
            self._downstream_card_url,
            policy=DiscoveryPolicy(
                allowed_schemes=frozenset({"http"}),
                allowed_ports=frozenset({parsed.port}),
                allow_loopback=True,
                allow_private_networks=True,
                request_timeout=2.0,
            ),
            correlation_id=context.correlation_id,
        )
        client = A2AClient.from_descriptor(descriptor)
        try:
            message = Message(
                message_id=f"{authorization.task_id}-downstream",
                role=Role.ROLE_USER,
                parts=[
                    Part(
                        text=json.dumps(
                            {
                                "skillId": stable_skill_id(descriptor.name, "process"),
                                "arguments": {"value": value},
                            },
                            separators=(",", ":"),
                        )
                    )
                ],
                reference_task_ids=[authorization.task_id],
            )
            metadata: dict[str, Any] = {"correlationId": context.correlation_id}
            remaining_timeout = context.remaining_timeout()
            if remaining_timeout is not None:
                metadata["timeoutSeconds"] = remaining_timeout
            message.metadata.update({"x-conducto": metadata})
            delegated_scopes = sorted(authorization.principal.scopes & _DELEGATED_SCOPES)
            events = [
                event
                async for event in await client.send_message(
                    SendMessageRequest(message=message),
                    context=ClientCallContext(
                        service_parameters={"x-demo-scopes": ",".join(delegated_scopes)}
                    ),
                )
            ]
        finally:
            await client.close()
        if len(events) != 1 or events[0].task.status.state != TaskState.TASK_STATE_COMPLETED:
            raise RuntimeError("downstream agent did not complete the delegated task")
        artifact = events[0].task.artifacts[0]
        value = json.loads(artifact.parts[0].text)
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise RuntimeError("downstream agent returned an invalid deterministic result")
        _record_observation(
            "agent-a",
            "delegate",
            {
                "correlation_id": context.correlation_id,
                "delegated_scopes": delegated_scopes,
                "downstream_lineage": value.get("lineage", []),
                "downstream_task_id": value.get("task_id", ""),
                "task_id": authorization.task_id,
                "value": value.get("value", ""),
            },
        )
        return value


def build_app() -> Any:
    """Build either direct or FastAPI-composed ASGI hosting from environment."""
    port = _port()
    agent_name = _environment("CONDUCTO_DEMO_AGENT", "agent-b")
    downstream_card_url = os.environ.get("CONDUCTO_DEMO_DOWNSTREAM_CARD_URL", "")
    if agent_name == "agent-a":
        if not downstream_card_url:
            raise RuntimeError("CONDUCTO_DEMO_DOWNSTREAM_CARD_URL is required for agent-a")
        agent: BaseAgent = DelegatingAgent(downstream_card_url)
    elif agent_name == "agent-b":
        agent = DownstreamAgent()
    else:
        raise RuntimeError("CONDUCTO_DEMO_AGENT must be agent-a or agent-b")
    conducto_app = create_a2a_app(
        agent=agent,
        runtime=Runtime(),
        public_url=f"http://{_LOOPBACK_HOST}:{port}",
        identity_resolver=resolve_demo_identity,
        security_config=A2AHostSecurityConfig(
            liveness_path="/livez",
            readiness_path="/readyz",
        ),
    )
    if os.environ.get("CONDUCTO_DEMO_COMPOSITION", "direct") == "direct":
        return conducto_app
    if os.environ["CONDUCTO_DEMO_COMPOSITION"] != "fastapi":
        raise RuntimeError("CONDUCTO_DEMO_COMPOSITION must be direct or fastapi")
    from fastapi import FastAPI

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        """Forward the embedding application's lifespan to the Conducto host."""
        await conducto_app.startup()
        try:
            yield
        finally:
            await conducto_app.drain()
            await conducto_app.aclose()

    application = FastAPI(lifespan=lifespan)
    application.mount("/", conducto_app)
    return application


app = build_app()
