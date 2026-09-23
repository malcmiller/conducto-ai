"""Discover and invoke a hosted demo agent from an installed Conducto wheel.

This module is the third process in the installed-package A2A example. It
uses only the public discovery and client APIs; it neither imports hosted agent
objects nor defines a protocol route.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from a2a.types.a2a_pb2 import Message, Part, Role, SendMessageRequest, TaskState

from conducto.transport import A2AClient, DiscoveryPolicy, discover_agent


def _parse_arguments() -> argparse.Namespace:
    """Parse the bounded local-process arguments for this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-card-url", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--correlation-id", required=True)
    return parser.parse_args()


async def _run(card_url: str, value: str, correlation_id: str) -> dict[str, Any]:
    """Discover Agent A and return its delegated result through the A2A client."""
    parsed = urlparse(card_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise ValueError("agent-card-url must be an explicit 127.0.0.1 HTTP URL with a port")
    descriptor = await discover_agent(
        card_url,
        policy=DiscoveryPolicy(
            allowed_schemes=frozenset({"http"}),
            allowed_ports=frozenset({parsed.port}),
            allow_loopback=True,
            allow_private_networks=True,
            request_timeout=2.0,
        ),
        correlation_id=correlation_id,
    )
    skill_id = next(
        skill["id"] for skill in dict(descriptor.card)["skills"] if skill["name"] == "delegate"
    )
    if not isinstance(skill_id, str):
        raise RuntimeError("Agent Card delegate skill has an invalid identifier")
    message = Message(
        message_id=f"{correlation_id}-message",
        role=Role.ROLE_USER,
        parts=[
            Part(
                text=json.dumps(
                    {"skillId": skill_id, "arguments": {"value": value}},
                    separators=(",", ":"),
                )
            )
        ],
    )
    message.metadata.update({"x-conducto": {"correlationId": correlation_id}})
    client = A2AClient.from_descriptor(descriptor)
    try:
        events = [
            event async for event in await client.send_message(SendMessageRequest(message=message))
        ]
    finally:
        await client.close()
    if len(events) != 1 or events[0].task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError("Agent A did not complete the delegated task")
    value_result = json.loads(events[0].task.artifacts[0].parts[0].text)
    if not isinstance(value_result, dict):
        raise RuntimeError("Agent A returned an invalid result")
    return value_result


def main() -> int:
    """Run the installed-wheel orchestrator and emit one JSON final result."""
    arguments = _parse_arguments()
    print(
        json.dumps(
            asyncio.run(_run(arguments.agent_card_url, arguments.value, arguments.correlation_id)),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
