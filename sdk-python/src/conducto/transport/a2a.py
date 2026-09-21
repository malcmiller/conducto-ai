"""Safe direct discovery and client construction for Conducto's A2A profile."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

import httpx
from google.protobuf.json_format import MessageToDict

from conducto.core.a2a_profile import (
    A2A_JSONRPC_BINDING,
    A2A_PROTOCOL_VERSION,
    A2AProtocolError,
    parse_agent_card,
)

from .errors import CompatibilityError, DiscoveryError, LimitExceededError, ProtocolError

_DEFAULT_PORTS = {"http": 80, "https": 443}
_METADATA_NETWORK = ipaddress.ip_network("169.254.169.254/32")


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    """Explicit network policy for direct, non-governed Agent Card discovery.

    Attributes:
        allowed_schemes: URL schemes accepted for card and advertised endpoints.
        allowed_ports: Destination ports accepted by the client.
        allow_loopback: Whether loopback addresses are permitted for local development.
        allow_private_networks: Whether RFC1918 and related private addresses are permitted.
        allow_link_local: Whether link-local addresses are permitted.
        max_card_bytes: Maximum response size accepted for an Agent Card.
        request_timeout: Bound applied to one card retrieval.
    """

    allowed_schemes: frozenset[str] = frozenset({"https"})
    allowed_ports: frozenset[int] = frozenset({443})
    allow_loopback: bool = False
    allow_private_networks: bool = False
    allow_link_local: bool = False
    max_card_bytes: int = 65_536
    request_timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.allowed_schemes or any(
            scheme not in {"http", "https"} for scheme in self.allowed_schemes
        ):
            raise ValueError("allowed_schemes must contain http and/or https")
        if not self.allowed_ports or any(port < 1 or port > 65535 for port in self.allowed_ports):
            raise ValueError("allowed_ports must contain valid TCP ports")
        if self.max_card_bytes <= 0 or self.request_timeout <= 0:
            raise ValueError("card and request limits must be positive")


@dataclass(frozen=True, slots=True)
class RemoteAgentDescriptor:
    """Immutable snapshot produced by direct Agent Card discovery."""

    card_url: str
    endpoint_url: str
    name: str
    version: str
    capabilities: tuple[str, ...]
    card: MappingProxyType[str, Any]


_DEFAULT_DISCOVERY_POLICY = DiscoveryPolicy()


class A2AClient:
    """Thin wrapper around the official A2A SDK client for a discovered descriptor.

    This API deliberately does not register a remote object in ``OrchestratorAgent``;
    direct remote activation is separate from the governed catalog boundary.
    """

    def __init__(self, descriptor: RemoteAgentDescriptor, client: Any) -> None:
        self.descriptor = descriptor
        self._client = client

    async def send_message(self, request: Any, *, context: Any = None) -> Any:
        """Send a non-retried A2A message through the official SDK client."""
        return self._client.send_message(request, context=context)

    async def get_task(self, request: Any, *, context: Any = None) -> Any:
        """Read a remote task through the official SDK client."""
        return await self._client.get_task(request, context=context)

    async def list_tasks(self, request: Any, *, context: Any = None) -> Any:
        """List remote tasks through the official SDK client."""
        return await self._client.list_tasks(request, context=context)

    async def cancel_task(self, request: Any, *, context: Any = None) -> Any:
        """Cancel a remote task through the official SDK client."""
        return await self._client.cancel_task(request, context=context)

    async def close(self) -> None:
        """Close the official SDK client's owned transport resources."""
        await self._client.close()


async def discover_agent(
    card_url: str,
    *,
    policy: DiscoveryPolicy = _DEFAULT_DISCOVERY_POLICY,
    http_client: httpx.AsyncClient | None = None,
) -> RemoteAgentDescriptor:
    """Retrieve and validate a public Agent Card under an explicit SSRF policy.

    Redirects are disabled. The configured card location and the card's advertised
    JSON-RPC endpoint are independently validated, including fresh DNS resolution.
    """
    _validate_url(card_url, policy)
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=False)
    try:
        try:
            async with client.stream(
                "GET",
                card_url,
                headers={"accept": "application/json"},
                follow_redirects=False,
                timeout=policy.request_timeout,
            ) as response:
                if response.is_redirect:
                    raise DiscoveryError("Agent Card redirects are not permitted")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "application/json" not in content_type.lower():
                    raise DiscoveryError(
                        "Agent Card response must have application/json content type"
                    )
                declared_size = response.headers.get("content-length")
                if declared_size and int(declared_size) > policy.max_card_bytes:
                    raise LimitExceededError("Agent Card exceeds configured size limit")
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > policy.max_card_bytes:
                        raise LimitExceededError("Agent Card exceeds configured size limit")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"Unable to retrieve Agent Card: {exc}") from exc
        try:
            payload = response.json() if not chunks else __import__("json").loads(b"".join(chunks))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProtocolError("Agent Card is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ProtocolError("Agent Card must be a JSON object")
        try:
            card = parse_agent_card(payload)
        except A2AProtocolError as exc:
            raise CompatibilityError(str(exc)) from exc
        interfaces = [
            interface
            for interface in card.supported_interfaces
            if interface.protocol_binding == A2A_JSONRPC_BINDING
            and interface.protocol_version == A2A_PROTOCOL_VERSION
        ]
        if len(interfaces) != 1:
            raise CompatibilityError(
                "Agent Card must advertise exactly one A2A 1.0 JSON-RPC interface"
            )
        endpoint_url = interfaces[0].url
        _validate_url(endpoint_url, policy)
        if not _same_authority(card_url, endpoint_url):
            raise DiscoveryError("Agent Card endpoint must use the configured card authority")
        return RemoteAgentDescriptor(
            card_url=card_url,
            endpoint_url=endpoint_url,
            name=card.name,
            version=card.version,
            capabilities=tuple(skill.id for skill in card.skills),
            card=MappingProxyType(MessageToDict(card, preserving_proto_field_name=False)),
        )
    finally:
        if owns_client:
            await client.aclose()


def _validate_url(url: str, policy: DiscoveryPolicy) -> None:
    """Validate URL syntax, port, and all currently resolved destination addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in policy.allowed_schemes or not parsed.hostname:
        raise DiscoveryError("Agent URL scheme or host is disallowed by discovery policy")
    try:
        port = parsed.port or _DEFAULT_PORTS[parsed.scheme]
    except ValueError as exc:
        raise DiscoveryError("Agent URL has an invalid port") from exc
    if port not in policy.allowed_ports:
        raise DiscoveryError("Agent URL port is disallowed by discovery policy")
    try:
        resolved = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DiscoveryError("Agent URL host could not be resolved") from exc
    addresses = {entry[4][0] for entry in resolved}
    if not addresses:
        raise DiscoveryError("Agent URL host resolved to no addresses")
    for address in addresses:
        try:
            value = ipaddress.ip_address(address)
        except ValueError as exc:
            raise DiscoveryError("Agent URL resolved to an invalid address") from exc
        if value.is_loopback and not policy.allow_loopback:
            raise DiscoveryError("loopback destinations require explicit opt-in")
        if value.is_link_local and not policy.allow_link_local:
            raise DiscoveryError("link-local destinations require explicit opt-in")
        if value in _METADATA_NETWORK:
            raise DiscoveryError("metadata-service destinations are never permitted")
        is_non_public = value.is_private or value.is_reserved or value.is_unspecified
        if is_non_public and not policy.allow_private_networks:
            raise DiscoveryError("private destinations require explicit opt-in")


def _same_authority(left: str, right: str) -> bool:
    """Return whether two URLs have the same normalized scheme, host, and port."""
    first, second = urlparse(left), urlparse(right)
    return (
        first.scheme == second.scheme
        and first.hostname == second.hostname
        and (first.port or _DEFAULT_PORTS[first.scheme])
        == (second.port or _DEFAULT_PORTS[second.scheme])
    )
