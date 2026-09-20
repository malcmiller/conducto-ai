"""A2A Agent Card construction and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .decorators import AgentMetadata
from .registration import AgentRegistrationError, RegisteredMethod

A2A_AGENT_CARD_SPEC_VERSION = "0.3.0"
DEFAULT_INPUT_MODES = ("text",)
DEFAULT_OUTPUT_MODES = ("text",)


def stable_skill_id(agent_name: str, capability_name: str) -> str:
    """Return the stable public identifier for a reflected capability.

    Args:
        agent_name: Agent name used to derive the capability identifier.
        capability_name: Capability name used to derive the identifier.

    Returns:
        A stable hashed capability identifier prefixed with the Conducto namespace.
    """
    value = f"{agent_name}:{capability_name}".encode()
    return f"conducto-{hashlib.sha256(value).hexdigest()[:16]}"


def build_agent_card(
    *,
    agent_type_name: str,
    metadata: AgentMetadata,
    registered_capabilities: Mapping[str, RegisteredMethod],
    url: str,
    preferred_transport: str,
    security_schemes: Mapping[str, Any] | None,
    security_requirements: Sequence[Mapping[str, Sequence[str]]] | None,
    default_input_modes: Sequence[str],
    default_output_modes: Sequence[str],
    capabilities: Mapping[str, bool] | None,
) -> dict[str, Any]:
    """Build a validated Agent Card without mutating agent state.

    Args:
        agent_type_name: Concrete agent class name used in validation errors.
        metadata: Declared A2A metadata for the agent.
        registered_capabilities: Capability registrations discovered for the agent.
        url: Public URL advertised in the agent card.
        preferred_transport: Preferred transport name used in the card.
        security_schemes: Optional security scheme definitions.
        security_requirements: Optional security requirements for the card.
        default_input_modes: Default input modes for advertised skills.
        default_output_modes: Default output modes for advertised skills.
        capabilities: Optional A2A capability flags to enable.

    Returns:
        The validated A2A Agent Card payload.
    """
    validate_card_metadata(agent_type_name, metadata, url, preferred_transport)
    input_modes = validate_modes(default_input_modes, "input")
    output_modes = validate_modes(default_output_modes, "output")
    normalized_security_schemes = validate_security_schemes(security_schemes)
    normalized_security = validate_security_requirements(security_requirements)

    agent_capabilities = {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    }
    if capabilities is not None and not isinstance(capabilities, Mapping):
        raise AgentRegistrationError("capabilities must be a mapping")
    if capabilities is not None:
        unknown = set(capabilities) - set(agent_capabilities)
        if unknown:
            raise AgentRegistrationError(
                "Unsupported A2A capability flag(s): " + ", ".join(sorted(unknown))
            )
        if any(not isinstance(value, bool) for value in capabilities.values()):
            raise AgentRegistrationError("A2A capability flags must be booleans")
        agent_capabilities.update(capabilities)

    skills: list[dict[str, Any]] = []
    parameter_schemas: dict[str, dict[str, Any]] = {}
    # Registration already inserts capabilities in deterministic attribute
    # order; preserve that order because it is part of the Agent Card wire output.
    for capability_name, registered in registered_capabilities.items():
        export = registered.capability
        assert export is not None
        if not export.description:
            raise AgentRegistrationError(
                f"{agent_type_name}.{registered.attribute_name} capability "
                "description is required for an A2A Agent Card"
            )
        skill_id = stable_skill_id(metadata.name, capability_name)
        skills.append(
            {
                "id": skill_id,
                "name": capability_name,
                "description": export.description,
                "tags": [skill_id],
                "inputModes": list(input_modes),
                "outputModes": list(output_modes),
            }
        )
        parameter_schemas[skill_id] = registered.parameter_schema

    return {
        "protocolVersion": A2A_AGENT_CARD_SPEC_VERSION,
        "name": metadata.name,
        "description": metadata.description,
        "url": url,
        "preferredTransport": preferred_transport,
        "version": metadata.version,
        "capabilities": agent_capabilities,
        "defaultInputModes": list(input_modes),
        "defaultOutputModes": list(output_modes),
        "skills": skills,
        "securitySchemes": normalized_security_schemes,
        "security": normalized_security,
        "x-conducto": {
            "parameters": parameter_schemas,
            "skillIdStrategy": "conducto-<sha256(agent-name:capability-name)[:16]>",
        },
    }


def serialize_agent_card(card: Mapping[str, Any]) -> str:
    """Serialize an Agent Card canonically.

    Args:
        card: Mapping describing the agent card payload.

    Returns:
        A stable JSON representation of the card.
    """
    return json.dumps(card, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def validate_card_metadata(
    agent_type_name: str,
    metadata: AgentMetadata,
    url: str,
    preferred_transport: str,
) -> None:
    """Validate the metadata required for a publishable Agent Card.

    Args:
        agent_type_name: Concrete agent class name used in error messages.
        metadata: Declared A2A metadata.
        url: Public URL advertised by the card.
        preferred_transport: Preferred transport string used in the card.

    Raises:
        AgentRegistrationError: If metadata is incomplete or malformed.
    """
    if not isinstance(url, str) or not is_absolute_http_url(url):
        raise AgentRegistrationError("Agent Card url must be an absolute http or https URL")
    if not isinstance(preferred_transport, str) or not preferred_transport.strip():
        raise AgentRegistrationError("Agent Card preferred_transport cannot be empty")
    if preferred_transport != preferred_transport.strip():
        raise AgentRegistrationError(
            "Agent Card preferred_transport cannot contain surrounding whitespace"
        )
    if not metadata.name.strip():
        raise AgentRegistrationError("Agent Card agent name cannot be empty")
    if not metadata.version.strip():
        raise AgentRegistrationError("Agent Card agent version cannot be empty")
    if not metadata.description:
        raise AgentRegistrationError(
            f"{agent_type_name} requires a description for an A2A Agent Card"
        )


def is_absolute_http_url(value: Any) -> bool:
    """Check whether a value is an absolute HTTP or HTTPS URL.

    Args:
        value: Candidate URL value to validate.

    Returns:
        True when the value is a valid absolute HTTP(S) URL.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def validate_modes(modes: Sequence[str], label: str) -> tuple[str, ...]:
    """Normalize and validate mode declarations for A2A cards.

    Args:
        modes: Sequence of offered modes.
        label: Name of the mode family being validated.

    Returns:
        The normalized tuple of non-empty mode strings.

    Raises:
        AgentRegistrationError: If the mode list is malformed or empty.
    """
    if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
        raise AgentRegistrationError(f"default_{label}_modes must be a sequence")
    normalized = tuple(mode.strip() for mode in modes if isinstance(mode, str))
    if len(normalized) != len(modes) or not normalized or any(not mode for mode in normalized):
        raise AgentRegistrationError(f"default_{label}_modes must contain non-empty strings")
    return normalized


def validate_security_schemes(
    security_schemes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize a security scheme definition mapping.

    Args:
        security_schemes: Mapping of security scheme names to scheme definitions.

    Returns:
        A normalized copy of the security scheme mapping.

    Raises:
        AgentRegistrationError: If any scheme is structurally invalid.
    """
    if security_schemes is None:
        return {}
    if not isinstance(security_schemes, Mapping):
        raise AgentRegistrationError("security_schemes must be a mapping")

    validated: dict[str, Any] = {}
    for name, scheme in security_schemes.items():
        if not isinstance(name, str) or not name.strip():
            raise AgentRegistrationError("security_schemes names must be non-empty strings")
        if not isinstance(scheme, Mapping):
            raise AgentRegistrationError(f"security scheme '{name}' must be an object")
        scheme_type = scheme.get("type")
        if scheme_type == "apiKey":
            if not isinstance(scheme.get("name"), str) or not scheme["name"].strip():
                raise AgentRegistrationError(
                    f"apiKey security scheme '{name}' requires a non-empty name"
                )
            if scheme.get("in") not in {"header", "query", "cookie"}:
                raise AgentRegistrationError(
                    f"apiKey security scheme '{name}' requires in=header, query, or cookie"
                )
        elif scheme_type == "http":
            if not isinstance(scheme.get("scheme"), str) or not scheme["scheme"].strip():
                raise AgentRegistrationError(
                    f"http security scheme '{name}' requires a non-empty scheme"
                )
        elif scheme_type == "oauth2":
            _validate_oauth2_scheme(name, scheme)
        elif scheme_type == "openIdConnect":
            if not is_absolute_http_url(scheme.get("openIdConnectUrl")):
                raise AgentRegistrationError(
                    f"openIdConnect security scheme '{name}' requires an absolute URL"
                )
        else:
            raise AgentRegistrationError(
                f"security scheme '{name}' has unsupported type {scheme_type!r}"
            )
        validated[name] = dict(scheme)
    return validated


def _validate_oauth2_scheme(name: str, scheme: Mapping[str, Any]) -> None:
    """Validate one OAuth2 scheme definition.

    Args:
        name: Security scheme name as declared in the card.
        scheme: Scheme definition to validate.

    Raises:
        AgentRegistrationError: If the OAuth2 flow metadata is malformed.
    """
    flows = scheme.get("flows")
    if not isinstance(flows, Mapping) or not flows:
        raise AgentRegistrationError(f"oauth2 security scheme '{name}' requires non-empty flows")
    for flow_name, flow in flows.items():
        if flow_name not in {
            "authorizationCode",
            "clientCredentials",
            "implicit",
            "password",
        } or not isinstance(flow, Mapping):
            raise AgentRegistrationError(f"oauth2 security scheme '{name}' has an invalid flow")
        scopes = flow.get("scopes")
        if not isinstance(scopes, Mapping) or any(
            not isinstance(scope, str) or not isinstance(description, str)
            for scope, description in scopes.items()
        ):
            raise AgentRegistrationError(
                f"oauth2 security scheme '{name}' flow '{flow_name}' "
                "requires a scope-description mapping"
            )
        if flow_name in {"authorizationCode", "implicit"} and not is_absolute_http_url(
            flow.get("authorizationUrl")
        ):
            raise AgentRegistrationError(
                f"oauth2 security scheme '{name}' flow '{flow_name}' "
                "requires an absolute authorizationUrl"
            )
        if flow_name != "implicit" and not is_absolute_http_url(flow.get("tokenUrl")):
            raise AgentRegistrationError(
                f"oauth2 security scheme '{name}' flow '{flow_name}' requires an absolute tokenUrl"
            )


def validate_security_requirements(
    security_requirements: Sequence[Mapping[str, Sequence[str]]] | None,
) -> list[dict[str, list[str]]]:
    """Validate and normalize security requirement sets for a card.

    Args:
        security_requirements: Ordered requirement objects describing required schemes.

    Returns:
        A normalized list of security requirement mappings.

    Raises:
        AgentRegistrationError: If the requirement structure is malformed.
    """
    if security_requirements is None:
        return []
    if isinstance(security_requirements, (str, bytes)) or not isinstance(
        security_requirements, Sequence
    ):
        raise AgentRegistrationError("security_requirements must be a sequence")

    validated: list[dict[str, list[str]]] = []
    for index, requirement in enumerate(security_requirements):
        if not isinstance(requirement, Mapping) or not requirement:
            raise AgentRegistrationError(f"security requirement {index} must be a non-empty object")
        normalized: dict[str, list[str]] = {}
        for scheme_name, scopes in requirement.items():
            if not isinstance(scheme_name, str) or not scheme_name.strip():
                raise AgentRegistrationError(
                    f"security requirement {index} has an invalid scheme name"
                )
            if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
                raise AgentRegistrationError(
                    f"security requirement '{scheme_name}' scopes must be a sequence"
                )
            if any(not isinstance(scope, str) for scope in scopes):
                raise AgentRegistrationError(
                    f"security requirement '{scheme_name}' scopes must be strings"
                )
            normalized[scheme_name] = list(scopes)
        validated.append(normalized)
    return validated
