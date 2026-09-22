"""Exact deployment grants binding authenticated actors to remote identities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from conducto.core.catalog import DeploymentType
from conducto.security.tokens import ValidatedIdentity

from .models import Operation, RegistrationRequest


@dataclass(frozen=True, slots=True)
class RegistrationGrant:
    """Application-owned exact identity and endpoint binding, never caller supplied.

    Each grant covers one issuer, actor, owner, environment, and logical agent.
    Different instances may use different grants with the same identity tuple.
    The card name and both URLs must match exactly; redirects and URL credentials
    are forbidden. Scopes must additionally contain ``registration:<operation>``.
    """

    issuer: str
    subject_id: str
    owner: str
    environment: str
    agent_id: str
    card_name: str
    agent_card_url: str = field(repr=False)
    endpoint_url: str = field(repr=False)
    operations: frozenset[Operation] = frozenset[Operation](
        {"register", "renew", "drain", "deregister", "status"}
    )
    max_lease_seconds: float = 300
    deployment_types: frozenset[DeploymentType] = frozenset({DeploymentType.REMOTE_CONTAINER})
    trust_policy_ref: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete identity bindings and unsafe configured URLs."""
        if not all(
            (
                self.issuer,
                self.subject_id,
                self.owner,
                self.environment,
                self.agent_id,
                self.card_name,
            )
        ):
            raise ValueError("registration grant requires exact identity bindings")
        if not math.isfinite(self.max_lease_seconds) or self.max_lease_seconds <= 0:
            raise ValueError("max_lease_seconds must be finite and positive")
        for url in (self.agent_card_url, self.endpoint_url):
            parts = urlsplit(url)
            if (
                parts.scheme not in {"https", "http"}
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
                or parts.query
                or parts.fragment
            ):
                raise ValueError("registration URLs must be absolute and credential-free")
        object.__setattr__(self, "operations", frozenset(self.operations))
        object.__setattr__(self, "deployment_types", frozenset(self.deployment_types))

    def permits(self, identity: ValidatedIdentity, request: RegistrationRequest) -> bool:
        """Match verified identity and exact resource scope before network access."""
        return (
            identity.issuer == self.issuer
            and identity.subject == self.subject_id
            and request.owner == self.owner
            and request.environment == self.environment
            and request.agent_id == self.agent_id
            and request.operation in self.operations
            and f"registration:{request.operation}" in identity.scopes
        )
