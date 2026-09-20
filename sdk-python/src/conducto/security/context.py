"""Immutable, transport-independent authorization context contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import AuthorizationDeniedError, MissingAuthorizationContextError


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity facts; credentials and raw tokens are excluded."""

    subject_id: str
    issuer: str
    audience: str | tuple[str, ...]
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.subject_id, self.issuer)):
            raise ValueError("principal subject_id and issuer are required")
        audiences = (self.audience,) if isinstance(self.audience, str) else self.audience
        if not audiences or not all(isinstance(value, str) and value for value in audiences):
            raise ValueError("principal audience is required")
        object.__setattr__(
            self, "audience", tuple(audiences) if len(audiences) > 1 else audiences[0]
        )
        object.__setattr__(self, "claims", _freeze(self.claims))
        object.__setattr__(self, "roles", frozenset(self.roles))
        object.__setattr__(self, "scopes", frozenset(self.scopes))


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Immutable facts supplied to authorization and approval policy callbacks."""

    principal: Principal
    task_id: str
    correlation_id: str
    policy_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.correlation_id:
            raise ValueError("task_id and correlation_id are required")
        object.__setattr__(self, "policy_metadata", _freeze(self.policy_metadata))


def delegate_context(
    parent: AuthorizationContext | None,
    candidate: AuthorizationContext | None,
) -> AuthorizationContext | None:
    """Inherit or restrict authorization for nested local invocation."""
    if candidate is None:
        return parent
    if not isinstance(candidate, AuthorizationContext):
        raise MissingAuthorizationContextError("invalid authorization context")
    if parent is None:
        return candidate
    if not isinstance(parent, AuthorizationContext):
        raise MissingAuthorizationContextError("invalid active authorization context")
    parent_principal = parent.principal
    child_principal = candidate.principal
    if (
        child_principal.subject_id != parent_principal.subject_id
        or child_principal.issuer != parent_principal.issuer
        or child_principal.audience != parent_principal.audience
        or not child_principal.roles.issubset(parent_principal.roles)
        or not child_principal.scopes.issubset(parent_principal.scopes)
        or candidate.task_id != parent.task_id
        or candidate.correlation_id != parent.correlation_id
    ):
        raise AuthorizationDeniedError("delegated authorization is broader than its caller")
    return candidate
