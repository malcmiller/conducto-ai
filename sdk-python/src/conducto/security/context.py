"""Immutable, transport-independent authorization context contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


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
