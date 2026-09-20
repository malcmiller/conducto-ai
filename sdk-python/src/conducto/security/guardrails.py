"""Metadata decorators and deterministic guardrail discovery."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
_ATTRIBUTE = "__conducto_guardrails__"


@dataclass(frozen=True, slots=True)
class ScopeRequirement:
    """A cumulative exact-match scope requirement."""

    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class ApprovalRequirement:
    """An approval requirement with an optional typed application callback."""

    role: str
    condition: Callable[..., bool] | None = None


@dataclass(frozen=True, slots=True)
class Guardrails:
    """Normalized guardrails attached to one callable."""

    scopes: tuple[str, ...] = ()
    approvals: tuple[ApprovalRequirement, ...] = ()


def _target(value: Any) -> Callable[..., Any]:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    if not callable(value):
        raise TypeError("Guardrails can only decorate callables")
    return cast(Callable[..., Any], inspect.unwrap(value))


def _merge(
    value: Any, *, scope: str | None = None, approval: ApprovalRequirement | None = None
) -> None:
    target = _target(value)
    current = get_guardrails(target)
    scopes = set(current.scopes)
    if scope is not None:
        scopes.add(scope)
    approvals = current.approvals + ((approval,) if approval is not None else ())
    setattr(
        target,
        _ATTRIBUTE,
        Guardrails(tuple(sorted(scopes)), tuple(sorted(approvals, key=lambda item: item.role))),
    )


def require_scope(*scopes: str) -> Callable[[F], F]:
    """Require every declared opaque, case-sensitive scope."""
    if not scopes or any(not isinstance(scope, str) or not scope for scope in scopes):
        raise ValueError("at least one non-empty scope is required")

    def decorate(value: F) -> F:
        for scope in scopes:
            _merge(value, scope=scope)
        return value

    return decorate


def require_approval(role: str, condition: Callable[..., bool] | None = None) -> Callable[[F], F]:
    """Require approval from ``role`` when the typed condition returns true."""
    if not isinstance(role, str) or not role:
        raise ValueError("approval role is required")
    if condition is not None and not callable(condition):
        raise TypeError("approval condition must be callable")

    def decorate(value: F) -> F:
        _merge(value, approval=ApprovalRequirement(role, condition))
        return value

    return decorate


def get_guardrails(value: Any) -> Guardrails:
    """Return normalized guardrails from a callable or descriptor."""
    metadata = getattr(_target(value), _ATTRIBUTE, None)
    return metadata if isinstance(metadata, Guardrails) else Guardrails()


def discover_guardrails(value: Any) -> Guardrails:
    """Discover guardrails in deterministic scope and declaration order."""
    return get_guardrails(value)
