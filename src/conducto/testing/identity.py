"""Insecure identity fixtures for local development and tests."""

from __future__ import annotations

from dataclasses import dataclass

from conducto.a2a import A2AAuthenticatedIdentity, A2AAuthenticationRequest
from conducto.security import AuthorizationContext, Principal


@dataclass(frozen=True, slots=True)
class AllowAllIdentityResolver:
    """Development and test fixture only; this resolver authenticates no one.

    Never use this resolver in production. It accepts every request and returns
    one fixed local principal.

    Args:
        subject_id: Subject identifier for the local fixture principal.
    """

    subject_id: str = "anonymous-local-client"

    def __call__(self, request: A2AAuthenticationRequest) -> A2AAuthenticatedIdentity:
        """Accept a request unconditionally with a local, unauthenticated identity."""
        return A2AAuthenticatedIdentity(
            authorization=AuthorizationContext(
                principal=Principal(
                    subject_id=self.subject_id,
                    issuer="local-development",
                    audience="local-a2a",
                ),
                task_id=request.task_id,
                correlation_id=request.correlation_id,
            )
        )


__all__ = ["AllowAllIdentityResolver"]
