"""Optional A2A ASGI server dependency profile and checks."""

from __future__ import annotations

from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .errors import A2ADependencyError

A2A_SERVER_EXTRA = "a2a-server"
"""Conducto package extra that installs the optional A2A ASGI server dependencies."""

A2A_SERVER_STARLETTE_PACKAGE = "starlette"
"""Distribution name of the ASGI framework used by the official A2A SDK server routes."""

A2A_SERVER_STARLETTE_SPECIFIER = ">=0.27"
"""Minimum Starlette release required by the official A2A SDK's ``http-server`` extra."""


def require_a2a_server_dependency() -> str:
    """Validate that the optional Starlette-based A2A server dependency is importable.

    Returns:
        The installed Starlette version.

    Raises:
        A2ADependencyError: If Starlette is absent or outside the supported range.
    """
    try:
        installed = metadata.version(A2A_SERVER_STARLETTE_PACKAGE)
        version = Version(installed)
    except (metadata.PackageNotFoundError, InvalidVersion) as error:
        raise A2ADependencyError(
            f"The A2A ASGI server adapter requires {A2A_SERVER_STARLETTE_PACKAGE}"
            f"{A2A_SERVER_STARLETTE_SPECIFIER}; install conducto-ai[{A2A_SERVER_EXTRA}]",
            extra=A2A_SERVER_EXTRA,
        ) from error
    if version not in SpecifierSet(A2A_SERVER_STARLETTE_SPECIFIER):
        raise A2ADependencyError(
            f"The A2A ASGI server adapter requires {A2A_SERVER_STARLETTE_PACKAGE}"
            f"{A2A_SERVER_STARLETTE_SPECIFIER}; install conducto-ai[{A2A_SERVER_EXTRA}]",
            extra=A2A_SERVER_EXTRA,
        )
    return installed


__all__ = [
    "A2A_SERVER_EXTRA",
    "A2A_SERVER_STARLETTE_PACKAGE",
    "A2A_SERVER_STARLETTE_SPECIFIER",
    "require_a2a_server_dependency",
]
