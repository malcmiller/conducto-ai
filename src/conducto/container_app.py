"""Application-selection contracts for the immutable Conducto container host.

The host never imports an application directly from untrusted runtime input.
Instead, deployment images bake a manifest that maps a bounded application key
to a server-owned ``module:attribute`` factory reference.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from starlette.types import ASGIApp

    from conducto.container import ContainerConfig
else:
    ASGIApp = Any

DEFAULT_APPLICATION_KEY = "reference"
"""Manifest key used when ``CONDUCTO_APPLICATION`` is unset."""

APPLICATION_ENVIRONMENT_VARIABLE = "CONDUCTO_APPLICATION"
"""Environment variable selecting one manifest key."""

APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE = "CONDUCTO_APPLICATIONS_MANIFEST"
"""Optional environment variable overriding the baked-in manifest path."""

DEFAULT_APPLICATIONS_MANIFEST_PATH = "/etc/conducto/applications.json"
"""Default baked-in manifest path for deployment images."""

DEFAULT_MANIFEST_VERSION = "1"
"""Current manifest schema version."""

_APPLICATION_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SECRET_NAMES = frozenset({"token", "secret", "api_key", "authorization", "password"})
_DEFAULT_MANIFEST_DOCUMENT: dict[str, Any] = {
    "manifestVersion": DEFAULT_MANIFEST_VERSION,
    "applications": {
        DEFAULT_APPLICATION_KEY: "conducto.container:build_reference_app",
    },
}


class ContainerApplication(Protocol):
    """Build a configured A2A ASGI application for the container host."""

    def __call__(self, config: ContainerConfig) -> ASGIApp:
        """Return the configured ASGI application for ``config``."""


@dataclass(frozen=True, slots=True)
class ContainerApplicationsManifest:
    """Validated immutable application manifest metadata."""

    manifest_version: str
    applications: Mapping[str, str] = field(repr=False)
    source: str
    source_path: str | None = None

    def __post_init__(self) -> None:
        """Freeze applications into a read-only mapping."""
        object.__setattr__(self, "applications", MappingProxyType(dict(self.applications)))


@dataclass(frozen=True, slots=True)
class ResolvedContainerApplication:
    """Validated application selection resolved from a baked-in manifest."""

    key: str
    target: str
    manifest_version: str
    manifest_source: str
    manifest_path: str | None
    factory: ContainerApplication = field(repr=False)

    def build(self, config: ContainerConfig) -> ASGIApp:
        """Build the selected ASGI application from ``config``.

        Args:
            config: Immutable container configuration selected for this process.

        Returns:
            The constructed ASGI application.

        Raises:
            ValueError: If the application factory cannot build a valid ASGI app.
        """
        try:
            application = self.factory(config)
        except ValueError:
            raise
        except (ImportError, RuntimeError, TypeError) as error:
            detail = _redact_message(str(error)).strip() or type(error).__name__
            raise ValueError(f"application {self.key!r} failed to build: {detail}") from error
        if not callable(application):
            raise ValueError(f"application {self.key!r} factory must return an ASGI application")
        return application


def load_container_applications_manifest(
    path: str | Path | None = None,
) -> ContainerApplicationsManifest:
    """Load the validated container-application manifest.

    Args:
        path: Optional explicit manifest path. When omitted, this function uses
            ``CONDUCTO_APPLICATIONS_MANIFEST`` when set, then the default baked
            path when it exists, and finally the embedded reference-only
            manifest.

    Returns:
        The immutable validated manifest.

    Raises:
        ValueError: If the manifest path, JSON, or schema is invalid.
    """
    if path is not None:
        explicit = str(path).strip()
        if not explicit:
            raise ValueError("applications manifest path must not be empty")
        return _load_manifest_path(Path(explicit))

    configured_path = os.environ.get(APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE)
    if configured_path is not None:
        if configured_path != configured_path.strip() or not configured_path:
            raise ValueError(
                f"{APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE} must be a non-empty file path"
            )
        return _load_manifest_path(Path(configured_path))

    default_path = Path(DEFAULT_APPLICATIONS_MANIFEST_PATH)
    if default_path.is_file():
        return _load_manifest_path(default_path)
    return _manifest_from_document(
        _DEFAULT_MANIFEST_DOCUMENT,
        source="embedded-default",
        source_path=None,
    )


def resolve_container_application(
    config: ContainerConfig,
    *,
    selection: str | None = None,
    manifest: ContainerApplicationsManifest | None = None,
) -> ResolvedContainerApplication:
    """Resolve the selected application factory from the validated manifest.

    Args:
        config: Immutable container configuration. The value is validated by the
            caller and is not mutated by this function.
        selection: Optional explicit application key. When omitted, this
            function reads ``CONDUCTO_APPLICATION`` and falls back to the
            reference application key.
        manifest: Optional preloaded manifest. When omitted, this function
            loads the configured manifest.

    Returns:
        A validated manifest selection plus the loaded factory callable.

    Raises:
        ValueError: If the selected key is unknown, malformed, or resolves to
            an invalid import target.
    """
    del config
    loaded_manifest = manifest or load_container_applications_manifest()
    requested = (
        selection if selection is not None else os.environ.get(APPLICATION_ENVIRONMENT_VARIABLE)
    )
    key = DEFAULT_APPLICATION_KEY if requested is None else requested
    if key != key.strip() or not key:
        raise ValueError(f"{APPLICATION_ENVIRONMENT_VARIABLE} must be a non-empty application key")
    _validate_application_key(key)
    target = loaded_manifest.applications.get(key)
    if target is None:
        choices = ", ".join(sorted(loaded_manifest.applications))
        raise ValueError(f"unknown application {key!r}; available keys: {choices}")
    factory = _resolve_factory(key, target)
    return ResolvedContainerApplication(
        key=key,
        target=target,
        manifest_version=loaded_manifest.manifest_version,
        manifest_source=loaded_manifest.source,
        manifest_path=loaded_manifest.source_path,
        factory=factory,
    )


def safe_container_application_metadata(
    config: ContainerConfig,
    resolved: ResolvedContainerApplication,
) -> dict[str, Any]:
    """Return safe container metadata suitable for diagnostics and inspection."""
    return {
        "application": {
            "agent_id": config.agent_id,
            "agent_version": config.agent_version,
            "key": resolved.key,
            "target": resolved.target,
            "manifest_version": resolved.manifest_version,
            "manifest_source": resolved.manifest_source,
            "manifest_path": resolved.manifest_path,
        },
        "image": {
            "version": os.environ.get("CONDUCTO_IMAGE_VERSION", "dev"),
            "source_revision": os.environ.get("CONDUCTO_SOURCE_REVISION", "unknown"),
        },
    }


def _load_manifest_path(path: Path) -> ContainerApplicationsManifest:
    """Load and validate one manifest file from ``path``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("applications manifest could not be read") from error
    try:
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("applications manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("applications manifest must contain a JSON object")
    return _manifest_from_document(document, source="file", source_path=str(path))


def _manifest_from_document(
    document: dict[str, Any],
    *,
    source: str,
    source_path: str | None,
) -> ContainerApplicationsManifest:
    """Validate one manifest document and freeze its application mapping."""
    manifest_version = document.get("manifestVersion", DEFAULT_MANIFEST_VERSION)
    applications = document.get("applications")
    if applications is None and "applications" not in document:
        applications = document
        manifest_version = DEFAULT_MANIFEST_VERSION
    if not isinstance(manifest_version, str) or not manifest_version.strip():
        raise ValueError("applications manifest version must be a non-empty string")
    if isinstance(applications, (str, bytes)) or not isinstance(applications, dict):
        raise ValueError("applications manifest 'applications' field must be an object")
    if not applications:
        raise ValueError("applications manifest must declare at least one application")
    normalized: dict[str, str] = {}
    for key, target in applications.items():
        if not isinstance(key, str):
            raise ValueError("application manifest keys must be strings")
        _validate_application_key(key)
        if not isinstance(target, str):
            raise ValueError(f"application {key!r} target must be a string")
        normalized[key] = _validate_target(key, target)
    return ContainerApplicationsManifest(
        manifest_version=manifest_version.strip(),
        applications=normalized,
        source=source,
        source_path=source_path,
    )


def _resolve_factory(key: str, target: str) -> ContainerApplication:
    """Import and validate one server-owned application factory target."""
    module_name, attribute_name = target.split(":", maxsplit=1)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        raise ValueError(
            f"application {key!r} module could not be imported: {module_name}"
        ) from error
    except ImportError as error:
        detail = _redact_message(str(error)).strip() or module_name
        raise ValueError(f"application {key!r} module import failed: {detail}") from error
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(
            f"application {key!r} attribute could not be resolved: {attribute_name}"
        ) from error
    if not callable(factory):
        raise ValueError(f"application {key!r} target must resolve to a callable factory")
    return cast(ContainerApplication, factory)


def _validate_application_key(key: str) -> None:
    """Reject malformed application keys before application readiness."""
    if key != key.strip() or not key:
        raise ValueError("application keys must be non-empty and free of surrounding whitespace")
    if not _APPLICATION_KEY_PATTERN.fullmatch(key):
        raise ValueError(
            "application keys must contain only lowercase letters, digits, and hyphens"
        )


def _validate_target(key: str, target: str) -> str:
    """Validate one ``module:attribute`` target string."""
    if target != target.strip() or not target:
        raise ValueError(f"application {key!r} target must be a non-empty module:attribute")
    parts = target.split(":")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"application {key!r} target must be formatted as module:attribute")
    return target


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys in one parsed JSON object."""
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"applications manifest contains duplicate key(s): {joined}")
    return result


def _redact_message(message: str) -> str:
    """Return ``message`` with secret-like substrings replaced."""
    redacted = message
    for name in _SECRET_NAMES:
        redacted = redacted.replace(name, "<redacted>")
    return redacted


__all__ = [
    "APPLICATIONS_MANIFEST_ENVIRONMENT_VARIABLE",
    "APPLICATION_ENVIRONMENT_VARIABLE",
    "ContainerApplication",
    "ContainerApplicationsManifest",
    "DEFAULT_APPLICATIONS_MANIFEST_PATH",
    "DEFAULT_APPLICATION_KEY",
    "DEFAULT_MANIFEST_VERSION",
    "ResolvedContainerApplication",
    "load_container_applications_manifest",
    "resolve_container_application",
    "safe_container_application_metadata",
]
