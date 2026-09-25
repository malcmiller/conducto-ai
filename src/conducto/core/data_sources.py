"""Credential-free declarations and registries for external data sources."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    "DataSourceMetadata",
    "DataSourceRegistrationError",
    "DataSourceRegistry",
    "DataSourceSnapshot",
    "MissingDataSourceError",
    "data_source",
    "get_data_source_metadata",
]

_DATA_SOURCE_METADATA_ATTRIBUTE = "__conducto_data_source_metadata__"
T = TypeVar("T")


class DataSourceRegistrationError(ValueError):
    """Raised when a data-source declaration is invalid or duplicated."""


class MissingDataSourceError(DataSourceRegistrationError):
    """Raised when a capability references an undeclared data source."""


@dataclass(frozen=True, slots=True)
class DataSourceMetadata:
    """Immutable contract metadata for one external data source.

    Attributes:
        name: Stable logical name used by capability declarations.
        kind: Connector category, such as ``fabric_ontology`` or ``sharepoint``.
        description: Optional human-readable purpose of the source.
        read_scopes: Canonically ordered scopes permitting reads.
        write_scopes: Canonically ordered scopes permitting writes.
        read_only: Whether the source is declared as read-only.
        binding_id: Opaque identifier resolved by trusted runtime configuration.
    """

    name: str
    kind: str
    description: str | None
    read_scopes: tuple[str, ...]
    write_scopes: tuple[str, ...]
    read_only: bool
    binding_id: str

    def __post_init__(self) -> None:
        """Validate and freeze declaration fields."""
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "kind", _required_text(self.kind, "kind"))
        object.__setattr__(self, "description", _optional_text(self.description, "description"))
        object.__setattr__(self, "read_scopes", _normalize_scopes(self.read_scopes, "read_scopes"))
        object.__setattr__(
            self, "write_scopes", _normalize_scopes(self.write_scopes, "write_scopes")
        )
        if not isinstance(self.read_only, bool):
            raise DataSourceRegistrationError("read_only must be a boolean")
        if self.read_only and self.write_scopes:
            raise DataSourceRegistrationError("read-only data sources cannot declare write_scopes")
        binding_id = _required_text(self.binding_id, "binding_id")
        if not binding_id.isascii() or any(
            not (character.isalnum() or character in "._-") for character in binding_id
        ):
            raise DataSourceRegistrationError(
                "binding_id must be an opaque identifier containing only letters, digits, '.', "
                "'_' or '-'"
            )
        object.__setattr__(self, "binding_id", binding_id)

    @property
    def declared_scopes(self) -> tuple[str, ...]:
        """Return all declared access scopes in stable order."""
        return tuple(sorted(set(self.read_scopes) | set(self.write_scopes)))

    def to_dict(self) -> dict[str, object]:
        """Return deterministic registry metadata without connector handles."""
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "read_scopes": list(self.read_scopes),
            "write_scopes": list(self.write_scopes),
            "declared_scopes": list(self.declared_scopes),
            "read_only": self.read_only,
            "binding_id": self.binding_id,
        }

    def to_audit_metadata(self) -> dict[str, str | bool | list[str] | None]:
        """Return a stable, credential-free audit projection of this source."""
        return {
            "name": self.name,
            "kind": self.kind,
            "binding_id": self.binding_id,
            "read_only": self.read_only,
            "declared_scopes": list(self.declared_scopes),
        }


@dataclass(frozen=True, slots=True)
class DataSourceSnapshot:
    """Atomic immutable snapshot of the configured data-source declarations.

    Attributes:
        revision: Monotonic mutation counter for this source registry.
        data_sources: Source declarations in stable name order.
    """

    revision: int
    data_sources: tuple[DataSourceMetadata, ...]

    def __post_init__(self) -> None:
        """Freeze declarations in stable name order."""
        object.__setattr__(
            self,
            "data_sources",
            tuple(sorted(self.data_sources, key=lambda source: source.name)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot in deterministic JSON-compatible form."""
        return {
            "revision": self.revision,
            "data_sources": [source.to_dict() for source in self.data_sources],
        }


def data_source(
    *,
    name: str,
    kind: str,
    description: str | None = None,
    read_scopes: Collection[str] = (),
    write_scopes: Collection[str] = (),
    read_only: bool = True,
    binding_id: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Declare credential-free metadata for an external system.

    Decorating a class records only immutable metadata; the class, its methods,
    and any clients it may construct are never retained by the registry.

    Args:
        name: Stable logical name referenced by ``@uses_data_source``.
        kind: Connector category, for example ``fabric_ontology`` or ``onelake``.
        description: Optional human-readable purpose.
        read_scopes: Scopes required to read from the source.
        write_scopes: Scopes required to write to the source.
        read_only: Whether the source is declared read-only.
        binding_id: Opaque runtime binding identifier. A stable opaque value is
            generated from ``name`` and ``kind`` when omitted.

    Returns:
        A decorator that attaches immutable declaration metadata and returns the
        decorated class unchanged.

    Raises:
        DataSourceRegistrationError: If metadata is invalid or the target is not
            a class.
    """
    normalized_name = _required_text(name, "name")
    normalized_kind = _required_text(kind, "kind")
    normalized_description = _optional_text(description, "description")
    normalized_reads = _normalize_scopes(read_scopes, "read_scopes")
    normalized_writes = _normalize_scopes(write_scopes, "write_scopes")
    if not isinstance(read_only, bool):
        raise DataSourceRegistrationError("read_only must be a boolean")
    if read_only and normalized_writes:
        raise DataSourceRegistrationError("read-only data sources cannot declare write_scopes")
    normalized_binding_id = (
        _required_text(binding_id, "binding_id")
        if binding_id is not None
        else "ds-"
        + hashlib.sha256(f"{normalized_name}:{normalized_kind}".encode()).hexdigest()[:24]
    )
    metadata = DataSourceMetadata(
        name=normalized_name,
        kind=normalized_kind,
        description=normalized_description,
        read_scopes=normalized_reads,
        write_scopes=normalized_writes,
        read_only=read_only,
        binding_id=normalized_binding_id,
    )

    def decorate(source_type: type[T]) -> type[T]:
        """Attach source metadata to a declaration class."""
        if not isinstance(source_type, type):
            raise DataSourceRegistrationError("@data_source can only decorate classes")
        setattr(source_type, _DATA_SOURCE_METADATA_ATTRIBUTE, metadata)
        return source_type

    return decorate


def get_data_source_metadata(source_type: type[object]) -> DataSourceMetadata | None:
    """Return metadata declared directly on a data-source class.

    Args:
        source_type: Class to inspect.

    Returns:
        Its direct metadata declaration, or ``None`` when it is not decorated.

    Raises:
        DataSourceRegistrationError: If the class contains malformed metadata.
    """
    if not isinstance(source_type, type):
        raise DataSourceRegistrationError("data-source declaration must be a class")
    metadata = source_type.__dict__.get(_DATA_SOURCE_METADATA_ATTRIBUTE)
    if metadata is not None and not isinstance(metadata, DataSourceMetadata):
        raise DataSourceRegistrationError("invalid Conducto data-source metadata")
    return metadata


class DataSourceRegistry:
    """Own immutable source declarations and publish deterministic snapshots."""

    def __init__(self) -> None:
        """Create an empty registry that owns metadata only."""
        self._lock = threading.RLock()
        self._data_sources: dict[str, DataSourceMetadata] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return the monotonic source-registry revision."""
        with self._lock:
            return self._revision

    def register(self, declaration: type[object] | DataSourceMetadata) -> DataSourceMetadata:
        """Register a decorated class or immutable declaration.

        Args:
            declaration: A class decorated with ``@data_source`` or its metadata.

        Returns:
            The immutable metadata added to the registry.

        Raises:
            DataSourceRegistrationError: If the declaration is invalid or its
                name is already registered.
        """
        metadata = (
            declaration
            if isinstance(declaration, DataSourceMetadata)
            else get_data_source_metadata(declaration)
        )
        if metadata is None:
            raise DataSourceRegistrationError(
                "data-source class must be decorated with @data_source"
            )
        with self._lock:
            if metadata.name in self._data_sources:
                raise DataSourceRegistrationError(
                    f"Data source '{metadata.name}' is already registered"
                )
            self._data_sources[metadata.name] = metadata
            self._revision += 1
        return metadata

    def resolve(self, names: Collection[str]) -> tuple[DataSourceMetadata, ...]:
        """Resolve declared names or fail with a deterministic missing-name error.

        Args:
            names: Names referenced by one or more capabilities.

        Returns:
            Matching declarations in canonical name order.

        Raises:
            MissingDataSourceError: If any requested name is not registered.
        """
        if isinstance(names, str) or not isinstance(names, Collection):
            raise DataSourceRegistrationError("data-source names must be a collection of strings")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise DataSourceRegistrationError(
                "data-source names must contain only non-empty strings"
            )
        normalized_names = tuple(sorted({name.strip() for name in names}))
        with self._lock:
            missing = tuple(name for name in normalized_names if name not in self._data_sources)
            if missing:
                raise MissingDataSourceError(
                    "Missing data source declaration(s): " + ", ".join(missing)
                )
            return tuple(self._data_sources[name] for name in normalized_names)

    def snapshot(self) -> DataSourceSnapshot:
        """Return a coherent immutable snapshot ordered by source name."""
        with self._lock:
            return DataSourceSnapshot(
                revision=self._revision,
                data_sources=tuple(self._data_sources.values()),
            )


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise DataSourceRegistrationError(f"{field} must be a non-empty string")
    return normalized


def _optional_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


def _normalize_scopes(values: Collection[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Collection):
        raise DataSourceRegistrationError(f"{field} must be a collection of non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise DataSourceRegistrationError(f"{field} must contain only non-empty strings")
    return tuple(sorted({value.strip() for value in values}))
