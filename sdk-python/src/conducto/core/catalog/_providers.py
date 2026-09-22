"""Provider-neutral catalog loading, with typed source and parsing failures."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ._models import (
    CatalogEntry,
    CatalogError,
    CatalogProviderUnavailableError,
    CatalogValidationError,
    DeploymentType,
)


class CatalogProvider(Protocol):
    """Contract for any catalog backend that can enumerate admission entries."""

    def list_entries(self) -> Sequence[CatalogEntry]:
        """Return the current set of candidate entries for admission."""
        ...


class InMemoryCatalogProvider:
    """Reference catalog provider backed by an explicit in-memory entry list."""

    def __init__(self, entries: Sequence[CatalogEntry] = ()) -> None:
        """Initialize the provider with an initial, possibly empty, entry set."""
        self._entries: tuple[CatalogEntry, ...] = tuple(entries)

    def list_entries(self) -> tuple[CatalogEntry, ...]:
        """Return the current in-memory entries."""
        return self._entries

    def replace(self, entries: Sequence[CatalogEntry]) -> None:
        """Atomically replace the provider's complete entry set."""
        self._entries = tuple(entries)


class StaticFileCatalogProvider:
    """Load a JSON array of ``CatalogEntry`` objects from explicit configuration.

    Deployment types are their enum string values. Loading does not admit
    entries or change lifecycle; admission remains the catalog's responsibility.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind this provider to a static catalog configuration file path."""
        self._path = Path(path)

    def list_entries(self) -> tuple[CatalogEntry, ...]:
        """Read and parse the configured file into catalog entries.

        Raises:
            CatalogProviderUnavailableError: If the file cannot be read or
                does not contain a valid JSON array of catalog entries.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as error:
            raise CatalogProviderUnavailableError(
                f"Could not read catalog file '{self._path}': {error}"
            ) from error
        try:
            documents = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CatalogProviderUnavailableError(
                f"Catalog file '{self._path}' is not valid JSON: {error}"
            ) from error
        if not isinstance(documents, list):
            raise CatalogProviderUnavailableError(
                f"Catalog file '{self._path}' must contain a JSON array"
            )
        return tuple(_entry_from_document(document) for document in documents)


def load_entries(provider: CatalogProvider) -> Sequence[CatalogEntry]:
    """Normalize source failures without suppressing existing typed catalog errors."""
    try:
        return provider.list_entries()
    except CatalogError:
        raise
    except Exception as error:
        raise CatalogProviderUnavailableError(
            f"Catalog provider {provider!r} is unavailable: {error}"
        ) from error


def _entry_from_document(document: Any) -> CatalogEntry:
    if not isinstance(document, Mapping):
        raise CatalogProviderUnavailableError("Catalog file entries must be JSON objects")
    try:
        deployment_type = DeploymentType(document["deployment_type"])
    except (KeyError, ValueError) as error:
        raise CatalogProviderUnavailableError(
            f"Catalog file entry has an invalid deployment_type: {error}"
        ) from error
    try:
        return CatalogEntry(
            agent_id=document["agent_id"],
            instance_id=document["instance_id"],
            owner=document["owner"],
            deployment_type=deployment_type,
            agent_card_url=document["agent_card_url"],
            agent_card=document["agent_card"],
            provenance=document.get("provenance"),
            trust_policy_ref=document.get("trust_policy_ref"),
            supported_versions=frozenset(document.get("supported_versions", ())),
            transports=frozenset(document.get("transports", ())),
            lease_seconds=float(document.get("lease_seconds", 60.0)),
            signature=document.get("signature"),
        )
    except KeyError as error:
        raise CatalogProviderUnavailableError(
            f"Catalog file entry is missing required field {error}"
        ) from error
    except CatalogValidationError as error:
        raise CatalogProviderUnavailableError(str(error)) from error
