"""Governed remote-agent admission, immutable discovery, and instance lifecycle."""

from ._admission import ProvenanceVerifier
from ._lifecycle import AgentCatalog
from ._models import (
    AgentInstanceRecord,
    CatalogAgentRecord,
    CatalogCapabilityDescriptor,
    CatalogEntry,
    CatalogError,
    CatalogLifecycleState,
    CatalogProviderUnavailableError,
    CatalogSnapshot,
    CatalogValidationError,
    DeploymentType,
    UnknownCatalogAgentError,
    UnknownCatalogInstanceError,
)
from ._providers import CatalogProvider, InMemoryCatalogProvider, StaticFileCatalogProvider

__all__ = [
    "AgentCatalog",
    "AgentInstanceRecord",
    "CatalogAgentRecord",
    "CatalogCapabilityDescriptor",
    "CatalogEntry",
    "CatalogError",
    "CatalogLifecycleState",
    "CatalogProvider",
    "CatalogProviderUnavailableError",
    "CatalogSnapshot",
    "CatalogValidationError",
    "DeploymentType",
    "InMemoryCatalogProvider",
    "ProvenanceVerifier",
    "StaticFileCatalogProvider",
    "UnknownCatalogAgentError",
    "UnknownCatalogInstanceError",
]
