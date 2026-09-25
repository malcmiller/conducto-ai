"""Deployment-owned provisioning and lifecycle contracts for data sources.

A declared data source is metadata. This package defines how the resource behind
that declaration is created, populated, verified queryable, and retired, and who
owns each step:

* A deployment provisions, ingests, indexes, verifies, and retires.
* An agent reads through governed retrieval and never owns the source's
  existence, so it can outlive any single index it was bound to.

Importing this package performs no I/O, imports no backend SDK, and resolves no
credential. Concrete behavior is supplied by adapters; the in-memory reference
adapter demonstrates the full lifecycle without network access.
"""

from .errors import (
    DataSourceLifecycleError,
    DataSourceNotReadyError,
    IngestionError,
    LifecycleCancelledError,
    LifecycleTimeoutError,
    PartialIngestionError,
    ProvisioningError,
    ReadinessError,
    ReadinessProbeError,
    RetirementError,
)
from .ingestion import ContentBatch, ContentItem, DataSourceIngestor, IngestionProgress
from .lifecycle import DataSourceBackend, DataSourceLifecycle, ReadinessCheckedRetriever
from .provisioning import (
    DataSourceDescription,
    DataSourceProvisioner,
    DataSourceState,
    IndexingState,
    LifecycleBudget,
    ProvisionedBinding,
    ProvisioningConfig,
    resolve_budget,
)
from .readiness import (
    ReadinessCheck,
    ReadinessGate,
    ReadinessPolicy,
    ReadinessProbe,
    ReadinessVerdict,
    resolve_readiness_policy,
)

__all__ = [
    "ContentBatch",
    "ContentItem",
    "DataSourceBackend",
    "DataSourceDescription",
    "DataSourceIngestor",
    "DataSourceLifecycle",
    "DataSourceLifecycleError",
    "DataSourceNotReadyError",
    "DataSourceProvisioner",
    "DataSourceState",
    "IndexingState",
    "IngestionError",
    "IngestionProgress",
    "LifecycleBudget",
    "LifecycleCancelledError",
    "LifecycleTimeoutError",
    "PartialIngestionError",
    "ProvisionedBinding",
    "ProvisioningConfig",
    "ProvisioningError",
    "ReadinessCheck",
    "ReadinessCheckedRetriever",
    "ReadinessError",
    "ReadinessGate",
    "ReadinessPolicy",
    "ReadinessProbe",
    "ReadinessProbeError",
    "ReadinessVerdict",
    "RetirementError",
    "resolve_budget",
    "resolve_readiness_policy",
]
