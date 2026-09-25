"""Tests for credential-free data-source declarations and capability bindings."""

from __future__ import annotations

import json

import pytest

from conducto import (
    BaseAgent,
    DataSourceRegistrationError,
    DataSourceRegistry,
    MissingDataSourceError,
    a2a_capability,
    data_source,
    requires_scope,
    uses_data_source,
)
from conducto.core.catalog import AgentCatalog, CatalogEntry, DeploymentType
from conducto.core.registry import AgentRegistry


@data_source(
    name="fabric_customer_ontology",
    kind="fabric_ontology",
    description="Customer ontology metadata.",
    read_scopes={"customer.read"},
)
class CustomerOntology:
    """Declaration for a customer ontology; it owns no client or credentials."""


@data_source(
    name="onelake_documents",
    kind="onelake",
    read_scopes={"document.read"},
)
class OneLakeDocuments:
    """Declaration for read-only document storage."""


def test_declarations_are_immutable_metadata_and_serialize_deterministically() -> None:
    registry = DataSourceRegistry()
    registry.register(OneLakeDocuments)
    registry.register(CustomerOntology)
    snapshot = registry.snapshot()

    assert [source.name for source in snapshot.data_sources] == [
        "fabric_customer_ontology",
        "onelake_documents",
    ]
    source = snapshot.data_sources[0]
    assert source.read_scopes == ("customer.read",)
    assert source.read_only
    assert source.binding_id.startswith("ds-")
    assert "endpoint" not in source.to_dict()
    assert json.dumps(snapshot.to_dict(), sort_keys=True) == json.dumps(
        registry.snapshot().to_dict(), sort_keys=True
    )
    audit_metadata = source.to_audit_metadata()
    assert audit_metadata == {
        "name": "fabric_customer_ontology",
        "kind": "fabric_ontology",
        "binding_id": source.binding_id,
        "read_only": True,
        "declared_scopes": ["customer.read"],
    }
    assert json.dumps(audit_metadata, sort_keys=True) == json.dumps(
        source.to_audit_metadata(), sort_keys=True
    )

    with pytest.raises(DataSourceRegistrationError, match="already registered"):
        registry.register(CustomerOntology)


def test_capability_dependencies_flow_through_registry_card_and_catalog() -> None:
    class CustomerAgent(BaseAgent):
        """Looks up a customer without exposing connector state to a model."""

        @a2a_capability(name="lookup_customer", description="Find a customer.")
        @requires_scope("customer.read")
        @uses_data_source("onelake_documents", "fabric_customer_ontology")
        def lookup(self, customer_id: str) -> str:
            return customer_id

    registry = AgentRegistry()
    registry.register_data_source(OneLakeDocuments)
    registry.register_data_source(CustomerOntology)
    agent = CustomerAgent()
    registry.register(agent)

    snapshot = registry.snapshot()
    capability = snapshot.capabilities[0]
    assert [source.name for source in snapshot.data_sources] == [
        "fabric_customer_ontology",
        "onelake_documents",
    ]
    assert capability.data_sources == ("fabric_customer_ontology", "onelake_documents")
    assert capability.to_dict()["data_sources"] == [
        "fabric_customer_ontology",
        "onelake_documents",
    ]

    card = agent.get_agent_card("https://example.test/a2a")
    card_json = agent.get_agent_card_json("https://example.test/a2a")
    assert json.loads(card_json) == card
    x_conducto = card["capabilities"]["extensions"][0]["params"]["x-conducto"]
    skill_id = card["skills"][0]["id"]
    assert x_conducto["dataSourceDependencies"][skill_id] == [
        "fabric_customer_ontology",
        "onelake_documents",
    ]
    assert "binding_id" not in card_json

    catalog = AgentCatalog(clock=lambda: 100.0)
    catalog.register_instance(
        CatalogEntry(
            agent_id="org.customers",
            instance_id="instance-1",
            owner="org",
            deployment_type=DeploymentType.REMOTE_CONTAINER,
            agent_card_url="https://example.test/a2a/agent-card.json",
            agent_card=card,
        )
    )
    catalog_snapshot = catalog.snapshot()
    assert catalog_snapshot.capabilities[0].data_sources == capability.data_sources
    serialized = json.dumps(catalog_snapshot.to_dict(), sort_keys=True)
    assert "example.test" not in serialized
    assert json.loads(serialized)["agents"][0]["capabilities"][0]["data_sources"] == [
        "fabric_customer_ontology",
        "onelake_documents",
    ]


def test_missing_dependency_fails_at_agent_registration() -> None:
    class MissingSourceAgent(BaseAgent):
        """Declares one unavailable source."""

        @a2a_capability(name="lookup", description="Finds a record.")
        @uses_data_source("missing_source")
        def lookup(self) -> str:
            return "value"

    registry = AgentRegistry()
    with pytest.raises(MissingDataSourceError, match="missing_source"):
        registry.register(MissingSourceAgent())


def test_data_source_declarations_reject_invalid_metadata() -> None:
    with pytest.raises(DataSourceRegistrationError, match="read-only"):
        data_source(
            name="readonly",
            kind="storage",
            write_scopes={"storage.write"},
        )

    with pytest.raises(DataSourceRegistrationError, match="at least one"):
        uses_data_source()

    with pytest.raises(DataSourceRegistrationError, match="opaque identifier"):
        data_source(
            name="invalid_binding",
            kind="storage",
            binding_id="https://example.test",
        )
