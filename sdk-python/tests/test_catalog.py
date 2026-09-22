"""Tests for the governed remote-agent catalog lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from conducto.core.catalog import (
    AgentCatalog,
    CatalogEntry,
    CatalogLifecycleState,
    CatalogProviderUnavailableError,
    CatalogValidationError,
    DeploymentType,
    InMemoryCatalogProvider,
    StaticFileCatalogProvider,
    UnknownCatalogAgentError,
    UnknownCatalogInstanceError,
)


def _card(
    *,
    name: str = "demo-agent",
    version: str = "1.0.0",
    skill_id: str = "skill-1",
    input_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extensions: list[dict[str, Any]] = []
    if input_schema is not None:
        extensions.append(
            {
                "uri": "https://conducto.ai/a2a/extensions/parameters/v1",
                "description": "params",
                "required": False,
                "params": {"x-conducto": {"parameters": {skill_id: input_schema}}},
            }
        )
    return {
        "name": name,
        "description": "a demo agent",
        "version": version,
        "supportedInterfaces": [
            {
                "url": "https://example.org/a2a",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
                "tenant": "",
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
            "extensions": extensions,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": skill_id,
                "name": "echo",
                "description": "echo text",
                "tags": [],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
                "examples": [],
                "securityRequirements": [],
            }
        ],
        "securitySchemes": {},
        "securityRequirements": [],
        "signatures": [],
    }


def _entry(
    *,
    agent_id: str = "org.demo",
    instance_id: str = "inst-1",
    card: Mapping[str, Any] | None = None,
    lease_seconds: float = 30.0,
    trust_policy_ref: str | None = None,
    signature: str | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        agent_id=agent_id,
        instance_id=instance_id,
        owner="org",
        deployment_type=DeploymentType.REMOTE_CONTAINER,
        agent_card_url="https://example.org/a2a/agent-card.json",
        agent_card=card if card is not None else _card(),
        lease_seconds=lease_seconds,
        trust_policy_ref=trust_policy_ref,
        signature=signature,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_register_instance_admits_agent_and_capabilities() -> None:
    catalog = AgentCatalog(clock=_Clock())
    record = catalog.register_instance(_entry())

    assert record.agent_id == "org.demo"
    assert record.lifecycle is CatalogLifecycleState.ACTIVE
    assert [capability.capability_id for capability in record.capabilities] == ["skill-1"]

    snapshot = catalog.snapshot()
    assert [agent.agent_id for agent in snapshot.agents] == ["org.demo"]
    assert [capability.capability_id for capability in snapshot.capabilities] == ["skill-1"]


def test_new_capability_becomes_visible_without_redeployment() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry())
    assert len(catalog.snapshot().capabilities) == 1

    updated_card = _card(version="1.1.0", skill_id="skill-2")
    catalog.register_instance(_entry(card=updated_card))

    capability_ids = {capability.capability_id for capability in catalog.snapshot().capabilities}
    assert "skill-2" in capability_ids


def test_multiple_instances_share_one_logical_agent() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry(instance_id="inst-1"))
    catalog.register_instance(_entry(instance_id="inst-2"))

    (record,) = catalog.snapshot().agents
    assert {instance.instance_id for instance in record.instances} == {"inst-1", "inst-2"}


def test_crashed_instance_ages_out_without_removing_healthy_sibling() -> None:
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(instance_id="inst-1", lease_seconds=10))
    catalog.register_instance(_entry(instance_id="inst-2", lease_seconds=100))

    clock.now = 11
    expired = catalog.expire_instances()
    assert expired == (("org.demo", "inst-1"),)

    (record,) = catalog.snapshot().agents
    assert [instance.instance_id for instance in record.instances] == ["inst-2"]


def test_snapshot_excludes_lease_expired_instances_without_explicit_expiry() -> None:
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(lease_seconds=10))
    assert len(catalog.snapshot().agents) == 1

    clock.now = 11
    assert catalog.snapshot().agents == ()


def test_lease_renewal_keeps_instance_eligible() -> None:
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(lease_seconds=10))

    clock.now = 9
    catalog.renew_lease("org.demo", "inst-1", lease_seconds=10)

    clock.now = 15
    assert len(catalog.snapshot().agents) == 1

    clock.now = 20
    assert catalog.snapshot().agents == ()


def test_renew_lease_requires_known_agent_and_instance() -> None:
    catalog = AgentCatalog(clock=_Clock())
    with pytest.raises(UnknownCatalogAgentError):
        catalog.renew_lease("missing", "inst-1")

    catalog.register_instance(_entry())
    with pytest.raises(UnknownCatalogInstanceError):
        catalog.renew_lease("org.demo", "missing-instance")


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        (lambda catalog: catalog.quarantine("org.demo"), CatalogLifecycleState.QUARANTINED),
        (lambda catalog: catalog.disable("org.demo"), CatalogLifecycleState.DISABLED),
        (lambda catalog: catalog.revoke("org.demo"), CatalogLifecycleState.REVOKED),
    ],
)
def test_lifecycle_transitions_exclude_agent_from_discovery(transition: Any, expected: Any) -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry())
    transition(catalog)

    assert catalog.get("org.demo").lifecycle is expected
    assert catalog.snapshot().agents == ()


def test_reactivate_restores_a_quarantined_agent() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry())
    catalog.quarantine("org.demo")
    assert catalog.snapshot().agents == ()

    catalog.reactivate("org.demo")
    assert [agent.agent_id for agent in catalog.snapshot().agents] == ["org.demo"]


def test_remove_deletes_the_registration() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry())
    catalog.remove("org.demo")

    assert catalog.get("org.demo") is None
    assert catalog.snapshot().agents == ()
    with pytest.raises(UnknownCatalogAgentError):
        catalog.remove("org.demo")


def test_identity_change_is_rejected_without_replacing_the_trusted_registration() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry())

    with pytest.raises(CatalogValidationError, match="identity changed"):
        catalog.register_instance(_entry(card=_card(name="different-agent")))

    # The original trusted registration must remain untouched.
    assert catalog.get("org.demo").card_digest is not None
    assert [agent.agent_id for agent in catalog.snapshot().agents] == ["org.demo"]


def test_capability_schema_change_without_version_bump_is_rejected() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry(card=_card(input_schema={"type": "object"})))

    changed_schema_card = _card(input_schema={"type": "string"})
    with pytest.raises(CatalogValidationError, match="without a version change"):
        catalog.register_instance(_entry(card=changed_schema_card))


def test_capability_schema_change_with_version_bump_is_admitted() -> None:
    catalog = AgentCatalog(clock=_Clock())
    catalog.register_instance(_entry(card=_card(version="1.0.0", input_schema={"type": "object"})))

    changed_schema_card = _card(version="2.0.0", input_schema={"type": "string"})
    catalog.register_instance(_entry(card=changed_schema_card))

    (capability,) = catalog.snapshot().capabilities
    assert capability.input_schema == {"type": "string"}


def test_provenance_is_required_and_verified_when_trust_policy_applies() -> None:
    catalog = AgentCatalog(
        clock=_Clock(),
        provenance_verifier=lambda entry: entry.signature == "valid-signature",
    )

    with pytest.raises(CatalogValidationError, match="requires signed provenance"):
        catalog.register_instance(_entry(trust_policy_ref="policy-v1"))

    with pytest.raises(CatalogValidationError, match="failed provenance verification"):
        catalog.register_instance(_entry(trust_policy_ref="policy-v1", signature="wrong-signature"))

    record = catalog.register_instance(
        _entry(trust_policy_ref="policy-v1", signature="valid-signature")
    )
    assert record.lifecycle is CatalogLifecycleState.ACTIVE


def test_invalid_agent_card_is_rejected() -> None:
    catalog = AgentCatalog(clock=_Clock())
    with pytest.raises(CatalogValidationError):
        catalog.register_instance(_entry(card={"name": "broken"}))


def test_catalog_entry_requires_identity_fields() -> None:
    with pytest.raises(CatalogValidationError):
        CatalogEntry(
            agent_id="",
            instance_id="inst-1",
            owner="org",
            deployment_type=DeploymentType.IN_PROCESS,
            agent_card_url="https://example.org/card.json",
            agent_card=_card(),
        )
    with pytest.raises(CatalogValidationError):
        CatalogEntry(
            agent_id="org.demo",
            instance_id="inst-1",
            owner="org",
            deployment_type=DeploymentType.IN_PROCESS,
            agent_card_url="not-a-url",
            agent_card=_card(),
        )


def test_in_memory_provider_refresh_admits_entries() -> None:
    provider = InMemoryCatalogProvider([_entry()])
    catalog = AgentCatalog(clock=_Clock())

    snapshot = catalog.refresh(provider)
    assert [agent.agent_id for agent in snapshot.agents] == ["org.demo"]

    provider.replace([_entry(), _entry(agent_id="org.other", instance_id="inst-9")])
    snapshot = catalog.refresh(provider)
    assert {agent.agent_id for agent in snapshot.agents} == {"org.demo", "org.other"}


def test_static_file_provider_loads_entries_from_json(tmp_path: Path) -> None:
    document = [
        {
            "agent_id": "org.demo",
            "instance_id": "inst-1",
            "owner": "org",
            "deployment_type": "remote_container",
            "agent_card_url": "https://example.org/a2a/agent-card.json",
            "agent_card": _card(),
            "lease_seconds": 60,
        }
    ]
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(document), encoding="utf-8")

    provider = StaticFileCatalogProvider(catalog_file)
    catalog = AgentCatalog(clock=_Clock())
    snapshot = catalog.refresh(provider)

    assert [agent.agent_id for agent in snapshot.agents] == ["org.demo"]


def test_static_file_provider_reports_missing_file() -> None:
    provider = StaticFileCatalogProvider("/nonexistent/catalog.json")
    with pytest.raises(CatalogProviderUnavailableError):
        provider.list_entries()


def test_static_file_provider_reports_invalid_json(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text("not json", encoding="utf-8")

    provider = StaticFileCatalogProvider(catalog_file)
    with pytest.raises(CatalogProviderUnavailableError):
        provider.list_entries()


def test_static_file_provider_reports_non_array_document(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    provider = StaticFileCatalogProvider(catalog_file)
    with pytest.raises(CatalogProviderUnavailableError):
        provider.list_entries()


def test_provider_unavailable_is_explicit_and_does_not_admit_entries() -> None:
    class BrokenProvider:
        def list_entries(self) -> tuple[CatalogEntry, ...]:
            raise RuntimeError("registry offline")

    catalog = AgentCatalog(clock=_Clock())
    with pytest.raises(CatalogProviderUnavailableError):
        catalog.refresh(BrokenProvider())

    assert catalog.snapshot().agents == ()


def test_capability_providers_filters_unhealthy_and_inactive_agents() -> None:
    clock = _Clock()
    catalog = AgentCatalog(clock=clock)
    catalog.register_instance(_entry(agent_id="org.a", instance_id="inst-1", lease_seconds=10))
    catalog.register_instance(_entry(agent_id="org.b", instance_id="inst-2", lease_seconds=100))

    assert {agent.agent_id for agent in catalog.capability_providers("skill-1")} == {
        "org.a",
        "org.b",
    }

    clock.now = 11
    assert {agent.agent_id for agent in catalog.capability_providers("skill-1")} == {"org.b"}
