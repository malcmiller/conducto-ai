"""Agent Card preparation and admission policy, separate from lease mutation."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from ..a2a_profile import CONDUCTO_PARAMETER_EXTENSION_URI, A2AProtocolError, parse_agent_card
from ..agent_card import capability_parameter_map
from ..decorators import CapabilityBudget, CapabilityPolicyMetadata
from ..gateway_models import canonical_json, thaw_json
from ._models import CatalogCapabilityDescriptor, CatalogEntry, CatalogValidationError

ProvenanceVerifier = Callable[[CatalogEntry], bool]


@dataclass(frozen=True, slots=True)
class PreparedEntry:
    """Validated card metadata ready for admission under the catalog lock.

    ``logical_digest`` excludes instance interface URLs and card signatures so
    otherwise identical replicas can coexist without changing logical metadata.
    """

    card_name: str
    card_digest: str
    capabilities: tuple[CatalogCapabilityDescriptor, ...]
    logical_digest: str


def prepare_entry(entry: CatalogEntry) -> PreparedEntry:
    """Parse and index the immutable card before acquiring the lifecycle lock."""
    raw_card = dict(thaw_json(entry.agent_card))
    try:
        parse_agent_card(raw_card)
    except A2AProtocolError as error:
        raise CatalogValidationError(
            f"Invalid Agent Card for '{entry.agent_id}': {error}"
        ) from error
    logical_card = dict(raw_card)
    logical_card.pop("signatures", None)
    logical_card["supportedInterfaces"] = [
        {key: value for key, value in interface.items() if key != "url"}
        for interface in raw_card.get("supportedInterfaces", ())
    ]
    return PreparedEntry(
        str(raw_card.get("name", "")),
        hashlib.sha256(canonical_json(entry.agent_card).encode()).hexdigest(),
        _capabilities_from_card(raw_card),
        hashlib.sha256(canonical_json(logical_card).encode()).hexdigest(),
    )


class AdmissionPolicy:
    """Enforce provenance and compatibility against the currently locked registration."""

    def __init__(self, verifier: ProvenanceVerifier | None, require_provenance: bool) -> None:
        self._verifier = verifier
        self._require_provenance = require_provenance

    def validate(
        self,
        *,
        entry: CatalogEntry,
        prepared: PreparedEntry,
        existing_card_name: str | None,
        existing_capabilities: tuple[CatalogCapabilityDescriptor, ...],
    ) -> None:
        """Reject untrusted or incompatible updates before any state is published."""
        requires_provenance = self._require_provenance or entry.trust_policy_ref is not None
        if requires_provenance:
            if not entry.signature or self._verifier is None:
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' requires signed provenance verification"
                )
            if not self._verifier(entry):
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' failed provenance verification"
                )
        if existing_card_name and prepared.card_name != existing_card_name:
            raise CatalogValidationError(
                f"Agent '{entry.agent_id}' Agent Card identity changed from "
                f"'{existing_card_name}' to '{prepared.card_name}'; reject rather than "
                "silently replace a trusted registration"
            )
        new_by_id = {capability.capability_id: capability for capability in prepared.capabilities}
        for capability in existing_capabilities:
            updated = new_by_id.get(capability.capability_id)
            if updated is None:
                continue
            if (
                updated.version == capability.version
                and updated.input_schema != capability.input_schema
            ):
                raise CatalogValidationError(
                    f"Agent '{entry.agent_id}' capability '{capability.capability_id}' "
                    "changed its input schema without a version change"
                )


def _capabilities_from_card(card: Mapping[str, Any]) -> tuple[CatalogCapabilityDescriptor, ...]:
    version = str(card.get("version", ""))
    parameter_map = capability_parameter_map(card)
    skills = card.get("skills", [])
    dependencies = _data_source_dependencies(card, {skill.get("id") for skill in skills})
    descriptors: list[CatalogCapabilityDescriptor] = []
    for skill in skills:
        capability_id = skill.get("id")
        input_modes = tuple(skill.get("inputModes", ()))
        policy = _capability_policy(card, capability_id)
        descriptors.append(
            CatalogCapabilityDescriptor(
                capability_id=capability_id,
                name=str(skill.get("name", capability_id)),
                description=skill.get("description"),
                tags=frozenset(skill.get("tags", ())),
                input_schema=parameter_map.get(capability_id, {}),
                output_schema=None,
                version=version,
                modality=input_modes[0] if input_modes else "text/plain",
                required_scopes=tuple(
                    sorted(set(_required_scopes(skill)) | set(policy.required_scopes))
                ),
                policy=policy,
                data_sources=dependencies.get(capability_id, ()),
            )
        )
    return tuple(descriptors)


def _data_source_dependencies(
    card: Mapping[str, Any], skill_ids: set[Any]
) -> dict[str, tuple[str, ...]]:
    """Read and validate stable dependency names from the Conducto card extension."""
    extension = next(
        (
            item
            for item in card.get("capabilities", {}).get("extensions", ())
            if (
                isinstance(item, Mapping)
                and item.get("uri") == CONDUCTO_PARAMETER_EXTENSION_URI
                and isinstance(item.get("params"), Mapping)
            )
        ),
        None,
    )
    if extension is None:
        return {}
    conducto = extension["params"].get("x-conducto")
    if not isinstance(conducto, Mapping):
        return {}
    declared = conducto.get("dataSourceDependencies", {})
    if not isinstance(declared, Mapping):
        raise CatalogValidationError("dataSourceDependencies must be an object")
    unknown_skills = set(declared) - skill_ids
    if unknown_skills:
        raise CatalogValidationError(
            "dataSourceDependencies references unknown skill(s): "
            + ", ".join(sorted(str(skill_id) for skill_id in unknown_skills))
        )
    dependencies: dict[str, tuple[str, ...]] = {}
    for skill_id, names in declared.items():
        if not isinstance(skill_id, str):
            raise CatalogValidationError("dataSourceDependencies keys must be skill IDs")
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise CatalogValidationError(
                f"Capability '{skill_id}' data-source dependencies must be a sequence"
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise CatalogValidationError(
                f"Capability '{skill_id}' data-source dependencies must be non-empty strings"
            )
        dependencies[skill_id] = tuple(sorted({name.strip() for name in names}))
    return dependencies


def _capability_policy(card: Mapping[str, Any], capability_id: Any) -> CapabilityPolicyMetadata:
    """Read optional Conducto policy metadata for one Agent Card skill."""
    if not isinstance(capability_id, str):
        return CapabilityPolicyMetadata()
    extension = next(
        (
            item
            for item in card.get("capabilities", {}).get("extensions", ())
            if (
                isinstance(item, Mapping)
                and item.get("uri") == CONDUCTO_PARAMETER_EXTENSION_URI
                and isinstance(item.get("params"), Mapping)
            )
        ),
        None,
    )
    if extension is None:
        return CapabilityPolicyMetadata()
    conducto = extension["params"].get("x-conducto")
    if not isinstance(conducto, Mapping):
        return CapabilityPolicyMetadata()
    policies = conducto.get("capabilityPolicies")
    policy = policies.get(capability_id) if isinstance(policies, Mapping) else None
    if not isinstance(policy, Mapping):
        return CapabilityPolicyMetadata()
    required_scopes = _policy_scopes(policy, capability_id)
    budget = policy.get("budget")
    max_cost = budget.get("maxCostUsd") if isinstance(budget, Mapping) else None
    try:
        declared_budget = (
            CapabilityBudget(
                max_model_calls=budget.get("maxModelCalls"),
                max_tool_calls=budget.get("maxToolCalls"),
                max_cost_usd=Decimal(str(max_cost)) if max_cost is not None else None,
            )
            if isinstance(budget, Mapping)
            else None
        )
    except InvalidOperation as error:
        raise CatalogValidationError(
            f"Capability '{capability_id}' has an invalid policy budget"
        ) from error
    return CapabilityPolicyMetadata(
        required_scopes=required_scopes,
        side_effect=policy.get("sideEffect") if isinstance(policy.get("sideEffect"), str) else None,
        timeout_seconds=(
            float(policy["timeoutSeconds"])
            if isinstance(policy.get("timeoutSeconds"), int | float)
            else None
        ),
        budget=declared_budget,
        data_classification=(
            policy.get("dataClassification")
            if isinstance(policy.get("dataClassification"), str)
            else None
        ),
    )


def _policy_scopes(policy: Mapping[str, Any], capability_id: str) -> tuple[str, ...]:
    """Validate and normalize declared policy scopes from an Agent Card extension."""
    scopes = policy.get("requiredScopes", ())
    if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
        raise CatalogValidationError(
            f"Capability '{capability_id}' policy requiredScopes must be a sequence "
            "of non-empty strings"
        )
    if any(not isinstance(scope, str) or not scope.strip() for scope in scopes):
        raise CatalogValidationError(
            f"Capability '{capability_id}' policy requiredScopes must be a sequence "
            "of non-empty strings"
        )
    return tuple(sorted({scope.strip() for scope in scopes}))


def _required_scopes(skill: Mapping[str, Any]) -> tuple[str, ...]:
    scopes: set[str] = set()
    for requirement in skill.get("securityRequirements", ()) or ():
        if not isinstance(requirement, Mapping):
            continue
        schemes = requirement.get("schemes", {})
        if not isinstance(schemes, Mapping):
            continue
        for scheme in schemes.values():
            if isinstance(scheme, Mapping):
                scopes.update(scheme.get("list", ()))
    return tuple(sorted(scopes))
