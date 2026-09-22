"""Agent Card preparation and admission policy, separate from lease mutation."""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..a2a_profile import A2AProtocolError, parse_agent_card
from ..agent_card import capability_parameter_map
from ..gateway_models import canonical_json, thaw_json
from ._models import CatalogCapabilityDescriptor, CatalogEntry, CatalogValidationError

ProvenanceVerifier = Callable[[CatalogEntry], bool]


@dataclass(frozen=True, slots=True)
class PreparedEntry:
    """Validated card metadata ready for admission under the catalog lock."""

    card_name: str
    card_digest: str
    capabilities: tuple[CatalogCapabilityDescriptor, ...]


def prepare_entry(entry: CatalogEntry) -> PreparedEntry:
    """Parse and index the immutable card before acquiring the lifecycle lock."""
    raw_card = dict(thaw_json(entry.agent_card))
    try:
        parse_agent_card(raw_card)
    except A2AProtocolError as error:
        raise CatalogValidationError(
            f"Invalid Agent Card for '{entry.agent_id}': {error}"
        ) from error
    return PreparedEntry(
        str(raw_card.get("name", "")),
        hashlib.sha256(canonical_json(entry.agent_card).encode()).hexdigest(),
        _capabilities_from_card(raw_card),
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
    descriptors: list[CatalogCapabilityDescriptor] = []
    for skill in card.get("skills", []):
        capability_id = skill.get("id")
        input_modes = tuple(skill.get("inputModes", ()))
        descriptors.append(
            CatalogCapabilityDescriptor(
                capability_id=capability_id,
                description=skill.get("description"),
                tags=frozenset(skill.get("tags", ())),
                input_schema=parameter_map.get(capability_id, {}),
                output_schema=None,
                version=version,
                modality=input_modes[0] if input_modes else "text/plain",
                required_scopes=_required_scopes(skill),
            )
        )
    return tuple(descriptors)


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
