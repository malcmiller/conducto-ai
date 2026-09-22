"""Issue and authenticate runtime-bound authority without resolving target objects."""

import hashlib
import hmac
import math
import time
import uuid

from ..gateway_models import (
    CapabilityBinding,
    CapabilityDescriptor,
    GatewayFailureCode,
    canonical_json,
)


class BindingAuthority:
    """Own binding signatures and expiration; registry admission stays atomic elsewhere."""

    def __init__(self, runtime_id: str, secret: bytes, ttl: float) -> None:
        self._runtime_id = runtime_id
        self._secret = secret
        self._ttl = ttl

    def issue(
        self,
        descriptor: CapabilityDescriptor,
        revision: int,
        generation: int,
    ) -> CapabilityBinding:
        """Sign a single snapshot candidate without granting additional authority."""
        issued_at = time.monotonic()
        values = {
            "agent_id": descriptor.agent_id,
            "capability_id": descriptor.capability_id,
            "schema_digest": descriptor.schema_digest,
            "registry_revision": revision,
            "registration_generation": generation,
            "runtime_id": self._runtime_id,
            "issued_at": issued_at,
            "expires_at": issued_at + self._ttl,
            "nonce": uuid.uuid4().hex,
        }
        signature = hmac.new(
            self._secret, canonical_json(values).encode(), hashlib.sha256
        ).hexdigest()
        return CapabilityBinding(
            agent_id=descriptor.agent_id,
            capability_id=descriptor.capability_id,
            schema_digest=descriptor.schema_digest,
            registry_revision=revision,
            registration_generation=generation,
            runtime_id=self._runtime_id,
            issued_at=issued_at,
            expires_at=issued_at + self._ttl,
            nonce=str(values["nonce"]),
            signature=signature,
        )

    def validate(self, binding: CapabilityBinding) -> GatewayFailureCode | None:
        """Authenticate structure, runtime ownership, lifetime, and signed fields."""
        if not isinstance(binding, CapabilityBinding):
            return GatewayFailureCode.INVALID_BINDING
        if (
            not all(
                isinstance(value, str) and bool(value)
                for value in (
                    binding.agent_id,
                    binding.capability_id,
                    binding.schema_digest,
                    binding.runtime_id,
                    binding.nonce,
                    binding.signature,
                )
            )
            or type(binding.registry_revision) is not int
            or binding.registry_revision < 0
            or type(binding.registration_generation) is not int
            or binding.registration_generation < 1
            or type(binding.issued_at) not in (int, float)
            or not math.isfinite(binding.issued_at)
            or type(binding.expires_at) not in (int, float)
            or not math.isfinite(binding.expires_at)
            or binding.expires_at <= binding.issued_at
            or len(binding.signature) != 64
            or any(character not in "0123456789abcdef" for character in binding.signature)
        ):
            return GatewayFailureCode.INVALID_BINDING
        if binding.runtime_id != self._runtime_id:
            return GatewayFailureCode.FOREIGN_RUNTIME
        if binding.expires_at <= time.monotonic():
            return GatewayFailureCode.EXPIRED_BINDING
        values = {
            "agent_id": binding.agent_id,
            "capability_id": binding.capability_id,
            "schema_digest": binding.schema_digest,
            "registry_revision": binding.registry_revision,
            "registration_generation": binding.registration_generation,
            "runtime_id": binding.runtime_id,
            "issued_at": binding.issued_at,
            "expires_at": binding.expires_at,
            "nonce": binding.nonce,
        }
        expected = hmac.new(
            self._secret, canonical_json(values).encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(binding.signature, expected):
            return GatewayFailureCode.INVALID_BINDING
        return None
