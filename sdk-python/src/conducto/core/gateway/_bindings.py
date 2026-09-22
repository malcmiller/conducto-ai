"""Issue and authenticate runtime-bound authority without resolving target objects."""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..gateway_models import (
    CapabilityBinding,
    CapabilityDescriptor,
    GatewayFailureCode,
    canonical_json,
)


@dataclass(frozen=True, slots=True)
class _StoredBinding:
    expires_at: float
    state: object | None


class BindingAuthority:
    """Own binding signatures and expiration; registry admission stays atomic elsewhere."""

    def __init__(
        self,
        runtime_id: str,
        secret: bytes,
        ttl: float,
        *,
        state_store: dict[str, object] | None = None,
        state_lock: threading.Lock | None = None,
        clock: Callable[[], float],
    ) -> None:
        self._runtime_id = runtime_id
        self._secret = secret
        self._ttl = ttl
        self._state_store: dict[str, object] = state_store if state_store is not None else {}
        self._state_lock = state_lock if state_lock is not None else threading.Lock()
        self._clock = clock

    def issue(
        self,
        descriptor: CapabilityDescriptor,
        revision: int,
        generation: int,
        *,
        state: object | None = None,
    ) -> CapabilityBinding:
        """Sign a single snapshot candidate without granting additional authority."""
        issued_at = self._clock()
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
        with self._state_lock:
            self._prune_locked(now=issued_at)
            self._state_store[str(values["nonce"])] = _StoredBinding(
                expires_at=issued_at + self._ttl,
                state=state,
            )
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
        now = self._clock()
        if binding.expires_at <= now:
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

    def resolve(
        self,
        binding: CapabilityBinding,
    ) -> tuple[GatewayFailureCode | None, object | None]:
        """Authenticate a binding and return its immutable discovery snapshot state."""
        failure = self.validate(binding)
        if failure is not None:
            return failure, None
        with self._state_lock:
            stored = self._state_store.get(binding.nonce)
            if stored is None or not isinstance(stored, _StoredBinding):
                return GatewayFailureCode.INVALID_BINDING, None
            now = self._clock()
            if stored.expires_at <= now:
                self._state_store.pop(binding.nonce, None)
                return GatewayFailureCode.EXPIRED_BINDING, None
            return None, stored.state

    def _prune_locked(self, *, now: float) -> None:
        expired = [
            nonce
            for nonce, binding in self._state_store.items()
            if isinstance(binding, _StoredBinding) and binding.expires_at <= now
        ]
        for nonce in expired:
            self._state_store.pop(nonce, None)
