"""Private configuration validation shared by first-party provider adapters."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from conducto.core.provider import ProviderError
from conducto.providers._http import DEFAULT_MAX_RESPONSE_BYTES


class GenerationDefaults(TypedDict):
    """Validated common factory settings accepted by both async adapters."""

    context_window: int | None
    max_output_tokens: int | None
    seed: int | None
    timeout: float | None
    max_response_bytes: int
    sampling: Mapping[str, int | float]
    provider_options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Bind shared validation to an adapter's typed configuration failure."""

    name: str
    error_type: type[ProviderError]

    def validate_options(
        self, values: Mapping[str, Any], allowed: set[str], label: str
    ) -> dict[str, Any]:
        """Return allowlisted options or fail without exposing their values."""
        unsupported = set(values) - allowed
        if unsupported:
            raise self.error_type(f"Unsupported {self.name} {label} options: {sorted(unsupported)}")
        return dict(values)

    def optional_int(self, value: Any, label: str) -> int | None:
        """Validate an optional integer, rejecting booleans."""
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise self.error_type(f"{label} must be an integer")
        return value

    def optional_positive_int(self, value: Any, label: str) -> int | None:
        """Validate an optional positive integer."""
        result = self.optional_int(value, label)
        if result is not None and result <= 0:
            raise self.error_type(f"{label} must be positive")
        return result

    def optional_positive_float(self, value: Any, label: str) -> float | None:
        """Validate an optional finite positive numeric setting."""
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise self.error_type(f"{label} must be positive")
        try:
            result = float(value)
        except OverflowError:
            raise self.error_type(f"{label} must be finite") from None
        if not math.isfinite(result):
            raise self.error_type(f"{label} must be finite")
        return result

    def credential(self, reference: str | None, *, label: str) -> str | None:
        """Resolve an environment credential without including it in errors."""
        if reference is None:
            return None
        value = os.environ.get(reference)
        if value is None:
            raise self.error_type(f"{label} is not available")
        if not value.strip():
            raise self.error_type(f"{label} is empty")
        return value

    def generation_defaults(self, values: dict[str, Any]) -> GenerationDefaults:
        """Pop common generation settings while leaving adapter-specific keys intact."""
        context_window = self.optional_positive_int(
            values.pop("context_window", None), "context_window"
        )
        max_output_tokens = self.optional_positive_int(
            values.pop("max_output_tokens", None), "max_output_tokens"
        )
        seed = self.optional_int(values.pop("seed", None), "seed")
        timeout = self.optional_positive_float(values.pop("timeout", None), "timeout")
        max_response_bytes = self.optional_positive_int(
            values.pop("max_response_bytes", DEFAULT_MAX_RESPONSE_BYTES), "max_response_bytes"
        )
        if max_response_bytes is None:
            raise self.error_type("max_response_bytes must be positive")
        return GenerationDefaults(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            seed=seed,
            timeout=timeout,
            max_response_bytes=max_response_bytes,
            sampling=cast(Mapping[str, int | float], extract_prefixed(values, "sampling_")),
            provider_options=extract_prefixed(values, "option_"),
        )

    def http_kwargs(
        self,
        *,
        token: str | None,
        headers: Mapping[str, str],
        timeout: float | None,
        transport_options: Mapping[str, Any],
        tls_options: Mapping[str, Any],
        proxy_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build bounded-client settings, sending credentials only as headers."""
        client_headers = dict(headers)
        if token is not None:
            # Replace case-insensitive duplicates rather than sending two credentials.
            client_headers = {
                key: value
                for key, value in client_headers.items()
                if key.lower() != "authorization"
            }
            client_headers["Authorization"] = f"Bearer {token}"
        kwargs: dict[str, Any] = {}
        if client_headers:
            kwargs["headers"] = client_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        pool_keys = {"max_connections", "max_keepalive_connections", "keepalive_expiry"}
        transport = self.validate_options(
            transport_options, pool_keys | {"follow_redirects"}, "transport"
        )
        if "follow_redirects" in transport:
            kwargs["follow_redirects"] = transport["follow_redirects"]
        limits = {key: transport[key] for key in pool_keys if key in transport}
        if limits:
            try:
                import httpx
            except ImportError as error:
                raise self.error_type(
                    f"httpx is required for {self.name} transport limits"
                ) from error
            kwargs["limits"] = httpx.Limits(**limits)
        kwargs.update(self.validate_options(tls_options, {"verify", "cert"}, "tls"))
        kwargs.update(self.validate_options(proxy_options, {"proxy"}, "proxy"))
        return kwargs


def extract_prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Pop provider defaults sharing a prefix, preserving their order."""
    return {
        key.removeprefix(prefix): values.pop(key) for key in tuple(values) if key.startswith(prefix)
    }
