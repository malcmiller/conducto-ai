"""Readiness policy, probes, and TTL-bounded verification for data sources.

A declared data source is only useful once it exists and holds a queryable
corpus. This module defines when that verification runs, how long an affirmative
verdict may be reused, and how a failed verification is reported.

Notes:
    Verification never repairs a data source. A failed verdict is reported as a
    typed failure; remediation, reprovisioning, and re-ingestion remain
    deployment decisions.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Flag, auto
from typing import Any, Protocol

from .errors import (
    DataSourceNotReadyError,
    LifecycleCancelledError,
    LifecycleTimeoutError,
    ReadinessProbeError,
)
from .provisioning import LifecycleBudget, resolve_budget

__all__ = [
    "ReadinessCheck",
    "ReadinessGate",
    "ReadinessPolicy",
    "ReadinessProbe",
    "ReadinessVerdict",
    "resolve_readiness_policy",
]


class ReadinessCheck(Flag):
    """Selects when a declared data source is verified queryable.

    Attributes:
        NONE: No verification. A deliberate, documented opt-out for in-memory
            sources and test fixtures that must be selected explicitly.
        ON_START: Verify during host startup so a host never reports ready over a
            missing or empty index.
        ON_INVOKE: Verify before serving an invocation so a source retired,
            emptied, or re-provisioned after startup is caught.

    Notes:
        The members combine. ``ON_START | ON_INVOKE`` is the recommended
        production setting, because startup verification and invocation-time
        verification cover different failures and neither subsumes the other.
    """

    NONE = 0
    ON_START = auto()
    ON_INVOKE = auto()


@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    """Declarative readiness verification policy for one data source.

    Attributes:
        checks: Combinable moments at which verification runs. Defaults to
            :attr:`ReadinessCheck.ON_START`, the cheap fail-fast option.
        readiness_ttl: Bound on how long an affirmative verdict may be reused by
            an ``ON_INVOKE`` check. A zero duration probes on every invocation.

    Notes:
        Only affirmative verdicts are cacheable, and the cache is invalidated by
        re-provisioning or re-ingestion of the source.
    """

    checks: ReadinessCheck = ReadinessCheck.ON_START
    readiness_ttl: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        """Validate the policy shape and its reuse bound."""
        if not isinstance(self.checks, ReadinessCheck):
            raise ValueError("checks must be a ReadinessCheck flag set")
        if not isinstance(self.readiness_ttl, timedelta):
            raise ValueError("readiness_ttl must be a timedelta")
        seconds = self.readiness_ttl.total_seconds()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("readiness_ttl must be a non-negative finite duration")

    @property
    def ttl_seconds(self) -> float:
        """Return the affirmative-verdict reuse bound in seconds."""
        return self.readiness_ttl.total_seconds()

    def includes(self, moment: ReadinessCheck) -> bool:
        """Return whether verification runs at the given moment.

        Args:
            moment: A single check moment, such as ``ReadinessCheck.ON_INVOKE``.

        Returns:
            ``True`` when the policy selects that moment.
        """
        return bool(self.checks & moment)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe projection of this policy."""
        return {
            "checks": sorted(
                member.name
                for member in ReadinessCheck
                if member is not ReadinessCheck.NONE
                and member.name is not None
                and bool(self.checks & member)
            ),
            "readiness_ttl_seconds": self.ttl_seconds,
        }


def resolve_readiness_policy(policy: ReadinessPolicy | None) -> ReadinessPolicy:
    """Return the declared policy or the documented default.

    Args:
        policy: Explicitly declared policy, or ``None``.

    Returns:
        ``policy`` when declared, otherwise a policy that checks
        :attr:`ReadinessCheck.ON_START`. ``ReadinessCheck.NONE`` is never
        reachable by omission.
    """
    return policy if policy is not None else ReadinessPolicy()


@dataclass(frozen=True, slots=True)
class ReadinessVerdict:
    """Result of one readiness evaluation.

    Attributes:
        data_source: Logical name the verdict belongs to.
        ready: Whether the source is provisioned, populated, and queryable.
        reason: Stable snake_case reason code for the verdict.
        evaluated_at: Monotonic timestamp of the evaluation.
        revision: Provisioning revision observed, when the source exists.
        document_count: Indexed document count observed, when known.
        from_cache: Whether an affirmative verdict was reused instead of probed.
    """

    data_source: str
    ready: bool
    reason: str
    evaluated_at: float
    revision: int | None = None
    document_count: int | None = None
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, credential-free projection of this verdict."""
        return {
            "data_source": self.data_source,
            "ready": self.ready,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
            "revision": self.revision,
            "document_count": self.document_count,
            "from_cache": self.from_cache,
        }


class ReadinessProbe(Protocol):
    """Structural contract for verifying that a data source is queryable."""

    async def probe(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict:
        """Evaluate whether one data source is provisioned and queryable.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            An affirmative or negative verdict. A probe that cannot reach a
            conclusion raises instead of returning an assumed pass.

        Raises:
            ReadinessProbeError: If the probe failed.
            LifecycleTimeoutError: If the probe exceeded its budget.
            LifecycleCancelledError: If cancellation was requested.
        """
        ...


class ReadinessGate:
    """Applies a readiness policy with a bounded affirmative-verdict cache.

    Args:
        probe: Backend probe used to evaluate a data source.
        policy: Declarative policy, or ``None`` to use the documented default of
            :attr:`ReadinessCheck.ON_START`.
        clock: Monotonic timestamp source used for TTL accounting.

    Notes:
        Only affirmative verdicts are cached, and only for ``readiness_ttl``. A
        negative or failed verdict is never cached and never suppresses a later
        check. :meth:`invalidate` is called by the lifecycle whenever a source is
        re-provisioned or re-ingested.
    """

    __slots__ = ("_clock", "_policy", "_probe", "_verdicts")

    def __init__(
        self,
        *,
        probe: ReadinessProbe,
        policy: ReadinessPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._policy = resolve_readiness_policy(policy)
        self._clock = clock
        self._verdicts: dict[str, tuple[ReadinessVerdict, float]] = {}

    @property
    def policy(self) -> ReadinessPolicy:
        """Return the resolved readiness policy."""
        return self._policy

    def invalidate(self, data_source: str) -> None:
        """Drop any cached affirmative verdict for one data source.

        Args:
            data_source: Logical name whose cached verdict is no longer valid.
        """
        self._verdicts.pop(data_source, None)

    def cached_verdict(self, data_source: str) -> ReadinessVerdict | None:
        """Return a still-valid cached affirmative verdict, if any.

        Args:
            data_source: Logical name to look up.

        Returns:
            The cached verdict when it has not expired, otherwise ``None``.
        """
        entry = self._verdicts.get(data_source)
        if entry is None:
            return None
        verdict, expires_at = entry
        if self._clock() >= expires_at:
            self._verdicts.pop(data_source, None)
            return None
        return verdict

    async def verify_on_start(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict | None:
        """Verify a data source during host startup.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            The affirmative verdict, or ``None`` when the policy does not select
            ``ON_START`` and no probe was performed.

        Raises:
            DataSourceNotReadyError: If the source is absent, empty, or not
                queryable. Startup fails and the host stays not-ready rather than
                degrading to serving.
            ReadinessProbeError: If the probe failed, timed out, or was
                cancelled.
        """
        return await self._verify(data_source, moment=ReadinessCheck.ON_START, budget=budget)

    async def verify_on_invoke(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None = None,
    ) -> ReadinessVerdict | None:
        """Verify a data source before serving one invocation.

        Args:
            data_source: Logical name of the declared data source.
            budget: Optional deadline and cancellation bound.

        Returns:
            The affirmative verdict, reused from cache while it is within
            ``readiness_ttl``, or ``None`` when the policy does not select
            ``ON_INVOKE``.

        Raises:
            DataSourceNotReadyError: If the source is absent, empty, or not
                queryable. The invocation is refused; no empty, partial, or
                success-shaped degraded result is produced.
            ReadinessProbeError: If the probe failed, timed out, or was
                cancelled.
        """
        return await self._verify(data_source, moment=ReadinessCheck.ON_INVOKE, budget=budget)

    async def _verify(
        self,
        data_source: str,
        *,
        moment: ReadinessCheck,
        budget: LifecycleBudget | None,
    ) -> ReadinessVerdict | None:
        """Probe or reuse a verdict for one check moment."""
        if not self._policy.includes(moment):
            return None
        if moment is ReadinessCheck.ON_INVOKE and (cached := self.cached_verdict(data_source)):
            return cached
        verdict = await self._probe_once(data_source, budget=budget)
        if not verdict.ready:
            self.invalidate(data_source)
            raise DataSourceNotReadyError(
                "Declared data source is not queryable",
                data_source=data_source,
                reason=verdict.reason,
            )
        ttl_seconds = self._policy.ttl_seconds
        if ttl_seconds > 0:
            self._verdicts[data_source] = (verdict, self._clock() + ttl_seconds)
        return verdict

    async def _probe_once(
        self,
        data_source: str,
        *,
        budget: LifecycleBudget | None,
    ) -> ReadinessVerdict:
        """Run one probe, mapping budget and backend failures to typed errors."""
        resolved = resolve_budget(budget)
        try:
            return await resolved.run(
                self._probe.probe(data_source, budget=resolved),
                data_source=data_source,
                operation="readiness_probe",
            )
        except LifecycleTimeoutError as error:
            self.invalidate(data_source)
            raise ReadinessProbeError(
                "Readiness probe exceeded its deadline",
                data_source=data_source,
                reason="readiness_timeout",
            ) from error
        except LifecycleCancelledError as error:
            self.invalidate(data_source)
            raise ReadinessProbeError(
                "Readiness probe was cancelled",
                data_source=data_source,
                reason="readiness_cancelled",
            ) from error
        except ReadinessProbeError:
            self.invalidate(data_source)
            raise
