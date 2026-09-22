"""Canonical runtime adapter for accepted inbound A2A capability requests."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeGuard

from a2a.types.a2a_pb2 import Message
from google.protobuf.json_format import MessageToDict

from conducto.core.agent import BaseAgent
from conducto.core.agent_card import stable_skill_id
from conducto.core.gateway._bindings import BindingAuthority
from conducto.core.gateway_models import (
    CapabilityBinding,
    GatewayFailureCode,
    RegistrationLifecycle,
    canonical_json,
    freeze_json,
)
from conducto.core.invocation_results import (
    InvocationApprovalRequired,
    InvocationAuditFailure,
    InvocationAuthorizationFailure,
    InvocationBindingFailure,
    InvocationCancelled,
    InvocationInternalFailure,
    InvocationResult,
    InvocationSchemaMismatch,
    InvocationStaleBinding,
    InvocationTargetNotFound,
    InvocationTargetUnavailable,
    InvocationTimeout,
    InvocationValidationFailure,
)
from conducto.core.model_config import ModelReference, RunConfig, freeze_metadata
from conducto.core.run_context import CancellationState, DelegationBudget
from conducto.core.runtime import Runtime
from conducto.core.telemetry import SPAN_A2A_SERVER, extract_trace_context, start_span
from conducto.security import ApprovalDecision, AuditDeliveryError, AuthorizationContext
from conducto.security.errors import SecurityError
from conducto.security.tokens import TokenValidationError

from .handler import A2ARequestContext

_CONDUCTO_METADATA_KEY = "x-conducto"
_DEFAULT_MAX_TIMEOUT = 300.0
_DEFAULT_MAX_DELEGATION_DEPTH = 8
_DEFAULT_MAX_DELEGATION_CALLS = 32


@dataclass(frozen=True, slots=True)
class A2AAuthenticationRequest:
    """Immutable facts supplied to an application-owned identity resolver.

    Raw headers are available only at this boundary and are never copied into
    run metadata, results, logs, audit events, or task artifacts.
    """

    task_id: str
    context_id: str
    message_id: str
    request_id: str
    correlation_id: str
    headers: Mapping[str, str] = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze request facts before application authentication runs."""
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class A2AAuthenticatedIdentity:
    """Authenticated authority and optional approved-resume decision.

    Attributes:
        authorization: Principal and policy facts produced by the injected
            authentication resolver.
        allowed_capabilities: Optional resolver-owned root capability allowlist.
        delegation_budget: Optional resolver-owned root delegation budget.
        approval_decision: Optional authenticated approval decision.
    """

    authorization: AuthorizationContext
    allowed_capabilities: frozenset[str] | None = None
    delegation_budget: DelegationBudget | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    approval_decision: ApprovalDecision | None = None

    def __post_init__(self) -> None:
        """Freeze the capability allowlist."""
        if self.allowed_capabilities is not None:
            object.__setattr__(
                self,
                "allowed_capabilities",
                frozenset(self.allowed_capabilities),
            )


class A2AIdentityResolver(Protocol):
    """Application-owned authentication boundary for inbound A2A requests."""

    def __call__(
        self, request: A2AAuthenticationRequest
    ) -> A2AAuthenticatedIdentity | Awaitable[A2AAuthenticatedIdentity]:
        """Authenticate one request without exposing credentials to the runtime."""
        ...


@dataclass(frozen=True, slots=True)
class A2ACapabilityBinding:
    """Advertised A2A skill paired with an opaque runtime capability binding."""

    skill_id: str
    capability: CapabilityBinding = field(repr=False)


@dataclass(frozen=True, slots=True)
class _InvocationRequest:
    binding: A2ACapabilityBinding
    arguments: Mapping[str, Any]
    correlation_id: str
    run_config: RunConfig
    model_reference: ModelReference | None
    requested_capabilities: frozenset[str] | None
    requested_budget: tuple[int, int, int | None, float | None] | None
    fingerprint: str
    metadata: Mapping[str, Any]
    deadline_expired: bool

    @property
    def argument_digest(self) -> str:
        """Return a stable digest used to bind approved argument values."""
        return hashlib.sha256(canonical_json(self.arguments).encode()).hexdigest()


@dataclass(slots=True)
class _ReplayRecord:
    fingerprint: str
    execution: asyncio.Task[InvocationResult]
    result: InvocationResult | None = None

    def capture(self, execution: asyncio.Task[InvocationResult]) -> None:
        """Cache a completed result without retaining request payloads."""
        if execution.cancelled():
            return
        if execution.exception() is None:
            self.result = execution.result()


@dataclass(frozen=True, slots=True)
class _ApprovalBinding:
    argument_digest: str
    skill_id: str
    principal_namespace: str


class A2ARuntimeHandler:
    """Bind validated A2A messages to one agent through canonical ``Runtime``.

    Args:
        runtime: Runtime that owns model resolution, security, context, and execution.
        agent: Hosted agent. It is registered with ``runtime.agent_registry`` when
            absent; a conflicting registration fails construction.
        identity_resolver: Application-owned authentication and identity resolver.
        clock: UTC timestamp source used to convert inbound deadlines to timeouts.
        max_timeout: Maximum timeout accepted from transport metadata.
        max_delegation_depth: Maximum transport-requested delegation depth.
        max_delegation_calls: Maximum transport-requested delegation calls.

    Notes:
        This adapter never calls reflected methods. It revalidates an immutable
        runtime binding and dispatches exclusively through ``Runtime.invoke`` or
        ``Runtime.resume_approval``. Replay identifiers remain retained for the
        handler lifetime so completed requests cannot execute again.
    """

    def __init__(
        self,
        *,
        runtime: Runtime,
        agent: BaseAgent,
        identity_resolver: A2AIdentityResolver,
        clock: Callable[[], float] = time.time,
        max_timeout: float = _DEFAULT_MAX_TIMEOUT,
        max_delegation_depth: int = _DEFAULT_MAX_DELEGATION_DEPTH,
        max_delegation_calls: int = _DEFAULT_MAX_DELEGATION_CALLS,
    ) -> None:
        if not math.isfinite(max_timeout) or max_timeout <= 0:
            raise ValueError("max_timeout must be a finite positive number")
        if max_delegation_depth < 0 or max_delegation_calls < 0:
            raise ValueError("delegation limits cannot be negative")
        self._runtime = runtime
        self._agent = agent
        self._identity_resolver = identity_resolver
        self._clock = clock
        self._max_timeout = max_timeout
        self._max_delegation_depth = max_delegation_depth
        self._max_delegation_calls = max_delegation_calls
        self._replay_lock = asyncio.Lock()
        self._replays: dict[str, _ReplayRecord] = {}
        self._cancellations: dict[str, CancellationState] = {}
        self._cancelled_tasks: set[str] = set()
        self._approval_bindings: dict[str, _ApprovalBinding] = {}
        self._bindings = self._build_bindings()

    async def handle_message(
        self,
        message: Message,
        *,
        task_id: str,
        context_id: str,
        request_context: A2ARequestContext,
    ) -> InvocationResult:
        """Authenticate, bind, deduplicate, and invoke one accepted A2A message."""
        correlation_id = _correlation_id(message, task_id)
        parsed = self._parse_request(
            message,
            correlation_id,
            task_id=task_id,
            context_id=context_id,
            request_id=request_context.request_id,
            safe_metadata=request_context.safe_metadata,
        )
        if isinstance(parsed, InvocationValidationFailure | InvocationTargetNotFound):
            return parsed
        auth_request = A2AAuthenticationRequest(
            task_id=task_id,
            context_id=context_id,
            message_id=message.message_id,
            request_id=request_context.request_id,
            correlation_id=correlation_id,
            headers=request_context.headers,
            metadata=parsed.metadata,
        )
        trace = extract_trace_context(request_context.headers)
        with start_span(
            SPAN_A2A_SERVER,
            kind="server",
            remote_context=trace.context,
            attributes={
                "conducto.protocol": "a2a",
                "conducto.transport": "jsonrpc",
                "conducto.task.id": task_id,
                "conducto.correlation_id": correlation_id,
                "conducto.invalid_remote_context": trace.invalid_remote_context,
            },
        ) as span:
            identity = await self._authenticate(auth_request)
            if not isinstance(identity, A2AAuthenticatedIdentity):
                await self._discard_pending_cancellation(task_id)
                return identity
            if (
                identity.authorization.task_id != task_id
                or identity.authorization.correlation_id != correlation_id
            ):
                await self._discard_pending_cancellation(task_id)
                span.set_outcome("denied", reason="identity_context_mismatch")
                return InvocationAuthorizationFailure(correlation_id, "identity_context_mismatch")
            principal_namespace = _principal_namespace(identity.authorization)
            replay_keys = (
                f"{principal_namespace}:request:{request_context.request_id}",
                f"{principal_namespace}:message:{message.message_id}",
            )
            result = await self._invoke_once(
                replay_keys,
                parsed,
                identity,
                task_id=task_id,
            )
            span.set_outcome(type(result).__name__)
            return result

    async def cancel(self, task_id: str) -> None:
        """Request cooperative cancellation of an active task invocation."""
        async with self._replay_lock:
            cancellation = self._cancellations.get(task_id)
            if cancellation is None:
                self._cancelled_tasks.add(task_id)
        if cancellation is not None:
            cancellation.cancel()

    async def _discard_pending_cancellation(self, task_id: str) -> None:
        """Release a pre-execution cancellation when execution will not start."""
        async with self._replay_lock:
            self._cancelled_tasks.discard(task_id)

    def binding_for_skill(self, skill_id: str) -> A2ACapabilityBinding | None:
        """Return the immutable advertised binding for deterministic inspection."""
        return self._bindings.get(skill_id)

    def _build_bindings(self) -> dict[str, A2ACapabilityBinding]:
        registry = self._runtime.agent_registry
        agent_id = self._agent.agent_metadata.name
        registration = registry.registration(agent_id)
        if registration is None:
            registry.register(self._agent)
            registration = registry.registration(agent_id)
        elif registration[0] is not self._agent:
            raise ValueError(f"Runtime registry contains a different agent for {agent_id!r}")
        assert registration is not None
        _, descriptor = registration
        (
            runtime_id,
            secret,
            state,
            state_lock,
            binding_clock,
        ) = self._runtime.gateway_binding_material()
        authority = BindingAuthority(
            runtime_id,
            secret,
            self._runtime.gateway_binding_ttl,
            state_store=state,
            state_lock=state_lock,
            clock=binding_clock,
        )
        self._binding_authority = authority
        return {
            stable_skill_id(agent_id, item.capability_id): A2ACapabilityBinding(
                stable_skill_id(agent_id, item.capability_id),
                authority.issue(
                    item,
                    registry.revision,
                    descriptor.generation,
                ),
            )
            for item in descriptor.capabilities
        }

    def _parse_request(
        self,
        message: Message,
        correlation_id: str,
        *,
        task_id: str,
        context_id: str,
        request_id: str,
        safe_metadata: Mapping[str, Any],
    ) -> _InvocationRequest | InvocationValidationFailure | InvocationTargetNotFound:
        if not message.message_id:
            return _validation_failure(correlation_id, "message_id", "Message id is required")
        if len(message.parts) != 1 or message.parts[0].WhichOneof("content") != "text":
            return _validation_failure(
                correlation_id,
                "parts",
                "Exactly one JSON text part is required",
            )
        try:
            payload = json.loads(message.parts[0].text)
        except (json.JSONDecodeError, UnicodeError):
            return _validation_failure(correlation_id, "parts", "Invocation JSON is invalid")
        if not isinstance(payload, dict):
            return _validation_failure(correlation_id, "parts", "Invocation JSON must be an object")
        skill_id = payload.get("skillId")
        arguments = payload.get("arguments")
        if not isinstance(skill_id, str) or not skill_id:
            return _validation_failure(correlation_id, "skillId", "Skill id is required")
        binding = self._bindings.get(skill_id)
        if binding is None:
            return InvocationTargetNotFound(
                correlation_id,
                self._agent.agent_metadata.name,
                skill_id,
            )
        if not isinstance(arguments, dict):
            return _validation_failure(
                correlation_id,
                "arguments",
                "Arguments must be an object",
            )
        raw_metadata = MessageToDict(message.metadata) if len(message.metadata) else {}
        conducto_metadata = raw_metadata.get(_CONDUCTO_METADATA_KEY, {})
        if not isinstance(conducto_metadata, dict):
            return _validation_failure(
                correlation_id,
                _CONDUCTO_METADATA_KEY,
                "Conducto metadata must be an object",
            )
        try:
            run_config, model_reference, deadline_expired = self._run_config(
                conducto_metadata,
                message,
                task_id=task_id,
                context_id=context_id,
                request_id=request_id,
                safe_metadata=safe_metadata,
            )
            requested_capabilities = _requested_capabilities(conducto_metadata)
            requested_budget = self._requested_budget(conducto_metadata)
            frozen_arguments = freeze_json(arguments)
            fingerprint = hashlib.sha256(
                canonical_json(
                    {
                        "skillId": skill_id,
                        "arguments": arguments,
                        "contextId": context_id,
                        "taskId": task_id,
                        "conducto": conducto_metadata,
                        "safeMetadata": safe_metadata,
                    }
                ).encode()
            ).hexdigest()
        except (TypeError, ValueError) as error:
            return _validation_failure(
                correlation_id,
                _CONDUCTO_METADATA_KEY,
                str(error),
            )
        return _InvocationRequest(
            binding=binding,
            arguments=frozen_arguments,
            correlation_id=correlation_id,
            run_config=run_config,
            model_reference=model_reference,
            requested_capabilities=requested_capabilities,
            requested_budget=requested_budget,
            fingerprint=fingerprint,
            metadata=run_config.metadata,
            deadline_expired=deadline_expired,
        )

    def _run_config(
        self,
        metadata: Mapping[str, Any],
        message: Message,
        *,
        task_id: str,
        context_id: str,
        request_id: str,
        safe_metadata: Mapping[str, Any],
    ) -> tuple[RunConfig, ModelReference | None, bool]:
        timeout = metadata.get("timeoutSeconds")
        deadline = metadata.get("deadline")
        deadline_expired = False
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
        ):
            raise ValueError("timeoutSeconds must be a finite number")
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, int | float):
                raise ValueError("deadline must be a finite Unix timestamp")
            remaining = float(deadline) - self._clock()
            if not math.isfinite(remaining):
                raise ValueError("deadline must be a finite Unix timestamp")
            deadline_expired = remaining <= 0
            timeout = remaining if timeout is None else min(float(timeout), remaining)
        if timeout is not None and float(timeout) <= 0:
            deadline_expired = True
        if timeout is not None:
            timeout = min(float(timeout), self._max_timeout)
        safe = metadata.get("metadata", {})
        if not isinstance(safe, dict):
            raise ValueError("metadata must be an object")
        attributes = {
            **safe,
            **safe_metadata,
            "a2a": {
                "context_id": context_id,
                "task_id": task_id,
                "message_id": message.message_id,
                "request_id": request_id,
                "lineage": list(message.reference_task_ids),
            },
        }
        reference = metadata.get("modelReference")
        if reference is not None and not isinstance(reference, str):
            raise ValueError("modelReference must be a string")
        model_reference = ModelReference(reference) if reference else None
        return (
            RunConfig(
                timeout=float(timeout) if timeout is not None and timeout > 0 else None,
                metadata=attributes,
            ),
            model_reference,
            deadline_expired,
        )

    async def _authenticate(
        self,
        request: A2AAuthenticationRequest,
    ) -> A2AAuthenticatedIdentity | InvocationResult:
        from conducto.transport.errors import AuthenticationError

        try:
            resolved = self._identity_resolver(request)
            identity = await resolved if inspect.isawaitable(resolved) else resolved
        except AuditDeliveryError as error:
            return InvocationAuditFailure(request.correlation_id, error.reason_code)
        except (SecurityError, TokenValidationError) as error:
            return InvocationAuthorizationFailure(request.correlation_id, error.reason_code)
        except AuthenticationError:
            return InvocationAuthorizationFailure(
                request.correlation_id,
                "authentication_failed",
            )
        except Exception as error:
            return InvocationInternalFailure(
                request.correlation_id,
                "authentication_failed",
                error,
            )
        if not isinstance(identity, A2AAuthenticatedIdentity):
            return InvocationInternalFailure(
                request.correlation_id,
                "authentication_failed",
                TypeError("identity resolver returned an invalid result"),
            )
        return identity

    async def _invoke_once(
        self,
        replay_keys: tuple[str, str],
        request: _InvocationRequest,
        identity: A2AAuthenticatedIdentity,
        *,
        task_id: str,
    ) -> InvocationResult:
        async with self._replay_lock:
            records = [self._replays.get(key) for key in replay_keys]
            existing = next((record for record in records if record is not None), None)
            if existing is not None:
                if existing.fingerprint != request.fingerprint or any(
                    record is not None and record is not existing for record in records
                ):
                    return InvocationAuthorizationFailure(
                        request.correlation_id,
                        "replay_detected",
                    )
                if existing.result is not None:
                    return existing.result
                execution = existing.execution
            else:
                cancellation = CancellationState()
                if task_id in self._cancelled_tasks:
                    cancellation.cancel()
                self._cancellations[task_id] = cancellation
                execution = asyncio.create_task(
                    self._execute(request, identity, cancellation=cancellation)
                )
                record = _ReplayRecord(request.fingerprint, execution)
                execution.add_done_callback(record.capture)
                for key in replay_keys:
                    self._replays[key] = record
        try:
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            await self.cancel(task_id)
            raise
        finally:
            if execution.done():
                async with self._replay_lock:
                    self._cancellations.pop(task_id, None)
                    self._cancelled_tasks.discard(task_id)

    async def _execute(
        self,
        request: _InvocationRequest,
        identity: A2AAuthenticatedIdentity,
        *,
        cancellation: CancellationState,
    ) -> InvocationResult:
        binding = request.binding.capability
        failure, _ = self._binding_authority.resolve(binding)
        if failure is GatewayFailureCode.EXPIRED_BINDING:
            refreshed = self._refresh_binding(request.binding)
            if refreshed is not None:
                binding = refreshed.capability
                failure = None
        if failure is not None:
            return InvocationBindingFailure(request.correlation_id, failure.value)
        (
            agent,
            _,
            lifecycle,
            healthy,
            generation_valid,
            schema_valid,
        ) = self._runtime.agent_registry.accept_binding(
            agent_id=binding.agent_id,
            capability_id=binding.capability_id,
            generation=binding.registration_generation,
            schema_digest=binding.schema_digest,
        )
        if agent is None or not generation_valid:
            return InvocationStaleBinding(
                request.correlation_id,
                binding.agent_id,
                binding.capability_id,
            )
        if not schema_valid:
            return InvocationSchemaMismatch(
                request.correlation_id,
                binding.agent_id,
                binding.capability_id,
            )
        if lifecycle is not RegistrationLifecycle.ACTIVE or not healthy:
            reason = lifecycle.value if lifecycle is not None else "unhealthy"
            if lifecycle is RegistrationLifecycle.ACTIVE and not healthy:
                reason = "unhealthy"
            return InvocationTargetUnavailable(
                request.correlation_id,
                binding.agent_id,
                binding.capability_id,
                reason,
            )
        allowed = _attenuate_capabilities(
            identity.allowed_capabilities,
            request.requested_capabilities,
        )
        exact_capability = f"{binding.agent_id}:{binding.capability_id}"
        if allowed is not None and not (
            binding.capability_id in allowed or exact_capability in allowed
        ):
            return InvocationAuthorizationFailure(
                request.correlation_id,
                "capability_not_allowed",
            )
        budget = _attenuate_budget(identity.delegation_budget, request.requested_budget)
        if cancellation.cancelled:
            return InvocationCancelled(request.correlation_id)
        if request.deadline_expired:
            return InvocationTimeout(request.correlation_id, 0.0)
        try:
            if identity.approval_decision is not None:
                approval = self._approval_bindings.get(identity.approval_decision.approval_id)
                if approval is None or approval != _ApprovalBinding(
                    request.argument_digest,
                    request.binding.skill_id,
                    _principal_namespace(identity.authorization),
                ):
                    return InvocationAuthorizationFailure(
                        request.correlation_id,
                        "approval_binding_mismatch",
                    )
                return await self._runtime.resume_approval(
                    agent,
                    binding.capability_id,
                    request.arguments,
                    identity.approval_decision,
                    authorization=identity.authorization,
                    model_reference=request.model_reference,
                    run_config=request.run_config,
                    allowed_capabilities=allowed,
                    delegation_budget=budget,
                    cancellation=cancellation,
                )
            result = await self._runtime.invoke(
                agent,
                binding.capability_id,
                request.arguments,
                correlation_id=request.correlation_id,
                model_reference=request.model_reference,
                run_config=request.run_config,
                authorization=identity.authorization,
                allowed_capabilities=allowed,
                delegation_budget=budget,
                cancellation=cancellation,
            )
            if isinstance(result, InvocationApprovalRequired):
                self._approval_bindings[result.challenge.approval_id] = _ApprovalBinding(
                    request.argument_digest,
                    request.binding.skill_id,
                    _principal_namespace(identity.authorization),
                )
            return result
        except AuditDeliveryError as error:
            return InvocationAuditFailure(request.correlation_id, error.reason_code)
        except SecurityError as error:
            return InvocationAuthorizationFailure(request.correlation_id, error.reason_code)
        except Exception as error:
            return InvocationInternalFailure(request.correlation_id, "internal_error", error)

    def _refresh_binding(
        self,
        advertised: A2ACapabilityBinding,
    ) -> A2ACapabilityBinding | None:
        """Renew an expired server-owned binding only when its target is unchanged."""
        binding = advertised.capability
        (
            agent,
            descriptor,
            lifecycle,
            healthy,
            generation_valid,
            schema_valid,
        ) = self._runtime.agent_registry.accept_binding(
            agent_id=binding.agent_id,
            capability_id=binding.capability_id,
            generation=binding.registration_generation,
            schema_digest=binding.schema_digest,
        )
        if (
            agent is None
            or descriptor is None
            or not generation_valid
            or not schema_valid
            or lifecycle is not RegistrationLifecycle.ACTIVE
            or not healthy
        ):
            return None
        capability = next(
            (
                item
                for item in descriptor.capabilities
                if item.capability_id == binding.capability_id
            ),
            None,
        )
        if capability is None:
            return None
        refreshed = A2ACapabilityBinding(
            advertised.skill_id,
            self._binding_authority.issue(
                capability,
                self._runtime.agent_registry.revision,
                descriptor.generation,
            ),
        )
        self._bindings[advertised.skill_id] = refreshed
        return refreshed

    def _requested_budget(
        self,
        metadata: Mapping[str, Any],
    ) -> tuple[int, int, int | None, float | None] | None:
        requested = _parse_requested_budget(metadata)
        if requested is None:
            return None
        max_depth, calls, tokens, cost = requested
        return (
            min(max_depth, self._max_delegation_depth),
            min(calls, self._max_delegation_calls),
            tokens,
            cost,
        )


def _correlation_id(message: Message, task_id: str) -> str:
    metadata = MessageToDict(message.metadata) if len(message.metadata) else {}
    conducto = metadata.get(_CONDUCTO_METADATA_KEY, {})
    if isinstance(conducto, dict):
        value = conducto.get("correlationId")
        if isinstance(value, str) and value:
            return value
    return task_id


def _principal_namespace(authorization: AuthorizationContext) -> str:
    """Return a stable non-secret replay namespace for one authenticated principal."""
    principal = authorization.principal
    audience = (
        principal.audience if isinstance(principal.audience, str) else "\0".join(principal.audience)
    )
    value = f"{principal.issuer}\0{principal.subject_id}\0{audience}"
    return hashlib.sha256(value.encode()).hexdigest()


def _validation_failure(
    correlation_id: str,
    location: str,
    message: str,
) -> InvocationValidationFailure:
    return InvocationValidationFailure(
        correlation_id,
        (MappingProxyType({"loc": (location,), "msg": message, "type": "value_error"}),),
    )


def _requested_capabilities(metadata: Mapping[str, Any]) -> frozenset[str] | None:
    value = metadata.get("allowedCapabilities")
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("allowedCapabilities must be an array of non-empty strings")
    return frozenset(value)


def _parse_requested_budget(
    metadata: Mapping[str, Any],
) -> tuple[int, int, int | None, float | None] | None:
    value = metadata.get("budget")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("budget must be an object")
    max_depth = value.get("maxDepth", 8)
    calls = value.get("calls", 32)
    tokens = value.get("tokens")
    cost = value.get("cost")
    if not _is_json_integer(max_depth) or not _is_json_integer(calls):
        raise ValueError("budget maxDepth and calls must be integers")
    if tokens is not None and not _is_json_integer(tokens):
        raise ValueError("budget tokens must be an integer")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, int | float)
        or not math.isfinite(float(cost))
    ):
        raise ValueError("budget cost must be numeric")
    normalized_depth = int(max_depth)
    normalized_calls = int(calls)
    normalized_tokens = int(tokens) if tokens is not None else None
    DelegationBudget(
        max_depth=normalized_depth,
        calls=normalized_calls,
        tokens=normalized_tokens,
        cost=cost,
    )
    return (
        normalized_depth,
        normalized_calls,
        normalized_tokens,
        float(cost) if cost is not None else None,
    )


def _is_json_integer(value: Any) -> TypeGuard[int | float]:
    """Return whether a protobuf-JSON number represents an exact integer."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value).is_integer()
    )


def _attenuate_capabilities(
    authenticated: frozenset[str] | None,
    requested: frozenset[str] | None,
) -> frozenset[str] | None:
    if authenticated is None:
        return requested
    if requested is None:
        return authenticated
    return authenticated & requested


def _attenuate_budget(
    authenticated: DelegationBudget | None,
    requested: tuple[int, int, int | None, float | None] | None,
) -> DelegationBudget | None:
    authenticated_snapshot = (
        authenticated.snapshot(current_depth=0, remaining_time=None)
        if authenticated is not None
        else None
    )
    if authenticated_snapshot is None and requested is None:
        return None
    if authenticated_snapshot is None:
        assert requested is not None
        max_depth, calls, tokens, cost = requested
    elif requested is None:
        max_depth = authenticated_snapshot.depth
        calls = authenticated_snapshot.calls
        tokens = authenticated_snapshot.tokens
        cost = authenticated_snapshot.cost
    else:
        requested_depth, requested_calls, requested_tokens, requested_cost = requested
        max_depth = min(authenticated_snapshot.depth, requested_depth)
        calls = min(authenticated_snapshot.calls, requested_calls)
        tokens = _minimum_optional(authenticated_snapshot.tokens, requested_tokens)
        cost = _minimum_optional(authenticated_snapshot.cost, requested_cost)
    return DelegationBudget(
        max_depth=max_depth,
        calls=calls,
        tokens=tokens,
        cost=cost,
    )


def _minimum_optional[T: int | float](left: T | None, right: T | None) -> T | None:
    """Return the stricter optional numeric bound."""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


__all__ = [
    "A2AAuthenticatedIdentity",
    "A2AAuthenticationRequest",
    "A2ACapabilityBinding",
    "A2AIdentityResolver",
    "A2ARuntimeHandler",
]
