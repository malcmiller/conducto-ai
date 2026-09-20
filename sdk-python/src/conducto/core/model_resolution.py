"""Model precedence, policy evaluation, and capability validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .model_config import (
    AgentModelConfig,
    ModelReference,
    ModelRequirement,
    ModelResolutionSource,
    RunConfig,
    RuntimeConfig,
    normalize_reference,
)
from .provider import ModelConfiguration, ModelProvider, ProviderCapabilities
from .provider_registry import ProviderRegistry
from .run_context import ModelPolicy, ModelPolicyContext, RunContext
from .runtime_errors import (
    IncompatibleProviderCapabilitiesError,
    MissingModelDefaultError,
    ModelOverrideDeniedError,
)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Credential-free description of the model selected for a run."""

    reference: ModelReference
    provider: str
    source: ModelResolutionSource


@dataclass(frozen=True, slots=True)
class _ResolvedModelBinding:
    """Runtime-private model binding that owns provider access."""

    model: ResolvedModel
    client: ModelProvider = field(repr=False, compare=False)
    configuration: ModelConfiguration = field(repr=False, compare=False)


class ModelResolver:
    """Resolve models using stable precedence, policy, and capability rules."""

    def __init__(self, provider_registry: ProviderRegistry) -> None:
        self.provider_registry = provider_registry

    def resolve_binding(
        self,
        *,
        agent_id: str,
        agent_config: AgentModelConfig,
        run_config: RunConfig,
        runtime_config: RuntimeConfig,
        policy: ModelPolicy | None,
        call_override: ModelReference | str | None = None,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> _ResolvedModelBinding | None:
        """Resolve the best available model binding for an invocation.

        Args:
            agent_id: Name of the invoking agent.
            agent_config: Agent-level model requirements and defaults.
            run_config: Run-level model override and metadata.
            runtime_config: Runtime-wide defaults and policy settings.
            policy: Optional model selection policy hook.
            call_override: Optional per-call model override.
            required_capabilities: Capabilities required by the invocation.

        Returns:
            The resolved binding for the selected model, if one is available.

        Raises:
            MissingModelDefaultError: If a model is required but unavailable.
            ModelOverrideDeniedError: If the selected model fails the policy hook.
            IncompatibleProviderCapabilitiesError: If the provider lacks required features.
        """
        call_reference = normalize_reference(call_override)
        candidates = (
            (call_reference, ModelResolutionSource.CALL_OVERRIDE),
            (run_config.model, ModelResolutionSource.RUN_OVERRIDE),
            (agent_config.default_model, ModelResolutionSource.AGENT_DEFAULT),
            (runtime_config.default_model, ModelResolutionSource.RUNTIME_DEFAULT),
        )
        selected = next(((ref, source) for ref, source in candidates if ref is not None), None)
        if selected is None:
            if agent_config.requirement is ModelRequirement.REQUIRED or required_capabilities:
                raise MissingModelDefaultError(f"Agent '{agent_id}' requires a model")
            return None

        reference, source = selected
        registration = self.provider_registry.resolve(reference)
        self.validate_capabilities(
            reference,
            registration.client.capabilities,
            required_capabilities,
        )
        policy_context = ModelPolicyContext(
            agent_id=agent_id,
            model_reference=reference,
            provider=registration.provider,
            source=source,
            caller=run_config.caller,
            environment=run_config.environment,
            cost_tier=run_config.cost_tier,
            data_classification=run_config.data_classification,
            metadata=run_config.metadata,
        )
        if policy is not None and not policy(policy_context):
            raise ModelOverrideDeniedError(
                f"Model reference '{reference}' is denied for agent '{agent_id}'"
            )
        return _ResolvedModelBinding(
            ResolvedModel(reference, registration.provider, source),
            registration.client,
            registration.configuration,
        )

    def resolve_for_call_binding(
        self,
        context: RunContext,
        override: ModelReference | str | None,
        *,
        runtime_config: RuntimeConfig,
        policy: ModelPolicy | None,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> _ResolvedModelBinding | None:
        """Resolve a model binding using the active invocation context.

        Args:
            context: Active run context for the current invocation.
            override: Optional call-level model override.
            runtime_config: Runtime defaults and policy settings.
            policy: Optional model selection policy hook.
            required_capabilities: Capabilities required by the invocation.

        Returns:
            The active binding, or a newly resolved binding for the override.
        """
        if override is None:
            binding = context._binding
            if binding is not None:
                self.validate_capabilities(
                    binding.model.reference,
                    binding.client.capabilities,
                    required_capabilities,
                )
            return binding
        return self.resolve_binding(
            agent_id=context.agent_id,
            agent_config=AgentModelConfig(requirement=ModelRequirement.REQUIRED),
            run_config=context.policy_context,
            runtime_config=runtime_config,
            policy=policy,
            call_override=override,
            required_capabilities=required_capabilities,
        )

    @staticmethod
    def validate_capabilities(
        reference: ModelReference,
        capabilities: ProviderCapabilities,
        required: frozenset[str],
    ) -> None:
        """Ensure the provider supports every required capability.

        Args:
            reference: Resolved model reference being checked.
            capabilities: Advertised provider capabilities.
            required: Capability names that must be present and enabled.

        Raises:
            IncompatibleProviderCapabilitiesError: If any required capability is missing.
        """
        unsupported = sorted(
            name
            for name in required
            if not hasattr(capabilities, name) or not bool(getattr(capabilities, name))
        )
        if unsupported:
            raise IncompatibleProviderCapabilitiesError(
                f"Model reference '{reference}' lacks capabilities: {', '.join(unsupported)}"
            )
