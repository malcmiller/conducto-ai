"""Public BaseAgent facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from .agent_card import (
    DEFAULT_INPUT_MODES,
    DEFAULT_OUTPUT_MODES,
    build_agent_card,
    serialize_agent_card,
)
from .model_config import AgentModelConfig, ModelReference, ModelRequirement
from .provider import ChatMessage, ProviderToolDefinition, StructuredOutputRequest
from .registration import (
    RegisteredMethod,
    register_decorated_methods,
    resolve_agent_metadata,
)
from .run_context import require_run_context

if TYPE_CHECKING:
    from .delegation import DelegationConfig, DelegationOutcome
    from .model_gateway import ModelCallResult

__all__ = ["BaseAgent"]
ResponseT = TypeVar("ResponseT", bound=BaseModel)


def _optional_structured_output() -> StructuredOutputRequest:
    """Build the permissive contract required by the provider protocol."""
    return StructuredOutputRequest(
        name="completion",
        schema={"type": "object"},
        required=False,
    )


class BaseAgent:
    """Base class for reflected Conducto agents.

    Agents declare model references or an ``AgentModelConfig``; provider clients
    and provider-specific model configuration belong to the runtime registry.

    Attributes:
        agent_metadata: Declared A2A metadata for the concrete agent type.
        _agent_config: Resolved model policy for the agent.
        _registered_methods: Methods discovered by reflection and metadata registration.
        _capabilities: Capability exports keyed by capability name.
        _tools: Tool exports keyed by tool name.
    """

    def __init__(
        self,
        *,
        model_reference: ModelReference | str | None = None,
        agent_config: AgentModelConfig | None = None,
        delegation_config: DelegationConfig | None = None,
    ) -> None:
        self.agent_metadata = resolve_agent_metadata(type(self))
        declared_reference = model_reference or self.agent_metadata.default_model
        if isinstance(declared_reference, str):
            declared_reference = ModelReference(declared_reference)
        self._agent_config = agent_config or AgentModelConfig(
            default_model=declared_reference,
            requirement=(
                ModelRequirement.REQUIRED
                if self.agent_metadata.model_required
                else ModelRequirement.NONE
            ),
        )
        self._delegation_config = delegation_config
        self._registered_methods: dict[str, RegisteredMethod] = {}
        self._capabilities: dict[str, RegisteredMethod] = {}
        self._tools: dict[str, RegisteredMethod] = {}
        (
            self._registered_methods,
            self._capabilities,
            self._tools,
        ) = register_decorated_methods(self)

    @property
    def agent_config(self) -> AgentModelConfig:
        """Return the resolved model policy for this reflected agent.

        Returns:
            The effective model policy is applied during invocation and routing.
        """
        return self._agent_config

    @property
    def delegation_config(self) -> DelegationConfig | None:
        """Return this agent's immutable opt-in delegation configuration."""
        return self._delegation_config

    async def run_delegation(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_type: type[ResponseT],
        config: DelegationConfig | None = None,
    ) -> DelegationOutcome[ResponseT]:
        """Run the reusable delegation loop in the active invocation context."""
        from .delegation import run_delegation

        effective = config or self._delegation_config
        if effective is None:
            raise ValueError("Delegation is not configured for this agent")
        return await run_delegation(
            require_run_context(),
            messages,
            config=effective,
            response_type=response_type,
        )

    async def complete(
        self,
        prompt: str | Sequence[ChatMessage],
        *,
        model: ModelReference | str | None = None,
        tools: Sequence[ProviderToolDefinition] = (),
        structured_output: StructuredOutputRequest | None = None,
    ) -> ModelCallResult:
        """Complete a prompt through the active runtime's governed model path.

        This method requires an active capability invocation. The runtime owns
        provider resolution, client leases, deadlines, cancellation, security,
        instruction composition, and audit provenance. Agent code must not
        construct provider clients directly.

        Args:
            prompt: A user prompt string or provider-neutral message sequence.
            model: Optional model reference override for this completion.
            tools: Optional provider-neutral tools available to the model.
            structured_output: Optional native structured-output contract. When
                omitted, the runtime sends a permissive non-required object
                contract because the provider protocol requires one.

        Returns:
            The provider result and invocation metadata.

        Raises:
            asyncio.CancelledError: If the active invocation is cancelled.
            NoActiveRunContextError: If called outside an active runtime invocation.
            TimeoutError: If the active invocation's deadline has expired.
            ValueError: If no prompt messages are provided.
        """
        messages = (
            (ChatMessage(role="user", content=prompt),)
            if isinstance(prompt, str)
            else tuple(prompt)
        )
        if not messages:
            raise ValueError("Completion requires at least one message")
        return await require_run_context().models.require(model).complete(
            messages,
            structured_output=structured_output or _optional_structured_output(),
            tools=tools,
        )

    @property
    def registered_methods(self) -> tuple[RegisteredMethod, ...]:
        """Return all decorated methods discovered on the agent.

        Returns:
            A tuple of registered methods in insertion order.
        """
        return tuple(self._registered_methods.values())

    @property
    def capabilities(self) -> dict[str, RegisteredMethod]:
        """Return a copy of the registered capability exports.

        Returns:
            A mapping of capability names to their registered metadata.
        """
        return dict(self._capabilities)

    @property
    def tools(self) -> dict[str, RegisteredMethod]:
        """Return a copy of the registered internal tool exports.

        Returns:
            A mapping of tool names to their registered metadata.
        """
        return dict(self._tools)

    def get_agent_card(
        self,
        url: str,
        *,
        preferred_transport: str = "JSONRPC",
        security_schemes: Mapping[str, Any] | None = None,
        security_requirements: Sequence[Mapping[str, Sequence[str]]] | None = None,
        default_input_modes: Sequence[str] = DEFAULT_INPUT_MODES,
        default_output_modes: Sequence[str] = DEFAULT_OUTPUT_MODES,
        capabilities: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Build an A2A capability card for this reflected agent.

        Args:
            url: The URL advertised for the agent's endpoint.
            preferred_transport: Preferred transport name for the card.
            security_schemes: Optional OAuth or scheme metadata for the endpoint.
            security_requirements: Optional security requirement objects.
            default_input_modes: Default input MIME modes for the agent card.
            default_output_modes: Default output MIME modes for the agent card.
            capabilities: Optional capability advertisement flags.

        Returns:
            A JSON-serializable A2A agent card for the reflected agent.
        """
        return build_agent_card(
            agent_type_name=type(self).__name__,
            metadata=self.agent_metadata,
            registered_capabilities=self._capabilities,
            url=url,
            preferred_transport=preferred_transport,
            security_schemes=security_schemes,
            security_requirements=security_requirements,
            default_input_modes=default_input_modes,
            default_output_modes=default_output_modes,
            capabilities=capabilities,
        )

    def get_agent_card_json(self, url: str, **kwargs: Any) -> str:
        """Serialize the reflected agent card to canonical JSON.

        Args:
            url: The advertised endpoint URL for the card.
            **kwargs: Additional keyword arguments forwarded to: meth:`get_agent_card`.

        Returns:
            The JSON-encoded A2A agent card.
        """
        return serialize_agent_card(self.get_agent_card(url, **kwargs))
