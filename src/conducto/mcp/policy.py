"""Immutable, default-deny export policy for Conducto MCP tool projection."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import McpPolicyError
from .naming import normalize_tool_name
from .profile import (
    MAX_EXPORTED_TOOLS,
    MAX_TOOL_DESCRIPTION_LENGTH,
    MAX_TOOL_NAME_LENGTH,
    MAX_TOOL_SCHEMA_BYTES,
)


@dataclass(frozen=True, slots=True)
class McpExportRule:
    """Exact ``agent_id:capability_id`` allowlist entry with an optional alias.

    Attributes:
        agent_id: Canonical Conducto agent identifier.
        capability_id: Canonical Conducto capability identifier.
        alias: Explicit MCP tool name replacing the derived default name.

    Notes:
        A rule never changes canonical schemas, required scopes, approval
        policy, or the callable executed by the runtime.
    """

    agent_id: str
    capability_id: str
    alias: str | None = None

    def __post_init__(self) -> None:
        """Validate identity fields and any explicitly configured alias."""
        if not self.agent_id.strip() or not self.capability_id.strip():
            raise McpPolicyError("Export rules require an agent_id and capability_id")
        if self.alias is not None:
            alias = self.alias.strip()
            if not alias or alias != self.alias:
                raise McpPolicyError("Export aliases cannot be empty or padded")
            if alias != normalize_tool_name(alias):
                raise McpPolicyError(
                    f"Export alias '{alias}' must already be a normalized MCP tool name"
                )

    @property
    def target(self) -> tuple[str, str]:
        """Return the exact allowlisted capability identity."""
        return self.agent_id, self.capability_id


@dataclass(frozen=True, slots=True)
class McpCapabilityQuery:
    """Bounded allowlist entry admitting one agent's matching capabilities.

    Attributes:
        agent_id: Canonical Conducto agent identifier.
        tags: Tags that every admitted capability must declare.
        limit: Maximum number of capabilities this query may admit.

    Notes:
        Queries never assign aliases. Exceeding ``limit`` fails exporter
        construction instead of silently truncating the exported tool list.
    """

    agent_id: str
    tags: frozenset[str] = frozenset()
    limit: int = 20

    def __post_init__(self) -> None:
        """Validate the queried agent identity and bounds."""
        if not self.agent_id.strip():
            raise McpPolicyError("Export queries require an agent_id")
        if any(not tag.strip() for tag in self.tags):
            raise McpPolicyError("Export query tags cannot be empty")
        if self.limit < 1 or self.limit > MAX_EXPORTED_TOOLS:
            raise McpPolicyError(f"Export query limit must be between 1 and {MAX_EXPORTED_TOOLS}")
        object.__setattr__(self, "tags", frozenset(self.tags))


@dataclass(frozen=True, slots=True)
class McpExportPolicy:
    """Immutable default-deny projection policy for one MCP server instance.

    Attributes:
        rules: Exact capability allowlist entries.
        queries: Bounded per-agent capability allowlist entries.
        max_tools: Maximum number of exported tools.
        max_name_length: Maximum exported tool name length.
        max_description_length: Maximum exported tool description length.
        max_schema_bytes: Maximum canonical JSON size of one exported schema.

    Notes:
        Nothing is exported unless an exact rule or bounded query admits it.
        The configuration is immutable for the lifetime of a server instance.
    """

    rules: tuple[McpExportRule, ...] = ()
    queries: tuple[McpCapabilityQuery, ...] = ()
    max_tools: int = MAX_EXPORTED_TOOLS
    max_name_length: int = MAX_TOOL_NAME_LENGTH
    max_description_length: int = MAX_TOOL_DESCRIPTION_LENGTH
    max_schema_bytes: int = MAX_TOOL_SCHEMA_BYTES

    def __post_init__(self) -> None:
        """Normalize entries and reject duplicate or out-of-range configuration."""
        rules = tuple(self.rules)
        queries = tuple(self.queries)
        if not rules and not queries:
            raise McpPolicyError("An MCP export policy must allowlist at least one capability")
        targets = [rule.target for rule in rules]
        if len(set(targets)) != len(targets):
            raise McpPolicyError("Export rules cannot allowlist one capability twice")
        query_agents = [query.agent_id for query in queries]
        if len(set(query_agents)) != len(query_agents):
            raise McpPolicyError("Export queries cannot allowlist one agent twice")
        if self.max_tools < 1 or self.max_tools > MAX_EXPORTED_TOOLS:
            raise McpPolicyError(f"max_tools must be between 1 and {MAX_EXPORTED_TOOLS}")
        if self.max_name_length < 1 or self.max_name_length > MAX_TOOL_NAME_LENGTH:
            raise McpPolicyError(f"max_name_length must be between 1 and {MAX_TOOL_NAME_LENGTH}")
        if (
            self.max_description_length < 1
            or self.max_description_length > MAX_TOOL_DESCRIPTION_LENGTH
        ):
            raise McpPolicyError(
                f"max_description_length must be between 1 and {MAX_TOOL_DESCRIPTION_LENGTH}"
            )
        if self.max_schema_bytes < 1 or self.max_schema_bytes > MAX_TOOL_SCHEMA_BYTES:
            raise McpPolicyError(f"max_schema_bytes must be between 1 and {MAX_TOOL_SCHEMA_BYTES}")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "queries", queries)

    def rule_for(self, agent_id: str, capability_id: str) -> McpExportRule | None:
        """Return the exact rule admitting one capability, when one exists.

        Args:
            agent_id: Canonical agent identifier.
            capability_id: Canonical capability identifier.

        Returns:
            The matching rule, or ``None`` when no exact rule admits the target.
        """
        for rule in self.rules:
            if rule.target == (agent_id, capability_id):
                return rule
        return None

    def query_for(self, agent_id: str) -> McpCapabilityQuery | None:
        """Return the bounded query configured for one agent, when one exists.

        Args:
            agent_id: Canonical agent identifier.

        Returns:
            The matching query, or ``None`` when the agent has no query.
        """
        for query in self.queries:
            if query.agent_id == agent_id:
                return query
        return None


__all__ = ["McpCapabilityQuery", "McpExportPolicy", "McpExportRule"]
