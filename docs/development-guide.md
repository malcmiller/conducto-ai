# Python SDK development guide

The Python SDK is structured around deterministic, testable behavior and is
the reference implementation for Conducto contracts.

## Local setup

From the repository root:

```bash
uv sync --locked --group dev
```

The project requires Python 3.12+ and uses `uv` as the preferred package manager.

## Common commands

### Format

```bash
uv run ruff format .
uv run ruff format --check .
```

### Lint

```bash
uv run ruff check .
```

### Type-check

```bash
uv run mypy src examples scripts tests/acceptance
```

### Unit and schema tests

```bash
uv run pytest -m "not acceptance"
```

### Acceptance tests

```bash
uv run pytest -m acceptance
```

### Build the package

```bash
uv build
```

## Test philosophy

The tests in `tests/` cover both the implementation and the public contract:

- `test_decorators.py` verifies metadata and decorator behavior
- `test_agent.py` exercises reflected Agent Cards and validation rules
- `test_orchestrator.py` covers registry, routing, invocation, and timeout behavior
- `tests/golden/test_agent_card_golden.py` locks important wire-format outputs
- `tests/acceptance/test_quickstart.py` verifies the documented end-to-end flow

This gives the project a layered validation strategy: low-level behavior, contract compatibility, and user-story acceptance.

## Contribution workflow

1. Start from a feature branch.
2. Keep changes focused and aligned with the public contract.
3. Update or add tests when behavior changes.
4. Run the smallest relevant validation command while developing, then the complete applicable
   validation gate in [`AGENTS.md`](../AGENTS.md).
5. Submit a pull request and ensure CI passes.

## CI expectations

The repository's workflow uses a single required Python CI check that runs formatting, linting, static typing, unit tests, build verification, and acceptance tests across the supported matrix.

For local reproduction, use the commands above before opening a PR.

## Working inside this repository

The most important code paths are:

- `src/conducto/core/decorators.py` — metadata declaration
- `src/conducto/core/agent.py` — reflection and Agent Card generation
- `src/conducto/core/registry.py` — local agent registration and snapshots
- `src/conducto/core/gateway/` — governed discovery, binding revalidation, schema compatibility, and tool projection
- `src/conducto/core/orchestrator.py` — direct and model-mediated routing facade
- `src/conducto/core/runtime.py` — execution composition and invocation context
- `src/conducto/core/provider/` — cohesive model, message, result, tool, schema, and retry contracts

If you are extending the SDK, change the smallest surface area that preserves the public API and update tests that assert the public contract.
