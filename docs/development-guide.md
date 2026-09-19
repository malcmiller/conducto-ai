# Development guide

This repository is small enough to work with conventionally, but it is structured to support deterministic, testable behavior. The Python SDK is the primary implementation and is the best place to begin if you want to extend the framework.

## Local setup

From the repository root:

```bash
cd sdk-python
uv sync --locked --group dev
```

The project requires Python 3.12+ and uses `uv` as the preferred package manager.

## Common commands

### Format

```bash
uv run ruff format --check .
```

### Lint

```bash
uv run ruff check .
```

### Type-check

```bash
uv run mypy src
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

The tests in `sdk-python/tests/` cover both the implementation and the public contract:

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
4. Run the smallest relevant validation command.
5. Submit a pull request and ensure CI passes.

## CI expectations

The repository's workflow uses a single required Python CI check that runs formatting, linting, static typing, unit tests, build verification, and acceptance tests across the supported matrix.

For local reproduction, use the commands above before opening a PR.

## Working inside this repository

The most important code paths are:

- `sdk-python/src/conducto/core/decorators.py` — metadata declaration
- `sdk-python/src/conducto/core/agent.py` — reflection and Agent Card generation
- `sdk-python/src/conducto/core/orchestrator.py` — runtime registry and invocation
- `sdk-python/src/conducto/core/provider.py` — model provider abstraction and routing contracts

If you are extending the SDK, change the smallest surface area that preserves the public API and update tests that assert the public contract.
