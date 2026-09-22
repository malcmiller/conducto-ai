# Validation and debugging

## Required local gate

Run from the repository root:

```bash
uv sync --locked --group dev
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src examples scripts tests/acceptance
uv run pytest
git diff --check
git status --short
```

Do not report completion when an applicable command is red or was not run.

## Package-facing changes

When public imports, package metadata, dependencies, extras, examples, or
installed behavior change:

```bash
uv build
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python examples/quickstart.py
uv run --no-project --with "$wheel" python scripts/smoke_test.py
```

Use the PowerShell equivalents in the [SDK quick start](../quickstart.md) on
Windows.

## Start focused, finish complete

During development, run the smallest tests that cover the behavior. Before
finishing, run the complete gate.

Examples:

```bash
uv run pytest tests/test_a2a_runtime.py
uv run pytest tests/test_gateway.py tests/test_gateway_hybrid.py
uv run pytest -m acceptance
```

## Debug by boundary

| Symptom | First boundary to inspect |
|---|---|
| Agent not found | Registry/catalog snapshot and gateway policy |
| Binding rejected | Binding generation, target lifecycle, schema signature |
| Arguments rejected | Reflected Pydantic model and capability schema |
| Authorization denied | Principal, scopes, allowed capabilities, guardrails |
| Approval does not resume | Approval ID, argument digest, principal binding |
| Wrong model | Model reference, registry snapshot, run override, policy |
| Remote card rejected | Discovery policy, profile version, endpoint authority |
| A2A task stuck | Repository transition and cancellation path |
| Optional import fails | Extra metadata, lazy import, missing-dependency error |
| Source works, wheel fails | Build inclusion, public export, runtime dependency |

## Concurrency debugging

Avoid adding sleeps to “fix” a race. Use events, barriers, injected clocks, and
controlled fake providers. Assert both the final result and the number of times
business logic executed.

Inspect ownership:

- Is mutable state shared intentionally?
- Is a snapshot copied before returning?
- Can cancellation arrive before work starts?
- Can two callers claim the same task or binding?
- Can cleanup race with accepted work?

## Documentation checks

For documentation-only work:

- resolve every local link;
- verify commands against the current repository layout;
- verify symbol names against public exports;
- render Mermaid diagrams;
- inspect Markdown formatting;
- run `git diff --check`.

Executable examples or generated documentation require their normal tests.
