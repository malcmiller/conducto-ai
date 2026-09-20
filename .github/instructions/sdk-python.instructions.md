---
applyTo: "sdk-python/**"
---

# Python SDK development instructions

Apply these instructions to every change under `sdk-python/`.

## Implementation

- Use Python 3.12 or newer and preserve strict type safety.
- Keep changes focused on the requested behavior and follow the existing module boundaries.
- Prefer small, cohesive modules and reusable helpers over duplicating logic.
- Preserve public imports from `conducto` and `conducto.core` unless a breaking change is
  explicitly requested.
- Preserve deterministic ordering, immutable public contracts, typed failures, cancellation,
  timeout, logging, and serialization behavior.
- Do not add a runtime dependency unless the change requires it.
- Never hide failures with broad exception handling, silent fallbacks, or success-shaped results.

## Docstrings and documentation

- After writing or changing code, always review and update its docstrings.
- Add docstrings to public modules, classes, methods, functions, protocols, dataclasses, and
  exceptions.
- Use PEP 257 principles and the repository's Google-style sections: `Args:`, `Returns:`,
  `Raises:`, `Attributes:`, and `Notes:`.
- Keep the first line concise and descriptive. Document behavior, contracts, side effects,
  invariants, error conditions, and ownership boundaries rather than restating the signature.
- Update surrounding documentation when behavior, configuration, public APIs, examples, or
  architecture changes.
- Do not add repetitive docstrings or comments to trivial private helpers.

## Tests

- Add or update focused tests for every behavior change and regression fix.
- Test public behavior rather than private implementation details where practical.
- Preserve deterministic tests: no network access, credentials, wall-clock sleeps, or external
  model downloads in required test suites.
- Update golden fixtures only for intentional public contract changes.
- Use the `acceptance` marker for end-to-end SDK flows and the `golden` marker for pinned wire
  contracts.

## Required validation

Run commands from `sdk-python/` after code changes. Do not consider the work complete until all
applicable commands pass without errors or warnings:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

- Run the smallest relevant tests while developing, followed by the full commands above before
  completion.
- Resolve all Ruff, mypy, test, and available IDE/code-inspection errors and warnings in files
  touched by the change.
- Do not suppress an inspection, add `noqa`, weaken a type, or exclude a file merely to make a
  check pass. Suppress only a verified false positive, use the narrowest suppression possible,
  and document why it is necessary.
- If a required command cannot run because of an environment or tooling limitation, report the
  exact command and blocker; do not claim validation passed.

## Packaging changes

When package behavior or exports change, also verify the built artifact:

```bash
uv build
```

Run the installed-wheel smoke test documented in `sdk-python/README.md` whenever public imports,
package metadata, or runtime dependencies change.
