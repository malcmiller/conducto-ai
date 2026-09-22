# Repository agent instructions

These instructions apply to the entire repository. The additive Python rules in
`.github/instructions/python.instructions.md` apply throughout this Python-only codebase.

## Product direction

Conducto is a capability-first multi-agent framework. It should let developers define an agent
once, expose typed capabilities, and invoke those capabilities through the same governed contract
whether the target is in-process, behind A2A transport, containerized, or hosted in Microsoft
Foundry.

Development is intentionally Python-first:

1. Establish the complete Python reference behavior.
2. Preserve language-neutral wire contracts and conformance fixtures.
3. Implement .NET parity from those stable contracts.
4. Add cross-organization federation only after local, remote, and cross-language behavior is
   proven.

Do not introduce .NET-driven abstractions into unfinished Python contracts or make local Python
development depend on cloud services.

## Architecture boundaries

- Agents declare metadata and typed capabilities; they do not own discovery infrastructure,
  transport clients, credentials, or global registries.
- Registries and catalogs own identity, capability metadata, lifecycle, and snapshots. They do not
  call models or execute capabilities.
- Gateways own policy-filtered discovery, target binding, transport selection, and normalized
  invocation.
- The runtime owns execution context, model resolution, deadlines, cancellation, security,
  auditing, and result construction.
- Transport adapters preserve the local invocation contract. They must not create alternate
  validation, guardrail, error, or serialization paths.
- Model-facing tools must be bounded, schema-validated, policy-approved, and backed by opaque
  bindings. Never expose mutable agent objects, callables, credentials, or endpoints to a model.
- Child calls may preserve or attenuate authority, deadlines, and budgets; they must never amplify
  them.
- Public failures must remain typed and explicit. Do not hide failures with broad catches, silent
  defaults, or success-shaped fallbacks.

## Implementation practices

- Read the relevant issue, dependencies, nearby tests, and existing implementation before editing.
- Prefer focused changes that solve the root problem without unrelated refactoring.
- Reuse existing contracts and helpers before adding parallel abstractions.
- Keep public APIs typed, deterministic, documented, and backward compatible unless the issue
  explicitly authorizes a breaking change.
- Update public exports, examples, architecture documentation, and tests when behavior changes.
- Keep modules cohesive. Reconsider a module around 400 lines and split it when it contains
  multiple responsibilities; line count alone is not a reason to fragment cohesive code.
- Never commit credentials, tokens, connection strings, private endpoints, generated secrets, or
  sensitive payloads.
- Required tests must be deterministic and must not require internet access, cloud credentials,
  model downloads, or wall-clock sleeps.
- Real provider and cloud tests must remain explicitly opt-in.

## Mandatory completion gate

Validation is part of implementation, not optional follow-up. An agent must not describe code work
as complete until every applicable command has run successfully with no errors or warnings.

For any Python code change, run from the repository root:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src examples scripts tests/acceptance
uv run pytest
```

Also run from the repository root:

```bash
git diff --check
git status --short
```

Then inspect the complete diff and resolve all applicable IDE, analyzer, type-checking, lint,
deprecation, and test warnings in changed files.

Additional requirements:

- Run focused tests during development, then the complete gate above before finishing.
- If public imports, package metadata, runtime dependencies, examples, or installed behavior change,
  also run `uv build` and the installed-wheel quick start and smoke test documented in
  `README.md`.
- Documentation-only changes require `git diff --check` and a manual check that commands, links,
  paths, and claims match the repository; they do not require the Python suite unless executable
  examples or generated documentation are affected.
- Do not add `noqa`, type ignores, warning suppressions, exclusions, or weaker checker settings just
  to pass validation. Use the narrowest suppression only for a confirmed false positive and explain
  it in code.
- If a command cannot run, report the exact command, output or blocker, and remaining uncertainty.
  Never imply that validation passed.
- In the final response, list the validation commands actually run and their outcomes.

## Python-specific rules

Follow `.github/instructions/python.instructions.md` in addition to this file. In particular:

- Use Python 3.12 or newer and strict typing.
- Review and update docstrings after every code change.
- Follow PEP 257 and the repository's Google-style docstring sections.
- Preserve deterministic ordering, immutable public contracts, cancellation, timeout, structured
  logging, and serialization behavior.
- Update golden fixtures only for intentional contract changes.

## Roadmap and dependency discipline

- GitHub issue `## Dependencies` sections and native `blocked by` relationships must agree.
- Refer to dependencies by story ID in issue bodies and by the corresponding issue in native GitHub
  relationships; do not confuse `Story 3.1` with issue `#3`.
- Do not begin a blocked story by bypassing its prerequisite contract.
- When moving or renumbering stories, update downstream references, native relationships,
  milestone descriptions, and status labels, then verify there are no missing references, cycles,
  or forward-milestone dependencies.
