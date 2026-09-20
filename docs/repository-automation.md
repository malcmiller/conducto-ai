# Repository automation and agent guidance

Repository automation consists of development-agent instructions and GitHub Actions workflows. It
supports the project but does not define Conducto's product architecture.

## File map

```text
.
├── AGENTS.md
└── .github/
    ├── instructions/
    │   └── sdk-python.instructions.md
    └── workflows/
        └── python-ci.yml
```

## Coding-agent instructions

Repository-wide requirements live in [`AGENTS.md`](../AGENTS.md). They define:

- the Python-first product sequence
- architecture and security boundaries
- implementation and modularity expectations
- mandatory validation and repository inspection
- roadmap dependency discipline

The path-scoped
[`sdk-python.instructions.md`](../.github/instructions/sdk-python.instructions.md) adds Python SDK
requirements for typing, docstrings, tests, Ruff, mypy, packaging, and installed-wheel validation.

The files are intentionally complementary:

1. `AGENTS.md` applies repository-wide.
2. `.github/instructions/sdk-python.instructions.md` adds requirements for `sdk-python/**`.
3. An agent must satisfy both when changing the Python SDK.

Validation is part of implementation. Agents must not report Python code work as complete without
running the applicable checks and reporting their outcomes. A tooling or environment failure must
be reported as a blocker rather than represented as a passing check.

## Python workflow

[`python-ci.yml`](../.github/workflows/python-ci.yml) validates the Python reference SDK on pull
requests and pushes to `main`.

| Job | Responsibility |
|---|---|
| `changes` | Detect whether Python SDK or workflow files changed. |
| `quality` | Check Ruff formatting and lint plus strict mypy across the supported matrix. |
| `test` | Run unit and schema/golden tests and publish JUnit diagnostics. |
| `acceptance` | Run the installed-user-oriented Python acceptance scenarios. |
| `package` | Build distributions and run examples/smoke tests from the produced wheel. |
| `required` | Provide one stable branch-protection result for all jobs above. |

The local equivalent for a Python code change is:

```bash
cd sdk-python
uv sync --locked --group dev
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src examples scripts tests/acceptance
uv run pytest
```

Package or public-import changes also require:

```bash
uv build
```

Run the installed-wheel example and smoke test using the commands documented in
[`sdk-python/README.md`](../sdk-python/README.md).

## Path filtering

The expensive Python jobs run when `sdk-python/**` or `python-ci.yml` changes. The `required` job
always runs, including for documentation-only changes, so the protected branch receives a stable
status check rather than a missing check.

When another path can affect Python execution, packaging, or validation, add it to the `changes`
filter in the same pull request.

## Required status check

Configure **`Python CI (required)`** as the required status check for `main`. Do not require each
matrix job independently; those names and counts can change as supported Python and operating
system versions evolve.

In repository settings:

1. Open **Settings → Rules → Rulesets**.
2. Create or edit the ruleset targeting `main`.
3. Enable required status checks.
4. Add `Python CI (required)`.

The aggregate job fails when a required dependency fails or is cancelled and succeeds when
path-filtered jobs are legitimately skipped.

## Workflow security

- Pin third-party actions to immutable commit SHAs and retain the release tag in a comment.
- Grant the minimum job-level permissions; default to `contents: read`.
- Never expose publishing, model-provider, cloud, or deployment credentials to required test jobs.
- Use `uv sync --locked` so dependency resolution is reproducible.
- Keep required tests network-free, cloud-free, credential-free, and model-download-free.
- Do not upload prompts, credentials, tokens, environment variables, or sensitive capability data
  as workflow artifacts.

## Future workflows

When the .NET SDK is introduced in Milestone 8, add a path-filtered .NET workflow with equivalent
formatting, analyzers, tests, package, and consumer-smoke coverage. Either extend the repository
aggregate or add a stable repository-level required job that depends on both language aggregates.

Deployment and publishing workflows should remain separate from pull-request validation, use
environment protection, and request credentials only in the jobs that need them.
