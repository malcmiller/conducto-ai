# CI Workflows

## `workflows/python-ci.yml`

CI for the `sdk-python/` reference SDK. Triggered on every pull request and
on pushes to `main`. See `sdk-python/README.md` for the exact local commands
each job runs.

Jobs:

* `changes` – detects whether `sdk-python/**` (or the workflow itself)
  changed, using `dorny/paths-filter`.
* `quality` – `ruff format --check`, `ruff check`, `mypy` across the
  Python/OS matrix.
* `test` – unit tests and schema/golden-fixture tests across the matrix,
  publishing JUnit XML as build artifacts (no secrets or prompts included).
* `acceptance` – the Milestone 1 quick-start acceptance suite
  (`tests/acceptance/`), a stable, story-level gate independent of
  fine-grained unit coverage.
* `package` – `uv build` plus an installed-wheel smoke test
  (`scripts/smoke_test.py`) run against a clean, dev-dependency-free
  environment built only from the produced wheel.
* `required` – always runs (`if: always()`) and aggregates the results of
  every job above. It succeeds when dependent jobs succeeded **or were
  skipped** (e.g. a documentation-only change), and fails if any dependent
  job failed or was cancelled.

### Required status check / repository ruleset

Mark **`Python CI (required)`** (the `required` job's display name) as the
required status check for the `main` branch:

1. Repository **Settings → Rules → Rulesets** (or, for classic branch
   protection, **Settings → Branches → Branch protection rules**).
2. Create/edit a ruleset targeting `main` (or the default branch).
3. Enable **"Require status checks to pass"** and add `Python CI (required)`.
4. Do **not** add the individual `quality` / `test` / `acceptance` /
   `package` / `changes` jobs as required checks — only the aggregate job.
   This keeps the required-check set stable even as the matrix or job list
   evolves, and guarantees documentation-only PRs still satisfy it.

Equivalently, via the GitHub API/CLI:

```bash
gh api repos/{owner}/{repo}/rulesets \
  --method POST \
  -f name="main-required-checks" \
  -f target="branch" \
  -f enforcement="active" \
  -f 'conditions[ref_name][include][]=refs/heads/main' \
  -f 'rules[][type]=required_status_checks' \
  -f 'rules[][parameters][required_status_checks][][context]=Python CI (required)'
```

### Security & supply-chain notes

* Third-party actions are pinned to immutable commit SHAs, not tags.
* Workflow permissions default to `contents: read`; no job requests write
  access, and no publishing/deployment credentials are used.
* `uv sync --locked` fails the build if `uv.lock` is out of date, so
  dependency resolution is reproducible between CI and local machines.
* Test/build artifacts uploaded for diagnostics (JUnit XML, wheels/sdists)
  contain only test results and build outputs — never prompts, credentials,
  or environment variables.

### Adding .NET CI (future, non-Milestone-1)

When `sdk-dotnet/` parity work begins, add a sibling `dotnet-ci.yml`
workflow (or a `dotnet` job gated by its own `changes` filter on
`sdk-dotnet/**`) following the same pattern: a matrix job per
OS/`dotnet` version, and add that job to the `required` job's `needs:` list
in `python-ci.yml` (or introduce a top-level `Repo CI (required)` job that
depends on both language aggregates). Until then, `.NET` is intentionally
out of scope for the Milestone 1 required check.
