# Conducto Python SDK quick start

This guide proves the Milestone 1 local-agent flow from a clean checkout, using
the public application facade, explicit domain contracts, and deterministic
`conducto.testing.FakeModel` routing.
No network access, model downloads, or credentials are required.

## 1. Start from a clean checkout

```bash
git clone https://github.com/malcmiller/conducto-ai.git
cd conducto-ai
git status --short
```

`git status --short` should print nothing before you compare local results with
CI.

## 2. Install development dependencies

```bash
uv sync --locked --group dev
```

## 3. Run the CI-equivalent checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src examples scripts tests/acceptance
uv run pytest -m "not acceptance"
uv run pytest -m acceptance
```

The acceptance suite exercises this flow:

```text
request
  -> model selects one registered agent and capability
  -> generated arguments are validated
  -> capability executes
  -> result is serialized into a typed envelope
  -> model and invocation provenance share one correlation ID
```

## 4. Build both distributions

```bash
uv build
python -c "from pathlib import Path; print('\n'.join(path.name for path in sorted(Path('dist').iterdir())))"
```

The `dist` directory should contain both a wheel (`.whl`) and source
distribution (`.tar.gz`).

To force a clean build first, remove `dist` with the command for your shell:

```bash
python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
uv build
```

## 5. Run the example from an installed wheel

Create a clean environment that does not install development dependencies:

```bash
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python examples/quickstart.py
```

Expected output:

```text
approval required: role=finance; granting local demo approval
quickstart result: agent=InvoiceAgent capability=classify_invoice value={"amount": 6000.0, "approved": false, "decision": "review", "vendor_id": "vendor-42"} correlation_id=quickstart-local-001
```

On Windows PowerShell, use:

```powershell
$wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
uv run --no-project --with "$wheel" python examples\quickstart.py
```

## 6. Run the installed-wheel smoke test

```bash
wheel=$(ls dist/*.whl)
uv run --no-project --with "$wheel" python scripts/smoke_test.py
```

PowerShell:

```powershell
$wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
uv run --no-project --with "$wheel" python scripts\smoke_test.py
```

The smoke test fails if `conducto` imports from the checkout's `src` directory instead of the
installed wheel. Install the wheel with the optional `mcp` extra
(`--with "$wheel[mcp]"`) to also drive the exported MCP stdio tools with an
official MCP SDK client; without the extra that section reports that it was
skipped.

## What the example demonstrates

`examples/quickstart.py` defines `InvoiceAgent` and `IncidentAgent`, publishes
deterministic Agent Cards for both, registers them with one `OrchestratorAgent`,
uses `FakeModel` to select a target capability, validates generated arguments,
invokes the selected local capability, receives an immutable
`InvocationSuccess` envelope, and checks structured model/invocation logs under
one caller-supplied correlation ID.

The quickstart imports application entry points from `conducto`, provider and
invocation contracts from their `conducto.core` packages, and the packaged fake
from `conducto.testing`. It registers the fake on a `ProviderRegistry` owned by
the application's runtime. It does not contact live providers, read
credentials, or depend on development-only packages.

## Troubleshooting

| Symptom                                                  | Fix                                                                                                                 |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `uv: command not found`                                  | Install `uv` from <https://docs.astral.sh/uv/> and reopen the shell.                                                |
| `ModuleNotFoundError: conducto`                          | Build the wheel first, then run with `uv run --no-project --with "$wheel" ...`.                                     |
| Smoke test says `conducto imported from source checkout` | Run the documented `uv run --no-project --with "$wheel"` command instead of activating the development environment. |
| Acceptance tests fail after Agent Card changes           | Review the generated card diff; Agent Cards are a public wire contract pinned to the Conducto A2A `1.0` profile.    |
| PowerShell does not understand `wheel=$(ls dist/*.whl)`  | Use the PowerShell commands shown above.                                                                            |
