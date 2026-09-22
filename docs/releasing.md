# Releasing `conducto-ai`

The canonical version is `project.version` in `pyproject.toml`. A production
tag is exactly `v<version>`; prerelease versions use the same PEP 440 value
after removing `v`. The workflow never selects or changes a version and never
creates a tag.

## Administrator setup

Before the first production release, an administrator must:

1. Create and verify the `conducto-ai` project on PyPI.
2. Add a PyPI Trusted Publisher for this repository and the
   `Python release` workflow, using the `pypi` GitHub environment.
3. Configure required reviewers for the protected `pypi` environment.
4. Protect `v*` tags so only the release process can create them.
5. Mark **Python CI (required)** and the release validation checks required for
   the default branch.

No API token, password, private key, or registry credential is stored in GitHub.
The only publication credential is the short-lived OIDC token issued to the
approved `publish` job.

## Rehearsal and production release

Pull requests run the complete gate, build, archive inspection, installed-wheel
smoke test, optional-extra matrix, SBOM, checksum, and sdist rebuild checks.
`workflow_dispatch` with its default `publish=false` performs the same
non-publishing rehearsal. Select a protected `v<version>` tag and set
`publish=true` only after the administrator gate is complete. The workflow
builds one wheel/sdist pair, carries it by digest through every job, refuses an
existing PyPI version, and creates the GitHub Release only after PyPI accepts
those exact files. Prereleases are marked as GitHub prereleases.

The release bundle contains `SHA256SUMS`, `package-inventory.json`, and an SPDX
2.3 SBOM. Verify files with `sha256sum -c SHA256SUMS`; compare the package
version, filenames, and GitHub attestation subjects before installation.

## Failure recovery and security response

A failed validation or PyPI publication produces no release and leaves the
immutable workflow artifact and diagnostics available for a rerun. A GitHub
Release failure after PyPI succeeds is repaired by rerunning only the release
attachment step; it must not rebuild or republish. PyPI versions and artifacts
are immutable: replace a bad release with a new version rather than overwriting
it. Yank or deprecate a version in PyPI when appropriate and document the
reason in the GitHub Release.

For a suspected compromise, pause tag creation and the `pypi` environment,
revoke the trusted-publisher binding, review workflow runs and attestations,
rotate administrator and recovery credentials, and contact PyPI support. Verify
provenance from the GitHub attestation, compare SHA-256 digests with the
published files, and prefer a new version after remediation. Recovery must not
introduce a long-lived publishing secret.
