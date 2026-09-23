# Conducto reference OCI image

The root `Dockerfile` is a production-oriented reference image for a single
installed Conducto wheel. It is model-neutral and publishes a deterministic
`echo` capability so image tests do not need a provider, model weights, cloud
credentials, or another service. Story 6.11 owns multi-container A2A topology.

## Build and identify an image

Build from a clean checkout with a revision and immutable version label:

```bash
REVISION="$(git rev-parse HEAD)"
docker build \
  --build-arg SOURCE_REVISION="$REVISION" \
  --build-arg IMAGE_VERSION="0.1.2-$REVISION" \
  --tag conducto-agent:0.1.2-"$REVISION" .
docker image inspect conducto-agent:0.1.2-"$REVISION" \
  --format '{{index .RepoDigests 0}}'
```

Promotion records the resulting `repo@sha256:...` digest, source revision,
platform, wheel filename, lockfile checksum, SBOM, provenance attestation, and
scanner result. Never promote a mutable tag.

The Dockerfile uses a pinned Python 3.12 slim base, `uv.lock`, a wheel built in
an isolated stage, and a runtime virtual environment containing only locked
runtime dependencies plus the wheel. The runtime stage has no package manager,
compiler, source checkout, tests, or development dependencies.

## Configuration

`CONDUCTO_CONFIG_FILE` may point to a JSON object. Environment variables override
file values and use the field names below in uppercase with the
`CONDUCTO_` prefix:

| Field | Default | Purpose |
| --- | --- | --- |
| `AGENT_ID`, `AGENT_VERSION` | reference values | Agent identity |
| `BIND_HOST`, `BIND_PORT` | `0.0.0.0`, `8000` | Listener |
| `PUBLIC_URL`, `A2A_ENDPOINT` | loopback URL, `/a2a` | Agent Card and RPC |
| `PROVIDER_TYPE`, `MODEL_REFERENCE`, `PROVIDER_ENDPOINT` | `local`, `local`, unset | Model-neutral deployment selection |
| `REQUEST_TIMEOUT_SECONDS`, `SHUTDOWN_GRACE_SECONDS` | `30`, `30` | Request and termination bounds |
| `TRUST_ROOTS_FILE`, `ISSUER`, `AUDIENCE`, `SCOPES` | unset | Deployment trust policy metadata |
| `LOG_FORMAT` | `json` | Structured logging selection |
| `WRITABLE_PATH`, `MAX_CONCURRENCY`, `MAX_REQUEST_BYTES` | `/tmp/conducto`, `64`, `262144` | Filesystem and resource bounds |

Sensitive values must be mounted as files and referenced by a corresponding
`*_FILE` variable. They are not accepted in image metadata, Agent Cards,
configuration diagnostics, or logs. The reference identity resolver is
deliberately credential-free; a real application must inject verification
based on its own trusted issuer and audience.

Startup validates configuration before readiness. `/livez` reports process
liveness, `/readyz` reports configuration/dependency readiness, and `/a2a`
is unavailable while starting, draining, or closed. Uvicorn receives SIGTERM,
the A2A host flips readiness before drain, cancels bounded in-flight work,
flushes retained tasks and closers, then exits with the server's deterministic
status. Invalid configuration exits `78`; SIGINT exits `130`.

## Hardening contract

Run with a read-only root and explicitly writable temporary/state paths:

```bash
docker run --read-only --tmpfs /tmp/conducto:rw,noexec,nosuid,size=64m \
  --mount type=tmpfs,destination=/var/lib/conducto \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --pids-limit=128 --memory=512m --cpus=1 \
  -p 8000:8000 conducto-agent@sha256:...
```

The process runs as numeric UID/GID `65532`, binds only the documented port,
needs no Linux capability, and does not claim cross-platform support: verify
each built image for the target `linux/amd64` or `linux/arm64` platform.

## Inspection and promotion policy

Inspect the wheel and lock inputs before the image build:

```bash
uv build
uv run python scripts/release_verify.py --wheel dist/conducto_ai-0.1.2-py3-none-any.whl \
  --sdist dist/conducto_ai-0.1.2.tar.gz --output-dir reports/release
docker history --no-trunc conducto-agent:TAG
docker inspect conducto-agent:TAG
```

Generate an SPDX/CycloneDX SBOM with the organization's pinned scanner (for
example, Syft), attach SLSA/in-toto provenance from the CI builder, and run the
organization's vulnerability scanner against the exact digest. Promotion is
blocked by any critical or high finding with an available fix, and by any
critical finding without a documented exception. Exceptions require an owner,
reason, compensating control, expiry date, and approval; they cannot suppress
scanner output or silently bypass a critical result. Rebuild when the pinned
base or locked dependency set changes. These updates must not change the
agent capability contract.

Host responsibilities are image build, signature/attestation verification,
resource limits, and secret-file/workload-identity mounting. Container
responsibilities are the non-root process, probes, bounded shutdown, and
documented writable paths. An orchestrator is responsible for routing,
readiness-based load balancing, identity injection, and restart policy. No
environment-specific discovery is embedded in the image.
