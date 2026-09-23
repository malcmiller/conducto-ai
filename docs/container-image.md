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
  --format 'local-image={{.Id}}'
```

Local builds expose an immutable image ID. After pushing to a registry, record
the resulting `repo@sha256:...` digest with `docker buildx imagetools inspect`
or registry inspection. Promotion records that digest, source revision,
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

## Reference image versus deployment image

The root image remains the verified Story 6.7 reference host. Its default
embedded manifest exposes only the built-in `reference` application mapped to
`conducto.container:build_reference_app`, which preserves the deterministic
single-capability `ReferenceAgent` used for image verification.

Deployment images can extend that same verified digest with an image-baked
application manifest at `/etc/conducto/applications.json`. The container host
loads that manifest, validates its schema, and selects one application with the
`CONDUCTO_APPLICATION` environment variable. The selection is a manifest key
such as `reference`, `orchestrator`, `agent-a`, or `agent-b`; it is never a raw
Python import path from runtime input.

The host supports this structured manifest format:

```json
{
  "manifestVersion": "1",
  "applications": {
    "reference": "conducto.container:build_reference_app",
    "orchestrator": "container_host_application.orchestrator:build_app",
    "agent-a": "container_host_application.agent_a:build_app",
    "agent-b": "container_host_application.agent_b:build_app"
  }
}
```

`CONDUCTO_APPLICATIONS_MANIFEST` may override the manifest path when a
deployment image intentionally bakes it somewhere else, but the manifest must
still be part of the image and validated by the host before readiness. Unknown,
empty, malformed, or duplicate keys fail startup with redacted diagnostics.

The container host exposes safe startup metadata through `--check-config`,
including the selected application key, configured agent identity/version, the
manifest version, and the image revision metadata. No provider credential,
secret, or model runtime detail is emitted.

## Example deployment image

`examples/container_host_application/` demonstrates how to build one immutable
deployment image that can host an orchestrator plus Agent A and Agent B from
the same verified base-image digest.

Build the base reference image first:

```bash
docker build --tag conducto-agent:local .
```

Then build the deployment image from that base:

```bash
docker build \
  --build-arg BASE_IMAGE=conducto-agent:local \
  --file examples/container_host_application/Dockerfile \
  --tag conducto-container-host:local \
  examples/container_host_application
```

The example Dockerfile only copies the example package sources needed to build a
wheel plus the manifest file. It does not copy the repository checkout into the
runtime image.

Validate each hosted role with `--check-config`:

```bash
docker run --rm conducto-container-host:local --check-config
docker run --rm -e CONDUCTO_APPLICATION=orchestrator conducto-container-host:local --check-config
docker run --rm -e CONDUCTO_APPLICATION=agent-a conducto-container-host:local --check-config
docker run --rm -e CONDUCTO_APPLICATION=agent-b conducto-container-host:local --check-config
```

Per-service configuration can use the same image digest with different
environment values:

```bash
docker run --rm -p 8100:8000 \
  -e CONDUCTO_APPLICATION=orchestrator \
  -e CONDUCTO_AGENT_ID=story-6-11-orchestrator \
  -e CONDUCTO_AGENT_VERSION=0.1.0 \
  conducto-container-host@sha256:...

docker run --rm -p 8101:8000 \
  -e CONDUCTO_APPLICATION=agent-a \
  -e CONDUCTO_AGENT_ID=story-6-11-agent-a \
  -e CONDUCTO_AGENT_VERSION=0.1.0 \
  conducto-container-host@sha256:...

docker run --rm -p 8102:8000 \
  -e CONDUCTO_APPLICATION=agent-b \
  -e CONDUCTO_AGENT_ID=story-6-11-agent-b \
  -e CONDUCTO_AGENT_VERSION=0.1.0 \
  conducto-container-host@sha256:...
```

This allowlisted key-based selection is the security boundary: operators choose
from vetted manifest entries already present in the verified image, while the
host refuses arbitrary runtime import paths, shell commands, mounted code, or
remote URLs. The deployment image also remains model-neutral: it bundles no
provider runtime, model server, or model weights.
