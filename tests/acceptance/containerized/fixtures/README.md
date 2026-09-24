# Containerized acceptance fixtures

This directory contains two standalone, deterministic **test-double** servers used by the
Conducto containerized multi-agent acceptance suite (Story 6.11): an OAuth/JWKS identity
fixture and an OpenAI-compatible model fixture. Both ship as their own minimal Docker images,
independent of the hardened Story 6.7 agent image (`Dockerfile` + `src/conducto/container.py`)
and of the Story 6.14 (#121) deployment-image application package. Neither fixture is wired
into a Compose topology or exercised by a real Conducto agent here — that is Story 6.11b and
later.

## Identity fixture (`identity/`)

A minimal RFC 8693 token-exchange and JWKS server. It issues and can validate ES256-signed
bearer tokens against the exact contract `conducto.security.tokens.JWTBearerTokenValidator`
expects, without any external identity provider or real credentials.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/livez` | Process liveness. |
| `GET` | `/readyz` | Readiness; reports ready once scenario configuration has loaded. |
| `GET` | `/.well-known/jwks.json` | Publishes the trusted signing key as a JWK set. |
| `POST` | `/token` | RFC 8693 token exchange (`application/x-www-form-urlencoded`). |

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDENTITY_FIXTURE_ISSUER` | `https://identity.fixture.invalid` | Issuer (`iss`) claim value. |
| `IDENTITY_FIXTURE_BIND_HOST` | `0.0.0.0` | Listen address. |
| `IDENTITY_FIXTURE_BIND_PORT` | `8080` | Listen port. |
| `IDENTITY_FIXTURE_SCENARIO` | `default` | Named scenario (see below). |
| `IDENTITY_FIXTURE_SIGNING_KEY_FILE` | unset (generates an ephemeral key) | Path to a mounted PEM-encoded EC P-256 private key. |

### Scenarios (`IDENTITY_FIXTURE_SCENARIO`)

| Name | Behavior |
| --- | --- |
| `default` | Issues a valid token for the requested audience and scope. |
| `invalid-signature` | Issues a token signed with an untrusted key while keeping the trusted key's `kid`, so real validators resolve the correct key and then fail signature verification specifically. |
| `expired-token` | Issues a token whose `exp` is already in the past. |
| `wrong-audience` | Issues a token bound to an audience that never matches the caller's requested audience. |
| `insufficient-scope` | Issues a token with an empty `scope` claim. |

## Model fixture (`model/`)

A minimal OpenAI-compatible server implementing only the routes
`conducto.providers.openai_compatible.OpenAICompatibleProvider` calls. It returns no real
inference, requires no model weights, and makes no network calls.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/livez` | Process liveness. |
| `GET` | `/readyz` | Readiness; reports ready once scenario configuration has loaded. |
| `GET` | `/v1/models` | Lists the single configured deterministic model id (used for provider readiness checks). |
| `POST` | `/v1/chat/completions` | Returns one scenario-shaped completion. Request-driven: a request with non-empty `tools` and no `tool`-role message returns a native tool call; otherwise it returns a terminal response. |

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_FIXTURE_BIND_HOST` | `0.0.0.0` | Listen address. |
| `MODEL_FIXTURE_BIND_PORT` | `8081` | Listen port. |
| `MODEL_FIXTURE_MODEL` | `fixture-model` | Model id listed by `/v1/models` and echoed by completions. |
| `MODEL_FIXTURE_SCENARIO` | `default` | Named scenario (see below). |
| `MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS` | `5.0` | Delay applied only when `MODEL_FIXTURE_SCENARIO=timeout`, so tests can bound the wait without rebuilding the image. |

### Scenarios (`MODEL_FIXTURE_SCENARIO`)

| Name | Behavior |
| --- | --- |
| `default` | Deterministic successful native tool call, then a schema-valid terminal response. |
| `malformed-response` | Returns HTTP 200 with a body that is not valid JSON. |
| `timeout` | Delays the response by `MODEL_FIXTURE_TIMEOUT_DELAY_SECONDS`. |
| `duplicate-tool-call` | Returns two native tool calls that share the same call id. |
| `error-500` | Returns HTTP 500 with an OpenAI-style error envelope. |

## Running locally

```bash
# Identity fixture
cd tests/acceptance/containerized/fixtures/identity
docker build -t identity-fixture .
docker run --rm -p 8080:8080 -e IDENTITY_FIXTURE_SCENARIO=default identity-fixture

# Model fixture
cd tests/acceptance/containerized/fixtures/model
docker build -t model-fixture .
docker run --rm -p 8081:8081 -e MODEL_FIXTURE_SCENARIO=default model-fixture
```

## Running the tests

```bash
uv run pytest tests/acceptance/containerized/fixtures -m acceptance
```

Docker build/run smoke tests in each `tests/test_*_fixture.py` module are automatically skipped
when Docker (with Linux containers) is unavailable.
