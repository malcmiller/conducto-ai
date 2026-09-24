# syntax=docker/dockerfile:1.7
# Standalone deterministic OAuth/JWKS identity test double. Independent of the
# hardened Story 6.7 Conducto agent image: no conducto.container dependency,
# no --check-config entrypoint, and its own minimal runtime image.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${PYTHON_IMAGE} AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
WORKDIR /src
RUN python -m pip install --no-cache-dir uv==0.8.14
COPY pyproject.toml ./pyproject.toml
COPY src ./src
RUN uv build --wheel --out-dir /wheel

FROM ${PYTHON_IMAGE} AS runtime
RUN python -m pip install --no-cache-dir uv==0.8.14 \
 && uv venv /opt/venv --python /usr/local/bin/python
COPY --from=build /wheel/identity_fixture-*.whl /tmp/wheel/
RUN uv pip install --python /opt/venv/bin/python --no-cache /tmp/wheel/identity_fixture-*.whl \
 && rm -rf /tmp/wheel \
 && python -m pip uninstall --yes uv \
 && groupadd --gid 65532 identity \
 && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin identity \
 && mkdir -p /tmp/identity-fixture \
 && chown -R 65532:65532 /tmp/identity-fixture

ENV PATH="/opt/venv/bin:${PATH}" \
    IDENTITY_FIXTURE_BIND_HOST=0.0.0.0 \
    IDENTITY_FIXTURE_BIND_PORT=8080

WORKDIR /opt/identity-fixture
USER 65532:65532
EXPOSE 8080
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=5s --timeout=2s --start-period=5s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=1)"]
ENTRYPOINT ["python", "-m", "identity_fixture.server"]
