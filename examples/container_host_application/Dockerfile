# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=conducto-agent:latest
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${PYTHON_IMAGE} AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
WORKDIR /src
RUN python -m pip install --no-cache-dir uv==0.8.14
COPY pyproject.toml ./pyproject.toml
COPY src ./src
RUN uv build --wheel --out-dir /wheel

FROM ${BASE_IMAGE} AS runtime
USER root
RUN mkdir -p /etc/conducto /opt/conducto/wheels
COPY --from=build /usr/local/bin/uv /usr/local/bin/uv
COPY --from=build /wheel/container_host_application-*.whl /opt/conducto/wheels/
COPY conducto-applications.json /etc/conducto/applications.json
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/conducto/wheels/container_host_application-*.whl \
 && rm -rf /opt/conducto/wheels \
 && chown -R 65532:65532 /etc/conducto
USER 65532:65532
