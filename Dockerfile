# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${PYTHON_IMAGE} AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
WORKDIR /src
RUN python -m pip install --no-cache-dir uv==0.8.14
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv build --wheel --out-dir /wheel \
 && uv export --locked --no-dev --extra a2a-server --no-emit-project --format requirements.txt --output-file /wheel/requirements.txt \
 && uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python --requirement /wheel/requirements.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps /wheel/conducto_ai-*.whl

FROM ${PYTHON_IMAGE} AS runtime
ARG SOURCE_REVISION=unknown
ARG IMAGE_VERSION=dev
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp/conducto \
    CONDUCTO_IMAGE_VERSION=${IMAGE_VERSION} \
    CONDUCTO_SOURCE_REVISION=${SOURCE_REVISION}
LABEL org.opencontainers.image.title="Conducto Python agent runtime" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/conducto
COPY --from=build /opt/venv /opt/venv
RUN mkdir -p /tmp/conducto /var/lib/conducto \
 && chown -R 65532:65532 /tmp/conducto /var/lib/conducto /opt/conducto
USER 65532:65532
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import json,os,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('CONDUCTO_BIND_PORT','8000')+'/livez', timeout=2); assert json.load(r)['status']=='alive'"]
ENTRYPOINT ["python", "-m", "conducto.container"]
