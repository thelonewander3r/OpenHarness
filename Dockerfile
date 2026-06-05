FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package (with OTLP exporter) plus httpx for healthcheck/test agent.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[otel]" httpx

# Runtime assets: model registry (hot-reloadable, also volume-mounted in compose)
# and the test agent used by the `test` compose profile.
COPY config ./config
COPY test_agent.py ./

RUN useradd --create-home harness
USER harness

EXPOSE 3100

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://127.0.0.1:3100/health', timeout=2).raise_for_status()"]

CMD ["decomphose"]
