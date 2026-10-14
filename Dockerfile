# syntax=docker/dockerfile:1

# Stage 1: Build virtual environment with uv
FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Install uv binary
COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy application source and build wheel/project
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# Stage 2: Runtime image
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Run OS upgrades for latest security patches and clean up package managers / unused caches
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root system user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

WORKDIR /app

# Pre-create mount points with appropriate ownership
RUN mkdir -p /media /data && \
    chown -R appuser:appgroup /media /data /app

# Copy virtual environment and source code
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

USER appuser:appgroup

VOLUME ["/media", "/data"]

ENTRYPOINT ["paperless-rocm-reindexer"]
CMD ["batch"]
