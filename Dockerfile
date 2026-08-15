# syntax=docker/dockerfile:1

# Stage 1: Build stage — install deps, playwright + chromium, bake fingerprint.
# Build tools and package caches stay in this stage and are discarded.
FROM python:3.12.14-slim AS builder

# Playwright's Chromium needs the headless browser system libraries at runtime,
# so they are installed here and copied into the runtime stage's libs dir.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Install Playwright + chromium into the appuser home so browsers are copied
# with the runtime image.
RUN uv pip install --system playwright
USER appuser
RUN playwright install chromium

# Install project dependencies
# The BuildKit cache mount persists uv's wheel cache across builds, so
# dependency changes (pyproject.toml) rebuild in seconds instead of re-downloading.
USER root
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system . watchfiles

# Dependency fingerprint: compared at runtime against the live uv.lock/pyproject.toml
# (bind-mounted). A mismatch means the image is stale and needs `make rebuild`.
RUN cat pyproject.toml uv.lock | sha256sum | cut -d' ' -f1 > /image_deps_sha256.txt

# Bake the git revision into the image for /api/v1/version.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="$GIT_SHA"
ENV IMAGE_GIT_SHA=$GIT_SHA

# Copy remaining code files and secure ownership in one shot
COPY . .
RUN chown -R appuser:appuser /app

# Stage 2: Runtime stage — slim image containing only the runtime deps,
# playwright browsers, and application code. No build tools, no package caches.
FROM python:3.12.14-slim AS runtime

ARG GIT_SHA=unknown

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright \
    IMAGE_GIT_SHA=$GIT_SHA

# Copy the venv-installed site-packages and playwright browsers from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /home/appuser/.cache/ms-playwright /home/appuser/.cache/ms-playwright

# Only the curl probe needed by HEALTHCHECK remains in the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Recreate the unprivileged user (builder's /etc/passwd entry is not copied).
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/ms-playwright \
    && chown -R appuser:appuser /home/appuser

# Copy the application code and fingerprint.
COPY --from=builder --chown=appuser:appuser /app /app
COPY --from=builder /image_deps_sha256.txt /image_deps_sha256.txt

USER appuser

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/docs || exit 1

CMD ["python", "main.py", "--help"]
