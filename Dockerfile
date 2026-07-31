FROM python:3.12-slim

# Install core and headless browser system libraries
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
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Install Playwright binaries
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
# (bind-mounted). A mismatch means the image is stale and needs `make docker-dev`.
RUN uv pip freeze > /image_deps.txt \
    && cat pyproject.toml uv.lock | sha256sum | cut -d' ' -f1 > /image_deps_sha256.txt

# Bake the git revision into the image for /api/v1/version.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="$GIT_SHA"
ENV IMAGE_GIT_SHA=$GIT_SHA

# Copy remaining code files and secure ownership in one shot
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/docs || exit 1

CMD ["python", "main.py", "--help"]
