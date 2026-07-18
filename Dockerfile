FROM python:3.12-slim

# Install core and headless browser system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
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

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create security user early
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Set environment path metrics
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Install Playwright binaries inside a cached layer to avoid redownloads on package updates
RUN uv pip install --system --no-cache playwright
USER appuser
RUN playwright install chromium

# Install project dependencies
USER root

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Copy remaining code files
COPY . .

# Secure file permissions
RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py", "--help"]