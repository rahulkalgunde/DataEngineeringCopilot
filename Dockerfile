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
USER root
COPY pyproject.toml .
RUN uv pip install --system .

# Copy remaining code files and secure ownership in one shot
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/docs || exit 1

CMD ["python", "main.py", "--help"]
