# ==============================================================================
# Dockerfile for DataEngineeringCopilot
# ==============================================================================
# Base image: Python 3.11 slim for smaller footprint
# ==============================================================================
FROM python:3.11-slim

# ==============================================================================
# System dependencies
# - build-essential: for compiling any C extensions
# - libpq-dev: for PostgreSQL client (if needed by langfuse/pgvector)
# - curl, gnupg: for installing additional packages
# - libglib2.0-0, libnss3, libnspr4, libatk1.0-0, libatk-bridge2.0-0, libcups2, libdrm2, libxkbcommon0, libxcomposite1, libxdamage1, libxfixes3, libxrandr2, libgbm1, libasound2: Playwright/Chromium dependencies
# ==============================================================================
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

# ==============================================================================
# Create non-root user for security
# ==============================================================================
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# ==============================================================================
# Install Python dependencies from requirements.txt
# ==============================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Install Playwright Chromium with system dependencies
# This is required for Crawl4AI to run safely inside the container
# ==============================================================================
RUN playwright install chromium --with-deps

# ==============================================================================
# Copy application code
# ==============================================================================
COPY . .

# ==============================================================================
# Set ownership to non-root user
# ==============================================================================
RUN chown -R appuser:appuser /app
USER appuser

# ==============================================================================
# Environment variables
# ==============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# ==============================================================================
# Default command (overridden by docker-compose)
# ==============================================================================
CMD ["python", "main.py", "--help"]
