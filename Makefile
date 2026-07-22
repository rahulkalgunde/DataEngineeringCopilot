PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest

.PHONY: install test test-quick test-unit test-unit-serial test-integration test-e2e test-ci test-ci-unit test-smoke lint format clean docker-up docker-down docker-ci-up

install:
	uv pip install -e ".[dev]"

# Full test suite — parallel by default
test:
	$(PYTEST) tests/ -v

# Fast feedback: unit tests only, no slow tests, parallel
test-quick:
	$(PYTEST) tests/unit/ -m "not slow" -v

# All unit tests (including slow)
test-unit:
	$(PYTEST) tests/unit/ -v

# Sequential unit tests (for debugging xdist issues)
test-unit-serial:
	$(PYTEST) tests/unit/ -v -n 0

# Integration tests — external services required (retry flaky service-connection tests)
test-integration:
	$(PYTEST) tests/integration/ -v --reruns 2 --reruns-delay 1

# E2E tests — full pipeline
test-e2e:
	$(PYTEST) tests/e2e/ -v

# CI: unit tests with coverage (parallel)
test-ci-unit:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing

# CI gate: unit + integration + e2e with coverage
test-ci:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
	$(PYTEST) tests/integration/ -v --reruns 2 --reruns-delay 1
	$(PYTEST) tests/e2e/ -v --reruns 2 --reruns-delay 1

# Quick sanity — smoke test
test-smoke:
	$(PYTEST) tests/unit/ -m "not slow" -q --no-header

lint:
	$(PYTHON) -m ruff check data_engineering_copilot/ tests/

format:
	$(PYTHON) -m ruff format data_engineering_copilot/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-ci-up:
	docker compose -f docker-compose.ci.yml up -d --wait
