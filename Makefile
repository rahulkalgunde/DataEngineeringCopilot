PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest

.PHONY: install test test-unit test-integration lint format clean docker-up docker-down

install:
	uv pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

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
