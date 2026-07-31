PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest
PROJECT_NAME := dataengineeringcopilot
GIT_SHA := $(shell git rev-parse --short HEAD)
IMAGE_TAG := dev-$(GIT_SHA)

.PHONY: install test test-quick test-unit test-unit-serial test-integration test-e2e test-ci test-ci-unit test-smoke test-eval lint format clean docker-up docker-down docker-status docker-rebuild docker-logs docker-logs-worker docker-health docker-stop-all docker-cleanup docker-prune docker-setup docker-dev docker-ci-up

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

# Integration tests — sequential by default (testcontainers + shared Docker services)
test-integration:
	$(PYTEST) tests/integration/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1
	$(PYTEST) tests/integration/ -m "not serial" -v -n 6 --dist worksteal --reruns 2 --reruns-delay 1

# Integration tests with controlled parallelism (xdist loadgroup for shared containers)
test-integration-parallel:
	$(PYTEST) tests/integration/ -m "not serial" -v -n 2 --dist=loadgroup --reruns 2 --reruns-delay 1

# E2E tests — full pipeline
test-e2e:
	$(PYTEST) tests/e2e/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1
	$(PYTEST) tests/e2e/ -m "not serial" -v -n 6 --dist worksteal --reruns 2 --reruns-delay 1

# CI: unit tests with coverage (parallel)
test-ci-unit:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing

# CI gate: unit + integration + e2e with coverage
test-ci:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
	$(PYTEST) tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1
	$(PYTEST) tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1

# Quick sanity — smoke test
test-smoke:
	$(PYTEST) tests/unit/ -m "not slow" -q --no-header

test-eval:
	$(PYTEST) tests/evaluation/ -v

streamlit:
	dec_venv/bin/streamlit run data_engineering_copilot/ui/streamlit_app.py

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
	docker compose --profile app up -d

docker-down:
	docker compose --profile app down

docker-status:
	@echo "=== Project Containers ==="
	docker compose --profile app ps --format "table {{.Name}}\t{{.Status}}\t{{.RunningFor}}"
	@echo ""
	@echo "=== Health Checks ==="
	@for svc in de_copilot_broker de_copilot_vectorstore de_copilot_ollama; do \
		echo -n "$$svc: "; \
		docker inspect --format='{{.State.Health.Status}}' "$$svc" 2>/dev/null || echo "no health check"; \
	done

docker-rebuild:
	docker compose --profile app build --no-cache
	docker compose --profile app up -d

docker-logs:
	docker compose --profile app logs --tail=100 -f

docker-logs-worker:
	docker compose --profile app logs --tail=50 -f celery_worker

docker-health:
	@echo "=== Component Health ==="
	@dec_venv/bin/dec health

docker-stop-all:
	docker compose --profile app stop
	@echo "All services stopped"

docker-cleanup:
	# NOTE: intentionally NOT --volumes — prunes images/cache/containers but keeps your data.
	docker system prune -f
	docker image prune -f
	@echo "Docker cleanup complete"

docker-prune:
	docker builder prune -f
	docker system prune -f
	@echo "Docker prune complete"

docker-setup: docker-up
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Pulling Ollama models..."
	docker exec de_copilot_ollama ollama pull nomic-embed-text || echo "Ollama not ready, pull manually"
	docker exec de_copilot_ollama ollama pull llama3.2:3b || echo "Ollama not ready, pull manually"
	docker exec de_copilot_ollama ollama pull qwen2.5-coder:7b || echo "Ollama not ready, pull manually"
	@echo "Setup complete. Run 'make docker-status' to verify."

# One-command dev ritual: rebuild base image with git-SHA tag, recreate app
# services. The celery worker auto-reloads via watchfiles (override file), so
# no manual restart is needed after code edits.
docker-dev:
	IMAGE_TAG=$(IMAGE_TAG) docker compose --profile app build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	IMAGE_TAG=$(IMAGE_TAG) docker compose --profile app up -d backend-api celery_worker
	@echo "Dev stack ready (image tag: $(IMAGE_TAG)). Check /api/v1/version for deps_fingerprint_ok."

docker-ci-up:
	docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait
