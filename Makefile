PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest
PROJECT_NAME := dataengineeringcopilot
COMPOSE := docker compose --profile app
GIT_SHA := $(shell git rev-parse --short HEAD)
IMAGE_TAG := dev-$(GIT_SHA)

.PHONY: install test test-quick test-unit test-unit-serial test-integration test-real test-e2e test-ci test-ci-unit test-smoke test-eval lint format clean docker-up docker-down docker-status docker-rebuild docker-logs docker-logs-worker docker-health docker-stop-all docker-build docker-pull docker-config docker-restart docker-shell docker-cleanup docker-prune docker-prune-stale docker-setup docker-dev docker-ci-up docker-ci-down

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

# Integration tests — split into 3 legs:
#   1. serial (shared PG/Redis/env state) — single process
#   2. light (no Ollama: qdrant/auth/progress/celery, fake embeddings) — 6-way parallel, no reruns
#   3. Ollama-heavy (rag/ollama/ingestion) — low concurrency to avoid Ollama contention timeouts
test-integration:
	$(PYTEST) tests/integration/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	$(PYTEST) tests/integration/ -m "not serial and not rag and not ollama and not ingestion" -v -n 6 --dist worksteal --durations=20 --durations-min=0.3
	$(PYTEST) tests/integration/ -m "not serial and (rag or ollama or ingestion)" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

# Integration tests with controlled parallelism (xdist loadgroup for shared containers)
test-integration-parallel:
	$(PYTEST) tests/integration/ -m "not serial" -v -n 2 --dist=loadgroup --reruns 2 --reruns-delay 1

# Real-infra gate: require ALL infra up (REQUIRE_INFRA=1). Any unreachable
# Qdrant/Ollama/Langfuse/Redis/PG causes a collection-time failure instead of
# a silent skip — this is how CI honestly reports unavailable services.
test-real:
	REQUIRE_INFRA=1 $(PYTEST) tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	REQUIRE_INFRA=1 $(PYTEST) tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

# E2E tests — full pipeline (all hit the real stack incl. Ollama, so the
# parallel leg runs at low concurrency)
test-e2e:
	$(PYTEST) tests/e2e/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	$(PYTEST) tests/e2e/ -m "not serial" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

# CI: unit tests with coverage (parallel)
test-ci-unit:
	$(PYTEST) tests/unit/ -v --durations=20 --durations-min=0.3 --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing

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
	$(COMPOSE) up -d

docker-down:
	$(COMPOSE) down

docker-status:
	@echo "=== Project Containers ==="
	$(COMPOSE) ps --format "table {{.Name}}\t{{.Service}}\t{{.Status}}\t{{.RunningFor}}"
	@echo ""
	@echo "=== Health Checks ==="
	@for svc in de_copilot_broker de_copilot_vectorstore de_copilot_ollama de_copilot_clickhouse de_copilot_observability de_copilot_postgres; do \
		echo -n "$$svc: "; \
		docker inspect --format='{{.State.Health.Status}}' "$$svc" 2>/dev/null || echo "no health check"; \
	done

docker-rebuild:
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

docker-logs:
	$(COMPOSE) logs --tail=100 -f

docker-logs-worker:
	$(COMPOSE) logs --tail=50 -f celery_worker

docker-health:
	@echo "=== Component Health ==="
	@dec_venv/bin/dec health

docker-stop-all:
	$(COMPOSE) stop
	@echo "All services stopped"

# Build the base image only (tagged dev-<git sha>) without starting anything.
docker-build:
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	@echo "Image de_copilot_base_image:$(IMAGE_TAG) built"

# Pull every third-party image referenced by the stack (skip any that fail).
docker-pull:
	docker compose pull --ignore-pull-failures
	@echo "All third-party images pulled"

# Validate both compose layouts (dev override is auto-loaded, CI file is explicit).
docker-config:
	$(COMPOSE) config --quiet && docker compose -f docker-compose.yml -f docker-compose.ci.yml config --quiet
	@echo "Compose config OK (dev + CI)"

docker-restart:
	$(COMPOSE) restart
	@echo "All services restarted"

# Open a shell in a service: make docker-shell svc=redis (default: celery_worker).
docker-shell:
	$(COMPOSE) exec $(or $(svc),celery_worker) sh -c "command -v bash >/dev/null 2>&1 && bash || sh"

# Project-scoped: stop and remove this project's containers only.
# Images, volumes (data), and build cache are kept.
docker-cleanup:
	$(COMPOSE) down
	@echo "Docker cleanup complete (project: $(PROJECT_NAME))"

# Project-scoped: remove this project's containers, their images, and the build
# cache. Volumes (data) are always kept — never deletes other projects' resources.
docker-prune:
	$(COMPOSE) down --rmi all
	docker builder prune -f
	@echo "Docker prune complete (project: $(PROJECT_NAME))"

# Remove only this project's stale, unused images (de_copilot_base_image:*).
# Never touches other projects' images, and keeps any image still referenced by
# a container (running or stopped). No-op when nothing matches.
docker-prune-stale:
	@images=$$(docker images --filter "reference=de_copilot_base_image:*" --format "{{.Repository}}:{{.Tag}}"); \
	if [ -z "$$images" ]; then \
		echo "No stale project images to prune (de_copilot_base_image:* not present)."; \
	else \
		for img in $$images; do \
			if docker ps -aq --filter "ancestor=$$img" | grep -q .; then \
				echo "  keep   $$img (in use by a container)"; \
			else \
				echo "  remove $$img"; \
				docker image rm "$$img" >/dev/null 2>&1 || echo "    ! could not remove $$img"; \
			fi; \
		done; \
	fi; \
	echo "Remaining project images:"; \
	docker images --filter "reference=de_copilot_base_image:*" --format "  {{.Repository}}:{{.Tag}}" || true

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
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up -d backend-api celery_worker
	@echo "Dev stack ready (image tag: $(IMAGE_TAG)). Check /api/v1/version for deps_fingerprint_ok."

docker-ci-up:
	docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait

docker-ci-down:
	docker compose -f docker-compose.yml -f docker-compose.ci.yml down -v
