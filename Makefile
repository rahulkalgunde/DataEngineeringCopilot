PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest
PROJECT_NAME := dataengineeringcopilot
COMPOSE := docker compose --profile app
GIT_SHA := $(shell git rev-parse --short HEAD)
IMAGE_TAG := dev-$(GIT_SHA)
DOCKER_TAG_FILE := .docker-tag

# ─── Core Docker Targets (use these) ────────────────────────────────────────

# First-time setup: build image, start stack, pull Ollama models
dev:
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	@echo "$(IMAGE_TAG)" > $(DOCKER_TAG_FILE)
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up -d --wait
	@echo "Pulling Ollama models (this may take a few minutes)..."
	@$(COMPOSE) exec -T ollama ollama pull nomic-embed-text || echo "⚠ Ollama not ready yet — run 'make pull-models' later"
	@$(COMPOSE) exec -T ollama ollama pull llama3.2:3b || echo "⚠ Ollama not ready yet — run 'make pull-models' later"
	@$(COMPOSE) exec -T ollama ollama pull qwen2.5-coder:7b || echo "⚠ Ollama not ready yet — run 'make pull-models' later"
	@echo ""
	@echo "✅ Dev stack ready. Image: $(IMAGE_TAG)"
	@echo "   Run 'make status' to verify health."

# Start everything (uses last built image — no rebuild)
up:
	@IMAGE_TAG=$$(cat $(DOCKER_TAG_FILE) 2>/dev/null || echo "latest") $(COMPOSE) up -d
	@echo "✅ Services started (image: $$(cat $(DOCKER_TAG_FILE) 2>/dev/null || echo 'latest'))"

# Stop everything
down:
	@$(COMPOSE) down
	@echo "✅ Services stopped"

# Rebuild image + restart app services (use after changing pyproject.toml/uv.lock)
rebuild:
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	@echo "$(IMAGE_TAG)" > $(DOCKER_TAG_FILE)
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up -d backend-api celery_worker
	@echo "✅ Rebuilt and restarted (image: $(IMAGE_TAG))"

# Follow logs (all services)
logs:
	@$(COMPOSE) logs --tail=100 -f

# Follow worker logs only
logs-worker:
	@$(COMPOSE) logs --tail=50 -f celery_worker

# Show container + health status
status:
	@echo "=== Containers ==="
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Service}}\t{{.Status}}\t{{.RunningFor}}"
	@echo ""
	@echo "=== Health Checks ==="
	@for svc in $$(docker ps --filter "label=com.docker.compose.project=dataengineeringcopilot" --format "{{.Names}}"); do \
		health=$$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}' "$$svc" 2>/dev/null); \
		if [ "$$health" = "healthy" ]; then echo "  ✅ $$svc: $$health"; \
		elif [ "$$health" = "no healthcheck" ]; then echo "  ℹ️  $$svc: $$health"; \
		else echo "  ❌ $$svc: $$health"; fi; \
	done
	@echo ""
	@echo "=== Deps Fingerprint ==="
	@echo "  Built image: $$(cat $(DOCKER_TAG_FILE) 2>/dev/null || echo 'unknown')"
	@echo "  Git SHA:     $(GIT_SHA)"

# Pull Ollama models (manual retry after dev)
pull-models:
	@$(COMPOSE) exec -T ollama ollama pull nomic-embed-text
	@$(COMPOSE) exec -T ollama ollama pull llama3.2:3b
	@$(COMPOSE) exec -T ollama ollama pull qwen2.5-coder:7b

# Health check via CLI
health:
	@dec_venv/bin/dec health

# Shell into a service: make shell svc=backend-api
shell:
	@$(COMPOSE) exec $(or $(svc),backend-api) sh -c "command -v bash >/dev/null 2>&1 && bash || sh"

# Validate compose config
config:
	@$(COMPOSE) config --quiet && docker compose -f docker-compose.yml -f docker-compose.ci.yml config --quiet
	@echo "✅ Compose config OK (dev + CI)"

# ─── Destructive Targets (require FORCE=1) ──────────────────────────────────

# Full rebuild: stop everything, remove containers + images + build cache
prune:
ifneq ($(FORCE),1)
	@echo "⚠️  This will remove all project containers, images, and build cache."
	@echo "   Data volumes are preserved. Use FORCE=1 to proceed."
	@exit 1
endif
	@$(COMPOSE) down --rmi all
	@docker builder prune -f
	@rm -f $(DOCKER_TAG_FILE)
	@echo "✅ Pruned all project resources"

# Remove only stale unused project images
prune-stale:
	@images=$$(docker images --filter "reference=de_copilot_base_image:*" --format "{{.Repository}}:{{.Tag}}"); \
	if [ -z "$$images" ]; then \
		echo "No stale project images to prune."; \
	else \
		for img in $$images; do \
			if docker ps -aq --filter "ancestor=$$img" | grep -q .; then \
				echo "  keep   $$img (in use)"; \
			else \
				echo "  remove $$img"; \
				docker image rm "$$img" >/dev/null 2>&1 || echo "    ! could not remove $$img"; \
			fi; \
		done; \
	fi
	@echo "Remaining project images:"
	@docker images --filter "reference=de_copilot_base_image:*" --format "  {{.Repository}}:{{.Tag}}" || true

# ─── CI Targets ─────────────────────────────────────────────────────────────

ci-up:
	docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait

ci-down:
	docker compose -f docker-compose.yml -f docker-compose.ci.yml down -v

# ─── Legacy Aliases (deprecated — use the targets above) ────────────────────

docker-dev: rebuild
docker-up: up
docker-down: down
docker-setup: dev
docker-status: status
docker-health: health
docker-rebuild:
	@echo "⚠️  'make docker-rebuild' rebuilds ALL images. Use 'make rebuild' for app-only."
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d
docker-logs: logs
docker-logs-worker: logs-worker
docker-stop-all:
	@$(COMPOSE) stop
	@echo "All services stopped"
docker-build:
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build --build-arg GIT_SHA=$(GIT_SHA) backend-api
	@echo "Image de_copilot_base_image:$(IMAGE_TAG) built"
docker-pull:
	@docker compose pull --ignore-pull-failures
docker-config: config
docker-restart:
	@$(COMPOSE) restart
	@echo "All services restarted"
docker-shell: shell
docker-cleanup:
	@$(COMPOSE) down
	@echo "Docker cleanup complete"
docker-prune: prune
docker-prune-stale: prune-stale
docker-ci-up: ci-up
docker-ci-down: ci-down

# ─── Python / Test Targets ──────────────────────────────────────────────────

install:
	uv pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -v

test-quick:
	$(PYTEST) tests/unit/ -m "not slow" -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-unit-serial:
	$(PYTEST) tests/unit/ -v -n 0

test-integration:
	$(PYTEST) tests/integration/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	$(PYTEST) tests/integration/ -m "not serial and not rag and not ollama and not ingestion" -v -n 6 --dist worksteal --durations=20 --durations-min=0.3
	$(PYTEST) tests/integration/ -m "not serial and (rag or ollama or ingestion)" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

test-integration-parallel:
	$(PYTEST) tests/integration/ -m "not serial" -v -n 2 --dist=loadgroup --reruns 2 --reruns-delay 1

test-real:
	REQUIRE_INFRA=1 $(PYTEST) tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	REQUIRE_INFRA=1 $(PYTEST) tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

test-e2e:
	$(PYTEST) tests/e2e/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
	$(PYTEST) tests/e2e/ -m "not serial" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3

test-ci-unit:
	$(PYTEST) tests/unit/ -v --durations=20 --durations-min=0.3 --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing

test-ci:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
	$(PYTEST) tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1
	$(PYTEST) tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1

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
