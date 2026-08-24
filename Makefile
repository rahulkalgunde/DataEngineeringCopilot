PYTHON := dec_venv/bin/python
PYTEST := $(PYTHON) -m pytest
DIFFCOVER := dec_venv/bin/diff-cover
DEC := dec_venv/bin/dec
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
	@IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up -d --wait || true
	@echo "Pulling Ollama models (this may take a few minutes)..."
	@$(COMPOSE) exec -T ollama ollama pull phi4-mini:3.8b || echo "⚠ Ollama not ready yet — run 'make pull-models' later"
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
	$(call confirm_destructive,⚠️ This will stop and remove all running containers.)
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
	@$(COMPOSE) exec -T ollama ollama pull phi4-mini:3.8b
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

# ─── Destructive Targets (require interactive confirmation or FORCE=1) ───────

# Interactive confirmation for destructive targets. Prompts [y/N] on a TTY;
# aborts unless answered 'y'. Non-interactive shells (CI) and scripts must set
# FORCE=1 to bypass the prompt.
define confirm_destructive
	@if [ "$${FORCE:-}" != "1" ]; then \
		if [ ! -t 0 ]; then \
			echo "Refusing to run: this operation is destructive. Set FORCE=1 to bypass confirmation."; \
			exit 1; \
		fi; \
		printf '%s [y/N] ' "$(1)"; \
		read -r answer; \
		case "$$answer" in \
			y|Y) ;; \
			*) echo "Aborted."; exit 1 ;; \
		esac; \
	fi
endef

# Full rebuild: stop everything, remove containers + images + build cache
prune:
	$(call confirm_destructive,⚠️ This will remove ALL project containers images and build cache. Data volumes are preserved.)
	@$(COMPOSE) down --rmi all
	@docker builder prune -f
	@rm -f $(DOCKER_TAG_FILE)
	@echo "✅ Pruned all project resources"

# Remove only stale unused project images
prune-stale:
	$(call confirm_destructive,⚠️ This will permanently remove unused de_copilot_base_image images.)
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
	$(call confirm_destructive,⚠️ This will tear down the CI stack INCLUDING volumes. Container data will be deleted.)
	docker compose -f docker-compose.yml -f docker-compose.ci.yml down -v

# ─── Legacy Aliases (deprecated — use the targets above) ────────────────────

docker-dev: rebuild
docker-up: up
docker-down: down
docker-setup: dev
docker-status: status
docker-health: health
docker-rebuild:
	$(call confirm_destructive,⚠️ This rebuilds ALL images with --no-cache, discarding the build cache.)
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
	$(call confirm_destructive,⚠️ This will stop and remove all running containers.)
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

# Changed-code coverage gate (diff-cover). Fails if lines touched vs
# origin/main are <90% covered by the unit suite. Needs coverage.xml from a
# preceding test run with --cov-report=xml. CI runs this after test-ci-unit;
# locally run `make test-unit-cov` first.
DIFF_THRESHOLD ?= 90
test-unit-cov:
	$(PYTEST) tests/unit/ -q --cov=data_engineering_copilot --cov-branch --cov-report=xml -n 6 --dist worksteal

test-cov-gate:
	@if [ ! -f coverage.xml ]; then \
		echo "coverage.xml missing — generating via test-unit-cov first"; \
		$(MAKE) --no-print-directory test-unit-cov; \
	fi
	$(DIFFCOVER) coverage.xml --compare-branch origin/main --fail-under=$(DIFF_THRESHOLD)

# Scoped mutation testing (local-only, NOT in CI — see [tool.mutmut] scope).
mutate:
	$(PYTHON) -m mutmut run

mutate-results:
	$(PYTHON) -m mutmut results

# S2: parallel eval sweep — split DATASET into LANES shards, one eval-generation
# process per shard. JUDGES is optional comma list for 3-judge majority.
LANES ?= 3
JUDGES ?=
eval-sweep-parallel:
	@test -n "$(DATASET)" || { echo "usage: make eval-sweep-parallel DATASET=tests/evaluation/golden/qa_spark.jsonl [LANES=3] [JUDGES=anyapi,gemini]"; exit 1; }
	$(PYTHON) scripts/split_jsonl.py $(DATASET) /tmp/opencode/esweep $(LANES)
	@rm -f /tmp/opencode/esweep.lane*.done
	@for i in $$(seq 1 $(LANES)); do \
		setsid env FORCE=1 dec_venv/bin/dec eval-generation \
			--dataset /tmp/opencode/esweep.lane$$i.jsonl \
			$(if $(JUDGES),--judges "$(JUDGES)",) \
			--concurrency $(CONCURRENCY) \
			--output /tmp/opencode/esweep.lane$$i.json \
			> /tmp/opencode/esweep.lane$$i.log 2>&1 < /dev/null & done
	@echo "LANES=$(LANES) launched; outputs -> /tmp/opencode/esweep.lane*.json (poll logs esweep.lane*.log)"

# G3: fail when derived golden artifacts are stale vs their provenance sources.
eval-data-stale:
	$(PYTHON) scripts/check_derived_staleness.py tests/evaluation/golden

# G5: fail when running containers' provider pins drift from .env.
env-verify:
	$(PYTHON) scripts/verify_container_env.py

# S7-lite: refresh provider catalog from live probes so catalog_auto_order and
# recommended orders reflect current latency/availability (~60-90s, all free).
catalog-refresh:
	$(DEC) probe-catalog --purpose evaluation --timeout 12 --output data/provider_catalog.json

# Shared-box coordination: register heavy runs in /tmp/opencode/ACTIVE_RUNS.md
# and run this before launching builds/tests/evals (see AGENTS.md Environment).
runcheck:
	@echo "=== Registered runs ==="; cat /tmp/opencode/ACTIVE_RUNS.md 2>/dev/null || echo "(none registered)"
	@echo "=== Live heavy processes ==="; ps aux | grep -E "make (rebuild|dev)|pytest|dec (evaluate|eval-|probe-|ingest|gen-)|docker build|refreeze" | grep -v grep | awk '{printf "  pid=%s cpu=%s%% %s %s %s\n", $$2, $$3, $$11, $$12, $$13}' | head -8 || true

test-ci:
	$(PYTEST) tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
	$(PYTEST) tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1
	$(PYTEST) tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1

test-smoke:
	$(PYTEST) tests/unit/ -m "not slow" -q --no-header

test-eval:
	$(PYTEST) tests/evaluation/ -v

# Dataset-quality gates (hermetic — no corpus/infra required). Runs in CI
# (test-eval job) so schema/slug/evidence violations fail every commit.
test-eval-data:
	$(PYTEST) tests/unit/test_eval_datasets_schema.py tests/unit/test_eval_schema.py \
		tests/unit/test_eval_coverage.py tests/unit/test_eval_run_metrics.py \
		tests/unit/test_synthetic_generator.py tests/unit/test_golden_schema_gate.py -v

# Corpus-coverage gate (local / real-infra): validates every recall eval row
# against the ACTIVE generation's indexed corpus.
eval-coverage:
	dec_venv/bin/dec eval-coverage

# Free (zero-LLM) layered integrity evaluation: corpus/chunk/embedding/vector-DB
# retrieval. Requires Qdrant + local embedder only — no paid LLM calls. Run after
# every code change.
eval-fast:
	dec_venv/bin/dec eval-fast

# Retrieval-only benchmark (Recall@K, MRR, Precision@K per intent).
eval-retrieval:
	dec_venv/bin/dec eval-retrieval --k 10

# Retrieval regression gate: run benchmark and compare against honest inscope baseline.
# Prints global Δ with 95% bootstrap CI + per-intent Δ vs max(0, baseline-0.05) where n>=5.
# Baseline is tests/evaluation/benchmarks/baseline_inscope.json (220 rows, R@10=0.259).
# Exits non-zero if any gate fails. Use in CI as a required status check.
eval-retrieval-gate:
	dec_venv/bin/dec eval-retrieval --dataset tests/evaluation/golden/recall_inscope.jsonl --compare-baseline tests/evaluation/benchmarks/baseline_inscope.json --k 10 --batch-size 55

# $0 reranker smoke: freeze 10 pools, replay ($0 for subsequent reruns)
eval-rerank-smoke:
	@mkdir -p /tmp
	@echo "=== eval-rerank: freeze pools ==="
	@dec_venv/bin/dec eval-rerank --pool-file /tmp/rerank_pool_smoke.json --k 10 || echo "⚠️  freeze step needs Qdrant+embedder — skipped"
	@echo "--- replay (frozen pools, $$0) ---"
	@dec_venv/bin/dec eval-rerank --pool-file /tmp/rerank_pool_smoke.json --k 10 || echo "⚠️  replay step needs pool file — skipped"

# $0 prompt-aug smoke: template mode, no LLM, no infra
eval-prompt-aug-smoke:
	dec_venv/bin/dec eval-prompt-aug --dataset tests/evaluation/golden/prompt_aug_eval_sample.jsonl --mode template

# Generate synthetic recall eval set for one source (deterministic + coverage-gated).
eval-gen-source:
	dec_venv/bin/dec gen-synthetic-eval --source "$(SOURCE)" $(if $(GENERATION),--generation $(GENERATION)) --limit $(or $(LIMIT),50) --out tests/evaluation/golden/recall_synthetic_$(shell echo $(SOURCE) | tr ' ' _ | tr '[:upper:]' '[:lower:]').jsonl

# Regenerate qa_*.jsonl datasets from recall_*.jsonl (offline template-based).
eval-dataset-regenerate:
	@echo "=== Regenerating qa datasets ==="
	@python scripts/generate_qa_datasets.py
	@echo "=== Done ==="

# Consolidate & validate the golden dataset.
eval-golden-consolidate:
	@echo "=== Consolidating golden dataset ==="
	@python scripts/consolidate_golden_dataset.py
	@python scripts/trim_and_create_oos_v2.py
	@python scripts/add_missing_queries.py
	@python scripts/fix_golden_urls.py
	@echo "=== Validating against corpus ==="
	@dec_venv/bin/dec eval-coverage --dataset tests/evaluation/golden/recall_all.jsonl --generation $(or $(GENERATION),)
	@echo "=== Done ==="

# Run the RAG optimization benchmark (full pipeline) and write report.
eval-rag-benchmark:
	@mkdir -p tests/evaluation/benchmarks
	@python -m data_engineering_copilot.evaluation.rag_optimization_benchmark --output $(OUTPUT) --generation $(or $(GENERATION),)

# Set a new baseline for retrieval regression gates.
eval-set-baseline:
	@mkdir -p tests/evaluation/benchmarks
	@dec_venv/bin/dec eval-retrieval --output-dir tests/evaluation/benchmarks --k 10
	@mv tests/evaluation/benchmarks/retrieval_eval.json $(OUTPUT)
	@echo "✅ Baseline written to $(OUTPUT)"

# Judge-vs-human calibration gate (labels required before first run).
eval-judge-calibrate:
	dec_venv/bin/dec eval-judge-calibrate

# Human labeling UI for judge calibration rows (zero LLM calls).
label-calibration:
	dec_venv/bin/streamlit run data_engineering_copilot/ui/label_calibration.py

# Auto-label calibration rows: 3-LLM majority vote (paid, ~240 calls).
label-calibration-auto:
	dec_venv/bin/python -m data_engineering_copilot.evaluation.majority_label

# Launch the Streamlit UI (requires a running stack: `make dev`).
streamlit:
	dec_venv/bin/streamlit run data_engineering_copilot/ui/streamlit_app.py

# Refresh the local Claude docs git mirror (network required). After running,
# paste the printed commit SHAs into pinned_sources.json `local_mirror` entries.
mirror-claude-docs:
	$(PYTHON) scripts/mirror_claude_docs.py

lint:
	$(PYTHON) -m ruff check data_engineering_copilot/ tests/
	@$(PYTHON) scripts/lint_env.py .env

format:
	$(PYTHON) -m ruff format data_engineering_copilot/ tests/

clean:
	$(call confirm_destructive,⚠️ This will delete all __pycache__ .pytest_cache *.egg-info directories and *.pyc files.)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: test-chunking test-chunking-serial
test-chunking:
	$(PYTEST) tests/unit/test_chunker_invariants.py tests/unit/test_chunking_metrics.py tests/unit/test_chunking_snapshots.py -v -n 6 --dist worksteal
test-chunking-serial:
	$(PYTEST) tests/unit/test_chunker_invariants.py tests/unit/test_chunking_metrics.py tests/unit/test_chunking_snapshots.py -v -n 0
