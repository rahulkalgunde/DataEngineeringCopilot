# Makefile Guide

The project `Makefile` wraps package setup, the test matrix, linting/formatting, and all Docker Compose operations. This guide documents every target, what it runs, and when to use it.

---

## Table of Contents

- [Overview](#overview)
- [Setup targets](#setup-targets)
- [Test targets](#test-targets)
- [Quality targets](#quality-targets)
- [Docker targets](#docker-targets)
- [Cheat sheet](#cheat-sheet)
- [Common workflows](#common-workflows)

---

## Overview

Key Makefile variables:

| Variable | Value | Notes |
|---|---|---|
| `PYTHON` | `dec_venv/bin/python` | Always the venv interpreter — never bare `python`/`pip`. |
| `PYTEST` | `$(PYTHON) -m pytest` | Pytest via the venv. |
| `PROJECT_NAME` | `dataengineeringcopilot` | Used for project-scoped Docker cleanup/prune. |
| `COMPOSE` | `docker compose --profile app` | The `--profile app` gates `backend-api` + `celery_worker`. |
| `GIT_SHA` | `git rev-parse --short HEAD` | Short git SHA for image tags. |
| `IMAGE_TAG` | `dev-$(GIT_SHA)` | e.g. `dev-1a2b3c4`. |

Run `make <target>`; pass variables inline where a target accepts them (e.g. `make docker-shell svc=redis`).

> **Pytest defaults**: `pyproject.toml` sets `addopts = "-n 6 --dist worksteal --strict-markers"` — tests are parallel (6 workers capped) by default. Use `-n 0` only to debug xdist ordering issues.

---

## Setup targets

### `make install`
Install the package in editable mode with dev extras.

```bash
make install
# Equivalent: uv pip install -e ".[dev]"
```

**When to use**: after a fresh clone, when `uv.lock`/dependencies change, or after switching branches.

### `make streamlit`
Launch the Streamlit UI using the venv interpreter.

```bash
make streamlit
# Equivalent: dec_venv/bin/streamlit run data_engineering_copilot/ui/streamlit_app.py
```

**When to use**: interactive UI development/testing. The API does not need to be running for the Streamlit app to start.

---

## Test targets

All test commands run via `$(PYTEST)`. Markers are defined in `pyproject.toml` (`unit`, `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`, `celery`, `e2e`, `evaluation`, `xdist_group`, `serial`).

### `make test`
Full test suite, parallel by default.

```bash
make test
# pytest tests/ -v
```

### `make test-quick`
Fast feedback — unit tests only, excluding `@slow`, parallel.

```bash
make test-quick
# pytest tests/unit/ -m "not slow" -v
```

**When to use**: the day-to-day fast loop (~15s).

### `make test-unit`
All unit tests, including `@slow`.

```bash
make test-unit
# pytest tests/unit/ -v
```

### `make test-unit-serial`
Unit tests run sequentially (`-n 0`).

```bash
make test-unit-serial
# pytest tests/unit/ -v -n 0
```

**When to use**: debugging xdist ordering/race issues (the AGENTS.md-recommended escape hatch).

### `make test-integration`
Integration tests split into three legs:

```bash
make test-integration
# 1) pytest tests/integration/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
# 2) pytest tests/integration/ -m "not serial and not rag and not ollama and not ingestion" -v -n 6 --dist worksteal --durations=20 --durations-min=0.3
# 3) pytest tests/integration/ -m "not serial and (rag or ollama or ingestion)" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
```

- Leg 1: serial tests (shared PG/Redis/env state), single process.
- Leg 2: light tests (no Ollama: qdrant/auth/progress/celery with fake embeddings), 6-way parallel, no reruns.
- Leg 3: Ollama-heavy tests, low concurrency to avoid Ollama contention timeouts.

**When to use**: after changes that touch infrastructure-touching code. Requires Docker services via testcontainers; silently skips when services are down (see `make test-real` for a hard gate).

### `make test-integration-parallel`
Integration tests with controlled parallelism using xdist `loadgroup` for shared containers.

```bash
make test-integration-parallel
# pytest tests/integration/ -m "not serial" -v -n 2 --dist=loadgroup --reruns 2 --reruns-delay 1
```

### `make test-real`
Real-infra gate: requires **all** infra up via `REQUIRE_INFRA=1`. Any unreachable Qdrant/Ollama/Langfuse/Redis/PG causes a collection-time failure instead of a silent skip — how CI honestly reports unavailable services.

```bash
make test-real
# REQUIRE_INFRA=1 pytest tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
# REQUIRE_INFRA=1 pytest tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
```

**When to use**: to prove infra health honestly before release/CI.

### `make test-e2e`
End-to-end tests across the full pipeline (all hit the real stack incl. Ollama):

```bash
make test-e2e
# 1) pytest tests/e2e/ -m "serial" -v -n 0 --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
# 2) pytest tests/e2e/ -m "not serial" -v -n 2 --dist loadgroup --reruns 2 --reruns-delay 1 --durations=20 --durations-min=0.3
```

### `make test-ci`
CI gate: unit + integration + e2e with coverage (no markers filter, sequential for the infra legs):

```bash
make test-ci
# pytest tests/unit/ -v --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
# pytest tests/integration/ -v -n 0 --reruns 2 --reruns-delay 1
# pytest tests/e2e/ -v -n 0 --reruns 2 --reruns-delay 1
```

### `make test-ci-unit`
CI unit tests with coverage only (parallel):

```bash
make test-ci-unit
# pytest tests/unit/ -v --durations=20 --durations-min=0.3 --cov=data_engineering_copilot --cov-report=xml --cov-report=term-missing
```

### `make test-smoke`
Quick sanity — unit tests, no slow, quiet output:

```bash
make test-smoke
# pytest tests/unit/ -m "not slow" -q --no-header
```

### `make test-eval`
RAG evaluation harness (mocked embedder, no infra):

```bash
make test-eval
# pytest tests/evaluation/ -v
```

**When to use**: verify evaluation tooling without spinning up Qdrant/Ollama. (Differs from `dec evaluate`, which runs the real pipeline against the golden dataset.)

---

## Quality targets

### `make lint`
Ruff lint over source + tests:

```bash
make lint
# dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/
```

### `make format`
Ruff format over source + tests:

```bash
make format
# dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/
```

### `make clean`
Remove Python caches and build artifacts:

```bash
make clean
# find . -type d -name __pycache__ -exec rm -rf {} +
# find . -type d -name .pytest_cache -exec rm -rf {} +
# find . -type d -name "*.egg-info" -exec rm -rf {} +
# find . -type f -name "*.pyc" -delete
```

**Warnings**: destructive to caches only; safe to run. It is **not** scoped to the project directory's Docker resources — that's `make docker-cleanup`/`docker-prune`.

---

## Docker targets

All Docker targets use `docker compose --profile app` (unless a `-f` compose file is specified explicitly). The `--profile app` gates `backend-api` + `celery_worker`.

### Core Targets (use these)

#### `make dev`
First-time setup: build image, start stack, pull Ollama models.

```bash
make dev
# IMAGE_TAG=dev-<sha> docker compose --profile app build --build-arg GIT_SHA=<sha> backend-api
# echo "dev-<sha>" > .docker-tag
# IMAGE_TAG=dev-<sha> docker compose --profile app up -d --wait
# docker exec ollama ollama pull nomic-embed-text
# docker exec ollama ollama pull llama3.2:3b
# docker exec ollama ollama pull qwen2.5-coder:7b
```

**When to use**: first-time local setup. Code changes are instant (bind-mounted); only dependency changes need `make rebuild`.

#### `make up`
Start everything (uses last built image from `.docker-tag`).

```bash
make up
# IMAGE_TAG=$(cat .docker-tag) docker compose --profile app up -d
```

**When to use**: start the stack without rebuilding. Always uses the correct image tag.

#### `make down`
Stop and remove containers (non-destructive; volumes kept).

```bash
make down
# docker compose --profile app down
```

#### `make rebuild`
Rebuild image + restart app services (use after changing `pyproject.toml`/`uv.lock`).

```bash
make rebuild
# IMAGE_TAG=dev-<sha> docker compose --profile app build --build-arg GIT_SHA=<sha> backend-api
# echo "dev-<sha>" > .docker-tag
# IMAGE_TAG=dev-<sha> docker compose --profile app up -d backend-api celery_worker
```

**When to use**: after dependency changes. The bind mount does NOT update installed packages — the image must be rebuilt.

#### `make logs`
Follow logs for all services.

```bash
make logs
# docker compose --profile app logs --tail=100 -f
```

#### `make status`
Show container + health status.

```bash
make status
# docker compose --profile app ps
# docker inspect for health checks
# Check .docker-tag for current image
```

### Additional Targets

#### `make logs-worker`
Follow Celery worker logs only.

```bash
make logs-worker
# docker compose --profile app logs --tail=50 -f celery_worker
```

#### `make pull-models`
Pull Ollama models (manual retry after `make dev`).

```bash
make pull-models
```

#### `make health`
Run `dec health` component health report.

```bash
make health
```

#### `make shell`
Open a shell inside a service. Defaults to `backend-api`; override with `svc=`.

```bash
make shell
make shell svc=redis
make shell svc=qdrant
```

#### `make config`
Validate both compose layouts.

```bash
make config
```

### Destructive Targets (require `FORCE=1`)

#### `make prune`
Remove all project containers, images, and build cache. Data volumes are preserved.

```bash
make prune           # Shows warning, requires FORCE=1
make FORCE=1 prune   # Actually prunes
```

#### `make prune-stale`
Remove only stale unused project images (`de_copilot_base_image:*`). Safe — keeps images in use.

```bash
make prune-stale
```

### CI Targets

```bash
make ci-up           # Start CI stack (infra only, --wait)
make ci-down         # Tear down CI stack (with volumes — destructive)
```

### Legacy Aliases (deprecated)

The old `docker-*` targets still work but are aliases for the new targets:

| Legacy | New | Notes |
|--------|-----|-------|
| `docker-dev` | `rebuild` | Same behavior |
| `docker-up` | `up` | Same behavior |
| `docker-down` | `down` | Same behavior |
| `docker-setup` | `dev` | Same behavior |
| `docker-status` | `status` | Same behavior |
| `docker-health` | `health` | Same behavior |
| `docker-rebuild` | — | ⚠️ Rebuilds ALL images (not just app) |
| `docker-logs` | `logs` | Same behavior |
| `docker-logs-worker` | `logs-worker` | Same behavior |
| `docker-shell` | `shell` | Same behavior |
| `docker-prune` | `prune` | Same behavior |
| `docker-prune-stale` | `prune-stale` | Same behavior |
| `docker-ci-up` | `ci-up` | Same behavior |
| `docker-ci-down` | `ci-down` | Same behavior |

---

## Cheat sheet

| Task | Command |
|---|---|
| Install deps | `make install` |
| Fast unit loop | `make test-quick` |
| All unit tests | `make test-unit` |
| Sequential unit tests (xdist debug) | `make test-unit-serial` |
| Integration (3 legs) | `make test-integration` |
| Real-infra hard gate | `make test-real` |
| E2E tests | `make test-e2e` |
| Eval harness (no infra) | `make test-eval` |
| Lint | `make lint` |
| Format | `make format` |
| Clean caches | `make clean` |
| **First-time setup** | **`make dev`** |
| **Start stack** | **`make up`** |
| **Rebuild after dep changes** | **`make rebuild`** |
| **Stop stack** | **`make down`** |
| **Watch logs** | **`make logs`** |
| **Check health** | **`make status`** |
| Follow worker logs | `make logs-worker` |
| Shell into a service | `make shell svc=redis` |
| Health report | `make health` |
| Remove containers + images | `make FORCE=1 prune` |
| Validate compose config | `make config` |
| CI layout up/down | `make ci-up` / `make ci-down` |
| Streamlit UI | `make streamlit` |

---

## Common workflows

**First-time setup**
```bash
make dev              # build image + start stack + pull models
make status           # verify everything is healthy
dec ingest --source "Apache Spark Documentation" --max-pages 20
```

**Day-to-day dev loop**
```bash
make up               # start without rebuilding
make test-quick       # fast unit feedback
dec ingest --source "Apache Spark Documentation" --max-pages 20
make logs-worker      # watch ingestion logs live
```

**After changing dependencies** (`pyproject.toml` / `uv.lock`)
```bash
make install          # update venv
make rebuild          # rebuild image so deps_fingerprint matches
```

**Pre-merge verification**
```bash
make lint
make format
make test-real        # REQUIRE_INFRA=1 integration + e2e — fails loudly if infra is missing
```

**Clean slate for local testing**
```bash
make FORCE=1 prune    # remove containers + images, keep data volumes
make dev              # fresh stack + Ollama models
make test-integration # (testcontainers; skips silently if infra down)
```
