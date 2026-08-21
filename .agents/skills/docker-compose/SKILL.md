---
name: docker-compose
description: Use for ANY task involving the DataEngineeringCopilot Docker stack — docker compose, docker-compose.yml, Makefile docker targets (make dev/up/down/rebuild/prune), container health, image rebuilds, dependency-hash staleness, Ollama model pulls, or the profile-gated backend-api/celery_worker. Triggers: docker, compose, container, image, rebuild, staleness, qdrant/redis/postgres/ollama service, make dev/up/down, Dockerfile, volumes.
---

# DataEngineeringCopilot Docker Compose

Project-specific guide for the `dataengineeringcopilot` Compose stack. The stack
is defined in `docker-compose.yml`, auto-merged with `docker-compose.override.yml`
(dev-only), and CI uses `-f docker-compose.yml -f docker-compose.ci.yml`.

## Project identity

- **Compose project name / network label**: `dataengineeringcopilot`
  (Docker inspect label `com.docker.compose.project=dataengineeringcopilot`).
- **Network**: single bridge network `de_copilot_net`.
- **Makefile wiring** (`Makefile:1-7`):
  - `COMPOSE := docker compose --profile app` — **all dev/CI targets run with `--profile app`**.
  - `IMAGE_TAG := dev-$(GIT_SHA)`; written to `.docker-tag` on build.
  - `GIT_SHA := git rev-parse --short HEAD`.

## Services (docker-compose.yml)

| Service | Image | Container | Host port | Notes |
|---------|-------|-----------|-----------|-------|
| `redis` | `redis:7-alpine` | `de_copilot_broker` | 6379 | `--requirepass ${REDIS_PASSWORD:-local_secure_password_123}`; health = `redis-cli -a <pw> ping` |
| `qdrant` | `qdrant/qdrant:v1.18.3` | `de_copilot_vectorstore` | 6333, 6334 | volume `qdrant_storage:/qdrant/storage`; health = TCP `/dev/tcp/localhost/6333` |
| `minio` | `minio/minio:RELEASE.2025-09-07T16-13-09Z` | `de_copilot_minio` | 9002:9000, 9001:9001 | creds `minioadmin`/`minioadmin`; bucket `local-dev` created by `minio-init` |
| `minio-init` | `minio/mc:...` | `de_copilot_minio_init` | — | one-shot, waits for minio then `mc mb --ignore-existing myminio/local-dev` |
| `ollama` | `ollama/ollama:0.32.4` | `de_copilot_ollama` | 11434 | volume `ollama_data`; env `OLLAMA_NUM_PARALLEL/MAX_QUEUE/KEEP_ALIVE` |
| `clickhouse` | `clickhouse/clickhouse-server:26.4-alpine` | `de_copilot_clickhouse` | 8123, 9000 | db/user `langfuse`; volumes `clickhouse_data`, `clickhouse_logs`; needs `cap_add: SYS_NICE` |
| `langfuse-postgres` | `postgres:16-alpine` | `de_copilot_langfuse_db` | 5432 | user/db `langfuse` |
| `langfuse` | `langfuse/langfuse:4` | `de_copilot_observability` | 3000 | **requires `LANGFUSE_ENCRYPTION_KEY`** (fails if unset); depends on langfuse-postgres, clickhouse, minio, minio-init, redis |
| `postgres` | `postgres:16-alpine` | `de_copilot_postgres` | 5433:5432 | user `copilot`, db `crawl_frontier` |
| `langfuse-worker` | `langfuse/langfuse-worker:4` | `de_copilot_langfuse_worker` | — | same env as `langfuse` |
| `backend-api` | `de_copilot_base_image:${IMAGE_TAG:-latest}` (built from `./Dockerfile`) | `de_copilot_api` | 8000 | **profile-gated**; uvicorn `--reload`; bind-mounts `.` → `/app`; `env_file: .env` |
| `celery_worker` | `de_copilot_base_image:${IMAGE_TAG:-latest}` | `de_copilot_celery_worker` | — | **profile-gated**; queue `ingestion`, `-c 1`; `shm_size: 2gb` |

### Profile gating

`backend-api` and `celery_worker` are behind `profiles: ["app"]`. A bare
`docker compose up` starts only infra; the app services require `--profile app`
(or the Makefile's `COMPOSE` var, which always includes it).

## Makefile docker targets (use these, not raw compose)

| Target | What it does |
|--------|--------------|
| `make dev` | **First-time setup**: build image with `GIT_SHA` arg, write `.docker-tag`, `up -d --wait`, pull Ollama models `nomic-embed-text`, `phi4-mini:3.8b`, `qwen2.5-coder:7b` |
| `make up` | Start stack with last built image (no rebuild) |
| `make down` | Stop + remove containers (interactive confirm; `FORCE=1` to bypass) |
| `make rebuild` | **Rebuild image + restart backend-api & celery_worker** — required after `pyproject.toml`/`uv.lock` changes |
| `make logs` / `make logs-worker` | Follow all logs / worker logs |
| `make status` | Container + health + image-fingerprint overview |
| `make pull-models` | Manual Ollama model pull retry |
| `make health` | `dec health` component check |
| `make shell svc=<name>` | Exec into a service (default `backend-api`) |
| `make config` | Validate compose config (dev + CI) |
| `make prune` | Destructive: down + remove all images + build cache (volumes kept) |
| `make streamlit` | Launch the Streamlit UI (`dec_venv/bin/streamlit run data_engineering_copilot/ui/streamlit_app.py`) — requires a running stack |
| `make prune-stale` | Remove unused `de_copilot_base_image:*` images |
| `make ci-up` / `make ci-down` | CI stack using `-f docker-compose.yml -f docker-compose.ci.yml` (`ci-down -v` deletes volumes!) |

Legacy aliases (`docker-dev`, `docker-up`, `docker-rebuild`, `docker-logs`, …)
exist but are deprecated — use the primary targets.

## Critical rules

- **Staleness**: the base image bakes a dependency hash. If `pyproject.toml` or
  `uv.lock` changes, the `.` bind mount alone is NOT enough — run `make rebuild`
  (or `make dev` on a fresh checkout). Otherwise the API/worker run stale deps.
- **Destructive targets** (`down`, `prune`, `prune-stale`, `ci-down`, `clean`)
  require interactive `[y/N]` confirmation; non-interactive shells must set
  `FORCE=1`. `ci-down` and `docker-compose down -v` destroy volumes — never run
  them casually.
- **Never** run `docker compose down -v`, `docker volume rm/prune` without
  explicit user approval (permissions in `opencode.json` deny them).
- **override file behavior**: `docker-compose.override.yml` auto-loads only when
  no `-f` is passed. It (a) runs `celery_worker` under `watchfiles` for
  auto-reload, and (b) sets `pull_policy: always` on the floating-tag images
  (`redis`, `clickhouse`, `langfuse`, `langfuse-worker`, `postgres`,
  `langfuse-postgres`) so dev always re-pulls; pinned images (`qdrant`, `ollama`,
  `minio`) are untouched.
- **CI compose** (`docker-compose.ci.yml`) omits the profile (no app services)
  and relaxes healthchecks for cold runners.
- **Env injection**: app services use `env_file: .env` plus explicit
  `environment:` entries (Redis/Qdrant/Ollama/Postgres URLs). `.env` is
  gitignored; required vars have `:-` defaults except `LANGFUSE_ENCRYPTION_KEY`
  which hard-fails when unset.

## Connection URLs (host-side probes)

- Redis: `redis://:local_secure_password_123@localhost:6379/0`
- Qdrant: `http://localhost:6333`
- Ollama: `http://localhost:11434`
- Postgres (crawl_frontier): `localhost:5433`, user `copilot`, db `crawl_frontier`
- Langfuse UI: `http://localhost:3000`

## Reference

- Full compose definition: `docker-compose.yml`
- Dev overrides: `docker-compose.override.yml`
- CI compose: `docker-compose.ci.yml`
- Makefile: `Makefile`
- Makefile guide: `docs/makefile_guide.md`
