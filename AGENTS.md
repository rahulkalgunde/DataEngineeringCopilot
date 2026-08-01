# DataEngineeringCopilot — Agent Guide

## Tech Stack
Python 3.12+, Pyright (standard mode), Ruff (lint+format), Pytest, structlog, `uv` only.

## Verification Loop (run in order after any code change)
1. `dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix`
2. `dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/`
3. `dec_venv/bin/python -m pyright data_engineering_copilot/ tests/`
4. `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v` (fast isolation first) — parallel by default via `-n 6` (capped; `-n auto` uses all cores and can destabilize the system); use `-n 0` ONLY to debug xdist-order issues

## Environment
- **VENV**: Always `dec_venv/bin/python` or `dec_venv/bin/dec`. Never bare `python`/`pip`.
- **Package mgmt**: `uv pip install -e ".[dev]"`. CI uses `uv sync --frozen --extra dev`.
- **Config load**: `.env` → `.env.secrets` → `.env.local`. Set `_env_file=None` does NOT reliably block env files — explicitly pass all relevant kwargs to override.
- **Default Redis**: `redis://:local_secure_password_123@localhost:6379/0`; Docker overrides host to `redis`.

## Entry Points
| Command | Runs |
|---|---|
| `dec <command>` | `main.py:main` → `cli.py` |
| API | `uvicorn data_engineering_copilot.api.app:app --reload --port 8000` |
| Streamlit | `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py` |
| Celery worker | `celery -A data_engineering_copilot.workers.tasks worker -Q ingestion -c 1 --loglevel=info` |

## CLI (`cli.py`)
| Command | Notes |
|---|---|
| `dec ask <question>` | Calls `build_rag_service()` directly (no API needed) |
| `dec ingest --source "X"` | POSTs to `http://localhost:8000/api/v1/ingest` (needs API+Celery) |
| `dec reset-index` | Full clean rebuild: recreates Qdrant collection + BM25 cache, clears Redis `crawl:*` keys, drops PG frontier tables |
| `dec reset-qdrant` | Deletes + recreates Qdrant collection w/ correct dimension + hybrid config, removes persisted BM25 cache |
| `dec evaluate` | RAG eval on `tests/evaluation/eval_dataset.jsonl` |
| `dec inspect-db` | Scrolls Qdrant points, shows source/chunk-type/URL distribution |
| `dec cancel <task-id>` | Cancels running ingestion via API |
| `dec profile` | Ingestion concurrency profiler |
| `dec monitor --task-id <id>` | Live ingestion dashboard (auto-refresh) |
| `dec status` / `dec health` / `dec config` | System status, health checks, config validation |

## API Routes (`api/routes.py`)
- `POST /api/v1/ingest` — Celery task dispatch via `SETNX` lock (60s TTL). Checks existing running task.
- `GET /api/v1/ingest/status/{task_id}` — polls Redis progress.
- `POST /api/v1/ask` — uses `get_rag_service()` (singleton from `services/rag_service_singleton.py`, not `factory.py`). 120s timeout.
- `POST /api/v1/ask/stream` — SSE streaming.
- `GET /api/v1/version` — git SHA + image build time + `deps_fingerprint_ok` (detects stale images).
- Middleware: `RateLimitMiddleware`, optional `ApiKeyAuthMiddleware`, CORS for `localhost:8501`.

## Testing
| Command | What |
|---|---|
| `make test-quick` | Unit, no `@slow`, parallel (~15s) |
| `make test-unit` | All unit tests |
| `make test-unit-serial` | Sequential (`-n 0`) |
| `make test-integration` | Splits into serial (`-m serial`, `-n 0`, 2 reruns) + parallel (`-m "not serial"`, `-n 6`, 2 reruns); needs Docker services |
| `make test-e2e` | Same split as integration; needs Docker stack |
| `make test-eval` | Mocked embedder, no infra |
| `make lint` / `make format` | Ruff only |

- `asyncio_mode = auto` — never `@pytest.mark.asyncio`. Default `addopts = "-n 6 --dist worksteal --strict-markers"` (6 capped; `-n auto` uses all 8 cores and can destabilize the system).
- Shared fixtures: `integration_settings`, `embeddings_provider`, `qdrant_store`, `ollama_client`, `populated_store`, `rag_service`, `api_client`.
- Integration/e2e conftests use testcontainers (Qdrant `v1.18.3`, Redis `7-alpine`, Ollama `0.32.4`) — no Docker Compose needed for tests.
- `unique_collection_name()` for per-test Qdrant isolation; auto-teardown.
- Auto-skip: `pytest_collection_modifyitems` checks service availability at collection time.
- Tests enforce Ollama-only providers — passing non-Ollama provider kwargs raises `RuntimeError`.
- All markers defined in `pyproject.toml`: `unit`, `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`, `celery`, `e2e`, `evaluation`, `xdist_group`, `serial`.

## Docker Services (full app stack)
`redis`, `qdrant`, `ollama`, `minio`, `clickhouse`, `langfuse` (incl. postgres + worker), `postgres` (crawl frontier), `backend-api`, `celery_worker`
- **Compose layout**: `docker-compose.yml` = single source of truth (all 12 services). `docker-compose.override.yml` = dev-only overrides (auto-loaded when no `-f` is passed). `docker-compose.ci.yml` = thin CI override (`-f docker-compose.yml -f docker-compose.ci.yml`).
- **Profiles**: `backend-api` + `celery_worker` are gated behind `--profile app`. Bare `docker compose up -d` starts infra only; use `make docker-dev` (or explicit service names) for the full stack. CI never starts them.
- `make docker-dev` = the one-command dev ritual: rebuilds `de_copilot_base_image` with a git-SHA tag (`dev-<sha>`), recreates both app services. Celery worker auto-reloads via `watchfiles` (dev override), so no manual restart after code edits.
- `make docker-setup` = start full stack + pulls `nomic-embed-text`, `llama3.2:3b`, `qwen2.5-coder:7b` into Ollama.
- CI: `make docker-ci-up` = `docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait`.
- Both `backend-api` and `celery_worker` share image `de_copilot_base_image:${IMAGE_TAG:-latest}`; both volume-mount `.:/app`.
- **Staleness detection**: the image bakes `/image_deps_sha256.txt` (sha256 of `pyproject.toml`+`uv.lock`). At startup the API checks it (`GET /api/v1/version` → `deps_fingerprint_ok`) and the Celery worker refuses to start on a mismatch. Dep changes ⇒ run `make docker-dev`; the bind mount does NOT update installed packages.
- Only `dec ingest` or manual testing needs full Docker Compose — tests use testcontainers only.

## Architecture
- **No LangChain/LlamaIndex** (except `langchain-text-splitters`).
- **Factory DI**: `build_rag_service()`, `build_async_ingestion_service()` in `factory.py`. Never instantiate manually.
- **Async only**: `SafeAsyncClientMixin` (uses `httpx.AsyncClient`; `aiohttp` is crawler-only).
- **Providers**: LLM → ollama, openrouter, nvidia, groq, cerebras, gemini. Embeddings → ollama, openrouter, nvidia, gemini. Switching providers requires `dec reset-qdrant` (dimension change).
- **Per-purpose LLM overrides**: Each pipeline stage (answer, rewrite, groundedness, intent, enrichment, evaluation, code) can use `{purpose}_llm_provider` / `{purpose}_llm_model`. Empty = fallback to global.
- **Adaptive fallback chain**: `AdaptiveLLMRouter` tries primary provider, falls through `llm_fallback_order`, ends at Ollama. Provider health registry tracks failures for cooldown.
- **Embedding dimension**: Model-dependent lookup in `embedding_model_dimensions` dict (`settings.py:124`).
- **Chunking** (`chunking_strategy`): `"sentence_preserving"` (default, `chunk_size_words * 5` chars), `"semantic"`, `"header_aware"`, `"fixed_size"`.
- **Hybrid search**: Enabled by default (dense + sparse, `hybrid_rrf_k=60`).
- **RAG pipeline** (`services/async_rag.py`): Query rewriting → vector retrieval → cross-encoder reranking → context assembly → LLM → groundedness verification. Two-tier query cache (exact + semantic).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`. Downloaded at runtime (~450MB). Singleton-cached in `services/reranker.py`.
- **Output parsing**: `parse_rag_response()` + `verify_citations()` in `services/structured_output.py`.
- **Crawl DB**: PostgreSQL via `PostgresCrawlFrontierDB` (set `CRAWL_DB_URL`).
- **No static type stubs**: `reportMissingTypeStubs = false`, `reportUnknownParameterType = false`, `reportPrivateImportUsage = false` in `pyrightconfig.json`.

## Ruff Config (`pyproject.toml`)
- `target-version = "py312"`, `line-length = 120`
- Lint: `select = ["E", "F", "W", "I", "UP", "B", "SIM"]`, `ignore = ["E501"]`
- `isort.known-first-party = ["data_engineering_copilot"]`

## CI Workflow (`.github/workflows/test.yml`)
- 4 job pipeline: `lint` → `test-unit` + `test-eval` (parallel) → `test-integration` → `test-e2e`.
- CI uses `uv sync --frozen --extra dev` (not `uv pip install -e .`).
- Integration/e2e jobs cache Docker images and Ollama models (`~/.ollama`).
- Ollama models pulled: `nomic-embed-text`, `llama3.2:3b`, `qwen2.5-coder:7b`.
- Also has identical disabled copy at `.github/workflows.disabled/test.yml`.

## Operational Gotchas
- **Qdrant health**: Use `GET /` (port 6333). `/health` returns 404.
- **Ollama `raw: True`**: Strips `<think>` tags. Empty response = output budget exhausted (increase `ollama_num_predict`).
- **OpenRouter rate limiter**: Per-provider `SlidingWindowRateLimiter` (20 RPM / 1000 RPD). Retries 429s up to 5x with exp backoff + `Retry-After`.
- **Worker time limits**: Soft 36000s (10h), Hard 43200s (12h). Zombie recovery via `task_failure` signal handler.
- **Qdrant batch splitting**: `upsert_chunks` splits into 256-chunk sub-batches (avoids 32MB payload limit).
- **URL dedup**: SHA-256 via `AsyncUrlRegistry` in Redis.
- **Ollama testcontainer caching**: Integration/e2e conftests mount `~/.ollama`; models cached across runs.
- **respx + Ollama embeddings**: `respx` passthrough returns empty bodies for `/api/embed`. To mock LLM with real embeddings, implement `LLMClientProtocol` instead of wire-mocking.

## Additional Instruction Sources
- `.clinerules/` — python-env-rules.md, guardrails.md, behavioral-constraints.md, memory-bank.md
- `opencode.json` — bash permission rules (which commands ask/allow/deny)
