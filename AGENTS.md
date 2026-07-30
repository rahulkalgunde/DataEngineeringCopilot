# DataEngineeringCopilot — Agent Guide

## Tech Stack
Python 3.12+, Pyright (standard mode), Ruff (lint+format), Pytest, structlog, `uv` only.

## Verification Loop (run in order after any code change)
1. `dec_venv/bin/python -m ruff check data_engineering_copilot/ tests/ --fix`
2. `dec_venv/bin/python -m ruff format data_engineering_copilot/ tests/`
3. `dec_venv/bin/python -m pytest tests/unit/<specific_test> -v` (fast isolation first)

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

## CLI
- `dec ask <question>` — calls `build_rag_service()` directly (no API needed)
- `dec ingest --source "X"` — POSTs to `http://localhost:8000/api/v1/ingest` (needs API+Celery running)
- `dec reset-index` — deletes Qdrant collection, recreates w/ correct dim, clears Redis `crawl:*` keys, drops crawl frontier PG tables
- `dec evaluate` — runs RAG eval on `tests/evaluation/eval_dataset.jsonl`
- `dec status` / `dec health` / `dec config` / `dec monitor --task-id <id>` / `dec profile`

## API Routes (`api/routes.py`)
- `POST /api/v1/ingest` — Celery task dispatch via `SETNX` lock (60s TTL). Checks existing running task.
- `GET /api/v1/ingest/status/{task_id}` — polls Redis progress.
- `POST /api/v1/ask` — uses `get_rag_service()` (singleton from `services/rag_service_singleton.py`, not `factory.py`). 120s timeout.
- `POST /api/v1/ask/stream` — SSE streaming.
- Middleware: `RateLimitMiddleware`, optional `ApiKeyAuthMiddleware`, CORS for `localhost:8501`.

## Testing
| Command | What |
|---|---|
| `make test-quick` | Unit, no `@slow`, parallel (~15s) |
| `make test-unit` | All unit tests |
| `make test-unit-serial` | Sequential (`-n 0`) |
| `make test-integration` | Sequential + 2 reruns; needs Docker (testcontainers) |
| `make test-e2e` | Full pipeline; needs Docker (testcontainers) |
| `make test-eval` | Mocked embedder, no infra |
| `make lint` / `make format` | Ruff only |

- `asyncio_mode = auto` — never `@pytest.mark.asyncio`. Default `addopts = "-n auto --dist worksteal --strict-markers"`.
- Shared fixtures in `tests/conftest.py`: `integration_settings`, `embeddings_provider`, `qdrant_store`, `ollama_client`, `populated_store`, `rag_service`, `api_client`.
- Integration/e2e conftests use testcontainers (Qdrant `v1.18.3`, Redis `7-alpine`, Ollama `0.32.4`) — no Docker Compose needed for tests.
- `unique_collection_name()` for per-test Qdrant isolation; auto-teardown.
- Auto-skip: `pytest_collection_modifyitems` checks service availability at collection time.

## Docker Services (full app stack)
`redis`, `qdrant`, `ollama`, `minio`, `clickhouse`, `langfuse` (incl. postgres + worker), `postgres` (crawl frontier), `backend-api`, `celery_worker`
- `make docker-setup` = `docker compose up -d` + pulls `nomic-embed-text`, `llama3.2:3b`, `qwen2.5-coder:7b` into Ollama.
- CI: `make docker-ci-up` (uses `docker-compose.ci.yml`, prefix `dec_ci_*`).
- Both `backend-api` and `celery_worker` share image `de_copilot_base_image`; worker volume-mounts `.:/app` — restart, don't rebuild.
- Only `dec ingest` or manual testing needs full Docker Compose — tests use testcontainers only.

## Architecture
- **No LangChain/LlamaIndex** (except `langchain-text-splitters`).
- **Factory DI**: `build_rag_service()`, `build_async_ingestion_service()` in `factory.py`. Never instantiate manually.
- **Async only**: `SafeAsyncClientMixin` (uses `httpx.AsyncClient`; `aiohttp` is crawler-only).
- **Providers**: LLM → ollama, openrouter, nvidia, groq, cerebras, gemini. Embeddings → ollama, openrouter, nvidia, gemini. Switching providers requires `dec reset-index` (dimension change).
- **Per-purpose LLM overrides**: Each pipeline stage (answer, rewrite, groundedness, intent, enrichment, evaluation, code) can use `{purpose}_llm_provider` / `{purpose}_llm_model`. Empty = fallback to global.
- **Embedding dimension**: Model-dependent lookup in `embedding_model_dimensions` dict (`settings.py:121`).
- **Chunking** (`chunking_strategy`): `"sentence_preserving"` (default, 375 words/~1875 chars), `"semantic"`, `"header_aware"`, `"fixed_size"`.
- **Hybrid search**: Enabled by default (dense + sparse, `hybrid_rrf_k=60`).
- **RAG pipeline** (`services/async_rag.py`): Query rewriting → vector retrieval → cross-encoder reranking → context assembly → LLM → groundedness verification. Two-tier query cache (exact + semantic).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`. Downloaded at runtime (~450MB). Singleton-cached in `services/reranker.py`.
- **Output parsing**: `parse_rag_response()` + `verify_citations()` in `services/structured_output.py`.
- **Crawl DB**: PostgreSQL via `PostgresCrawlFrontierDB` (set `CRAWL_DB_URL`).

## Operational Gotchas
- **Qdrant health**: Use `GET /` (port 6333). `/health` returns 404.
- **Ollama `raw: True`**: Strips `<think>` tags. Empty response = output budget exhausted (increase `ollama_num_predict`).
- **OpenRouter rate limiter**: Per-provider `SlidingWindowRateLimiter` (20 RPM / 1000 RPD). Retries 429s up to 5x with exp backoff + `Retry-After`.
- **Worker time limits**: Soft 36000s (10h), Hard 43200s (12h). Zombie recovery via `task_failure` signal handler.
- **Qdrant batch splitting**: `upsert_chunks` splits into 256-chunk sub-batches (avoids 32MB payload limit).
- **URL dedup**: SHA-256 via `AsyncUrlRegistry` in Redis.
- **Ollama testcontainer caching**: Integration/e2e conftests mount `~/.ollama`; models cached across runs.
- **respx + Ollama embeddings**: `respx` passthrough returns empty bodies for `/api/embed`. To mock LLM with real embeddings, implement `LLMClientProtocol` instead of wire-mocking.

## CI Workflows
- In `.github/workflows.disabled/test.yml` (disabled). Jobs: `lint`, `test-unit` (+coverage), `test-eval`, `test-integration`, `test-e2e`.
- CI pulls 3 Ollama models: `nomic-embed-text`, `llama3.2:3b`, `qwen2.5-coder:7b`.

## Session Guardrails
- Run `git status` at start; alert on uncommitted changes.
- Check `plans/` and `sessions/` for stale files.
- Save plans to `plans/PLAN_<desc>_<YYYY-MM-DD_HHmm>.md` before presenting.
- **Always list file names explicitly** in `git add` and `git commit` commands. Never use `-A`, `--all`, or `-a`.
- **Never commit, push, or `git add`**. Print exact commands for user to run.
- **Never run commands taking >15 min** (large Docker pulls, full integration suites). Print command, ask user.
- Validate fixes with `pytest <specific_test>` before broader suites.
- Verify file existence with `git status` before suggesting `git add`.

## Additional Instruction Sources
See `.clinerules/` for environment-specific rules (python env, guardrails, behavioral constraints).
