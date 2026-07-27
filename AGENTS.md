# DataEngineeringCopilot — Agent Guide

## Python & Environment
- **Virtual Env**: Always use `dec_venv/bin/python` or `dec_venv/bin/dec`. Never bare `python`/`pip`.
- **Package Management**: `uv pip install -e ".[dev]"` for dev install; CI uses `uv sync --frozen --extra dev`. `uv.lock` pins exact dependency versions.
- **Config Load Chain**: `.env` → `.env.secrets` → `.env.local` (later overrides earlier). `.env.local` is gitignored. See `.env.example` for all variable names.
- **Redis Auth**: Local Docker Redis requires password `local_secure_password_123` in connection string. Default Redis URL from settings (`redis://redis:6379/0`) is for inside-Docker; `.env` overrides for localhost.
- **Logging**: structlog with JSON output (production) or console (DEBUG). Controlled via `LOG_LEVEL` env var.

## Entry Points
- CLI: `dec <command>` (registered in `pyproject.toml` as `main:main`; also `main.py` → `data_engineering_copilot.cli:main`). Dispatches ingestion through FastAPI backend.
- API: `dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000`
- Streamlit UI: `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py`
- Celery worker: `celery -A data_engineering_copilot.workers.tasks worker --concurrency=4 -Q ingestion -E`

## CLI Commands
| Command | Purpose |
|---|---|
| `dec ask <question>` | Query the RAG pipeline |
| `dec ingest --source "<name>"` | Crawl & index docs (name must match `documentation_sources.json`) |
| `dec status [task-id]` | Ingestion progress from Redis |
| `dec health` | Verify all service connections |
| `dec config` | Show config (secrets redacted) |
| `dec reset-index` | Clear Qdrant + Redis |
| `dec evaluate` | Run RAG evaluation pipeline |
| `dec profile --sources "X" --max-pages 20 --load-sweep "10,20,50,100"` | Ingestion profiling (defaults: 10,20,50,100 load sweep) |

## Testing
- **Commands** (via Makefile):
  - `make test-quick`: unit tests excluding `@slow`, parallel (~15s)
  - `make test-unit`: all unit tests
  - `make test-unit-serial`: unit tests, no parallel (`-n 0`)
  - `make test-smoke`: quick sanity — unit tests, no slow, no header
  - `make test-integration`: sequential + 2 reruns; requires Docker services
  - `make test-integration-parallel`: integration tests with controlled parallelism (`-n 2 --dist=loadgroup`)
  - `make test-e2e`: full pipeline tests; requires Docker services
  - `make test-eval`: RAG quality evaluation (mocked embedder, no infra)
  - `make test-ci-unit`: unit tests with coverage (XML + term-missing)
  - `make lint` / `make format`: Ruff only
- **CI Gate**: `lint` → `test-ci-unit` (coverage) → `test-integration` → `test-e2e` (see `.github/workflows/test.yml`). The `make test-ci` target runs all three with coverage.
- **Pytest Config** (`pyproject.toml`): `asyncio_mode = "auto"` — never decorate with `@pytest.mark.asyncio`. `asyncio_default_fixture_loop_scope = "function"`. `addopts = "-n auto --dist worksteal --strict-markers"`. Coverage excludes `data_engineering_copilot/ui/*`.
- **Shared Fixtures** (`tests/conftest.py`): `integration_settings`, `embeddings_provider`, `qdrant_store`, `ollama_client`, `populated_store`, `rag_service`, `api_client`. Auto-skips integration-marked tests when Docker services unreachable.
- **Pre-commit**: ruff lint+fix, ruff-format, trailing-whitespace, end-of-file-fixer.

## Source Layout
```
data_engineering_copilot/
  cli.py                    CLI dispatcher (main entry point)
  api/app.py                FastAPI app
  config/settings.py        AppSettings (pydantic-settings), settings singleton
  config/documentation_sources.json  4 sources: Spark, Airflow, Databricks, Delta Lake
  config/logging.py         structlog setup
  domain/models.py          Dataclasses: DocumentChunk, Answer, RagConfig, etc.
  infrastructure/           Adapters: Qdrant, Ollama, OpenRouter, OpenAI, crawl, rate_limiter
  factory.py                DI: build_rag_service(), build_async_ingestion_service(), etc.
   services/                 24 modules: ingestion, RAG, chunker, reranker, query_rewriting, etc.
  workers/celery_app.py     Celery config, task routing to "ingestion" queue
  workers/tasks.py          async_ingest_task + legacy execute_background_ingestion
  ui/streamlit_app.py       Streamlit interface
  profiler/cli.py           Profiling CLI
  observability/            Langfuse, OpenTelemetry, structlog, token tracking
```

## Docker Services
- `redis`, `qdrant`, `ollama`, `minio`, `clickhouse`, `langfuse` (incl. postgres + worker), `backend-api`, `celery_worker`
- Commands: `make docker-up/down/status/rebuild/logs/health/setup/cleanup/stop-all`
- `make docker-setup` pulls Ollama models automatically after `docker-up`
- CI stack: `make docker-ci-up` (uses `docker-compose.ci.yml`, prefix `dec_ci_*`)
- Worker volume mount `.:/app` — code changes need worker restart, not rebuild.

## Architecture Constraints
- **No LangChain/LlamaIndex** (except `langchain-text-splitters` for syntax-aware chunking)
- **Factory DI**: Always use `build_rag_service()`, `build_async_ingestion_service()`, etc. Never instantiate manually.
- **Async Only**: `SafeAsyncClientMixin` in `infrastructure/async_client.py`. Use `httpx.AsyncClient` / `aiohttp`.
- **Provider Support**: LLM: ollama, openrouter. Embeddings: ollama, openrouter, openai. Switching requires `dec reset-index` (vector dimension changes).
- **Chunking**: Strategy in `settings.chunking_strategy` — `"sentence_preserving"` (default), `"semantic"`, `"header_aware"`, `"fixed_size"`.
- **Hybrid Search**: Enabled by default (keyword + vector). Configured via `hybrid_search_enabled` and `hybrid_rrf_k`.
- **RAG Pipeline**: Query rewriting → vector retrieval → cross-encoder reranking → context assembly → LLM → groundedness verification. Two-tier query cache (exact + semantic) with NumPy SIMD scoring.
- **Crawl Database**: SQLite at `data/crawl_frontier.db` via `CrawlFrontierDB`.

## Plan Mode Discipline
- **No edits in plan mode**: When the system says "Plan mode ACTIVE — READ-ONLY", do not modify any files. Only present the plan. Wait for the user to explicitly transition you to build mode before making changes.

## Operational Gotchas
- **Qdrant Health**: Use `GET /` (port 6333), not `/health` (returns 404).
- **Ollama `raw: True`**: Strips `<think>` tags. Empty response = output budget exhausted (increase `ollama_num_predict`).
- **OpenRouter Rate Limiter**: Shared singleton coordinates 20 RPM / 1000 RPD between LLM and embedding clients. Retries 429s up to 5x with exponential backoff + `Retry-After`.
- **Worker Time Limits**: Soft 36000s (10h), Hard 43200s (12h). Zombie recovery via `task_failure` signal handler marks FAILED in Redis.
- **Qdrant Batch Splitting**: `upsert_chunks` splits into 256-chunk sub-batches to avoid 32MB payload limit.
- **URL Dedup**: SHA-256 via `AsyncUrlRegistry` in Redis.
- **Ingestion Lock**: Released before flush to prevent unbounded chunk accumulation.
