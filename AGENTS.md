# DataEngineeringCopilot — Agent Guide

## Python & Environment
- **Virtual Env**: Always use `dec_venv/bin/python` or `dec_venv/bin/dec`. Never run bare `python` or `pip`.
- **Dependency Management**: Dev package installation uses `uv pip install -e ".[dev]"`.
- **Config Split**: Settings load from `.env` → `.env.secrets` → `.env.local` (later files override earlier). `.env.local` is gitignored for sensitive overrides.
- **Redis Auth**: `redis://:local_secure_password_123@localhost:6379/0` — password is required in all Redis connection strings.
- **Models**: LLM uses OpenRouter free tier, Embeddings use OpenRouter `nvidia/nemotron-3-embed-1b:free` (dimension=2048, hybrid search enabled).

## Testing & Verification
- **Commands**: 
  - `make test-quick`: Fast unit-test-only verification (~15s, ignores `@slow`).
  - `make test-integration`: Runs integration tests sequentially. Auto-skips if local Docker services are down.
  - `make test-e2e` / `make test-eval`: Runs full E2E / RAG evaluation tests.
  - `make lint` / `make format`: Uses Ruff for linting and formatting.
- **CI Pipeline Gating**: Order of jobs is `lint` -> `test-unit` -> `test-integration` -> `test-e2e`. Always run `make lint` and `make test-quick` before pushing.
- **Pytest Asyncio**: Configured as `asyncio_mode = "auto"`. **Never** decorate tests with `@pytest.mark.asyncio`.
- **Shared Fixtures (`tests/conftest.py`)**: Reuse `mock_ollama`, `mock_vector_store`, `mock_embedder` for unit tests; use `populated_store`, `api_client`, and `integration_settings` for integrations. Do not manually re-mock infrastructure.

## Running & CLI
- **Main CLI**: Use `dec_venv/bin/dec` (e.g., `dec ingest`, `dec ask`, `dec reset-index`).
- **CLI Commands**:
  - `dec health`: Verify all service connections (Redis, Qdrant, Ollama, OpenRouter).
  - `dec status`: Show ingestion job status from Redis with progress tracking.
  - `dec evaluate`: Run RAG evaluation pipeline.
  - `dec config`: Display current configuration (redacted secrets).
- **Ingestion**: Source names in `dec ingest --source "<name>"` must match names in `documentation_sources.json` exactly. CLI dispatches ingestion through the FastAPI backend.
- **Streamlit UI**: `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py`
- **FastAPI Backend**: `dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000`

## Services & Docker
- **Docker Commands** (via Makefile):
  - `make docker-up`: Start all services.
  - `make docker-down`: Stop all services.
  - `make docker-status`: Show all project containers, status, and health checks.
  - `make docker-rebuild`: Full rebuild with `--no-cache` then restart.
  - `make docker-logs`: Stream logs for all services (tail=100).
  - `make docker-logs-worker`: Stream worker-specific logs.
  - `make docker-health`: Run `dec health` to verify service connectivity.
  - `make docker-stop-all`: Stop all services without removing.
  - `make docker-cleanup`: Prune unused Docker images, containers, and volumes.
  - `make docker-setup`: Start services + pull Ollama models in one command.
- **Worker Health Check**: Uses Python script (`redis.ping()` + process check) instead of `celery inspect ping` which can hang.
- **Volume Mount**: Worker uses `.:/app` mount — code changes take effect after worker restart (no Docker rebuild needed).
- **CI Stack**: Start via `make docker-ci-up` (uses `docker-compose.ci.yml` and container prefix `dec_ci_*`).

## Rate Limiting
- **OpenRouter Rate Limiter**: Shared `OpenRouterRateLimiter` coordinates RPM (20 req/min) and RPD (1000 req/day) between embeddings and LLM clients.
- **429 Handling**: Both clients parse `Retry-After` header and retry up to 5 times with exponential backoff.

## Ingestion & Workers
- **Worker Time Limits**: Soft limit=36000s (10h), Hard limit=43200s (12h) — sufficient for large crawls.
- **Zombie Task Recovery**: Celery `task_failure` signal handler catches hard time limit kills and marks Redis status as FAILED (prevents stuck "in_progress" tasks).
- **Qdrant Batch Splitting**: `upsert_chunks` splits into 256-chunk sub-batches to prevent 32MB payload limit errors.
- **Ingestion Lock**: Lock is released before flush to prevent unbounded chunk accumulation.

## Profiler & Concurrency Tuning
- **Profiling CLI**: `dec profile --sources "Apache Spark" --max-pages 20 --load-sweep "100,500,1000,5000"`
- **How it works**: Monitors per-stage timing, CPU/memory, and rate limits. Tests page limits under fixed worker config.
- **API Dispatch**: Profiler dispatches ingestion through the FastAPI backend (production path), tracked via `dec status`.
- **Production Metrics**: Profiler report includes Redis metrics (pages crawled, chunks upserted, errors).

## Architectural Gotchas
- **No LangChain/LlamaIndex**: This is a custom pipeline. Do NOT import or use LangChain/LlamaIndex abstractions (except `langchain-text-splitters` which is explicitly used for syntax-aware document chunking).
- **Dependency Injection**: Always use `factory.py` (e.g., `build_rag_service()`, `build_async_ingestion_service()`). **Never** instantiate services manually.
- **Async & Transport**: Derived from `SafeAsyncClientMixin`. Use `httpx.AsyncClient` / `aiohttp`. Never write blocking sync code.
- **Ollama `raw: True`**: Strips `<think>` tags. An empty Ollama response means output budget exhaustion (increase `ollama_num_predict`).
- **Provider Swapping**: Switching LLM/Embedding providers requires running `dec reset-index` to clean databases and re-initialize Qdrant collection at the correct vector dimension.
- **Dedup & Cache**: URL deduplication uses SHA-256 in Redis `AsyncUrlRegistry`. RAG uses a two-tier in-memory `QueryCache` (with NumPy SIMD vector scoring and deque eviction).
- **Qdrant Health Check**: Uses `GET /` (returns 200), not `/health` (returns 404).
