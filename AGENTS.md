# DataEngineeringCopilot — Agent Guide

## Python & Environment
- **Virtual Env**: Always use `dec_venv/bin/python` or `dec_venv/bin/dec`. Never run bare `python` or `pip`.
- **Dependency Management**: Dev package installation uses `uv pip install -e ".[dev]"`.

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
- **Ingestion**: Source names in `dec ingest --source "<name>"` must match names in `documentation_sources.json` exactly.
- **Streamlit UI**: `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py`
- **FastAPI Backend**: `dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000`

## Services & Docker
- **Local Stack**: Start via `make docker-up` (runs Qdrant, Redis, Ollama, Langfuse/Clickhouse, worker, and backend).
- **Ollama Models**: Run `docker exec de_copilot_ollama ollama pull nomic-embed-text` and `llama3.2:3b` after starting.
- **CI Stack**: Start via `make docker-ci-up` (uses `docker-compose.ci.yml` and container prefix `dec_ci_*`).
- **Rebuild**: Rebuild backend/worker via `docker compose build --no-cache backend-api && docker compose up -d backend-api celery_worker`.

## Profiler & Concurrency Tuning
- **Profiling CLI**: `dec profile --sources "Apache Spark" --max-pages 20 --concurrency-sweep "1,2,4,8"`
- **How it works**: Monitors per-stage timing, CPU/memory, and rate limits. Uses Little's Law (`ConcurrencyTuner`) to recommend/apply stage scalability options (`SCALE_UP` / `SCALE_DOWN` / `RATE_LIMITED`).

## Architectural Gotchas
- **No LangChain/LlamaIndex**: This is a custom pipeline. Do NOT import or use LangChain/LlamaIndex abstractions (except `langchain-text-splitters` which is explicitly used for syntax-aware document chunking).
- **Dependency Injection**: Always use `factory.py` (e.g., `build_rag_service()`, `build_async_ingestion_service()`). **Never** instantiate services manually.
- **Async & Transport**: Derived from `SafeAsyncClientMixin`. Use `httpx.AsyncClient` / `aiohttp`. Never write blocking sync code.
- **Ollama `raw: True`**: Strips `<think>` tags. An empty Ollama response means output budget exhaustion (increase `ollama_num_predict`).
- **Provider Swapping**: Switching LLM/Embedding providers requires running `dec reset-index` to clean databases and re-initialize Qdrant collection at the correct vector dimension.
- **Dedup & Cache**: URL deduplication uses SHA-256 in Redis `AsyncUrlRegistry`. RAG uses a two-tier in-memory `QueryCache` (with NumPy SIMD vector scoring and deque eviction).
