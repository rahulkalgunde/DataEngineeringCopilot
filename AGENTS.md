# DataEngineeringCopilot — Agent Guide

## Python & Environment
- Always use `dec_venv/bin/python` or `dec_venv/bin/dec`. Never bare `python` or `pip`.
- Install: `uv pip install -e ".[dev]"`. CI uses `uv sync --frozen --extra dev`.
- Build system: `hatchling` (pyproject.toml). Entrypoint: `dec = "main:main"`.

## Commands & Testing
```bash
make install           # uv pip install -e ".[dev]"
make test              # all tests (parallel: -n auto --dist worksteal)
make test-quick        # unit only, no @slow (776 tests, ~15s)
make test-unit         # all unit tests (778)
make test-integration  # integration (sequential, --reruns 2)
make test-e2e          # end-to-end pipeline test
make test-eval         # RAG quality eval (mocked embedder, no infra)
make lint              # ruff check data_engineering_copilot/ tests/
make format            # ruff format data_engineering_copilot/ tests/
make test-ci-unit      # unit + coverage (xml + term-missing)
make test-ci           # unit + integration + e2e with coverage
make test-smoke        # fast sanity: unit not slow, -q --no-header
```
- Pytest: `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- `asyncio_default_fixture_loop_scope = "function"` — httpx clients recreate per test.
- Marker auto-skip: conftest checks Qdrant/Ollama/Langfuse/Redis availability; integration tests auto-skip when unreachable.
- Shared fixtures in `tests/conftest.py`: `integration_settings`, `embeddings_provider`, `qdrant_store`, `ollama_client`, `populated_store`, `rag_service`, `api_client`.

## Running & CLI
- Use `dec_venv/bin/dec` (installed via `pyproject.toml [project.scripts]`): `dec ingest`, `dec ask "..."`, `dec reset-index`, `dec ui`.
- `dec ingest --source "Apache Spark Documentation" --max-pages 40` — source names must match `documentation_sources.json` exactly.
- Streamlit: `dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py`
- FastAPI: `dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000`

## Services & Docker
- `make docker-up` → Redis (auth: `local_secure_password_123`), Qdrant (6333/6334), Ollama (11434), Langfuse (3000), MinIO, ClickHouse, Celery worker, API backend.
- After starting, pull models: `docker exec de_copilot_ollama ollama pull nomic-embed-text` and `llama3.2:3b`.
- CI stack: `make docker-ci-up` (`docker-compose.ci.yml`, containers prefixed `dec_ci_*`).

## Architecture & Gotchas
- **Pipeline** (no LangChain/LlamaIndex): `AsyncCrawler` → `MarkdownParser` → `Chunker` → `Embeddings` → `QdrantVectorStore`.
- **RAG pipeline** (stages): query rewriting → hybrid search (dense+sparse) → cross-encoder reranking → context compression → groundedness verification → RAGAS evaluation. All configurable via settings + feature flags.
- **Factory DI**: `factory.py` wires all components. Never instantiate `AsyncRagService` or `AsyncIngestionService` manually — use `build_rag_service()` / `build_async_ingestion_service()`.
- **Async everywhere**: CLI uses `asyncio.run()`. All infrastructure clients use `httpx.AsyncClient` / `aiohttp`. All infra clients derive from `SafeAsyncClientMixin` for loop-safe transport lifecycle.
- **Ollama raw mode**: `raw: True` in API payload; strips `<think>` tags from response. Empty = output budget exhausted (increase `ollama_num_predict`).
- **Chunking strategies**: `fixed_size`, `sentence_preserving` (default), `semantic`, `header_aware`. Set via `chunking_strategy` in settings.
- **Providers**: Ollama (default), OpenRouter, OpenAI for embeddings/LLM. Set via `llm_provider` / `embedding_provider`. Switching embedding provider requires `dec reset-index` (different vector dimensions).
- **Config**: `.env` sets `LANGFUSE_HOST` (note: mapped from `LANGFUSE_BASE_URL` in env). Settings via `pydantic-settings` frozen class.
- **Reset index**: `dec reset-index` deletes Qdrant collection, crawl frontier SQLite (`data/crawl_frontier.db`), Redis `crawl:url_registry:*` keys. Also recreates collection at correct dimension for active provider.
- **Dedup**: SHA-256 content hash via async Redis `AsyncUrlRegistry`.
- **Logging**: structlog JSON format. Logs under `logs/`. `LOG_LEVEL` env var for debug.
- **Observability**: Langfuse for traces, OpenTelemetry (`otel_telemetry.py`), bounded token/retrieval trackers.
- **Query cache**: `QueryCache` in `services/query_cache.py` — two-tier in-memory cache with NumPy SIMD vector scoring and `deque`-bounded eviction. Wired by default into the RAG pipeline.

## CI Pipeline (`.github/workflows/test.yml`)
Order: `lint` → `test-unit` (coverage) → `test-integration` (needs lint+unit) → `test-e2e` (needs integration). Docker image caching and Ollama model caching for integration jobs. Uses `uv sync --frozen --extra dev`.

## Testing Patterns
- Unit tests: mock infra with `AsyncMock`. `tests/unit/conftest.py` provides `mock_vector_store`, `mock_ollama`, `mock_embedder`. Auto-fakes `sentence_transformers` import.
- Integration tests: `require_qdrant()`, `require_ollama()`, etc. from conftest for conditional skip. Unique collection names per test for isolation. Teardown deletes collection.
- Eval tests: `make test-eval` — uses mocked embedder, no services needed.
- Markers: `unit`, `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`, `evaluation`, `xdist_group`.
