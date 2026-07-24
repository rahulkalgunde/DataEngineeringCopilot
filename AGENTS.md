# DataEngineeringCopilot — Agent Guide

## TDD
Write tests first. Run tests after every code change.

## Python & Environment
- Always use `dec_venv/bin/python` (project-root venv). Never `python` or `pip`.
- Install: `uv pip install -e ".[dev]"`. Only `uv`, never `pip` or `python -m venv`.

## Commands
```bash
make install          # uv pip install -e ".[dev]"
make test             # all tests (parallel: -n auto --dist worksteal)
make test-quick       # unit only, no @slow (~15s)
make test-unit        # all unit tests
make test-unit-serial # sequential (debug xdist issues)
make test-integration # integration (sequential, --reruns 2)
make test-integration-parallel  # integration with xdist loadgroup (2 workers)
make test-e2e         # end-to-end
make test-ci          # CI gate: unit (with coverage) + integration + e2e
make test-smoke       # quick sanity: unit, no @slow, no header
make lint             # ruff check data_engineering_copilot/ tests/
make format           # ruff format data_engineering_copilot/ tests/
make docker-up        # docker compose up -d (full stack)
make docker-ci-up     # docker compose -f docker-compose.ci.yml up -d --wait (infra only)
```
- Pytest markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`, `evaluation`, `xdist_group(name)`.
- Integration tests auto-skip when services unreachable (`tests/conftest.py`). Run with `-n 0 --reruns 2`.
- `asyncio_mode = "auto"` in pyproject.toml; `asyncio_default_fixture_loop_scope = "function"`. Async tests do NOT need `@pytest.mark.asyncio` — `auto` mode handles it.

## Running the App
```bash
dec_venv/bin/python main.py ingest --max-pages 40           # or: dec ingest
dec_venv/bin/python main.py ingest --source "Apache Spark"  # single source (exact name match)
dec_venv/bin/python main.py ask "question"                  # or: dec ask "..."
dec_venv/bin/python main.py reset-index                     # deletes Qdrant coll + frontier DB + Redis keys
dec_venv/bin/python main.py ui                              # prints streamlit command
dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py
dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000
```
- `dec` console script is a shortcut (defined in pyproject.toml): `dec ingest`, `dec ask "..."`.
- Source names must **exactly match** entries in `data_engineering_copilot/config/documentation_sources.json`.
- Celery worker: `celery -A data_engineering_copilot.workers.tasks worker --loglevel=info`

## Services & Docker
- `make docker-up` → Redis, Qdrant (6333/6334), Ollama (11434), Langfuse (3000), langfuse-worker, postgres, ClickHouse, MinIO, backend-api, celery_worker.
- Redis requires auth password `local_secure_password_123` (set in docker-compose).
- Pull Ollama models: `docker exec de_copilot_ollama ollama pull nomic-embed-text` and `llama3.2:3b`.
- CI uses `COMPOSE_PROJECT_NAME: dec_ci` (`.github/workflows/test.yml`). Containers are prefixed `dec_ci_*`.

## Architecture
```
CLI/UI → AsyncIngestionService → AsyncCrawler → MarkdownParser → Chunker → Embeddings → QdrantVectorStore
CLI/UI → AsyncRagService → Embeddings → QdrantVectorStore → Reranker → ContextAssembler → OllamaClient
```
- No LangChain/LlamaIndex — fully manual pipeline.
- **Entrypoints**: `main.py` → `cli.py`, `ui/streamlit_app.py`, `api/app.py` (FastAPI).
- **Factory** (`factory.py`): wires everything — `build_chunker()`, `build_async_ingestion_service()`, `build_rag_service()`.
- **Layers**: `config/` → `domain/` → `infrastructure/` → `services/` → `cli.py | ui/ | api/`
- **Phase 2 services**: `QueryRewriter`, `GroundednessVerifier`, `ContextCompressor` (wired in `build_rag_service`).
- **Reranker**: `sentence-transformers` CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), module-level singleton.
- **Cache**: two-tier query cache (exact + semantic similarity) in `services/query_cache.py`; plus `CrawlCache` in `infrastructure/crawl_cache.py`.
- **Observability**: `LangfuseTelemetryTracer` / `NoOpTelemetryTracer` fallback + `TokenTracker` + structlog.
- **Chunking strategies**: `sentence_preserving` (default), `fixed_size`, `semantic`, `header_aware` (splits on `#`/`##`/`###` headers).
- **Query intents**: `factual`, `how_to`, `debugging`, `comparative`, `api_lookup` (filters to `chunk_type="api"`), `code_example` (filters to `chunk_type="code"`).
- **Post-processors**: `ApiDocExtractor` (API metadata enrichment), `CodeBlockParser` (code block splitting with AST), `ContextualChunkEnricher` (LLM-based context prefixes, opt-in).

## Gotchas
- **Ollama raw mode**: `AsyncOllamaClient` sends `"raw": True` to skip Ollama's chat template, then strips `<think>` tags. Empty response means model exhausted output budget — increase `ollama_num_predict` or reduce context.
- **`.env` dead config**: `.env` sets `LANGFUSE_BASE_URL` but `AppSettings` reads `LANGFUSE_HOST`. Set `LANGFUSE_HOST` env var.
- **`reset-index`**: deletes Qdrant collection, crawl frontier SQLite DB (`data/crawl_frontier.db`), and Redis `crawl:url_registry:*` keys.
- **Content-hash dedup**: SHA-256 of page content; unchanged pages skipped via Redis `AsyncUrlRegistry`.
- **Langfuse tracing**: graceful fallback to `NoOpTelemetryTracer` if unavailable.
- **Semantic chunker**: gated by `chunking_strategy=semantic` + `enable_semantic_chunking=True` (default strategy is `sentence_preserving`).
- **FastAPI rate limit**: 60/min for `/ask`, 10/min for `/ingest`.
- **Logs**: structlog to `logs/app.log` and `logs/ingestion_refresh.log`.
- **No canonical URL normalization** — query-string variants may duplicate pages.

## Agent Behavioral Rules
- **One edit per turn**: Never modify more than ONE file at a time. Do not chain multiple write actions.
- **Read before writing**: Always `read` a file fully before editing. Never guess contents.
- **No speculative code**: Write complete, production-ready implementations. No placeholders or TODOs.
- **No unasked refactoring**: Fix only the explicit target requested. Do not clean up surrounding code.
- **Single-command rule**: Run exactly ONE terminal command at a time. Wait for output before next.
- **Anti-looping**: If you fail to resolve an error after 2 sequential attempts using the same tool, STOP and present current state.
- **Test-driven**: Write tests first. After code change, run unit tests.

## Session & Plan Conventions
- Check `git status` at start; alert user if uncommitted changes exist.
- Check `plans/` and `sessions/` for stale files to resume.
- Save plans to `plans/PLAN_<desc>_<YYYY-MM-DD_HHmm>.md` before presenting.
- Save session details to `sessions/SESSION_<desc>_<YYYY-MM-DD_HHmm>.md` on context loss.
- **Never** `git commit`, `push`, `add`, or history-modifying commands — only print the commands for the user.
- **Never** run commands that may take >15 min — print and ask.

## Ruff Config
- line-length=120, target=py312, select=E,F,W,I,UP,B,SIM, ignore=E501.
- Coverage omits `data_engineering_copilot/ui/*`.

## Pre-commit
- Hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files (500KB), ruff --fix, ruff-format.
- Commits may be blocked if files fail these checks. Run `make lint && make format` before committing.
