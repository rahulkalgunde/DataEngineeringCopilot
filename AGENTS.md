# DataEngineeringCopilot — Agent Guide

## TDD Policy
Write tests first. Never write implementation before tests exist. Run tests immediately after code changes.

## Session & Plan Guardrails
- Run `git status` at start. Alert user if uncommitted changes exist.
- Check `plans/` and `sessions/` for stale files — resume if applicable.
- Save plans to `plans/PLAN_<desc>_<YYYY-MM-DD_HHmm>.md` before presenting.
- Save session details to `sessions/SESSION_<desc>_<YYYY-MM-DD_HHmm>.md` after context loss.
- **Never run `git commit`, `git push`, `git add`, or any git command that changes history.** Only remind the user with the exact commands to run.
- **Never run commands that may take longer than 15 minutes** (e.g., pulling large Docker images, running full integration suites). Print the command and ask the user to run it.
- After each milestone, print the exact `git add`/`git commit`/`git push` commands and ask the user to run them.

## Commands
```bash
make install          # uv pip install -e ".[dev]"
make test             # all tests (parallel by default via -n auto --dist worksteal)
make test-quick       # unit tests only, no @slow tests (fast feedback, ~15s)
make test-unit        # all unit tests (including @slow)
make test-unit-serial # unit tests sequentially (for debugging xdist issues)
make test-integration # integration tests only
make test-e2e         # e2e tests only
make test-ci          # CI gate: unit + integration + e2e with coverage
make test-smoke       # quick sanity check
make lint             # ruff check data_engineering_copilot/ tests/
make format           # ruff format data_engineering_copilot/ tests/
make clean            # remove __pycache__, .pytest_cache, .pyc
make docker-up        # docker compose up -d (full stack)
make docker-down      # docker compose down
make docker-ci-up     # docker compose -f docker-compose.ci.yml up -d --wait (infra only, no app)
```

## Running Tests
```bash
dec_venv/bin/python -m pytest tests/unit/ -v                          # unit only
dec_venv/bin/python -m pytest tests/unit/test_chunker_improved.py -v  # single file
dec_venv/bin/python -m pytest tests/integration/ -v                   # integration
dec_venv/bin/python -m pytest tests/integration/ -v -m qdrant         # by marker
dec_venv/bin/python -m pytest tests/e2e/ -v                           # e2e
dec_venv/bin/python -m pytest tests/evaluation/ -v                    # RAG quality evaluation
```
- Parallel execution enabled by default (`-n auto --dist worksteal` via `addopts`). Use `-n 0` to debug sequentially.
- Pytest markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`, `evaluation`, `xdist_group(name)`.
- Integration tests auto-skip when services unreachable (`tests/conftest.py`). Run sequentially (`-n 0`) with `--reruns 2`.
- Coverage omits `data_engineering_copilot/ui/*`. Ruff: line-length=120, select=E,F,W,I,UP,B,SIM, ignore=E501.

## Package Management
- **NEVER** use `pip` or `python -m venv`. Use **`uv`** exclusively.
- Install: `uv pip install -e ".[dev]"`
- Always target `dec_venv/bin/python` (project-root venv).
- `.pre-commit-config.yaml` runs ruff lint+format, trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files on commit.

## Running the App
```bash
dec_venv/bin/python main.py ingest --max-pages 40
dec_venv/bin/python main.py ask "question"
dec_venv/bin/python main.py reset-index
dec_venv/bin/python main.py ui                        # prints Streamlit launch command
dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py
dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000
```
- `dec` console script works as shortcut: `dec ingest`, `dec ask "..."`, etc.
- Source selection uses **exact name match** against `config/documentation_sources.json`.
- Celery worker: `celery -A data_engineering_copilot.workers.tasks worker --loglevel=info`

## Infrastructure Dependencies
- `docker compose up -d` brings up: Redis, Qdrant (6333/6334), Ollama (11434), Langfuse (3000), langfuse-worker, langfuse-postgres, ClickHouse, MinIO (+ init), backend-api, celery_worker.
- `docker compose -f docker-compose.ci.yml up -d --wait` — infra only (no app services).
- **Ollama runs in Docker.** Pull models: `docker exec de_copilot_ollama ollama pull nomic-embed-text` and `llama3.2:3b`.
- Ollama models cached in `.ollama_cache/` (bind-mounted in CI).
- `AppSettings` defaults `redis_url` to `redis://redis:6379/0` (Docker-internal). Override for local dev.

## CI Pipeline
- `.github/workflows/test.yml` — 4 jobs: `lint`, `test-unit`, `test-integration`, `test-e2e`.
- Integration: full `docker-compose.ci.yml` stack. E2E: Qdrant + Redis + Ollama only.
- Docker images cached via `actions/cache` (`/tmp/docker-cache/`).

## Architecture
- **Pattern**: layered — config → domain → infrastructure → services → CLI/UI/API
- **No LangChain/LlamaIndex** — direct crawler/parser/chunker/embed/vector/query/generate pipeline.
- **Entry points**: `main.py` → `cli.py`, `ui/streamlit_app.py`, `api/app.py` (FastAPI).
- **Factory** (`factory.py`): wires everything. `build_chunker()` selects strategy (fixed_size/sentence_preserving/semantic); `build_async_ingestion_service()` and `build_rag_service()` construct full stacks.
- **Reranker**: `sentence-transformers` CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), module-level singleton in `services/reranker.py`.
- **Cache**: two-tier query cache (exact match + semantic similarity with configurable threshold/ttl) in `services/query_cache.py`.
- **Async infra**: `httpx.AsyncClient` with event-loop detection (recreates client on loop change for pytest function-scoped loops).

### Data Flow
```
CLI/UI → AsyncIngestionService → AsyncCrawler → MarkdownParser → Chunker → Embeddings → QdrantVectorStore
CLI/UI → AsyncRagService → Embeddings → QdrantVectorStore → Reranker → ContextAssembler → OllamaClient
```

## Key Gotchas
- **Ollama raw prompt**: `AsyncOllamaClient` sends `"raw": True` to skip Ollama's chat template, then strips `<think>` tags (`infrastructure/async_ollama_client.py:93`). Empty responses mean the model exhausted output budget — increase `ollama_num_predict` or reduce context.
- **`.env` dead config**: `.env` sets `LANGFUSE_BASE_URL` but `AppSettings` reads `LANGFUSE_HOST`. Set `LANGFUSE_HOST` env var.
- **`reset-index`** deletes Qdrant collection, crawl frontier SQLite DB (`data/crawl_frontier.db`), and Redis `crawl:url_registry:*` keys.
- **Content-hash dedup**: `AsyncIngestionService` computes SHA-256 and skips unchanged pages via `AsyncUrlRegistry` (Redis). Changed pages have old chunks deleted before re-index; removed pages leave orphaned chunks.
- **No canonical URL normalization** beyond fragments/slash dedupe — query-string variants may duplicate pages.
- **Langfuse tracing**: graceful fallback to `NoOpTelemetryTracer` if Langfuse unavailable — RAG still works.
- **Semantic chunker**: gated by `enable_semantic_chunking` (default True). Uses `min_semantic_similarity=0.5`. Falls back to `sentence_preserving` if disabled.
- **Rate limiting**: FastAPI middleware — 60/min for `/ask`, 10/min for `/ingest`.
- **Logging**: structlog to `logs/app.log` and `logs/ingestion_refresh.log`.

## Operational Guardrails
- Do not install new software or run `sudo` without explicit user permission.
- Never use `--force` or destructive delete commands without explicit permission.
