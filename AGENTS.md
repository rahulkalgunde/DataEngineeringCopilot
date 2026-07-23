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
```
- Parallel execution is enabled by default (`-n auto --dist worksteal` via `addopts`). Use `-n 0` to debug sequentially.
- Pytest markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`.
- Integration tests auto-skip when services are unreachable (`tests/conftest.py`).
- Coverage omits `data_engineering_copilot/ui/*`.
- Ruff: line-length=120, select=E,F,W,I,UP,B,SIM, ignore=E501.

## Package Management
- **NEVER** use `pip` or `python -m venv`. Use **`uv`** exclusively.
- Install: `uv pip install -e ".[dev]"`
- Target venv: `dec_venv/bin/python`
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
- `dec` console script (from `pyproject.toml`) works as shortcut after `pip install -e .`.
- Source selection uses **exact name match** against `config/documentation_sources.json`.

## Infrastructure Dependencies
- `docker compose up -d` brings up: Redis, Qdrant (6333/6334), Ollama (11434), Langfuse (3000), langfuse-worker, langfuse-postgres, PostgreSQL, ClickHouse, MinIO (+ minio-init), backend-api, celery_worker.
- `docker compose -f docker-compose.ci.yml up -d --wait` brings up infra only (no app services) — used in CI.
- **Ollama is now in Docker** — no need to start separately: `docker exec de_copilot_ollama ollama pull <model>`.

## CI Pipeline
- `.github/workflows/test.yml` — 4 jobs: `lint`, `test-unit`, `test-integration`, `test-e2e`.
- `test-integration` starts the full `docker-compose.ci.yml` stack (Qdrant, Redis, Ollama, Langfuse stack).
- `test-e2e` starts only Qdrant, Redis, and Ollama (Langfuse not needed).
- Ollama models (`nomic-embed-text`, `llama3.2:3b`) are pulled via `docker exec` after compose healthcheck passes.
- Ollama models are cached via `actions/cache` (`~/.ollama` bind-mounted to `.ollama_cache/`). First run pulls all models (~2.3 GB), subsequent runs restore from cache.
- Tests auto-skip via `tests/conftest.py` when a required service is unreachable (port-check with 3s timeout).

## Architecture
- **Pattern**: layered — config → domain → infrastructure → services → CLI/UI/API
- **No LangChain/LlamaIndex** — direct crawler/parser/chunker/embed/vector/query/generate pipeline.
- **Entry points**: `main.py` (CLI → `data_engineering_copilot.cli`), `data_engineering_copilot/ui/streamlit_app.py` (Streamlit), `data_engineering_copilot/api/app.py` (FastAPI).
- **Factory** (`factory.py`): wires everything. `build_chunker()` selects strategy; `build_async_ingestion_service()` and `build_rag_service()` construct full stacks.

### Data Flow
```
CLI/UI → AsyncIngestionService → AsyncCrawler → MarkdownParser → Chunker → Embeddings → QdrantVectorStore
CLI/UI → ProductionRagService → Embeddings → QdrantVectorStore → Reranker → ContextAssembler → OllamaClient
```

## Key Gotchas
- **Ollama raw prompt**: `AsyncOllamaClient` sends `"raw": True` to skip Ollama's chat template, then strips `<think>` tags from output (`infrastructure/async_ollama_client.py:59`). Empty-response errors mean the model exhausted its budget on reasoning — increase `ollama_num_predict` or reduce context.
- **`.env` dead config**: `.env` sets `LANGFUSE_BASE_URL` but `AppSettings` reads `LANGFUSE_HOST`. Set `LANGFUSE_HOST` for local Langfuse config.
- **`reset-index`** deletes Qdrant collection, crawl frontier SQLite DB (`data/crawl_frontier.db`), and Redis crawl registry keys.
- **Content-hash dedup**: `AsyncIngestionService` computes SHA-256 hashes and skips re-indexing unchanged pages via `UrlRegistry` (Redis). Chunks from removed pages are **not** cleaned up.
- **Stale chunks**: changed pages have old chunks deleted before re-indexing, but pages removed from the crawl leave orphaned chunks.
- **No canonical URL normalization** beyond fragments/slash dedupe — query-string variants may duplicate pages.
- **Langfuse tracing** in `ProductionRagService`: graceful fallback if Langfuse unavailable — RAG still works.

## Operational Guardrails (from `.clinerules`)
- Do not install new software or run `sudo` without explicit user permission.
- Never use `--force` or destructive delete commands without explicit permission.
