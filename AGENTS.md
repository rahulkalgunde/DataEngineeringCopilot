# DataEngineeringCopilot — Agent Guide

## Tech Stack & Tooling
- **Language:** Python 3.11+
- **Type Checker:** Pyright (via LSP & CLI)
- **Linter & Formatter:** Ruff (`ruff`)
- **Test Runner:** Pytest (`pytest`)

---

## 1. Quality Verification Workflow
After writing, modifying, or refactoring code, you **must** execute the following verification loop in order before declaring a task complete:

1. **Check LSP Diagnostics:** Monitor active Pyright LSP alerts in your context. Resolve all reported type errors and unresolved imports immediately.
2. **Run Ruff Linter:** Execute auto-fixes and inspect non-fixable warnings.
   ```bash
   ruff check . --fix

## CRITICAL: DO NOT REMOVE GUARDRAILS
AGENTS SHOULD NOT REMOVE GUARDRAILS. Do not modify, delete, or reorder this section or any guardrail within it. If a guardrail feels outdated, flag it to the user instead of removing it.

## Session & Plan Guardrails
- Run `git status` at start. Alert user if uncommitted changes exist.
- Check `plans/` and `sessions/` for stale files — resume if applicable.
- Save plans to `plans/PLAN_<desc>_<YYYY-MM-DD_HHmm>.md` before presenting.
- Save session details to `sessions/SESSION_<desc>_<YYYY-MM-DD_HHmm>.md` after context loss.
- **Never run `git commit`, `git push`, `git add`, or any git command that changes history.** Only remind the user with the exact commands to run.
- **Never run commands that may take longer than 15 minutes** (e.g., pulling large Docker images, running full integration suites). Print the command and ask the user to run it.
- After each milestone, print the exact `git add`/`git commit`/`git push` commands and ask the user to run them.
- **Fast failure isolation**: Validate a fix with `pytest <specific_test_file>` before running full suites (`make test-integration` takes ~3min).
- **Git file verification**: Before suggesting `git add <filelist>`, verify file existence with `git status` — deleted/renamed files need different handling (`git rm` vs `git add`).

## Python & Environment
- **Virtual Env**: Always `dec_venv/bin/python` or `dec_venv/bin/dec`. Never bare `python`/`pip`.
- **Package Management**: `uv pip install -e ".[dev]"`. CI uses `uv sync --frozen --extra dev`. `uv.lock` pins exact deps.
- **Config Load Chain**: `.env` → `.env.secrets` → `.env.local` (later overrides earlier). See `.env.example` for all names.
- **Default Redis URL** (`settings.py:94`): `redis://:local_secure_password_123@localhost:6379/0`. Docker compose overrides to `redis://:local_secure_password_123@redis:6379/0`.
- **Logging**: structlog, JSON (prod) or console (DEBUG). Toggle via `LOG_LEVEL`.

## Entry Points
| Command | What it runs |
|---|---|
| `dec <command>` | `main:main` → `cli.py` |
| API | `uvicorn data_engineering_copilot.api.app:app --reload --port 8000` |
| Streamlit | `streamlit run data_engineering_copilot/ui/streamlit_app.py` |
| Celery worker | `celery -A data_engineering_copilot.workers.tasks worker --concurrency=4 -Q ingestion -E` |

## CLI Commands
| Command | Notes |
|---|---|
| `dec ask <question>` | Calls `build_rag_service()` directly — no API needed |
| `dec ingest --source "X"` | **Dispatches through FastAPI** (`POST /api/v1/ingest`) — API + Celery must be running |
| `dec reset-index` | Deletes Qdrant collection (then recreates with correct dim), clears Redis crawl keys, drops crawl frontier tables |
| `dec health` / `dec config` | Service connectivity / validate config |
| `dec evaluate` | Runs RAG eval on `tests/evaluation/eval_dataset.jsonl` |
| `dec profile --sources "X" --load-sweep "10,20,50,100"` | Ingestion profiling |
| `dec status [task-id]` | Poll ingestion progress from Redis |

## API Routes (`routes.py`)
- `POST /api/v1/ingest` — dispatches Celery task. Uses atomic SETNX lock to prevent concurrent runs.
- `GET /api/v1/ingest/status/{task_id}` — polls Redis for progress.
- `POST /api/v1/ask` — singleton `get_rag_service()`. 120s timeout.
- `POST /api/v1/ask/stream` — SSE streaming variant.
- Middleware: `RateLimitMiddleware` (60/min ask, 10/min ingest), optional `ApiKeyAuthMiddleware`.
- **Always use `get_rag_service()` (singleton) in API code** — not `build_rag_service()`.

## Testing
| Command | What |
|---|---|
| `make test-quick` | Unit, no `@slow`, parallel (~15s) |
| `make test-unit` | All unit tests |
| `make test-unit-serial` | Sequential (`-n 0`) |
| `make test-smoke` | Unit, no slow, no header, quiet |
| `make test-integration` | Sequential + 2 reruns; needs Docker |
| `make test-integration-parallel` | `-n 2 --dist=loadgroup` |
| `make test-e2e` | Full pipeline; needs Docker |
| `make test-eval` | Mocked embedder, no infra |
| `make test-ci-unit` | With coverage (XML + term-missing) |
| `make lint` / `make format` | Ruff only |

- **Pytest quirks** (`pyproject.toml`): `asyncio_mode = auto` — never `@pytest.mark.asyncio`. Default `addopts = "-n auto --dist worksteal --strict-markers"`. Coverage omits `ui/`.
- **Shared fixtures** (`tests/conftest.py`): `integration_settings`, `embeddings_provider`, `qdrant_store`, `ollama_client`, `populated_store`, `rag_service`, `api_client`. Auto-skips integration tests when Docker services unreachable (checks at collection time via `pytest_collection_modifyitems`).
- **Test isolation**: `unique_collection_name()` creates per-test Qdrant collections; teardown deletes them.

## Docker Services
`redis`, `qdrant`, `ollama`, `minio`, `clickhouse`, `langfuse` (incl. postgres + worker), `postgres` (app crawl frontier), `backend-api`, `celery_worker`
- Commands: `make docker-up/down/status/rebuild/logs/health/setup/cleanup/stop-all`
- `make docker-setup` = `docker-up` + pulls `nomic-embed-text` + `llama3.2:3b` into Ollama.
- CI stack: `make docker-ci-up` (uses `docker-compose.ci.yml`, prefix `dec_ci_*`).
- Worker volume mount `.:/app` — code changes need worker restart, not rebuild.
- `backend-api` and `celery_worker` share the same Docker image (`de_copilot_base_image`).

## Architecture & Constraints
- **No LangChain/LlamaIndex** (except `langchain-text-splitters`).
- **Factory DI**: `build_rag_service()`, `build_async_ingestion_service()`, etc. in `factory.py`. Never instantiate manually.
- **Async Only**: `SafeAsyncClientMixin` in `infrastructure/async_client.py`. Uses `httpx.AsyncClient` / `aiohttp`.
- **Providers**: LLM → ollama, openrouter, nvidia. Embeddings → ollama, openrouter, nvidia, openai. Switching providers requires `dec reset-index` (dimensions change).
- **Chunking** (`settings.chunking_strategy`): `"sentence_preserving"` (default, 1875-char chunks), `"semantic"`, `"header_aware"`, `"fixed_size"`.
- **Hybrid Search**: Enabled by default (dense + sparse). Configured via `hybrid_search_enabled`, `hybrid_rrf_k=60`.
- **RAG Pipeline**: Query rewriting → vector retrieval → cross-encoder reranking → context assembly → LLM → groundedness verification. Two-tier query cache (exact + semantic) with NumPy SIMD scoring.
- **Crawl Database**: PostgreSQL via `PostgresCrawlFrontierDB` (set `CRAWL_DB_URL`).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`. Downloaded at runtime (~450MB). Singleton-cached in `services/reranker.py`.
- **Output parsing**: `parse_rag_response()` + `verify_citations()` in `services/structured_output.py`.

## Key Dependencies (heavy)
- `crawl4ai` → requires Playwright/Chromium (installed in Docker, not in dev venv).
- `sentence-transformers` → downloads cross-encoder model on first rerank.
- `qdrant-client`, `redis`, `celery`, `langfuse`, `structlog`.

## Operational Gotchas
- **Qdrant Health**: Use `GET /` (port 6333). `/health` returns 404.
- **Ollama `raw: True`**: Strips `<think>` tags from responses. Empty response = output budget exhausted (increase `ollama_num_predict` in settings).
- **OpenRouter Rate Limiter**: Shared singleton coordinates 20 RPM / 1000 RPD across LLM + embeddings. Retries 429s up to 5x with exp backoff + `Retry-After`.
- **Worker Time Limits**: Soft 36000s (10h), Hard 43200s (12h). Zombie recovery via `task_failure` signal handler marks FAILED in Redis.
- **Qdrant Batch Splitting**: `upsert_chunks` splits into 256-chunk sub-batches (avoids 32MB payload limit).
- **URL Dedup**: SHA-256 via `AsyncUrlRegistry` in Redis.
- **Ingestion Lock**: Atomic SETNX on `ingestion:dispatch_lock` (60s TTL). Released before flush to prevent unbounded chunk accumulation.
- **Streamlit**: UI uses SSE to poll `/api/v1/ingest/status/{task_id}`.
- **reset-index behavior**: Deletes Qdrant collection, recreates it with correct dim (provider-dependent), deletes all Redis `crawl:*` keys, and drops crawl frontier tables (PostgreSQL).
- **respx + Ollama embedding**: `respx` passthrough (`assert_all_mocked=False`) returns empty bodies for Ollama `/api/embed` calls. If you need to mock the LLM while using real Ollama for embeddings, implement `LLMClientProtocol` with a simple class instead of wire-mocking with `respx`.

## Plan Mode Discipline
- **No edits in plan mode**: When the system says "Plan mode ACTIVE — READ-ONLY", do not modify files. Only present the plan. Wait for explicit transition to build mode.
