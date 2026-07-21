# DataEngineeringCopilot — Agent Guide

# Strict TDD Policy

Write tests first for all features, refactors, and bug fixes. Never write implementation before tests exist.

## The TDD Protocol
1. **Analyze:** Read existing files to map out architecture.
2. **Write Tests First:** Cover happy path and edge cases (nulls, empty inputs, type mismatches).
3. **Verify Failure:** Run the test suite to ensure new tests fail.
4. **Implement:** Write minimum production code to make tests pass.
5. **Execute & Iterate:** Run tests immediately after editing. Fix failures, repeat.

## Session & Plan Guardrails

All implementation plans and session records are persisted to disk for traceability.

### Session Start Protocol
- Run `git status` at start. Alert user if uncommitted changes exist.
- Check `plans/` and `sessions/` for stale plan/session files — resume if applicable.

### Plan Files
- Save plans to `plans/PLAN_<short_description>_<YYYY-MM-DD_HHmm>.md` before presenting.
- Update status to IN PROGRESS before starting. Update throughout with decisions/outcomes.

### Session Files
- Save session details to `sessions/SESSION_<short_description>_<YYYY-MM-DD_HHmm>.md` after context loss.
- Save `sessions/SESSION_SUMMARY_<YYYY-MM-DD_HHmm>.md` before closing.

### Git Safety
- Check `git status` at session start. Alert user if uncommitted changes exist.
- Before multi-file change: `git checkout -b refactor/<description>`
- After each milestone, remind user to commit and push.

## Project Execution Commands
- Test runner: `pytest` via `dec_venv/bin/python -m pytest`
- Auto-run flag: permitted to run tests autonomously after code changes — no need to ask.

## Package Management
- **NEVER** use `pip` or `python -m venv`. Use **`uv`** exclusively.
- Install deps: `uv pip install -e ".[dev]"`
- Target venv: `dec_venv/bin/python`
- `.vscode/settings.json` sets interpreter to `dec_venv/bin/python`.

## Data Safety Rules (Matches `opencode.json` permissions)
- **NEVER** delete any data — Docker volumes, databases, indexed content, user files.
- **NEVER** run: `docker volume rm`, `docker volume prune`, `docker system prune`, `docker compose down -v`, `rm -rf` on any directory without explicit approval.
- Crash-looping container? Check logs first. Never assume delete+recreate is OK.

## Makefile
```bash
make install          # uv pip install -e ".[dev]"
make test             # run all tests
make test-unit        # unit tests only
make test-integration # integration tests only
make lint             # ruff check data_engineering_copilot/ tests/
make format           # ruff format data_engineering_copilot/ tests/
make clean            # remove __pycache__, .pytest_cache, .pyc files
make docker-up        # docker compose up -d
make docker-down      # docker compose down
```

## Running Tests
```bash
# Unit tests only (no external services needed) — 425 tests
dec_venv/bin/python -m pytest tests/unit/ -v

# Single test file
dec_venv/bin/python -m pytest tests/unit/test_chunker_improved.py -v

# Integration tests (require Qdrant + Ollama + Langfuse) — 56 tests
dec_venv/bin/python -m pytest tests/integration/ -v

# By service marker
dec_venv/bin/python -m pytest tests/integration/ -v -m qdrant
dec_venv/bin/python -m pytest tests/integration/ -v -m ollama
dec_venv/bin/python -m pytest tests/integration/ -v -m langfuse
dec_venv/bin/python -m pytest tests/integration/ -v -m rag
dec_venv/bin/python -m pytest tests/integration/ -v -m ingestion
dec_venv/bin/python -m pytest tests/integration/ -v -m api

# E2E (full pipeline)
dec_venv/bin/python -m pytest tests/e2e/ -v
```
- Pytest markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`.
- Integration tests auto-skip when services are unreachable.
- Coverage omits `data_engineering_copilot/ui/*`.
- Ruff config: line-length=120, select=E,F,W,I,UP,B,SIM, ignore=E501.

## Running the App
```bash
dec_venv/bin/python main.py ingest --max-pages 40
dec_venv/bin/python main.py ask "your question"
dec_venv/bin/python main.py reset-index
dec_venv/bin/python main.py ui                        # prints Streamlit launch command
dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py
dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000
```

## Infrastructure Dependencies
- `docker compose up -d` brings up: Redis, Qdrant (6333/6334), Langfuse (3000), PostgreSQL, ClickHouse, MinIO, backend-api, celery_worker.
- **Ollama runs outside Docker** — start separately:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

## Architecture
- **Pattern**: layered — config → domain → infrastructure → services → CLI/UI/API
- **No LangChain/LlamaIndex** — direct crawler/parser/chunker/embed/vector/query/generate pipeline
- **Entry points**: `main.py` (CLI → `data_engineering_copilot.cli`), `data_engineering_copilot/ui/streamlit_app.py` (Streamlit), `data_engineering_copilot/api/app.py` (FastAPI)
- **Factory** (`factory.py`): wires everything. `build_chunker()` selects strategy; `build_ingestion_service()` and `build_rag_service()` construct full stacks.

### Key Gotchas
- **Crawler** (`infrastructure/crawler.py`): BFS crawler. Preserves trailing slashes on directory URLs (required for Spark relative links). Dedupes by normalizing `/index.html` and trailing-slash variants.
- **Embeddings** (`infrastructure/embeddings.py`): Uses Ollama `/api/embed` endpoint only. Class name: `OllamaEmbeddings`.
- **Ollama Client** (`infrastructure/ollama_client.py`): HTTP POST to `/api/generate`. Uses `raw=True` to avoid Qwen thinking-only empty responses.
- **Chunker** (`services/chunker.py`): strategies = `fixed_size`, `sentence_preserving` (default), `semantic`. Deterministic chunk IDs: `<slug(source)>:<sha1(url)[:10]>:<0000-index>`.
- **Context Assembler** (`services/context_assembler.py`): deduplicates chunks (>70% overlap), truncates to `max_context_chars`.
- **Reranker** (`services/reranker.py`): cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) when `reranker_enabled=True`.
- **Workers** (`workers/tasks.py`): Celery with Redis broker. Uses `crawl4ai.AsyncWebCrawler`.

### Data Flow
```
CLI/UI → IngestionService → Crawler → Parser → Chunker → Embeddings → QdrantVectorStore
CLI/UI → RagAnswerService → Embeddings → QdrantVectorStore → Reranker → ContextAssembler → OllamaClient
```

## Settings (Defaults in `config/settings.py`)
- `ollama_model`: `llama3.2:3b` (NOT qwen3:4b or qwen3.5:9b)
- `embedding_model_name`: `nomic-embed-text`
- `embedding_dimension`: `768`
- `chunking_strategy`: `sentence_preserving`
- `chunk_size_words`: `375` / `chunk_overlap_words`: `90`
- `retrieval_top_k`: `15` / `reranker_top_k`: `5`
- `max_context_chars`: `4000` / `confidence_threshold`: `0.18`
- `max_pages_per_source`: `80` / `crawl_delay_seconds`: `0.2`
- `ollama_num_ctx`: `4096` / `ollama_num_predict`: `512` (with retry escalation up to `1024`)

## CLI Commands
```
dec_venv/bin/python main.py ingest                          # all sources
dec_venv/bin/python main.py ingest --source "Apache Spark Documentation"
dec_venv/bin/python main.py ingest --source X --source Y
dec_venv/bin/python main.py ingest --max-pages 40
dec_venv/bin/python main.py ask "question"                  # RAG Q&A
dec_venv/bin/python main.py reset-index                     # delete Qdrant collection
dec_venv/bin/python main.py ui
```
Source selection uses **exact name match** against `data_engineering_copilot/config/documentation_sources.json`.

## Known Pitfalls
- **Stale chunks**: no deletion/pruning for removed docs. Upsert overwrites by chunk ID, but chunks from changed pages may linger.
- **Ollama raw prompt**: must use `raw=True` to avoid Qwen thinking-only empty responses.
- **No canonical URL normalization** beyond fragments/slash dedupe — query-string variants may duplicate pages.
- **UI refresh is synchronous** — long crawls block the Streamlit session.
- **Langfuse tracing** in `ProductionRagService`: graceful fallback if Langfuse unavailable — RAG still works.

## Cline/Other Agent Compatibility
- `.clinerules/` directory contains Cline-specific rules (`behavioral-constraints.md`, `guardrails.md`, `memory-bank.md`, `python-env-rules.md`). These enforce single-edit-per-turn, extreme laziness, context conservation. Not authoritative for OpenCode but worth knowing if the repo is used with Cline.
- `.cursor/rules/` and `.cursorrules` do not exist; `.github/copilot-instructions.md` does not exist.
