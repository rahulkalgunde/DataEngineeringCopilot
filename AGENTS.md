# DataEngineeringCopilot — Agent Guide

# OpenCode Development Rules - Strict TDD Policy

You must strictly adhere to Test-Driven Development (TDD) principles for all feature requests, refactors, and bug fixes. Do not write implementation code before tests exist.

## The TDD Protocol
1. **Analyze:** Read existing files to map out architecture (use Plan Mode).
2. **Write Tests First:** Write comprehensive unit/integration tests covering the happy path and edge cases (nulls, empty inputs, type mismatches).
3. **Verify Failure:** Run the test suite using the `bash` tool to ensure the new tests fail. 
4. **Implement:** Write the minimum production code required to make the tests pass (use Build Mode).
5. **Execute & Iterate:** Run the test suite command immediately after editing files. If a test fails, analyze the output and fix the implementation. Repeat until tests pass.

## Project Execution Commands
- Python/PySpark Testing: `pytest`
- Auto-run flag: After writing or updating code, you are permitted to run the test suite autonomously without asking for confirmation.

## Package Management
- **NEVER** use `pip` or `python -m venv`. This project uses **`uv`** exclusively.
- Install deps: `uv pip install -e ".[dev]"`
- Target venv: `dec_venv/bin/python`

## Data Safety Rules
- **NEVER** delete any data — Docker volumes, databases, indexed content, user files, or any persistent state. This is an absolute rule with no exceptions.
- **NEVER** run `docker volume rm`, `docker volume prune`, `docker system prune`, `docker compose down -v`, or any command that removes volumes or persistent data.
- **NEVER** run `rm -rf` on any directory without explicit user approval first.
- If a container is crash-looping or a service fails, **always ask the user** before taking any action that could affect data. Never assume deleting and recreating is acceptable.
- When debugging issues, prefer checking logs (`docker logs <container>`), inspecting state, or updating image versions — never destructive remediation.
- **When in doubt, ask.** Data loss is irreversible; restarting a container is not.

## Makefile
```bash
make install          # install all deps
make test             # run all tests
make test-unit        # unit tests only
make test-integration # integration tests only
make lint             # ruff check (requires ruff)
make format           # ruff format (requires ruff)
make clean            # remove __pycache__, .pytest_cache, .pyc files
make docker-up        # docker compose up -d
make docker-down      # docker compose down
```

## Virtual Environment
- The active venv is `dec_venv/` at the project root.
- `.vscode/settings.json` sets the interpreter to `dec_venv/bin/python`.
- **Do NOT reference** `/home/rahul/PythonVenvs/data_eng_copilot_env` — it does not exist.

## Running Tests
```bash
# Unit tests only (no external services needed)
dec_venv/bin/python -m pytest tests/unit/ -v

# Single test file
dec_venv/bin/python -m pytest tests/unit/test_chunker_improved.py -v

# Integration tests (require running Qdrant + Ollama + Langfuse)
dec_venv/bin/python -m pytest tests/integration/ -v

# Integration tests by service
dec_venv/bin/python -m pytest tests/integration/ -v -m qdrant       # Qdrant only
dec_venv/bin/python -m pytest tests/integration/ -v -m ollama      # Ollama only
dec_venv/bin/python -m pytest tests/integration/ -v -m langfuse    # Langfuse only
dec_venv/bin/python -m pytest tests/integration/ -v -m rag         # Full RAG pipeline
dec_venv/bin/python -m pytest tests/integration/ -v -m ingestion   # Full ingestion pipeline
dec_venv/bin/python -m pytest tests/integration/ -v -m api         # FastAPI endpoints

# E2E tests (full pipeline)
dec_venv/bin/python -m pytest tests/e2e/ -v

# Run everything (unit + integration)
dec_venv/bin/python -m pytest tests/ -v
```
- `pyproject.toml` `[tool.pytest.ini_options]` defines markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`.
- Integration tests auto-skip when required services are unreachable.
- `pyproject.toml` `[tool.coverage.run]` omits `data_engineering_copilot/ui/*` from coverage.
- `pyproject.toml` `[tool.ruff]` configures ruff linting (E, F, W, I, UP, B, SIM rules) and formatting.

## Test Structure
```
tests/
├── conftest.py                 # shared fixtures, auto-skip hooks
├── unit/                       # ~170 tests, no external services
├── integration/                # 6 test files, require Qdrant/Ollama/Langfuse
└── e2e/                        # 1 test file, full pipeline
```

## Running the App
```bash
# CLI
dec_venv/bin/python main.py ingest --max-pages 40
dec_venv/bin/python main.py ask "your question"
dec_venv/bin/python main.py reset-index

# Streamlit UI
dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py

# FastAPI (for async ingestion via Celery)
dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000
```

## Infrastructure Dependencies (Docker Compose)
- `docker compose up -d` brings up: **Redis**, **Qdrant** (ports 6333/6334), **Langfuse** (port 3000), **PostgreSQL**, **ClickHouse**, **MinIO**, plus the **backend-api** and **celery_worker**.
- **Ollama** runs outside Docker — must be started separately and have models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

## Architecture (High-Level)
- **Pattern**: layered architecture (config → domain → infrastructure → services → CLI/UI/API)
- **No LangChain/LlamaIndex** — direct crawler/parser/chunker/embed/vector/query/generate pipeline
- **Entry points**: `main.py` (CLI), `data_engineering_copilot/ui/streamlit_app.py` (Streamlit), `data_engineering_copilot/api/app.py` (FastAPI)

### Key Components
- **Crawler** (`infrastructure/crawler.py`): BFS HTML crawler with domain/prefix allowlisting. Preserves trailing slashes for correct relative-link resolution.
- **HTML Parser** (`infrastructure/html_parser.py`): BeautifulSoup-based. Removes nav/footer/header/aside. Skips pages < 40 words.
- **Embeddings** (`infrastructure/embeddings.py`): Ollama `/api/embed` endpoint only. Class is `OllamaEmbeddings`.
- **Vector Store** (`infrastructure/qdrant_store.py`): Primary storage backend. Uses `qdrant-client` HTTP API. `hybrid_query` embeds + queries in one call.
- **Ollama Client** (`infrastructure/ollama_client.py`): HTTP POST to `/api/generate`. Uses `raw=True` to avoid Qwen thinking-only empty responses.
- **Chunker** (`services/chunker.py`): Supports `fixed_size`, `sentence_preserving`, and `semantic` strategies. Deterministic chunk IDs: `<slug(source)>:<sha1(url)[:10]>:<0000-index>`.
- **Semantic Chunker** (`services/semantic_chunker.py`): Embedding-based clustering with cosine similarity.
- **Context Assembler** (`services/context_assembler.py`): Deduplicates chunks (>70% overlap threshold) and truncates to `max_context_chars`.
- **Reranker** (`services/reranker.py`): Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) when `reranker_enabled=True`.
- **RAG Service** (`services/rag.py`): `ProductionRagService` with Langfuse tracing. Embed → retrieve → rerank → assemble context → Ollama generate.
- **Ingestion Service** (`services/ingestion.py`): Orchestrates crawl → parse → chunk → embed → Qdrant upsert.
- **Workers** (`workers/tasks.py`): Celery tasks with Redis broker. Uses `crawl4ai.AsyncWebCrawler` for async ingestion.
- **Factory** (`factory.py`): Wires everything together. `build_chunker()` selects strategy; `build_ingestion_service()` and `build_rag_service()` construct full stacks.

### Data Flow
```
CLI/UI → IngestionService → Crawler → Parser → Chunker → Embeddings → QdrantVectorStore
CLI/UI → RagAnswerService → Embeddings → QdrantVectorStore → Reranker → ContextAssembler → OllamaClient
```

## Settings (Actual Defaults in `settings.py`)
These are the **real** defaults — the ARCHITECTURE.md and older docs may be stale:

| Setting | Default | Notes |
|---------|---------|-------|
| `ollama_model` | `llama3.2:3b` | NOT qwen3:4b or qwen3.5:9b |
| `embedding_model_name` | `nomic-embed-text` | Ollama HTTP API |
| `embedding_dimension` | `768` | |
| `chunk_size_words` | `375` | |
| `chunk_overlap_words` | `90` | |
| `chunking_strategy` | `sentence_preserving` | Options: `fixed_size`, `sentence_preserving`, `semantic` |
| `min_semantic_similarity` | `0.5` | Semantic chunker only |
| `max_chunk_words` | `None` (auto: 1.5x chunk_size_words) | Semantic chunker only |
| `retrieval_top_k` | `15` | |
| `reranker_enabled` | `True` | |
| `reranker_top_k` | `5` | |
| `max_context_chars` | `4000` | |
| `confidence_threshold` | `0.18` | |
| `qdrant_url` | `http://localhost:6333` | |
| `crawl_delay_seconds` | `0.2` | |
| `max_pages_per_source` | `80` | |
| `ingestion_batch_chunk_size` | `256` | |
| `ollama_num_ctx` | `4096` | |
| `ollama_num_predict` | `512` | |
| `ollama_retry_context_ratio` | `0.5` | Retry with reduced context on failure |
| `ollama_retry_extra_num_predict` | `512` | Extra output budget on retry |
| `ollama_retry_max_num_predict` | `1024` | Max output budget on retry |

## Documentation Sources
Configured in `data_engineering_copilot/config/documentation_sources.json`:
- Apache Spark → `https://spark.apache.org/docs/latest/`
- Apache Airflow → `https://airflow.apache.org/docs/apache-airflow/stable/`
- Databricks → `https://docs.databricks.com/aws/en/`
- Delta Lake → `https://docs.delta.io/latest/`

Source selection uses **exact name match** (e.g., `--source "Apache Spark Documentation"`).

## Crawler Gotchas
- Preserves trailing slash on directory URLs — required for Spark relative links (e.g., `quick-start.html`).
- Dedupes by normalizing `/index.html` and trailing-slash variants to the same key.
- Skips `mailto:`, `tel:`, `javascript:` links.
- Only downloads responses with `Content-Type: text/html`.
- `pages_fetched` in crawler events includes failed downloads; service counts only yielded documents.

## CLI Commands
```
python main.py ingest                          # ingest all sources
python main.py ingest --source "Apache Spark Documentation"  # single source
python main.py ingest --source X --source Y    # multiple sources
python main.py ingest --max-pages 40           # cap pages per source
python main.py ask "question"                  # RAG Q&A
python main.py reset-index                     # delete Qdrant collection
python main.py ui                              # prints Streamlit launch command
```

## Langfuse Observability
- `.env` contains Langfuse keys (local dev instance at `localhost:3000`).
- `ProductionRagService` traces retrieval, generation, and confidence via Langfuse v3 API.
- Graceful fallback if Langfuse is unavailable — RAG still works without tracing.

## Known Pitfalls
- **Stale chunks**: No deletion/pruning for removed docs; upsert overwrites by chunk ID but stale chunks from changed pages may linger.
- **Ollama raw prompt**: Must use `raw=True` in OllamaClient to avoid Qwen models returning thinking-only empty responses.
- **No canonical URL normalization** beyond fragments/slash dedupe — query-string variants may duplicate pages.
- **UI refresh is synchronous** — long crawls block the Streamlit session.

## Project Files
- `AGENTS.md` — this file
- `ARCHITECTURE.md` — architecture docs (partially stale; trust source code over it)
- `PROJECT_CONTEXT.md` — older context doc (partially stale; ChromaDB refs fixed, but some details may be outdated)
- `pyproject.toml` — project metadata, deps, pytest/coverage/ruff config
- `data_engineering_copilot/config/logging.py` — root-level logging setup with file rolling (10MB, 5 backups)
- `docker-compose.yml` — full infra stack (Redis, Qdrant, Langfuse, PG, ClickHouse, MinIO) with health checks
- `Dockerfile` — multi-stage Python 3.12-slim with uv, default CMD: `python main.py --help`
- `.dockerignore` — excludes tests, logs, .git, venvs from Docker context
- `.pre-commit-config.yaml` — ruff linting + formatting, trailing whitespace, end-of-file fixer
- `.env` — Langfuse credentials for local dev
- `.streamlit/config.toml` — disables file watcher and usage stats
