# DataEngineeringCopilot — Agent Guide

## Package Management
- **NEVER** use `pip` or `python -m venv`. This project uses **`uv`** exclusively.
- Install deps: `uv pip install -r requirements.txt`
- Target venv: `${workspaceFolder}/dec_venv/bin/python`

## Virtual Environment
- The active venv is `dec_venv/` at the project root.
- `.vscode/settings.json` sets the interpreter to `${workspaceFolder}/dec_venv/bin/python`.
- **Do NOT reference** `/home/rahul/PythonVenvs/data_eng_copilot_env` — it does not exist.

## Running Tests
```bash
# Unit tests only (no external services needed)
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m "not integration"

# Single test file
${workspaceFolder}/dec_venv/bin/python -m pytest tests/test_chunker.py -v

# Integration tests (require running Qdrant + Ollama + Langfuse)
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m integration

# Integration tests by service
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m qdrant       # Qdrant only
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m ollama      # Ollama only
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m langfuse    # Langfuse only
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m rag         # Full RAG pipeline
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m ingestion   # Full ingestion pipeline
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m api         # FastAPI endpoints
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v -m "slow"      # Long-running tests

# Run everything (unit + integration)
${workspaceFolder}/dec_venv/bin/python -m pytest tests/ -v
```
- `pytest.ini` markers: `integration`, `slow`, `qdrant`, `ollama`, `langfuse`, `rag`, `ingestion`, `api`.
- Integration tests auto-skip when required services are unreachable.
- `.coveragerc` omits `data_engineering_copilot/ui/*` from coverage.
- No linting/typechecking tools are configured (no ruff, mypy, flake8).

## Running the App
```bash
# CLI
${workspaceFolder}/dec_venv/bin/python main.py ingest --max-pages 40
${workspaceFolder}/dec_venv/bin/python main.py ask "your question"
${workspaceFolder}/dec_venv/bin/python main.py reset-index

# Streamlit UI
${workspaceFolder}/dec_venv/bin/python -m streamlit run data_engineering_copilot/ui/streamlit_app.py

# FastAPI (for async ingestion via Celery)
${workspaceFolder}/dec_venv/bin/python -m uvicorn data_engineering_copilot.api.app:app --reload --port 8000
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
- **Embeddings** (`infrastructure/embeddings.py`): Ollama `/api/embed` endpoint only. Class is `SentenceTransformerEmbeddings` (historical name).
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
| `retrieval_top_k` | `15` | |
| `reranker_enabled` | `True` | |
| `reranker_top_k` | `5` | |
| `max_context_chars` | `4000` | |
| `confidence_threshold` | `0.18` | |
| `qdrant_url` | `http://localhost:6333` | |
| `crawl_delay_seconds` | `0.2` | |
| `max_pages_per_source` | `80` | |
| `ollama_num_ctx` | `4096` | |
| `ollama_num_predict` | `512` | |

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
python main.py reset-index                     # delete qdrant_db/
python main.py export-index --output out.zip  # zip qdrant_db/
python main.py import-index archive.zip        # restore qdrant_db/
python main.py ui                              # prints Streamlit launch command
```

## Langfuse Observability
- `.env` contains Langfuse keys (local dev instance at `localhost:3000`).
- `ProductionRagService` traces retrieval, generation, and confidence via Langfuse v3 API.
- Graceful fallback if Langfuse is unavailable — RAG still works without tracing.

## Known Pitfalls
- **Stale chunks**: No deletion/pruning for removed docs; upsert overwrites by chunk ID but stale chunks from changed pages may linger.
- **No retry/backoff** for transient HTTP errors in crawler or Ollama.
- **Ollama raw prompt**: Must use `raw=True` in OllamaClient to avoid Qwen models returning thinking-only empty responses.
- **No canonical URL normalization** beyond fragments/slash dedupe — query-string variants may duplicate pages.
- **UI refresh is synchronous** — long crawls block the Streamlit session.

## Project Files
- `AGENTS.md` — this file
- `ARCHITECTURE.md` — architecture docs (partially stale; trust source code over it)
- `PROJECT_CONTEXT.md` — older context doc (stale; references ChromaDB, wrong models)
- `logger_config.py` — root-level logging setup with file rolling (10MB, 5 backups)
- `docker-compose.yml` — full infra stack (Redis, Qdrant, Langfuse, PG, ClickHouse, MinIO)
- `Dockerfile` — Python 3.12-slim, Playwright Chromium, default CMD: `python main.py --help`
- `.env` — Langfuse credentials for local dev
- `.streamlit/config.toml` — disables file watcher and usage stats
