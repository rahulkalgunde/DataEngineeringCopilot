# DataEngineeringCopilot

Offline question answering for data engineering documentation using Ollama, llama3.2:3b, Qdrant, and Streamlit.

## Project Structure

```text
DataEngineeringCopilot/
  main.py
  README.md
  Makefile                    # Docker, test, lint commands
  AGENTS.md                   # Agent guide & architecture
  qdrant_db/ -> qdrant_db/
  data/
  logs/                       # Runtime logs
  data_engineering_copilot/
    config/
      documentation_sources.json
      settings.py
    domain/
      models.py
    infrastructure/
      rate_limiter.py          # OpenRouter RPM/RPD coordination
      async_openrouter_client.py
      async_openrouter_embeddings.py
      qdrant_store.py
      redis_store.py
      crawl_cache.py
      html_parser.py
      ollama_client.py
      embeddings.py
    services/
      chunker.py
      ingestion.py
      rag.py
    workers/
      celery_app.py            # Celery config & signal handlers
      tasks.py                 # Ingestion tasks with zombie recovery
    api/
      app.py                   # FastAPI backend
    ui/
      streamlit_app.py
    cli.py                     # CLI dispatcher
    utils/
      text.py
    profiler/
      cli.py                   # Profiler CLI
      report_generator.py      # Profiler report generation
  scripts/
    download_embedding_model.py
```

## Setup

# Package Management Constraints
- NEVER use standard 'pip' or 'python -m venv' commands.
- This project exclusively uses 'uv' as its Python package and environment manager.
- To create or manage virtual environments, use: `uv venv dec_venv`
- To install packages, use: `uv pip install -e ".[dev]"`
- To add a single package to the environment, use: `uv pip install <package_name>`
- Always ensure you target the correct local virtual environment binary path: `dec_venv/bin/python`

On windows machine, Install and start Ollama, then run the models:

```bash
ollama serve
ollama pull nomic-embed-text:latest
ollama pull qwen3.5:9b
```

Docker

1. Start Docker Desktop on windows machine
2. Login to wsl and go to Project Directory
3. Activate python venv `source dec_venv/bin/activate`
3. Run: `docker compose up -d`

Always Use Python virtual environment located at `dec_venv/` at the project root.

Linux/macOS:

```bash
uv venv dec_venv
source dec_venv/bin/activate
uv pip install -e ".[dev]"
```

No additional embedding model download is required. The system uses Ollama's `nomic-embed-text` model via HTTP API.

## Configuration

Settings load from three `.env` files in order (later files override earlier):

1. `.env` — defaults (committed)
2. `.env.secrets` — sensitive keys (gitignored)
3. `.env.local` — personal overrides (gitignored)

### Models
- **LLM**: OpenRouter free tier (`openrouter/free`)
- **Embeddings**: OpenRouter `nvidia/nemotron-3-embed-1b:free` (dimension=2048)
- **Local Ollama**: `nomic-embed-text` + `llama3.2:3b` (alternative)

## Docker Commands

```bash
make docker-up          # Start all services
make docker-down        # Stop all services
make docker-status      # Show containers, status, health checks
make docker-rebuild     # Full rebuild with --no-cache
make docker-logs        # Stream logs (tail=100)
make docker-logs-worker # Stream worker-specific logs
make docker-health      # Verify service connectivity (dec health)
make docker-stop-all    # Stop all services (preserve state)
make docker-cleanup     # Prune unused images/containers/volumes
make docker-setup       # Start + pull Ollama models
```

### Services
- redis, qdrant, minio, ollama, clickhouse, langfuse
- backend-api (FastAPI), celery_worker
- langfuse-postgres, langfuse-worker

## CLI Commands

```bash
# Ingestion
dec ingest --source "Apache Spark Documentation" --max-pages 1000
dec ingest --source "Apache Airflow Documentation"

# Query
dec ask "How does Delta Lake time travel work?"

# Status
dec status                    # Show ingestion job status
dec status <task-id>          # Show specific task
dec health                    # Verify all service connections
dec config                    # Show current configuration

# Reset
dec reset-index              # Clear Qdrant + Redis

# Profiling
dec profile --sources "Apache Spark" --max-pages 20 --load-sweep "100,500,1000,5000"
```

## Ingestion

The crawler downloads documentation pages and stores chunks in Qdrant. After ingestion, question answering is fully local: Qdrant reads from disk, Ollama runs `nomic-embed-text` and `llama3.2:3b` locally.

### When to Reset and Re-Ingest

Run `dec reset-index` before re-ingesting in these scenarios:

- **Switching embedding provider** (e.g. Ollama → OpenRouter) — different providers produce different vector dimensions
- **Changing embedding model** — models may use incompatible vector spaces
- **Changing chunking strategy** — new chunks have different boundaries than old ones
- **Corrupted or incomplete index** — if ingestion partially failed or Qdrant reports errors
- **Major documentation restructure** — if source URLs changed significantly

```bash
dec reset-index
dec ingest --max-pages 40
```

**Do NOT reset** for incremental ingestion — the system deduplicates via content hash and only updates changed pages.

### Switching Embedding Providers

Set these in `.env`:

```bash
# Switch from Ollama to OpenRouter
LLM_PROVIDER=openrouter
EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_EMBEDDING_DIMENSION=2048
```

Then reset and re-ingest:
```bash
dec reset-index
dec ingest --max-pages 20
```

The configured documentation sources are:

- Apache Spark Documentation
- Apache Airflow Documentation
- Databricks Documentation
- Delta Lake Documentation

Edit documentation source URLs in:

```text
data_engineering_copilot/config/documentation_sources.json
```

Each chunk stores:

- source name
- title
- original URL
- chunk id
- chunk text

## Rate Limiting

- **OpenRouter Rate Limiter**: Shared `OpenRouterRateLimiter` coordinates RPM (20 req/min) and RPD (1000 req/day) between embeddings and LLM clients.
- **429 Handling**: Both clients parse `Retry-After` header and retry up to 5 times with exponential backoff.

## Ingestion & Workers

- **Worker Time Limits**: Soft limit=36000s (10h), Hard limit=43200s (12h)
- **Zombie Task Recovery**: Celery `task_failure` signal handler catches hard time limit kills and marks Redis status as FAILED
- **Qdrant Batch Splitting**: `upsert_chunks` splits into 256-chunk sub-batches to prevent 32MB payload limit errors
- **Ingestion Lock**: Released before flush to prevent unbounded chunk accumulation

## Architecture

This project intentionally does not use LangChain or LlamaIndex.

- `config`: source URLs and runtime settings
- `domain`: simple dataclasses shared by the app
- `infrastructure`: adapters for HTTP crawling, HTML parsing, embeddings, Qdrant, Ollama, and rate limiting
- `services`: business workflows for ingestion and RAG answering
- `workers`: Celery tasks with zombie recovery
- `api`: FastAPI backend with ingestion endpoints
- `cli`: Command-line interface dispatcher
- `profiler`: Performance profiling and reporting
- `ui`: Streamlit interface

Local generation can take time on CPU. The timeout and generation limits are configured in `data_engineering_copilot/config/settings.py` as `ollama_timeout_seconds`, `ollama_num_ctx`, `ollama_num_predict`, `retrieval_top_k`, and `max_context_chars`.

If Ollama fails due to prompt or output length, the service automatically retries with reduced repository context and then with a larger output budget. You can tune this behavior with `ollama_retry_context_ratio`, `ollama_retry_extra_num_predict`, and `ollama_retry_max_num_predict` in the same settings file.

Default retry settings in `data_engineering_copilot/config/settings.py`:

```python
ollama_retry_context_ratio = 0.5
ollama_retry_extra_num_predict = 512
ollama_retry_max_num_predict = 1024
```

Runtime logs are written under `logs/` in the project workspace:

- `logs/app.log` captures CLI, Streamlit, ingestion, retrieval, vector store, and Ollama events for troubleshooting.
- `logs/ingestion_refresh.log` captures detailed UI refresh events and fetched documentation URLs.
