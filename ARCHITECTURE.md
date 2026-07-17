# DataEngineeringCopilot Architecture Documentation

## 1. Overview
- Offline Retrieval-Augmented Generation (RAG) assistant for data-engineering documentation.
- Sources are crawled, parsed, chunked, embedded, and stored in **QdrantDB**.
- Answers are generated locally via **Ollama** (llama3.2:3b model) without LangChain or LlamaIndex.
- Direct crawler/parser/chunker/embed/vector/query/generate pipeline.

## 2. High‑Level Architecture
```
+-------------------+   +-------------------+   +-------------------+
| CLI / UI          |   | Services Layer    |   | Infrastructure   |
| (main.py, Stream- |
| lit)              |   | (Ingestion, RAG,  |   | (Crawler, Parser, |
|                   |   |  FastAPI)         |   |  Embeddings,      |
+-------------------+   +-------------------+   |  Vector Store)    |
        |                        |                +-------------------+
        v                        v                         |
+-------------------+   +-------------------+               |
| FastAPI Gateway   |   | ProductionRagServ |               |
| (uvicorn)         |   |  (Langfuse)       |               |
+-------------------+   +-------------------+               |
        |                        |                       |
        v                        v                       v
+---------------------------------------------------------------+
| Docker‑Compose Stack: Redis (broker) | Qdrant (vector DB) | Langfuse |
+---------------------------------------------------------------+
```

## 3. Component Details

### 3.1 CLI (`main.py`)
- Commands: `ingest`, `ask`, `reset-index`, `ui`, `api` (starts FastAPI).
- Dispatches to factory to obtain services.

### 3.2 Factory (`data_engineering_copilot/factory.py`)
- `build_ingestion_service()` → `IngestionService` (Celery task wrapper).
- `build_rag_service()` → `ProductionRagService` (Langfuse tracing, Ollama HTTP).
- `build_vector_store()` → `QdrantVectorStore`.

### 3.3 Services
- **IngestionService** – coordinates async Celery task (`workers/tasks.py`) that runs:
  - `crawl4ai.AsyncWebCrawler` → raw HTML
  - `DocumentationHtmlParser` → clean text
  - `DocumentChunker` → overlapping word chunks
  - `SentenceTransformerEmbeddings` (Ollama `/api/embed`) → dense vectors
  - `QdrantVectorStore.upsert_chunks` → persistent storage
- **ProductionRagService** – embeds query, performs hybrid search (`QdrantVectorStore.hybrid_query`), applies confidence gating, builds prompt, calls Ollama via direct HTTP, sends tracing data to Langfuse.

### 3.4 Infrastructure
- **DocumentationCrawler** – BFS crawler
- **DocumentationHtmlParser** – 
- **DocumentChunker** – 
- **SentenceTransformerEmbeddings** – Ollama-only embedding provider using `/api/embed` endpoint.
- **QdrantVectorStore** (`data_engineering_copilot/infrastructure/qdrant_store.py`)
  - Adds `hybrid_query` for dense + BM25 sparse retrieval.
- **OllamaClient** – called directly by `ProductionRagService`.
- **LangfuseTracer** – wrapper used by `ProductionRagService` to log prompts, responses, and latency.

### 3.5 Async Workers (`workers/tasks.py`)
- Celery app configured to use Redis as broker.
- `ingest_task(source_names, max_pages)` runs the full ingestion pipeline asynchronously.

### 3.6 FastAPI Gateway (`api/routes.py`)
- Endpoints:
  - `POST /api/v1/ingest` – triggers Celery ingestion task.
  - `GET /api/v1/status` – health check for Redis, Qdrant, and Langfuse.
  - `GET /api/v1/metrics` – simple stats (chunk count, vector store size).

### 3.7 UI (`ui/streamlit_app.py`)
- Refactored to call FastAPI `/api/v1/ingest` instead of invoking services directly.
- Still provides sidebar source selection, max‑page input, and answer panel.

## 4. Configuration (`data_engineering_copilot/config/settings.py`)
| Setting | Default | Description |
|---------|----------|-------------|
| `embedding_model_name` | `nomic-embed-text:latest` | Embedding model |
| `ollama_model` | `llama3.2:3b` | LLM model |
| `chunk_size_words` | `420` | Chunk size |
| `chunk_overlap_words` | `80` | Overlap |
| `retrieval_top_k` | `2` | Top‑k dense retrieval |
| `confidence_threshold` | `0.35` | Minimum confidence |
| `qdrant_url` | `http://localhost:6333` | Qdrant service endpoint |
| `redis_url` | `redis://localhost:6379/0` | Celery broker |
| `langfuse_url` | `http://localhost:3000` | Langfuse UI |
| `langfuse_public_key` | `"<public_key>"` | Langfuse auth |
| `langfuse_secret_key` | `"<secret_key>"` | Langfuse auth |
| `crawl_delay_seconds` | `0.25` | Delay between page fetches |
| `max_pages_per_source` | `80` | Crawl limit |

## 5. Data Persistence
- **Qdrant** stores vector embeddings and metadata (`qdrant_db/`).
- Embedding model cache under `data/embedding_models/`.

## 6. RAG Pipeline (updated)
1. Query embedding via `SentenceTransformerEmbeddings.embed_query`.
2. Hybrid search in Qdrant (`dense + BM25`) → list of `RetrievedChunk`s.
3. Confidence gating (`1 - cosine_distance` >= `confidence_threshold`).
4. Prompt construction (max `max_context_chars`).
5. Ollama HTTP generate.
6. Langfuse tracing of request/response.

## 7. Deployment & Operations
- **Docker‑Compose** (`docker-compose.yml`) brings up Redis, Qdrant, Langfuse.
- Start services: `docker compose up -d`.
- Ensure Ollama is running with `nomic-embed-text` and `llama3.2:3b` model: `ollama pull nomic-embed-text && ollama pull llama3.2:3b`.
- Run Celery worker: `celery -A data_engineering_copilot.workers.tasks worker --loglevel=info`.
- Launch FastAPI: `uvicorn api.routes:app --reload`.
- Launch Streamlit UI: `streamlit run data_engineering_copilot/ui/streamlit_app.py`.

## 8. Testing Strategy
- Existing unit tests remain valid (crawler, parser, chunker, settings).
- Add new tests for:
  - `QdrantVectorStore` upsert & hybrid_query.
  - Celery ingestion task behavior.
  - FastAPI endpoints (status, ingest).
  - Langfuse tracing integration.

## 9. Remaining Gaps (post‑update)
- Implement deterministic metadata hashing & stale‑chunk pruning in Qdrant.
- Add BM25 index build step in Qdrant store.
- Provide Makefile targets for `infra-up`, `infra-down`.
- Extend documentation (README, ARCHITECTURE) to fully describe new stack.

## 10. References
- `AGENTS.md` – project description.
- `docker-compose.yml` – infra definition.

---
*Document generated for AI‑agent consumption. Structure uses explicit headings, tables, and ASCII diagrams to facilitate parsing and downstream processing.*