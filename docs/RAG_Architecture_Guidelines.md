# Enterprise Python RAG Pipeline: Production Architecture & Code Guidelines

> **Purpose**: This reference architecture document serves as the ground truth specification for code reviews, architectural conformance checks, and automated refactoring guidelines for Python-based Retrieval-Augmented Generation (RAG) applications with recursive web crawling capabilities.

---

## 1. Core System Architecture & Layering

The application **MUST** follow Clean Architecture principles with strict separation of concerns into four isolated layers. Circular dependencies across layers are strictly forbidden.

```
       +---------------------------------------------------------+
       |                Presentation Layer (API)                 |
       |            (FastAPI Routes, Pydantic Models)            |
       +---------------------------+-----------------------------+
                                   |
                                   v
       +---------------------------------------------------------+
       |               Application Layer (Use Cases)             |
       |        (Orchestration, Ingestion Engine, RAG Service)   |
       +---------------------------+-----------------------------+
                                   |
                                   v
       +---------------------------------------------------------+
       |                 Domain Layer (Core Logic)               |
       |       (Entities, Interfaces, Chunking, Prompt Templates)|
       +---------------------------+-----------------------------+
                                   |
                                   v
       +---------------------------------------------------------+
       |             Infrastructure Layer (Adapters)             |
       |      (Qdrant Client, Redis Client, Scraper, LLM APIs)   |
       +---------------------------------------------------------+
```

### Architectural Rules
1. **Dependency Inversion**: High-level modules (Application/Domain) MUST NOT depend on low-level modules (Infrastructure). All external services (Qdrant, Redis, LLMs, Scrapers) must be accessed via Abstract Base Classes (ABCs) or Protocols.
2. **Asynchronous First**: All I/O-bound operations (Web fetching, Redis access, Qdrant vector operations, LLM inference API calls) **MUST** be strictly asynchronous (`async`/`await` using `httpx`, `asyncio`, `qdrant-client[async]`, and `redis-py` async API).
3. **Immutability & Type Safety**: All data transfer objects (DTOs) and domain models **MUST** use `Pydantic v2` models or frozen Python `@dataclass` types with explicit type annotations.

---

## 2. Ingestion & Recursive Web Crawler Guidelines

### 2.1 Crawler Behavior & Politeness
- **Domain Scope Restrictions**: The recursive crawler **MUST** enforce exact domain and boundary checks (e.g., `allow_domains`, `max_depth`, `regex_include_patterns`, `regex_exclude_patterns`) before initiating any HTTP request.
- **Robots.txt Compliance**: The crawler **MUST** parse and respect `robots.txt` rules and implement configurable politeness delays (minimum 0.5s per domain).
- **Rate Limiting & Concurrency**:
  - Limit concurrent HTTP requests per target domain using `asyncio.Semaphore`.
  - Enforce timeout limits on HTTP requests (Connect timeout: 5.0s, Read timeout: 15.0s).
  - Use `httpx.AsyncClient` with connection pooling and automated exponential backoff retries (`tenacity` library).

### 2.2 Deduplication & State Tracking
- **URL Normalization**: Canonicalize all discovered URLs (strip tracking params like `utm_*`, lowercase scheme/host, strip trailing slashes, remove fragments) before processing.
- **State Store (Redis)**:
  - Discovered and visited state MUST be maintained in Redis sets (`crawler:visited:{job_id}`) with an atomic `SADD` operation to prevent infinite loops and duplicate crawling.
  - Content hashing (SHA-256 of cleaned text body) MUST be calculated and checked against Redis (`crawler:content_hash:{hash}`) to detect duplicate pages under different URLs.

### 2.3 Parsing & Extraction
- **Boilerplate Removal**: Raw HTML must be cleaned to extract main body content while stripping scripts, styles, headers, footers, navigation bars, and ads (using libraries like `readability-lxml` or `trafilatura`).
- **Metadata Extraction**: Extracted documents **MUST** capture:
  - Source URL (`source_url`)
  - Canonical URL (`canonical_url`)
  - Page Title (`title`)
  - Crawl Timestamp (`crawled_at` ISO 8601 UTC)
  - Depth level (`depth`)
  - Breadcrumb / Navigation Hierarchy where available.

---

## 3. RAG Pipeline Guidelines (Chunking, Embedding, Retrieval)

### 3.1 Chunking Strategy
- **Semantic Continuity**: Avoid arbitrary character slicing. Use recursive character splitting or markdown-aware text splitting (`RecursiveCharacterTextSplitter` or semantic header splitting).
- **Chunk Parameters**:
  - Target chunk size: 512 to 1024 tokens (~2000-4000 characters).
  - Chunk overlap: 10% - 15% (e.g., 50-100 tokens) to preserve context boundaries.
- **Metadata Inheritance**: Every chunk **MUST** inherit all parent document metadata plus:
  - `chunk_id` (Deterministic UUID derived from `source_url` + `chunk_index`).
  - `chunk_index` (Integer position in document).
  - `total_chunks` (Total chunks generated from parent document).

### 3.2 Embedding Generation
- **Batch Processing**: Never request embeddings one document at a time. Embeddings **MUST** be processed in optimal async batches (e.g., batch size 32-128).
- **Model Abstraction**: Embeddings generator must implement an abstract interface (`EmbeddingProvider`) allowing seamless swapping between OpenAI, Cohere, HuggingFace, or self-hosted models.

### 3.3 Retrieval & Generation
- **Hybrid Search**: Combine dense vector similarity search with sparse/BM25 search or payload filtering in Qdrant.
- **Re-ranking**: Implement a two-stage retrieval pattern: retrieve Top-K (e.g., K=30) from vector store, then pass through a Cross-Encoder or Re-ranker model (e.g., Cohere Rerank or `bge-reranker`) to select Top-N (e.g., N=5) for the prompt context.
- **Prompt Isolation**: System prompts **MUST** clearly isolate user queries from retrieved context blocks to prevent Prompt Injection attacks. Context blocks must be clearly formatted (e.g., `<context_doc id="1" url="..."> ... </context_doc>`).

---

## 4. Qdrant Vector DB Guidelines

### 4.1 Collection & Vector Specs
- **Distance Metric**: Default to `Cosine` or `Dot` distance depending on embedding model normalization.
- **HNSW Configuration**:
  - Set `m` (connections per node): 16 (default) to 32 (high precision).
  - Set `ef_construct`: 100 to 200 for balanced indexing speed and recall accuracy.
- **On-Disk Payload Storage**: Store payload vectors in RAM for fast search, but set `on_disk_payload: true` for large scale payload data to optimize memory.

### 4.2 Payload Indexing
- **Index Field Types**: Any field used in filtering queries (e.g., `source_url`, `job_id`, `created_at`, `domain`) **MUST** have an explicit payload index created (`KeywordIndex`, `IntegerIndex`, or `DatetimeIndex`).

### 4.3 Client Connection & Operations
- **Async Client Singleton**: Reuse a single global instance of `AsyncQdrantClient` initialized with connection pooling (`limits` and connection pool settings).
- **Batch Upserts**: Always use `client.upsert()` with `Batch` objects or lists of `PointStruct` instances (batch sizes of 100-500 points).
- **Payload Schema Validation**: Points payload MUST conform to a strict Pydantic model before upserting into Qdrant.

```json
{
    "id": "uuid-v5-hash-of-url-and-index",
    "vector": [0.012, -0.043, 0.891],
    "payload": {
        "text": "Extracted chunk content...",
        "source_url": "https://example.com/docs/api",
        "title": "API Reference Documentation",
        "job_id": "crawl-job-9821",
        "chunk_index": 2,
        "total_chunks": 10,
        "crawled_at": "2026-08-03T10:00:00Z"
    }
}
```

---

## 5. Redis Guidelines

### 5.1 Use Cases & Key Namespacing
Redis MUST be used for three distinct operational roles with explicit key prefixing:

| Purpose | Key Pattern Example | Data Structure | TTL Policy |
| :--- | :--- | :--- | :--- |
| **Crawler State** | `crawler:visited:{job_id}` | `SET` | 7 Days |
| **Content Hash** | `crawler:hash:{sha256}` | `STRING` | 30 Days |
| **Distributed Lock**| `lock:crawler:domain:{domain}` | `STRING` (Redlock) | Short (30s) |
| **Semantic / Query Cache** | `cache:query:{md5_hash}` | `STRING` (JSON) | 24 Hours |
| **Rate Limiters** | `ratelimit:{domain}:{window}` | `INCR` counter | Sliding window |

### 5.2 Caching Strategy & Eviction
- **TTL Enforcement**: Every key stored in Redis **MUST** have an explicit Time-To-Live (TTL). Unlimited keys are strictly forbidden.
- **MaxMemory Policy**: Redis instance MUST be configured with `maxmemory-policy volatile-lru` or `allkeys-lru` to handle out-of-memory scenarios gracefully.
- **Connection Management**: Use `redis.asyncio.ConnectionPool` or `redis.asyncio.Redis` instance managed via application lifespan context manager.

---

## 6. Docker & Containerization Guidelines

### 6.1 Multi-Stage Build Architecture
Dockerfiles **MUST** use multi-stage builds to produce lightweight runtime images devoid of build tools, compilers, or temporary caches.

```dockerfile
# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:$PATH"

COPY --from=builder /install /install

RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Docker Best Practices
- **Never Run as Root**: Always create and switch to an unprivileged non-root user (`appuser`).
- **`.dockerignore` Requirements**: `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.env`, `data/`, and test coverage files MUST be excluded.
- **Secret Management**: API keys (OpenAI, Qdrant, Redis password) MUST NEVER be baked into image layers. Pass them via environment variables or secret managers.

---

## 7. Testing & Quality Assurance Guidelines

### 7.1 Testing Pyramid
All code submissions must satisfy three levels of automated testing using `pytest`:

1. **Unit Tests (`tests/unit/`)**:
   - Focus on isolated domain logic (text splitters, custom parsers, prompt templates, URL normalizers).
   - **Zero external Network I/O permitted**. All LLM calls, Redis operations, and Vector DB queries **MUST** be mocked using `unittest.mock` or `pytest-mock`.
2. **Integration Tests (`tests/integration/`)**:
   - Validate interactions between services (Qdrant client queries, Redis caching, Async HTTP Scraper).
   - Use `testcontainers-python` to spin up ephemeral Redis and Qdrant instances during test runs.
3. **End-to-End (E2E) Tests (`tests/e2e/`)**:
   - Validate full crawl -> embed -> store -> retrieve pipeline on a sample static HTML fixture server.

### 7.2 RAG Evaluation Metrics (RAG Triad)
Continuous evaluation pipelines MUST measure:
- **Faithfulness**: Is the answer derived strictly from the retrieved context?
- **Answer Relevance**: Does the generated output directly address the user query?
- **Context Precision & Recall**: Did the retriever pull relevant web chunks and filter out noise?

---

## 8. Code Quality, Error Handling & Observability

### 8.1 Custom Exceptions
Never raise generic `Exception` or `RuntimeError`. Define an explicit domain exception hierarchy:

```python
class BaseRAGException(Exception):
    """Base exception for RAG application."""
    pass

class ScraperException(BaseRAGException):
    """Raised when web scraping or content parsing fails."""
    pass

class VectorDBException(BaseRAGException):
    """Raised during vector upsert or query operations."""
    pass

class RateLimitExceededException(BaseRAGException):
    """Raised when crawling or LLM API rate limit is reached."""
    pass
```

### 8.2 Logging & Tracing
- **Structured JSON Logging**: Use `structlog` or standard library JSON formatter for production logs.
- **Contextual Logging**: Every log message MUST contain contextual metadata (e.g., `job_id`, `url`, `chunk_count`, `duration_ms`).
- **Tracing**: Instrument critical spans (retrieval, re-ranking, LLM generation, crawling task) using OpenTelemetry (`opentelemetry-api`).
