# DataEngineeringCopilot — Complete RAG System Flow Guide

A visual and technical reference tracing every data transformation from HTML pages to LLM-generated answers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Ingestion Pipeline: URL to Vectors](#2-ingestion-pipeline-url-to-vectors)
3. [Query Pipeline: Question to Answer](#3-query-pipeline-question-to-answer)
4. [Streaming Flow](#4-streaming-flow)
5. [Configuration Reference](#5-configuration-reference)
6. [Observability & Caching](#6-observability--caching)
7. [Provider Fallback Chain](#7-provider-fallback-chain)
8. [Spark Index Generations](#8-spark-index-generations)
9. [Claude Docs Ingestion (llms.txt)](#9-claude-docs-ingestion-llmstxt)
10. [Indirect Prompt Injection Guard](#10-indirect-prompt-injection-guard)

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       User / Client                              │
│         curl │ Streamlit UI │ CLI (dec ask)                      │
└─────────┬─────────────────────────┬──────────────────────────────┘
          │                         │
          ▼                         ▼
┌──────────────────────┐  ┌─────────────────────┐
│   FastAPI / REST     │  │  CLI (dec ask)       │
│  POST /api/v1/ask    │  │  (bypasses API)      │
│  POST /api/v1/ask/stream │  │  calls factory      │
│  POST /api/v1/ingest │  │  directly            │
└──────┬──────────┬────┘  └──────┬──────────────┘
       │          │              │
       ▼          ▼              ▼
┌──────────────────────────────────────────────────┐
│              AsyncRagService                     │
│  answer() / answer_stream()                      │
│  ┌─────────┬─────────┬──────────┬──────────┐    │
│  │ Cache   │ Rewrite │ Retrieve │ Rerank   │    │
│  │ 2-tier  │  LLM    │ Qdrant   │ Cross-   │    │
│  │ L1+L2   │ HyDE    │ RRF      │ encoder  │    │
│  ├─────────┼─────────┼──────────┼──────────┤    │
│  │ Context │ Prompt  │ LLM     │ Post-    │    │
│  │ Assem.  │ Builder │ Client  │ process  │    │
│  └─────────┴─────────┴──────────┴──────────┘    │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│           Services / Infrastructure              │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ │
│  │ Qdrant  │ │ Redis   │ │ Ollama │ │ Postgres│ │
│  │ Vectors │ │ Cache   │ │ LLM    │ │ Crawl  │ │
│  │         │ │ Queue   │ │ Embed  │ │ Front. │ │
│  └─────────┘ └─────────┘ └────────┘ └────────┘ │
└──────────────────────────────────────────────────┘
```

**One-sentence summary:** A RAG system that crawls data-engineering documentation sites, stores them as hybrid searchable vectors in Qdrant, rewrites user questions into optimal search queries, retrieves relevant context, and generates grounded answers via a provider-fallback LLM chain.

**Key design decisions:**
- No LangChain/LlamaIndex — pure Python with structural-typing protocols (`domain/protocols.py`)
- Async-first — `httpx.AsyncClient`, `asyncio.TaskGroup`, `aiohttp`
- Per-purpose LLM routing — different models for answer, rewrite, groundedness, intent, code generation
- All LLM/embedding calls route through `ProviderFallbackChain` — never call a provider directly
- Annotate-only, fail-open — groundedness and guardrails never block answers, only annotate
- Header-aware chunking for documentation pages (preserves `heading_path`, siblings at same level)

**Tech stack:**
- Python 3.12, Pydantic, FastAPI, Celery
- Qdrant (vector DB), Redis (cache/queue), PostgreSQL (crawl frontier)
- Ollama (local LLM + embeddings), OpenRouter / NVIDIA / Gemini / Groq / Cloudflare / Cerebras / HuggingFace (cloud fallbacks), local-hf (local SentenceTransformer)
- Langfuse v4 + OpenTelemetry (observability), ClickHouse (Langfuse analytics), MinIO (dev object store)
- testcontainers (integration testing)

---

## 2. Ingestion Pipeline: URL to Vectors

### Step 0 API Entry (`api/routes.py:199-281`)

```
  Client                         FastAPI                       Celery
    │                              │                             │
    │ POST /api/v1/ingest          │                             │
    │ {source_names, max_pages}    │                             │
    │─────────────────────────────►│                             │
    │                              │ SETNX lock (60s TTL)       │
    │                              │ Check existing task status  │
    │                              │                             │
    │                              │ async_ingest_task.delay()   │
    │                              │────────────────────────────►│
    │                              │                             │ asyncio.run(
    │                              │                             │   service.ingest())
    │◄─────────────────────────────│                             │
    │ {task_id, state: "PENDING"}  │                             │
```

**Input:** `IngestRequest(source_names: list[str] | None, max_pages: int | None)`

**Key guard:** Atomic `SETNX ingestion:dispatch_lock` (line 223) prevents concurrent ingestion runs. If a task is already `PROCESSING` or `DISPATCHED`, returns 409. `ingestion:dispatch_lock` TTL is 60s (`routes.py:221-223`).

**Config:** `max_pages` defaults to `settings.max_pages_per_source = 100000` (`settings.py:651`), clamped 1-100000. Hard cap `max_pages_hard_cap = 100000` (`settings.py:652`).

**Status endpoints:**
- `GET /api/v1/ingest/status/{task_id}` (line 294) — poll progress
- `GET /api/v1/ingest/latest` (line 316) — latest task
- `POST /api/v1/ingest/{task_id}/cancel` (line 330) — cancel a running ingest

### Step 1 Celery Task (`workers/tasks.py:104-150`)

The Celery task `async_ingest_task` is configured with:
```python
@celery_app.task(
    bind=True,
    queue="ingestion",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    retry_backoff=True,
)
def async_ingest_task(self, source_names: list[str], max_pages: int | None):
```

It validates inputs via `_validate_ingest_inputs` (rejects empty `source_names`), propagates W3C trace context, and calls:
```python
service = build_async_ingestion_service()     # factory.py:1021
asyncio.run(service.ingest(source_names=..., max_pages_per_source=..., on_event=...))
```

Progress is persisted to Redis via `IngestionProgressTracker` (pollable from Streamlit/API).

### Step 2 Factory Wiring (`factory.py:1021-1085`)

`build_async_ingestion_service()` wires all 10+ components:

| Component | Source | Key Config |
|---|---|---|
| `redis_client` | `aioredis.from_url(redis_url)` | `settings.redis_url` |
| `AsyncDocumentationCrawler` | `build_async_crawler()` | concurrency=10 (max 40), delay=0.3s, per-domain=2 |
| `MarkdownParser` | direct instantiation | min_words=40 |
| `DocumentChunker` | `build_chunker()` | chunk_size=1000 chars, overlap=100 chars (for `sentence_preserving`) |
| embedder | `build_embedder()` | provider chain, 768-dim default (nomic-embed-text) |
| `AsyncQdrantVectorStore` | direct instantiation | collection=`data_engineering_docs`, hybrid=on |
| `ContextualChunkEnricher` | `LLMContextSummarizer` wrapper | batch_size=20, queued to Redis worker |
| `ApiDocExtractor` | direct instantiation | enabled when `api_extraction_enabled=True` |
| `CodeBlockParser` | direct instantiation | enabled when `code_block_parsing_enabled=True` |
| `ChunkFilter` | direct instantiation | enabled when `chunk_filtering_enabled=True` |

The chunker used depends on `chunking_strategy` (`settings.py:610`):
- `sentence_preserving` (default) → `DocumentChunker`
- `semantic` → `SemanticChunker` (needs `enable_semantic_chunking=True`)
- `header_aware` → `HeaderAwareChunker`

Spark pages additionally route through a `SparkChunker` when `_spark_chunker` is configured (`async_ingestion.py:171-173`).

### Step 3 Source Selection (`async_ingestion.py:459-481`)

Sources are loaded from `data_engineering_copilot/config/documentation_sources.json` via `settings.py`:

```python
@dataclass(frozen=True)
class DocumentationSource:
    name: str                    # e.g. "Apache Spark Documentation"
    start_urls: tuple[str, ...]  # e.g. ("https://spark.apache.org/docs/latest/",)
    allowed_domains: tuple[str, ...]
    url_prefixes: tuple[str, ...]
    priority: int = 1
```

If `source_names` is `None`, all sources are used. If specified, only matching names are selected.

### Step 4 Crawling (`infrastructure/async_crawler.py:213-360`)

```
  PostgreSQL Frontier              Crawler                          Network
       │                              │                              │
       │ discover(url, depth=0)       │                              │
       │◄─────────────────────────────│                              │
       │ INSERT INTO crawl_frontier   │                              │
       │ ON CONFLICT DO NOTHING       │                              │
       │                              │                              │
       │ get_pending(source, limit)   │                              │
       │◄─────────────────────────────│                              │
       │ UPDATE ... WHERE state='DISCOVERED'                         │
       │ RETURNING *                  │                              │
       │─────────────────────────────►│                              │
       │                              │                              │
       │                              │ HEAD /page? If-Modified-Since
       │                              │─────────────────────────────►│
       │                              │◄──── 304 Not Modified ──────│
       │                              │  (skip, rediscover children) │
       │                              │                              │
       │                              │ GET /page                    │
       │                              │─────────────────────────────►│
       │                              │◄─── 200 OK + HTML body ─────│
       │                              │                              │
       │                              │ Extract links (<a> tags)     │
       │ discover(child_url, parent)  │                              │
       │─────────────────────────────►│                              │
```

**BFS Frontier** (`infrastructure/crawl_db.py`):
- 4-state state machine: `DISCOVERED → FETCHING → PROCESSED | FAILED`
- Atomic `SELECT ... FOR UPDATE SKIP LOCKED` for worker assignment
- `reset_stranded()` on init recovers crashed workers
- `sitemap_edges` table maintains parent-child link graph
- Composite index on `(source_name, state, depth, created_at)`
- `frontier_max_attempts=3` (`settings.py`): FAILED URLs are re-discovered up to 3 times, then become terminal

**Crawler guards** (`async_crawler.py`):
- **Robots.txt** — `_get_robots_parser` + `_is_allowed_async` (lines 613-640) respect per-domain robots rules before fetching
- **Sitemap seeding** — `_seed_frontier` → `_try_sitemap` (line 477) + `_parse_sitemap` (line 489) harvest URLs from `sitemap.xml` first
- **SSRF guard** — `_is_private_ip` (line 353) blocks private/loopback targets
- **Domain priority** — per-domain semaphores with `_get_domain_priority` + `_get_priority_multiplier` (lines 172-181)
- **Conditional GET** — Two-phase per-page workflow:
  1. **HEAD** (line 384): Sends `If-None-Match: {etag}` / `If-Modified-Since: {last_modified}`. 304 = skip.
  2. **GET** (line 376): Full fetch with 3x retry, exponential backoff. Returns HTML.

**Link discovery** (lines 513-536): Fast `HTMLParser` based link extraction, BeautifulSoup fallback. Filters by scheme (http/https), `allowed_domains`, and `url_prefixes`. URLs normalized/deduped via `_clean_url` (line 565) and `_dedupe_key` (line 593).

**Output per page:** `RawDocument(source_name, url, html)` — the HTML string of a single page.

### Step 5 HTML Parsing (`infrastructure/html_to_markdown.py:12-72`)

```
  RawDocument.html (raw HTML string)
       │
       ▼
  BeautifulSoup parse
       │
       ├─ Remove: script, style, noscript, nav, footer, header, aside
       ├─ Extract: <main> → <article> → <body> (priority order)
       ├─ markdownify → ATX-style markdown (# headings)
       ├─ Clean: collapse newlines, strip trailing spaces
       └─ Filter: min_words=40, discard thin pages
       │
       ▼
  ParsedDocument(source_name, title, url, text)
```

**Title extraction** (line 66): prefers `<h1>` text, falls back to `<title>`, falls back to URL.

**Output:** `ParsedDocument(source_name, title, url, text=markdown_text)` or `None` (if < 40 words).

**Special parsers** for structured sources:
- `spark_html_parser.py` — Apache Spark docs-specific HTML extraction
- `rst_parser.py` — reStructuredText sources
- `native_document_parser.py` — native/SDK documentation formats

### Step 6 Content Hashing & Dedup (`async_ingestion.py:94-117`)

```
  ParsedDocument.text (str)
       │
       ├─ SHA-256(text) → content_hash
       │
       ├─ Redis lookup: crawl:url_registry:{source} → get stored hash
       │  OR Qdrant fallback: scroll by URL → get content_hash from payload
       │
       ├─ If hash matches → skip page (duplicate, emit event)
       ├─ If hash differs → delete all old Qdrant points for this URL
       └─ If no stored hash → first-time ingestion, proceed
```

The hash is stored in Redis after successful upsert (`async_ingestion.py:193-201`), keyed by `(url, source_name)`.

### Step 7 Chunking — 3 Strategies

#### 7a DocumentChunker (default: sentence_preserving / fixed_size)

**File:** `services/chunker.py:24-100`
**Config:** `chunk_size_chars=1000`, `chunk_overlap_chars=100`

```
Input:  ParsedDocument
Output: list[DocumentChunk]
ID fmt: {source_slug}:{url_sha1_10}:{index:04d}
         e.g. "spark_docs:a1b2c3d4e5:0003"
```

**Language-aware splitting** (line 44): Detects language from URL path:
- `/api/python/` or `/pyspark` → `Language.PYTHON` (class/def boundaries)
- `/api/scala/` → `Language.SCALA`
- `/api/java/` → `Language.JAVA`
- `/api/R/` → `Language.R`
- Otherwise → generic `RecursiveCharacterTextSplitter` on `["\n\n", "\n", " ", ""]`

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter`.

#### 7b HeaderAwareChunker

**File:** `services/header_aware_chunker.py:49-250`
**Config:** `chunk_size_words=375`, `overlap_words=90`, `min_chunk_words=int(chunk_size_words * 0.1) = 37`

```
Input:  ParsedDocument
Output: list[DocumentChunk] with enriched metadata
ID fmt: {source}:{url_sha1}:hdr:{index:04d}
```

**Algorithm:**
1. **Parse headers**: Regex `^(#{1,6})\s+(.*)` extracts heading hierarchy. Maintains a `heading_stack: list[tuple[int, str]]` tracking `(level, text)` to build `heading_path` (e.g., `("Overview", "Installation")`).
2. **Sibling preservation (fix)**: When a new header's level is `<=` the top of the stack, pop stack entries — same-level siblings (`### A`, `### B` under the same `## parent`) stay siblings, not nested parent/child.
3. **Flush on boundary**: Flushes when parent boundary changes or word count exceeds `chunk_size_words`. Adds overlap from previous chunk's tail.
4. **Content loss prevention (fix)**: Sub-`min_chunk_words` trailing sections are merged into the previous chunk (via `dataclasses.replace`) instead of dropped. A page whose entire body is below the minimum is still filtered (min_chunk_words contract preserved).

**Output fields populated:**
- `section_header` — e.g. `"Requirements"`
- `chunk_type` — `"text"`, `"code"`, or `"mixed"`
- `word_count` — actual word count
- `heading_path` — tuple of all ancestor headers

#### 7c SemanticChunker

**File:** `services/semantic_chunker.py:89-348`
**Config:** `min_semantic_similarity=0.5`, `chunk_size_words=375`, `overlap_words=90`, `min_chunk_words=37`, `max_chunk_words=562`

**Special path in ingestion** (`async_ingestion.py:174-200`):
```python
sentences = self.chunker.extract_sentences(parsed.text)      # NLTK sent_tokenize
embeddings = await self.embeddings.embed_texts(sentences)     # Embed ALL sentences first
chunks = await self.chunker.chunk(parsed, precomputed_embeddings=embeddings)
```

**Three-valued contract:** `extract_sentences` returns `None` (unsupported), `[]` (empty — page skipped), or `list` (content). Check `is None` explicitly.

**Algorithm:**
1. **Sentence extraction** via NLTK `sent_tokenize`
2. **Embed each sentence** (precomputed or on-the-fly)
3. **Greedy clustering** (line 147-204): For each sentence, compute cosine similarity to cluster centers. Add to most similar cluster if ≥ 0.5, else start new cluster.
4. **Merge clusters** (line 206-304): Merge respecting word-count limits (`max_chunk_words`), adding overlap at boundaries.

ID format: `{source}:{url_sha1}:semantic:{index:04d}`

### Step 8 Enrichment (`async_ingestion.py:200-260`)

After chunking, synchronous enrichers run:

1. **ChunkFilter** (`services/text_filter.py`): Drops low-quality chunks (too short, boilerplate text, noise patterns like log lines, excessive brackets). Enabled via `chunk_filtering_enabled=True` (`settings.py:703`).
2. **ApiDocExtractor** (`services/api_extractor.py`): Identifies API documentation chunks via regex patterns (`function_name(`, `class Name:`, `def method`) and tags them with `chunk_type="api"`. Prepends structured metadata: `[API: Module: X | Method: Y | Params: ... | Returns: ...]`. Enabled via `api_extraction_enabled=True`.
3. **CodeBlockParser** (`services/code_block_parser.py`): Identifies fenced code blocks, tags with `chunk_type="code"`, optionally splits large blocks at function/class boundaries. Enabled via `code_block_parsing_enabled=True`.

**Contextual enrichment** (`async_ingestion.py:202-207`): Uses `LLMContextSummarizer` to generate a document-level summary prepended to each chunk. Config: `contextual_enrichment_enabled=True` (`settings.py:695`). When running under Celery (`_task_id_explicit`), enrichment is **queued to Redis** for the background worker rather than done inline — the shared Redis client (`get_shared_redis_client()`) is used.

### Step 9 Batch Accumulation (`async_ingestion.py:312-438`)

```
Crawler → Queue[RawDocument | None] → Workers (4 concurrent) → Batch[Chunk]
                                          │
                                    When batch >= 256 chunks
                                          │
                                          ▼
                                    Flush Batch:
                                      enrich → embed_texts() → upsert_chunks()
```

**Data structures:**
- `queue: asyncio.Queue(maxsize=8)` — bounded queue with backpressure
- `shared["batch_chunks"]: list[DocumentChunk]` — batch accumulator (protected by `asyncio.Lock`)
- `embed_semaphore: Semaphore(max_parallel)` — limits concurrent embedding calls

### Step 10 Embedding (`infrastructure/async_embeddings.py:42-124`, fallback chain)

```
Input:  list[str] — chunk texts (up to 256 per batch)
Output: list[list[float]] — dense vectors (768-dim for nomic-embed-text)
```

**Provider:** Selected via `embedding_provider` + `embedding_fallback_order`. All calls route through the `ProviderFallbackChain` (see [Section 7](#7-provider-fallback-chain)):
- **Ollama** via `POST /api/embed`: `{"model": "nomic-embed-text", "input": texts}`
- **OpenRouter** via `POST /embeddings` — `nvidia/nemotron-3-embed-1b:free`
- **NVIDIA** via `POST /embeddings` — `nvidia/nemotron-3-embed-1b`
- **HuggingFace serverless** (`huggingface_serverless_embeddings.py`) — native `feature-extraction` pipeline route; prepends `query:`/`passage:` prefixes client-side (serverless backend ignores `prompt_name`)
- **local-hf** (`local_sentence_transformer_embeddings.py`) — local SentenceTransformer, offline

**Batch slicing:** Texts split into `embedding_batch_size=64` sub-batches (`settings.py:397`). Processed sequentially.

**Per-provider input limits:** `tokenizer_registry.py` maps model names to token budgets (`KNOWN_INPUT_LIMITS`, `_MODEL_TOKENIZER_REPO`). Texts exceeding the limit are rejected with a budget error rather than silently truncated.

**Retry:** 3 attempts, exponential backoff (1-10s), retries on `TimeoutException`, `ConnectError`, `OSError`.

**Dimension validation** (line 95): Checks each embedding length against `embedding_model_dimensions` lookup (`settings.py:411-419`):

| Model | Dimensions |
|---|---|
| `nomic-embed-text` | 768 |
| `mxbai-embed-large` | 1024 |
| `snowflake-arctic-embed2` | 1024 |
| `llama3.2:3b` | 3072 |
| `nvidia/nemotron-3-embed-1b` | 2048 |
| `nvidia/nemotron-3-embed-1b:free` | 2048 |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 |
| `text-embedding-004` | 768 |

### Step 11 Qdrant Upsert (`infrastructure/async_qdrant_store.py:160-204`)

```
Input:  list[DocumentChunk] + list[list[float]] (dense vectors)
Output: Qdrant points stored
```

**Point ID:** Deterministic UUID5 from chunk_id string (ensures idempotent upserts):
```python
str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
```

**Payload** (one per chunk, line 143-155):
```python
{
    "chunk_id": str,
    "source_name": str,
    "title": str,
    "url": str,
    "text": str,
    "content_hash": str,
    "section_header": str,
    "chunk_type": str,       # "text" | "code" | "api" | "table"
    "word_count": int,
    "heading_path": list[str],
}
```

**Hybrid mode** (default): Stores both dense and sparse vectors:
```python
vectors_config = {
    "dense": VectorParams(size=768, distance=COSINE),
}
sparse_vectors_config = {
    "sparse": SparseVectorParams(index=SparseIndexParams()),
}
```

BM25 sparse vectors are computed inline: `self._bm25.tokenize_query(chunk.text)` (`async_qdrant_store.py:187-189`).

**Sub-batching:** 256 points per batch (line 179) to avoid Qdrant's 32MB payload limit.

**Payload indexes** (line 105-136): Keyword indexes on `url`, `source_name`, `chunk_type`, `section_header` for fast filtering.

### Step 12 BM25 Fitting (`async_ingestion.py:255-258`)

After ALL sources finish, BM25 is fitted on the full corpus:
```python
if self._corpus_texts and hasattr(self.vector_store, "fit_bm25"):
    self.vector_store.fit_bm25(self._corpus_texts)
```

The `BM25Tokenizer` (`infrastructure/bm25_tokenizer.py:198-220`) persists to disk: `.bm25_cache/{collection_name}.json`. Includes vocab, IDF, corpus stats, and frozen flag. Loaded on startup.

**Silent degradation warning:** If BM25 is not yet fitted (during ingestion or after restart), hybrid search silently falls back to dense-only (`async_qdrant_store.py:237`: `use_hybrid = ... and self._bm25._frozen`).

---

## 3. Query Pipeline: Question to Answer

### Complete Flow Diagram

```
  User Question (str)
       │
       ▼
  ┌──────────────────────────────┐
  │ 1. CACHE CHECK               │
  │ Exact (SHA-256) → Semantic   │
  │ (NumPy SIMD) → Redis L2      │
  └──────────┬───────────────────┘
             │ miss
             ▼
  ┌──────────────────────────────┐
  │ 2. QUERY REWRITING           │
  │ Intent → Sub-queries → HyDE  │
  │ → Multi-query expansion      │
  └──────────┬───────────────────┘
             │ effective_query + all_queries
             ▼
  ┌──────────────────────────────┐
  │ 3. EMBEDDING                 │
  │ CachedEmbedder → fallback    │
  │ chain (768-dim)              │
  └──────────┬───────────────────┘
             │ query_embedding (list[float])
             ▼
  ┌──────────────────────────────┐
  │ 4. VECTOR SEARCH (Qdrant)    │
  │ Dense cosine + BM25 sparse   │
  │ → RRF fusion (k=60)          │
  └──────────┬───────────────────┘
             │ list[RetrievedChunk]
             ▼
  ┌──────────────────────────────┐
  │ 5. INPUT GUARDRAILS          │
  │ Drop injection-laden chunks  │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │ 6. RERANKING                 │
  │ Cross-encoder sigmoid        │
  │ → post-rerank confidence gate│
  └──────────┬───────────────────┘
             │ top-N chunks
             ▼
  ┌──────────────────────────────┐
  │ 7. CONTEXT COMPRESSION       │
  │ Jaccard dedup (0.85)         │
  │ → Relevance re-ranking       │
  └──────────┬───────────────────┘
             │
             ▼
  ┌──────────────────────────────┐
  │ 8. CONTEXT ASSEMBLY          │
  │ Lost-in-middle mitigation    │
  │ → Chunk boundary truncation  │
  └──────────┬───────────────────┘
             │ context_str (max_context_chars)
             ▼
  ┌──────────────────────────────┐
  │ 9. PROMPT BUILDING           │
  │ Density tag + intent-based   │
  │ output format (Langfuse)     │
  └──────────┬───────────────────┘
             │ full prompt
             ▼
  ┌──────────────────────────────┐
  │10. LLM GENERATION            │
  │ System msg + fallback chain  │
  │ Temperature 0.05             │
  └──────────┬───────────────────┘
             │ answer_text (raw str)
             ▼
  ┌──────────────────────────────┐
  │11. POST-PROCESSING           │
  │ JSON retry → Code validation │
  │ → Guardrails → PII → Cit.    │
  │ → Groundedness → Cache store │
  └──────────┬───────────────────┘
             │ Answer(text, sources, confidence, ...)
             ▼
  ┌──────────────────────────────┐
  │12. RESPONSE                  │
  │ AskResponse with metrics     │
  │ and stage_times              │
  └──────────────────────────────┘
```

### Step 1 Cache Check (`async_rag.py:363-415`)

```
Input:  question (str)
Lookup: aget(question, query_embedding)
Output: str | None (cached answer)
```

**Three-layer cascade** (in-process `query_cache.py`):
1. **L1 Exact** (`query_cache.py:171-177`): SHA-256 of normalized query. LRU cache (`OrderedDict`, max 1024 entries).
   - Normalization: lowercase, strip, remove non-word chars, collapse whitespace.
2. **L2 Semantic** (`query_cache.py:190-229`): NumPy batch dot product against cached unit vectors. Threshold = 0.95 (config: `semantic_cache_threshold`, `settings.py:680`). LRU cache (`deque`, max 512 entries). TTL = 3600s.
3. **L3 Redis** (`query_cache.py:251-339`): Async Redis client. Exact: `GET rag:cache:exact:{sha256}`. Semantic: `SCAN rag:cache:semantic:*` → `HGETALL` → cosine similarity. On hit, backfills L1/L2.

**Cache scope:** Keys are scope-fingerprinted (`scope_fingerprint`, `query_cache.py:58`) — a `CacheScope` (e.g. `PRIVATE` for a user) changes the key namespace so one user's answer is never served to another. Private-scope answers are not stored in Redis.

**Flow optimization:** Exact tier is checked FIRST — no embedding round-trip on an exact hit. Only on exact miss does the code pay the embedding cost for semantic lookup (`async_rag.py:376-390`).

**Cacheability:** `is_cacheable()` (`query_cache.py:136`) — only answers above `MIN_CACHE_CONFIDENCE` are written back.

On cache hit, returns immediately with `Answer(text=cached, confidence=...)` and records a lightweight `rag-query-pipeline-cache-hit` trace (`_record_cache_hit_trace`, `async_rag.py:298`) — skips the entire pipeline.

### Step 2 Query Rewriting (`query_rewriting.py:203-231`, `async_rag.py:430-500`)

```
Input:  question (str)
Output: RewrittenQuery(intent, decomposed_steps, hyde_query, filters)
        all_queries: list[str] (for multi-query retrieval)
```

#### 2a LLM Rewrite Prompt

When `async_rewrite()` is used (non-streaming path), this prompt is sent to the LLM:

```
You are a search query rewriter. Given a user question, produce a concise,
search-optimized query that would best retrieve relevant documentation.
Rules:
- Return ONLY the rewritten query, no preamble.
- Preserve the user's intent.
- Expand abbreviations and jargon where helpful.
- Output a single line, no more than 30 words.

User question: {question}

Rewritten query:
```

If the LLM returns empty or < 3 chars, falls back to rule-based `rewrite()`.

#### 2b Intent Classification (`query_rewriting.py:105-127`)

Six intents matched by regex priority:

| Priority | Intent | Regex (partial) |
|---|---|---|
| 1 | `api_lookup` | `DataFrame.`, `spark.read`, `.groupBy`, `.select`, `\.\w+\(` |
| 2 | `code_example` | `give me code`, `write a script`, `how to code` |
| 3 | `comparative` | `compare`, `vs`, `versus`, `difference between`, `pros and cons` |
| 4 | `debugging` | `why is`, `error`, `fail`, `bug`, `oom`, `exception`, `crash` |
| 5 | `how_to` | `how to`, `how do`, `step-by-step`, `guide`, `tutorial`, `configure` |
| 6 | `factual` | catch-all (`re.compile(r".*")`) |

If no regex matches and `intent_classification_llm_enabled=True` (`settings.py:689`), falls back to LLM:
```json
{"intent": "code_example" | "factual"}
```

#### 2c Multi-Step Decomposition

Each intent decomposes the query into sub-queries:

| Intent | Decomposition | Example Output |
|---|---|---|
| `comparative` | X, Y, differences | `("What is X?", "What is Y?", "What are the differences?")` |
| `how_to` | prerequisites, steps | `("What are prerequisites for X?", "Steps to X")` |
| `debugging` | cause, solution | `("What causes X?", "How to fix X?")` |
| `api_lookup` | signature, params, original | `("Signature of method?", "Parameters of method?", original)` |
| `code_example` | example, recommended way | `("Code example for X", "Recommended way")` |
| `factual` | original only | `(original_query,)` |

#### 2d Structured Filters

`api_lookup` intents can also extract structured metadata filters (`rewritten.filters`) — e.g. a specific `modules` list pins the exact API page. When a hard `modules` filter is present, the "api" chunk-type restriction is dropped because rendered reference pages carry `chunk_type="text"` (`async_rag.py:497-508`).

#### 2e HyDE Generation (`query_rewriting.py:264-348`)

HyDE = Hypothetical Document Embedding. Generate a hypothetical answer, then embed THAT as an ADDITIONAL query variant (not a replacement).

**Prompt:**
```
Write a short, authoritative paragraph that would perfectly answer
the following question. Do not address the user directly.

Question: {query}
```

The HyDE query is appended to `queries_to_run` and embedded separately (`async_rag.py:520-530`).

#### 2f Multi-Query Expansion (`query_rewriting.py:279-301`)

**Prompt:**
```
Generate {max_variations} different search queries that would find
the same information as this question. Return ONLY the queries,
one per line, no numbering.

Original question: {query}

Variations:
```

Called with `max_variations=settings.max_expansion_queries` (default 2). Results are appended to `all_queries`.

**Final `all_queries` list (`async_rag.py:432`):**
```python
all_queries = [question]          # always includes original
+ decomposed_steps                # 1-3 sub-queries from intent
+ expanded_variations             # up to 2 LLM-generated variants
```

Each query in `all_queries` gets its own embedding + vector store query. The original query is always first so it receives the fusion bonus.

### Step 3 Query Embedding (`async_rag.py:530`, `infrastructure/embedding_cache.py:43-66`)

```
Input:  effective_query (str)
Output: query_embedding (list[float], 768-dim)
```

The embedder is wrapped in `CachedEmbedder` (`factory.py`):
```python
embedder = CachedEmbedder(embedder)  # LRU cache, max 1024 entries
```

**Cache key:** SHA-256 of lowercased+stripped text.

**Cache hit:** Returns cached `list[float]` immediately.
**Cache miss:** Delegates to the fallback chain:
- **Ollama** (`async_embeddings.py:113-118`): `POST /api/embed` with `{"model": "nomic-embed-text", "input": [text]}`
- **OpenRouter/NVIDIA** (`async_openai_compatible_embeddings.py`): `POST /embeddings` with `{"model": "...", "input": [text]}`, truncating to per-model input limit (tokenizer_registry).
- **HuggingFace serverless**: native `feature-extraction` route; prefixes `query:` for queries and `passage:` for documents client-side.
- **local-hf**: local SentenceTransformer.

### Step 4 Vector Search (`infrastructure/async_qdrant_store.py:206-329`)

```
Input:  query_embedding (list[float])
        query_text (str) — for BM25 sparse
        source_filter (list[str] | None)
        chunk_type_filter (str | None)
        metadata_filters (MetadataFilter | None)
        top_k (int)
        fused_limit (int) — rerank pool cap
Output: list[RetrievedChunk]
```

#### 4a Filter Construction

```python
filter_conditions = []
if source_filter:
    filter_conditions.append(FieldCondition(key="source_name", match=MatchAny(any=source_filter)))
if chunk_type_filter:
    filter_conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter)))
if metadata_filters and not metadata_filters.is_empty:
    # e.g. modules → "url" contains /sql/..., build in metadata_filters
    ...
query_filter = Filter(must=filter_conditions) if filter_conditions else None
```

#### 4b Hybrid Search Path — default

When `hybrid_search_enabled=True` AND BM25 is fitted:

```python
prefetch = [
    Prefetch(query=query_embedding, using="dense", limit=top_k * 2, filter=query_filter),
    Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2, filter=query_filter),
]
query = RrfQuery(rrf=Rrf(k=60))  # RRF fusion
```

**RRF scoring:** Each result gets score = `1 / (k + rank_dense) + 1 / (k + rank_sparse)`. k=60 softens the rank contribution. Top results up to `fused_limit` returned.

**BM25 sparse vector:** `self._bm25.tokenize_query(query_text)` — Porter-stemmed tokens from the query extracted against the fitted vocabulary. Tokens not in the vocabulary are silently dropped (see `bm25_tokenizer.py`).

#### 4c Dense-Only Path — fallback

When BM25 is not fitted or hybrid is disabled:
```python
query = query_embedding
using = "dense"
```

#### 4d Result Construction

Each Qdrant hit becomes:
```python
RetrievedChunk(
    chunk=DocumentChunk(chunk_id, source_name, title, url, text, ...),
    distance=1.0 - confidence,
    confidence=min(1.0, max(0.0, score)),
)
```

Score is clamped to [0, 1].

#### 4e Multi-Query Merge (`async_rag.py:57-90`)

Per-query result sets are merged via `merge_retrieval_results(per_query_results, question)` — the original query's hits receive a fusion bonus. Query-level `api_lookup` (e.g. `def filter(`) gets an additional lexical bonus on the vector score.

**Unfiltered fallback:** If metadata filters removed everything, retry once without inferred filters (`async_rag.py:556-566`).

### Step 5 Input Guardrails (`async_rag.py:680-690`, `services/input_guardrails.py:28-60`)

**Indirect prompt injection guard.** Before any chunk reaches the prompt:
```python
scan_result = self.input_guardrails.scan_chunks(retrieved_chunks)
retrieved_chunks = scan_result.kept
```

Drops retrieved chunks that look like embedded instructions (e.g. "ignore previous instructions", prompt-esque payloads). If ALL chunks are rejected, returns the out-of-repository answer. Enabled via `input_guardrails_enabled=True` (`settings.py:705`).

### Step 6 Reranking (`async_rag.py:700-745`, `services/reranker.py:84-151`)

```
Input:  query (str), chunks (list[RetrievedChunk]), top_k
Output: list[RetrievedChunk] — reranked by cross-encoder score
```

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (`reranker_model`, `settings.py:622`) (~450MB, downloaded on first use). Loaded lazily off the event loop (`_ensure_reranker_ready`) — a cold model cache degrades to "no reranking" instead of failing the answer.

**Scoring:**
1. Prepare `(query, chunk_text)` pairs.
2. `self.model.predict(pairs)` — raw logits from cross-encoder.
3. Sigmoid normalization: `score = 1.0 / (1.0 + math.exp(-logit))`.
4. Sort descending by score.

**Rerank pool:** The reranker scores a candidate pool of `min(pre_rerank_count, _rerank_pool_size(retrieval_top_k, reranker_top_k))` — a broad pool (default `retrieval_top_k * 2 = 100`) narrowed by the cross-encoder.

**Query choice:** The reranker uses the `effective_query` (first decomposed step or rewritten query), NOT the original question — the rewritten query is more search-optimized.

**No MMR:** The old MMR diversity step was removed. The reranker is the single relevance decision; a second diversity objective could discard complementary evidence needed for multi-part questions (`async_rag.py:725-728`).

**Truncation:** `if len(retrieved_chunks) > reranker_top_k: retrieved_chunks = retrieved_chunks[:reranker_top_k]`

### Step 7 Post-Rerank Confidence Gate (`async_rag.py:259-295`)

```python
gate_threshold = reranker_confidence_threshold if rerank_used else confidence_threshold
if retrieved_chunks[0].confidence >= gate_threshold:
    return None  # pass
# else: out-of-repository answer + low-confidence review record
```

**Why two thresholds:** Cross-encoder sigmoid scores cluster lower than embedding/fused confidence (relevant pairs commonly land ~0.10-0.15). So when a reranker ran, the gate compares against `reranker_confidence_threshold=0.10`; without one, it uses `confidence_threshold=0.18`. On rejection it returns the out-of-repository `Answer` and records a low-confidence review.

### Step 8 Context Compression (`async_rag.py:739-742`, `context_compression.py:46-75`)

```
Input:  chunks (list[RetrievedChunk]), query (str)
Output: list[RetrievedChunk] — deduplicated + relevance scored
```

**Runs AFTER reranking** (avoids wasted work — compressed results were previously discarded by re-fetch).

1. **Jaccard dedup** (line 56): If two chunks have token-set Jaccard similarity ≥ 0.85, keep only the first. Tokenization: `re.findall(r"[a-z0-9_]+", text.lower())`.
2. **Relevance scoring** (lines 59-73):
   - `cosine = |A∩B| / √(|A|·|B|)` — token-set cosine
   - `overlap = |query_tokens ∩ chunk_tokens| / |query_tokens|`
   - `score = cosine × 0.6 + overlap × 0.4`
3. Return top `max_chunks` by score.

**Note:** `context_compression_enabled` defaults to `False` (`settings.py:692`). When enabled, the `ContextAssembler`'s internal dedup is skipped to avoid double work (`async_rag.py:747`, `deduplicate=self.context_compressor is None`).

### Step 9 Context Assembly (`async_rag.py:745-770`, `context_assembler.py:31-111`)

```
Input:  chunks (list[RetrievedChunk]), max_context_chars=12000
Output: context_str (str), source_names (list[str]), dropped_records (list[dict])
```

#### 9a Dedup (if compressor didn't run)

Jaccard-like word overlap with 70% threshold. Employs a 12-word filler list to avoid false positives on common words.

#### 9b Lost-in-the-Middle Mitigation

Reorders chunks so the most relevant appear at BOTH ends of the context:

```
Original (by confidence): [A, B, C, D, E]
Rearranged:               [A, E, B, D, C]
```

Algorithm: alternate picking from left (highest confidence) and right (lowest confidence). Only activates when > 3 chunks remain after dedup.

#### 9c Formatting & Truncation

Each chunk formatted:
```
[source_name > section_header] chunk text
```
or (no section header):
```
[source_name] chunk text
```

Truncation happens at chunk boundaries — if adding the next chunk would exceed `max_context_chars`, it's dropped entirely (unless it's the first chunk). Dropped records are returned as `dropped_records` (used for provenance — budget-dropped segments must not be claimed as retrieved).

### Step 10 Prompt Building (`async_rag.py:784-794`, `prompt_builder.py:110-165`)

```
Input:  context_str (str), safe_question (str), intent (str)
Output: full_prompt (str)
```

#### 10a Pre-processing

```python
safe_question = PromptBuilder.sanitize_query(question)
# 1. Strip triple backticks
# 2. Convert "## " → "# " (prevent heading injection)
# 3. Truncate to 2000 chars
```

If PII redaction is enabled (default), the question is also PII-redacted before prompt building.

#### 10b Context Density Tag

```python
density_tag = "HIGH"   if word_count > 100 and alpha_ratio > 0.7
density_tag = "MEDIUM" if word_count > 30  and alpha_ratio > 0.5
density_tag = "LOW"    otherwise
```

Injected into context as: `<chunk>\n[DENSITY: {density_tag}]\n{context}\n</chunk>`

#### 10c Prompt Source: Langfuse-Managed

The RAG answer prompt is **managed in Langfuse** under `rag-answer` (`observability/langfuse_prompts.py:85`); `_RAG_PROMPT_TEMPLATE` is the offline fallback (`register_fallback`). Rendered via:

```python
get_langfuse_prompt("rag-answer").compile(
    system_role=self.system_role,
    output_format=output_format,
    instructions=instructions,
    tagged_context=tagged_context,
    question=question,
)
```

#### 10d Intent-Based Prompt Selection

| Intent | Instructions | Output Format |
|---|---|---|
| `code_example`, `api_lookup` | `_CODE_INSTRUCTIONS`: brief explanation + runnable example | `_CODE_OUTPUT_FORMAT` (fenced code block) |
| Documentation + code keywords in query | `_DOCUMENTATION_INSTRUCTIONS_WITH_CODE` | `_CODE_OUTPUT_FORMAT` (safety net) |
| Everything else | `_DOCUMENTATION_INSTRUCTIONS` | `_DOC_OUTPUT_FORMAT` (JSON) |

The system prompt is sent as a real `role: "system"` message — `build_chat_messages(prompt)` (`llm_client.py:38-53`) splits on `SYSTEM_BLOCK_SEPARATOR` and places the system block before the user turn.

### Step 11 LLM Generation (`async_rag.py:800-880`, `llm_client.py:179-260`)

#### 11a Client Selection

```python
def _select_llm_client(self, intent: str) -> LLMClientProtocol:
    if self.code_llm_client and intent in CODE_INTENTS:
        return self.code_llm_client  # qwen2.5-coder:7b or configured code model
    return self.llm_client           # per-purpose chain (answer)
```

#### 11b API Payload

```python
payload = {
    "model": self.model,
    "messages": build_chat_messages(prompt),    # system + user split
    "temperature": temp,                        # Very low — nearly deterministic
    "max_tokens": max_tokens,                   # per-purpose, from settings
    **self._extra_body,                         # Ollama: {options: {num_ctx: 4096, num_predict: 512}}
}
```

#### 11c HTTP Request (`_http_post`, lines 282-303)

- **URL:** `POST {base_url}/v1/chat/completions`
- **Retry:** Single attempt per call — no client-side retry, no circuit breaker. On failure the adaptive router marks a category-based cooldown and fails over to the next provider in `llm_fallback_order` (ending at Ollama).
- **Rate limiter:** Non-blocking pre-flight gate before the request — an over-limit provider (OpenRouter/NVIDIA/HF) is skipped without a paid call. `retry_after` from `429` headers is parsed.
- **Auth:** Bearer token in `Authorization` header (empty string for Ollama).
- **Keep-alive:** `keep_alive` (default `"10m"`) forwarded for Ollama.

#### 11d JSON Retry

If the intent expects JSON output but parsing fails (`intent not in CODE_INTENTS` and no `answer` extracted), the generation is **retried once** with a stricter instruction suffix (`rag-json-retry-suffix` prompt, `langfuse_prompts.py:193`).

#### 11e Response Processing

```python
content = body["choices"][0]["message"]["content"]
# Strip <think>...</think> tags (Ollama raw mode)
# Record token usage: LLMUsage(prompt_tokens, completion_tokens, model)
```

### Step 12 Post-Processing (in order)

#### 12a Code Syntax Validation (`async_rag.py:1422`, code intents only)

1. Extract Python code blocks: `re.findall(r"```python\n(.*?)```", answer_text, re.DOTALL)`
2. Try `ast.parse()` on each block.
3. If any have syntax errors, send fix prompt to LLM:
   ```
   The following Python code has syntax errors. Fix ONLY the code,
   keeping the same structure and imports. Return valid Python only.

   Broken code:
   ```python
   {invalid_block}
   ```
   ```
4. Replace broken block with fixed version. Only validates Python (Scala/SQL unchecked).

#### 12b Output Guardrails (`async_rag.py:903-910`, `output_guardrails.py:29-117`)

`OutputGuardrails.verify(raw_answer, source_count)`:

1. **Try JSON parse** (lines 68-77): Strip markdown fences, `json.loads()`, validate via Pydantic.
2. **Fallback plain text** (lines 80-93): Accept text with code blocks. Extract citations from `Sources: ...` / `Cited: ...` lines.
3. **Quality checks** (lines 96-108):
   - Reject if empty.
   - If `source_count > 0`: reject if < 20 chars.
   - Reject boilerplate: "I cannot answer", "outside my knowledge", "I don't have enough", "I am not able to", "beyond my knowledge".
   - `INSUFFICIENT_CONTEXT` passes through regardless.
4. **INSUFFICIENT_CONTEXT handling** (`async_rag.py:907-913`): if status is `INSUFFICIENT_CONTEXT` and `missing_info` present, appends `\n\nMissing information: {missing_info}` to the answer.
5. **Fallback on rejection**: If guardrails reject, the raw output is used unchanged (fail-open).

#### 12c PII Redaction (`async_rag.py:941-944`, `pii_redactor.py:99-130`)

**Patterns:** email, phone (US), SSN, credit card, IP address

**Modes:**
- `"full"` (default): Replace with `[REDACTED_EMAIL]`, `[REDACTED_SSN]`, etc.
- `"masked"`: Partial mask (e.g., `j***@***.com`)
- `"none"`: Passthrough

**Applied both:** pre-LLM (on question) and post-LLM (on answer). Config: `pii_redaction_enabled=True`, `pii_redaction_mode="full"` (`settings.py:703-704`).

#### 12d Citation Verification (`async_rag.py:946`, `structured_output.py:16-56`)

```python
parsed = parse_rag_response(answer_text)       # Extract answer + citations from JSON/text
source_names = [c.chunk.source_name for c in retrieved_chunks]
verified = verify_citations(parsed.citations, source_names)  # Keep if source matches
```

#### 12e Groundedness Verification (`async_rag.py:959-985`, `groundedness.py:66-216`)

Uses LLM-as-judge to verify each claim in the answer against the context.

**NLI Prompt:**
```
You are a groundedness verifier. Given an answer and supporting context,
determine which claims in the answer are supported by the context.

For each claim in the answer, output a JSON array of objects:
[{"claim": "...", "supported": true/false, "evidence": "..."}]
Return ONLY the JSON array, no preamble.

ANSWER:
{answer}

CONTEXT (excerpted from documentation):
{context_excerpt}

JSON array:
```

**Context excerpt budgeting:** `budget = min(3000, max(500, answer_chars * 6))` — 6x the answer length, capped at 3000 chars.

**Scoring:** `score = supported_claims / total_claims`. Threshold = `groundedness_threshold` (default 0.6).

**Fail-open:** If LLM is unavailable, falls back to text-overlap heuristic (Jaccard-ish token overlap with `min_support_score=0.3`). On guardrails failure, appends warning note: `"[Note: Some claims may not be fully supported by the documentation.]"`.

#### 12f Token/Cost Telemetry

Token usage + cost are recorded to Langfuse: `generation_span.update(usage_details=..., cost_details=...)`, `trace.update(metadata={"token_usage": ..., "cost_usd": ...})`, and a `cost_usd` numeric score (`async_rag.py:846-884`). Cost estimated via `_estimate_cost(prompt_tokens, completion_tokens, model)`.

### Step 13 Response Assembly

**Service layer** (`async_rag.py:977-985`):
```python
result = Answer(
    text=answer_text,
    sources=tuple(c.chunk for c in _final_chunks),
    confidence=retrieved_chunks[0].confidence,
    groundedness_score=...,
    stage_times={"rewrite": 45.2, "retrieval": 120.5, "rerank": 89.3, ...},
)
```

**Provenance** (`async_rag.py:358-395`): When `provenance=[]` is passed (evaluation), the service appends a structured record with `query_variants`, `fused`, `rerank`, `final_context`, `dropped`, `expected_urls`, `candidate_pool_size`, `stage_times` (schema version 1).

**API layer** (`routes.py:400-465`):
```python
AskResponse(
    answer=parsed.answer,              # Cleaned answer text
    sources=[SourceRef(source_name, title, url, snippet[:200])],
    confidence=float,                  # Top chunk confidence
    groundedness_score=float,          # From groundedness verifier
    citations=[{"source": "...", ...}],  # Verified citations
    metrics={
        "chunks_retrieved": len(sources),
        "confidence": confidence,
        "time_rewrite": 45.2,
        "time_retrieval": 120.5,
        "time_rerank": 89.3,
        "time_generation": 1234.5,
        "time_total": 1489.5,
    },
)
```

---

## 4. Streaming Flow

### Endpoint: `POST /api/v1/ask/stream` (`routes.py:466-527`)

Returns `StreamingResponse` with `media_type="text/event-stream"`.

### SSE Event Sequence (`async_rag.py:1179-1390`)

| # | Event Type | JSON Payload | When |
|---|---|---|---|
| 1 | `status` | `{"message": "Sanitizing query"}` | Always |
| 2 | `status` | `{"message": "PII redacted: ..."}` | If PII detected in question |
| 3 | `status` | `{"message": "Rewriting query"}` | If rewriter enabled |
| 4 | `status` | `{"message": "Intent: code_example"}` | If rewriter enabled |
| 5 | `status` | `{"message": "Retrieving documents"}` | Always |
| 6 | `status` | `{"message": "Retrieved 5 chunks"}` | Always |
| 7 | `status` | `{"message": "Reranking"}` | If reranker enabled |
| 8 | `status` | `{"message": "Generating answer"}` | Always |
| 9-N | `token` | `{"content": "The "}` | One per LLM token |
| N+1 | `done` | `{"text": "...", "confidence": 0.87}` | Final |
| — | `error` | `{"message": "Generation failed"}` | On failure |
| — | EOF | `data: [DONE]\n\n` | Always (finally block) |

### Key Differences from Non-Streaming

| Feature | Non-Streaming | Streaming |
|---|---|---|
| Cache check | ✅ Before retrieval | ❌ Not performed |
| Multi-query expansion | ✅ 2 LLM-generated variants | ❌ Single query |
| LLM rewriting | ✅ `async_rewrite()` | ✅ `async_rewrite()` (fallback to rule-based on error) |
| HyDE | ✅ Generated and embedded | ❌ Skipped |
| Query decomposition | ✅ Multiple sub-queries | ❌ Single query |
| Metadata filters | ✅ Extracted + applied | ❌ Not applied |
| Code syntax validation | ✅ Post-generation | ❌ Not performed |
| Groundedness | ✅ LLM NLI check | ❌ Not performed |
| Output guardrails | ✅ Pydantic validation | ✅ Applied (after streaming) |
| PII redaction | ✅ Pre+post | ✅ Pre+post |
| Cache storage | ✅ After generation | ✅ After generation (exact + semantic) |
| Provenance | ✅ (opt-in) | ❌ |

### Token Streaming Mechanism (`llm_client.py:320-360`)

```
LLMClient.generate_stream(prompt)
  │
  ├─ payload = {..., "stream": True}
  ├─ httpx.AsyncClient.stream("POST", url, json=payload)
  │
  ▼
async for line in response.aiter_lines():
  if line.startswith("data: "):
    if line.strip() == "data: [DONE]": break
    chunk = json.loads(line[6:])
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
      yield delta["content"]     ← yields ONE token
```

**Fallback:** On any HTTP error, falls back to non-streaming `generate()` and yields the complete answer as a single token. The stream is rate-limited by the same per-provider pre-flight gate (`try_acquire`).

**Post-stream:** PII redaction → `OutputGuardrails.verify` → cache write (exact + semantic) → Langfuse confidence score + trace end.

**SSE Formatter** (`async_rag.py:183`):
```python
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
```

---

## 5. Configuration Reference

### 5.1 Infrastructure Settings

| Setting | Default | File:Line | Description |
|---|---|---|---|
| `collection_name` | `data_engineering_docs` | `settings.py:360` | Qdrant collection name (legacy) |
| `active_collection_name` | `""` | `settings.py:364` | Active generation collection (set by `dec spark-activate`) |
| `qdrant_url` | `http://localhost:6333` | `settings.py:367` | Qdrant HTTP API |
| `ollama_base_url` | `http://localhost:11434` | `settings.py:368` | Ollama container |
| `embedding_ollama_base_url` | `""` | `settings.py:371` | Override for embed-only Ollama |
| `llm_ollama_base_url` | `""` | `settings.py:372` | Override for LLM-only Ollama |
| `redis_url` | `redis://:local_secure_password_123@localhost:6379/0` | `settings.py:375` | Redis (cache, queue, URL registry) |
| `langfuse_url` | `http://langfuse:3000` | `settings.py:376` | Langfuse observability |

### 5.2 Provider Selection

| Setting | Default | Options | File:Line |
|---|---|---|---|
| `llm_provider` | `ollama` | `ollama`, `openrouter`, `nvidia`, `groq`, `gemini`, `cloudflare`, `cerebras`, ... | `settings.py:422` |
| `llm_model` | `llama3.2:3b` | Any model name | `settings.py:423` |
| `embedding_provider` | `ollama` | `ollama`, `openrouter`, `nvidia`, `huggingface`, `local-hf`, `gemini` | `settings.py:424` |
| `embedding_model_name` | `nomic-embed-text` | Any Ollama model | `settings.py:403` |
| `local_hf_embedding_model` | `nvidia/Nemotron-3-Embed-1B-BF16` | Any HF SentenceTransformer | `settings.py:408` |

### 5.3 Provider API Keys & Fallback Chains

| Setting | Default | Description |
|---|---|---|
| `openrouter_api_key` / `openrouter_model` / `openrouter_embedding_model` | `openrouter/free` / `nvidia/nemotron-3-embed-1b:free` | OpenRouter (`settings.py:449-454`) |
| `nvidia_api_key` / `nvidia_model` / `nvidia_embedding_model` | `nvidia/nemotron-3-embed-1b` | NVIDIA (`settings.py:457-464`) |
| `huggingface_api_key` / `huggingface_embedding_model` / `huggingface_base_url` | `nvidia/Nemotron-3-Embed-1B-BF16` / `https://router.huggingface.co/hf-inference` | HF serverless (`settings.py:506-511`) |
| `llm_fallback_order` | `["cloudflare", "groq", "nvidia", "gemini", "cerebras", "ollama"]` | LLM failover order (`settings.py:520-522`) |
| `embedding_fallback_order` | `["nvidia", "openrouter", "ollama"]` | Embedding failover order (`settings.py:526`) |
| `llm_fallback_call_timeout` | 30 | per-attempt timeout for non-primary fallback providers (`settings.py:524`) |
| `provider_cooldown_seconds` | 60 | cooldown after a provider fails (`settings.py:528`) |

### 5.4 Rate Limits

| Setting | Default | Description |
|---|---|---|
| `openrouter_rpm_limit` / `openrouter_rpd_limit` | 18 / 900 | OpenRouter sliding-window rate limits |
| `nvidia_rpm_limit` / `nvidia_rpd_limit` | 36 / 900 | NVIDIA |
| `huggingface_rpm_limit` / `huggingface_rpd_limit` | 4 / 900 | HF serverless (~270 req/hr free tier) |

### 5.5 Per-Purpose LLM Overrides

All empty string = fall back to global `llm_provider`/`llm_model`:

| Purpose | Settings Key | Default Provider | Used For |
|---|---|---|---|
| Answer | `answer_llm_provider/model` | `openrouter` | Main answer generation |
| Rewrite | `rewrite_llm_provider/model` | `groq` | Query rewriting |
| Groundedness | `groundedness_llm_provider/model` | `groq` | Claim verification |
| Intent | `intent_llm_provider/model` | `groq` | Intent classification |
| Enrichment | `enrichment_llm_provider/model` | `""` | Contextual chunk enrichment |
| Evaluation | `evaluation_llm_provider/model` | `groq` | Faithfulness evaluation |
| Code | `code_llm_provider/model` | `""` | Code-specific answers |

### 5.6 RAG Pipeline Settings

| Setting | Default | Range | File:Line |
|---|---|---|---|
| `retrieval_top_k` | 50 | 1-100 | `settings.py:620` |
| `reranker_enabled` | True | — | `settings.py:621` |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — | `settings.py:622` |
| `reranker_top_k` | 30 | 1-100 | `settings.py:623` |
| `max_context_chars` | 12000 | 500-100000 | `settings.py:624` |
| `max_context_tokens` | 4096 | — | `settings.py:693` |
| `max_expansion_queries` | 2 | 0-5 | `settings.py:625` |
| `context_compression_ratio` | 0.8 | — | `settings.py:626` |
| `groundedness_threshold` | 0.6 | 0.0-1.0 | `settings.py:627` |
| `confidence_threshold` | 0.18 | 0.0-1.0 | `settings.py:628` |
| `reranker_confidence_threshold` | 0.10 | 0.0-1.0 | cross-encoder gate (`settings.py:633`) |
| `hybrid_search_enabled` | True | — | `settings.py:677` |
| `hybrid_rrf_k` | 60 | 10-200 | `settings.py:678` |
| `semantic_cache_threshold` | 0.95 | 0.5-1.0 | `settings.py:680` |
| `semantic_cache_ttl` | 3600 | seconds | `settings.py:681` |
| `query_rewrite_enabled` | True | — | `settings.py:688` |
| `groundedness_enabled` | True | — | `settings.py:689` |
| `intent_classification_llm_enabled` | False | — | LLM fallback for intent (`settings.py:690`) |
| `context_compression_enabled` | False | — | `settings.py:692` |
| `input_guardrails_enabled` | True | — | Indirect prompt injection guard (`settings.py:705`) |
| `rbac_enabled` | False | — | Document-level access control (`settings.py:707`) |
| `temperature` | 0.05 | 0.0-1.0 | per-purpose, in llm_client |

### 5.7 Chunking Settings

| Setting | Default | File:Line |
|---|---|---|
| `chunking_strategy` | `sentence_preserving` | `settings.py:610` |
| `chunk_size_words` | 375 | `settings.py:611` |
| `chunk_overlap_words` | 90 | `settings.py:612` |
| `min_semantic_similarity` | 0.5 | `settings.py:614` |
| `max_chunk_words` | None (→ 1.5× chunk_size_words = 562) | `settings.py:615` |
| `enable_semantic_chunking` | True | `settings.py:617` |
| `embedding_batch_size` | 64 | `settings.py:397` |

### 5.8 Crawl Settings

| Setting | Default | File:Line |
|---|---|---|
| `max_pages_per_source` | 100000 | `settings.py:651` |
| `max_pages_hard_cap` | 100000 | `settings.py:652` |
| `recovery_max_pages` | 100000 | `settings.py:653` |
| `crawl_delay_seconds` | 0.3 | `settings.py:650` |
| `crawl_async_concurrency` | 10 | `settings.py:666` |
| `crawl_async_max_concurrency` | 40 | `settings.py:667` |
| `crawl_async_per_domain_concurrency` | 2 | `settings.py:668` |
| `crawl_async_conditional_get` | True | `settings.py:669` |
| `frontier_max_attempts` | 3 | `settings.py:672` |

### 5.9 Ollama Settings

| Setting | Default | File:Line |
|---|---|---|
| `ollama_timeout_seconds` | 180 | `settings.py:635` |
| `ollama_connect_timeout_seconds` | 5 | `settings.py:636` |
| `ollama_pool_timeout_seconds` | 5 | `settings.py:637` |
| `ollama_keep_alive` | `"10m"` | `settings.py:638` |
| `ollama_num_ctx` | 4096 | `settings.py:643` |
| `ollama_num_predict` | 512 | `settings.py:646` |

### 5.10 Index Generation Settings

| Setting | Default | Description |
|---|---|---|
| `index_generation` | `""` | Generation identity for reproducible corpus build |
| `index_require_hybrid` | True | Generation must have dense+BM25 to validate |
| `index_validation_min_points` | 1 | Minimum points for a valid generation |

### 5.11 Docker Services

| Service | Image | Port(s) |
|---|---|---|
| redis | `redis:7-alpine` | 6379 |
| qdrant | `qdrant/qdrant:v1.18.3` | 6333, 6334 |
| ollama | `ollama/ollama:0.32.4` | 11434 |
| minio | `minio/minio:RELEASE.2025-09-07` | 9000, 9001 |
| minio-init | `minio/mc:RELEASE.2025-08-13` | — |
| clickhouse | `clickhouse/clickhouse-server:26.4-alpine` | 8123, 9000 |
| langfuse | `langfuse/langfuse:4` | 3000 |
| langfuse-worker | `langfuse/langfuse-worker:4` | — |
| postgres (infra) | `postgres:16-alpine` | 5432 |
| postgres (app) | `postgres:16-alpine` | 5433 |
| backend-api | `de_copilot_base_image` (custom) | 8000 |
| celery_worker | `de_copilot_base_image` (custom) | (none) |

---

## 6. Observability & Caching

### 6.1 Telemetry Pipeline (`observability/telemetry.py:74-102`)

Priority chain:
```
OTelTelemetryTracer → LangfuseTelemetryTracer → NoOpTelemetryTracer
```

**OpenTelemetry** (`otel_telemetry.py:35-58`):
- Reads `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`)
- Does TCP health check before importing SDK
- Creates `TracerProvider` → `BatchSpanProcessor` → `OTLPSpanExporter`
- Creates spans named after pipeline stages

**Span hierarchy** in `answer()`:
```
trace: "rag-query-pipeline"
  ├── span: "query-rewriting"     (intent + decomposition + HyDE + expansion)
  ├── span: "retrieval"           (embedding + vector DB calls)
  ├── span: "reranking"           (cross-encoder)
  └── generation: "llm-generation"  (LLM API call, token + cost details)
```

Cache hits trace under `rag-query-pipeline-cache-hit`; streaming under `rag-query-pipeline-stream`.

**Span attributes:** `app.input` (truncated 2000), `app.model`, `app.output` (truncated 5000), `app.span_type`. Trace metadata carries `git_sha`, `app_env`, token usage, and `cost_usd`. A `confidence` numeric score is recorded per trace.

### 6.2 Token & Retrieval Tracking

**TokenTracker** (`token_tracker.py:18-55`):
- Thread-safe (`threading.Lock`)
- Tracks: prompt tokens, completion tokens, call count, per-model breakdown

**RetrievalTracker** (`token_tracker.py:58-109`):
- Ring buffer (`deque`, max 10,000 entries)
- Computes: mean, p50, p95, p99 of retrieval scores

- Exposed via Prometheus `/metrics` endpoint (`api/app.py:107-139`):
  ```
  rag_retrieval_score{quantile="0.5"}
  rag_retrieval_score{quantile="0.95"}
  rag_retrieval_score{quantile="0.99"}
  rag_retrieval_score{quantile="mean"}
  rag_retrieval_queries_total
  rag_token_usage_total{type="prompt"}
  rag_token_usage_total{type="completion"}
  rag_llm_calls_total
  ```

### 6.3 Cache Architecture (`query_cache.py:36-389`)

```
┌───────────────────────────────────────────────┐
│              QueryCache                       │
│                                                │
│  ┌─────────────┐  ┌────────────────────────┐  │
│  │ L1 Exact    │  │ L2 Semantic            │  │
│  │ SHA-256 key │  │ NumPy dot product      │  │
│  │ LRU 1024    │  │ threshold 0.95         │  │
│  │             │  │ LRU deque 512 + TTL    │  │
│  └──────┬──────┘  └───────────┬────────────┘  │
│         │                     │               │
│         └────────┬────────────┘               │
│                  ▼                            │
│  ┌──────────────────────────────┐             │
│  │ L3 Redis                     │             │
│  │ SET rag:cache:exact:{hash}   │             │
│  │ HSET rag:cache:semantic:{n}  │             │
│  └──────────────────────────────┘             │
└───────────────────────────────────────────────┘
```

**Cache key naming:**
- Exact: `rag:cache:exact:{sha256_hex}` (STRING, TTL 3600s)
- Semantic: `rag:cache:semantic:{counter}` (HASH with query, embedding, answer, TTL 7200s)
- Counter: `rag:cache:semantic:counter` (INCR, no TTL)
- Scope-fingerprinted when `CacheScope` is set (e.g. private/user-scoped)

**Read path** (`aget`, query_cache.py:251-339):
1. L1 exact match (in-memory `OrderedDict`)
2. L2 semantic match (in-memory `deque` + NumPy dot product)
3. Redis exact match (`GET`)
4. Redis semantic match (`SCAN` + `HGETALL` + cosine similarity)
5. On hit: backfill in-memory caches from Redis

**Write path** (`aset_exact` + `aset_semantic`):
1. Write to L1/L2
2. Write to Redis with TTL

**Alternative:** `redis_query_cache.py` provides a Redis-only `RedisQueryCache` (standalone sync/async API) used where the in-process `QueryCache` isn't wired.

### 6.4 Drift Detection (`drift_detector.py:65-172`)

Stores eval snapshots in `data/eval_history.jsonl`. Each snapshot contains timestamp, metrics dict, and eval dataset hash.

**Baseline:** Average of snapshots over last N days (default 7).
**Drift detection:** If current metric < baseline - threshold. Default thresholds:
- `faithfulness`: 0.8
- `context_recall`: 0.7
- `context_precision`: 0.6
- `answer_relevancy`: 0.7
- `overall`: 0.7

Wired into `dec evaluate` CLI command.

### 6.5 Health Checks (`provider_health.py`, `health_check.py`)

Provider health is scored as a weighted blend of success rate, latency, and recency:
```
health_score = success_rate_weight(0.6) + latency_weight(0.2) + recency_weight(0.2) - consecutive_failure_penalty(0.3)
```

Used by the fallback chain to skip unhealthy providers before attempting a call (`_provider_gate`, `provider_fallback.py:243`).

---

## 7. Provider Fallback Chain

**File:** `infrastructure/provider_fallback.py` (346 lines)

All LLM and embedding calls route through `ProviderFallbackChain[T, R]` — never call a provider directly. Build via `build_llm_fallback_chain()` / `build_embedding_fallback_chain()` in `factory.py`.

```
        ProviderFallbackChain.execute(request)
                    │
                    ▼
        for provider in fallback_order:
            │
            ├─ _provider_gate()  ── is it in cooldown? over rate limit? unhealthy?
            │        │ (skipped if gated)
            │        ▼
            ├─ _call_with_health()  ── record success/failure + latency
            │        │
            │        ▼
            │   Provider succeeds?  ── yes → return response
            │        │
            │        no → categorize error (rate-limit / auth / 5xx / timeout)
            │        │
            │        ▼
            └─ next provider in order (until chain exhausted)
```

**Providers in the default LLM chain:** `cloudflare → groq → nvidia → gemini → cerebras → ollama`. The last entry (Ollama) is the **degraded fallback** — it always ends there so answers still work with only local infra.

**Providers in the default embedding chain:** `nvidia → openrouter → ollama`. `.env` may extend it (e.g. `["nvidia", "openrouter", "huggingface", "local-hf"]`).

### 7.1 Error Categorization

`ErrorCategorizer` (`provider_fallback.py:67-111`) maps exceptions to `ProviderErrorCategory`:
- `RATE_LIMIT` (429, rate limit headers)
- `AUTH` (401/403)
- `SERVER_ERROR` (5xx)
- `TIMEOUT`
- `NETWORK`
- `INVALID_REQUEST`

Category determines whether the provider gets a cooldown and whether the request is retried on the next provider.

### 7.2 Provider Health & Cooldown

- `_provider_gate(provider)` (line 243): returns `(allowed, reason, wait_seconds)`. Skips providers in cooldown, over rate limit, or with degraded health.
- `_call_with_health(provider, request)` (line 270): records success/failure + latency into the health tracker after each attempt.
- `_degraded_available(provider)` (line 264): checks whether the local degraded fallback (Ollama) is reachable.

### 7.3 Rate Limiter

`SlidingWindowRateLimiter` (`infrastructure/rate_limiter.py`) — non-blocking pre-flight gate per provider:
```python
if rate_limiter is not None and not await rate_limiter.try_acquire():
    raise RateLimitError(wait_until=rate_limiter.wait_until_available())
```
Limits come from `{provider}_rpm_limit` / `{provider}_rpd_limit` settings. `429` `Retry-After` headers are parsed (`parse_retry_after`) and contribute to cooldown.

### 7.4 Fallback Embedder

`infrastructure/fallback_embedder.py` — thin `FallbackEmbedder` adapter (embeds with `inner` provider): fallback logic lives entirely in the chain; the adapter only bridges `embed_texts`/`embed_query` calls.

---

## 8. Spark Index Generations

Spark docs are built as **named, validated, atomically-switched index generations** — a reproducible-corpus approach distinct from incremental ingestion.

**Files:** `services/spark_index_builder.py`, `services/spark_chunker.py`, `services/spark_metadata.py`, `services/spark_rendered_chunker.py`, `services/spark_rendered_builder.py`, `infrastructure/spark_source_resolver.py`

### 8.1 Lifecycle

```
dec spark-build --generation <gen>     → build full dense+BM25 generation in Qdrant
dec spark-validate --generation <gen>  → validate built generation
dec spark-activate --generation <gen>  → atomically switch alias to validated gen
dec evaluate --spark                   → spark retrieval-recall evaluation
```

### 8.2 Build (`SparkIndexBuilder.build`)

1. **Pin source:** Resolve the Spark version + commit via `spark_source_resolver.py` (config in `config/spark_sources.json`, `config/spark_rendered_sources.json`).
2. **Chunk:** `spark_chunker.py` / `spark_rendered_chunker.py` produce generation-aware chunks from the rendered docs.
3. **Embed + BM25:** dense vectors + sparse BM25 tokenization inline.
4. **Write to a generation collection:** `data_engineering_docs__{source}-{generation}` (via `_spark_generation_collection(generation)`).
5. **Validate:** `validate_index_generation(len(normalized))` (`spark_index_builder.py:241`) enforces `index_require_hybrid=True` and `index_validation_min_points`.
6. **Report:** `IndexBuildReport(generation, chunk_count, coverage_records, validation)`.

### 8.3 State Management

- `.index_state/active.json` — current active generation
- `.index_state/history.jsonl` — activation history
- `.index_state/validation-{generation}.json` — per-generation validation artifacts
- `dec spark-activate` atomically switches the Qdrant collection alias (`data_engineering_docs`) to the validated generation collection.

### 8.4 Verification

`dec spark-validate --generation <gen>` runs retrieval-recall checks against the built generation. `dec evaluate --spark` reports faithfulness / context recall / precision / answer relevancy with drift detection (Section 6.4).

---

## 9. Claude Docs Ingestion (llms.txt)

**File:** `services/claude_docs_ingestion.py`

`dec ingest-claude-docs --site all|platform|code` ingests Anthropic Claude documentation in-process (no Celery) using the **llms.txt** convention.

```
Claude docs (llms.txt / llms-full.txt)
       │
       ▼
parse_llms_index(url, url_prefix)  ── parse markdown links → [(title, url)]
       │
       ▼
fetch_markdown_files(entries)      ── concurrent download with backoff
       │
       ▼
build_parsed_documents(site, entries, root_dir)  ── strip frontmatter → ParsedDocument
       │
       ▼
_chunk_embed_upsert(docs, chunker, embedder, store)
       │
       └─ batched: chunk → embed (with retry) → upsert → flush every N
```

**Key details:**
- Sources: `SOURCE_CODE` (docs.claude.com code API docs) and `SOURCE_PLATFORM` (Anthropic platform docs).
- Frontmatter is stripped (`strip_frontmatter`, line 101) before parsing.
- Uses the configured chunker (default HeaderAwareChunker — chosen specifically because Claude platform pages have nested `###` headings).
- `_embed_batch_with_retry` (line 297) routes embedding through the fallback chain with per-batch retry.
- Rendered markdown is stored directly (no HTML crawl), so parse failures are rare.
- CLI: `dec ingest-claude-docs --site all` (in-process, requires Qdrant + embedder).

**Supporting CLI commands:**
- `dec reenrich --source "Name"` — re-enrich failed summaries
- `dec retry-failed --source "Name" --category fetch` — retry failed pages
- `dec unskip --source "Name"` — unskip pages
- `dec reset-index` — full clean rebuild (Qdrant + BM25 + Redis + PG)
- `dec reset-qdrant` — recreate Qdrant collection + BM25 only
- `dec reset-crawler-db` — reset Redis/PG crawl state (keep Qdrant)

---

## 10. Indirect Prompt Injection Guard

**File:** `services/input_guardrails.py`

Retrieved documentation is untrusted input. Before any chunk reaches the prompt builder, `InputGuardrails.scan_chunks(retrieved_chunks)` inspects each chunk for **indirect prompt injection** patterns — embedded instructions that attempt to override the system prompt (e.g. "ignore previous instructions", prompt-style payloads hidden in doc text).

```python
scan_result = self.input_guardrails.scan_chunks(retrieved_chunks)
retrieved_chunks = scan_result.kept
```

**Behavior:**
- **Kept:** chunks that pass the scan continue through the pipeline unchanged.
- **Rejected:** injection-laden chunks are dropped. If ALL chunks are rejected, the pipeline returns the out-of-repository answer ("I cannot answer this question because it is outside my knowledge repository.") rather than generating from a poisoned context.
- Enabled via `input_guardrails_enabled=True` (`settings.py:705`).

Complements `prompt_injection.py` (query-side injection detection on user input) and the prompt-builder's own heading sanitization (`##` → `#`).

---

> **Document version:** 2026-08-12  
> **Codebase ref:** commit `ffcdfd1`  
> **Generated from:** deep architectural audit of the DataEngineeringCopilot codebase
