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

**One-sentence summary:** A RAG system that crawls data-engineering documentation sites, stores them as hybrid searchable vectors in Qdrant, rewrites user questions into optimal search queries, retrieves relevant context, and generates grounded answers via a local LLM.

**Key design decisions:**
- No LangChain/LlamaIndex — pure Python with structural-typing protocols (`domain/protocols.py`)
- Async-first — `httpx.AsyncClient`, `asyncio.TaskGroup`, `aiohttp`
- Per-purpose LLM routing — different models for answer, rewrite, groundedness, intent, code generation
- Annotate-only, fail-open — groundedness and guardrails never block answers, only annotate

**Tech stack:**
- Python 3.12, Pydantic, FastAPI, Celery
- Qdrant (vector DB), Redis (cache/queue), PostgreSQL (crawl frontier)
- Ollama (LLM + embeddings), OpenRouter (fallback LLM)
- Langfuse + OpenTelemetry (observability)
- testcontainers (integration testing)

---

## 2. Ingestion Pipeline: URL to Vectors

### Step 0 API Entry (`api/routes.py:57-117`)

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

**Key guard:** Atomic `SETNX ingestion:dispatch_lock` (line 69) prevents concurrent ingestion runs. If a task is already `PROCESSING` or `DISPATCHED`, returns 409.

**Config:** `max_pages` defaults to `settings.max_pages_per_source = 100000` (`settings.py:305`), clamped 1-100000.

### Step 1 Celery Task (`workers/tasks.py:88-131`)

The Celery task `async_ingest_task` is configured with:
```python
autoretry_for=(ConnectionError, TimeoutError), max_retries=3, countdown=10, retry_backoff=True
```

It calls:
```python
service = build_async_ingestion_service()     # factory.py:334
asyncio.run(service.ingest(source_names=..., max_pages_per_source=..., on_event=...))
```

### Step 2 Factory Wiring (`factory.py:334-387`)

`build_async_ingestion_service()` wires all 10+ components:

| Component | Source | Key Config |
|---|---|---|
| `redis_client` | `aioredis.from_url(redis_url)` | `settings.redis_url` |
| `AsyncDocumentationCrawler` | `build_async_crawler()` | concurrency=20, delay=0.2s, domain limit=3 |
| `MarkdownParser` | direct instantiation | min_words=40 |
| `DocumentChunker` | `build_chunker()` | chunk_size=1875 chars, overlap=450 chars |
| `AsyncOllamaEmbeddings` | `build_embedder()` | model=`nomic-embed-text`, dim=768 |
| `AsyncQdrantVectorStore` | direct instantiation | collection=`data_engineering_docs`, hybrid=on |
| `ContextualChunkEnricher` | `LLMContextSummarizer` wrapper | batch_size=20 |
| `ApiDocExtractor` | direct instantiation | enabled always |
| `CodeBlockParser` | direct instantiation | enabled always |
| `ChunkFilter` | direct instantiation | enabled always |

### Step 3 Source Selection (`async_ingestion.py:459-481`)

Sources are loaded from `data_engineering_copilot/config/documentation_sources.json` via `settings.py:233-237`:

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

### Step 4 Crawling (`infrastructure/async_crawler.py:143-233`)

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

**Conditional GET** — Two-phase per-page workflow:
1. **HEAD** (line 302-320): Sends `If-None-Match: {etag}` / `If-Modified-Since: {last_modified}`. 304 = skip.
2. **GET** (line 322-351): Full fetch with 3x retry, exponential backoff. Returns HTML.

**Link discovery** (line 405-448): Fast `HTMLParser` based link extraction, BeautifulSoup fallback. Filters by scheme (http/https), `allowed_domains`, and `url_prefixes`.

**Output per page:** `RawDocument(source_name, url, html)` — the HTML string of a single page.

### Step 5 HTML Parsing (`infrastructure/html_to_markdown.py:45-72`)

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

**Title extraction** (line 65-72): prefers `<h1>` text, falls back to `<title>`, falls back to URL.

**Output:** `ParsedDocument(source_name, title, url, text=markdown_text)` or `None` (if < 40 words).

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

**File:** `services/chunker.py:69-100`
**Config:** `chunk_size_chars=1875` (= `chunk_size_words * 5`), `chunk_overlap_chars=450`

```
Input:  ParsedDocument
Output: list[DocumentChunk]
ID fmt: {source_slug}:{url_sha1_10}:{index:04d}
         e.g. "spark_docs:a1b2c3d4e5:0003"
```

**Language-aware splitting** (line 76): Detects language from URL path:
- `/api/python/` or `/pyspark` → `Language.PYTHON` (class/def boundaries)
- `/api/scala/` → `Language.SCALA`
- `/api/java/` → `Language.JAVA`
- Otherwise → generic `RecursiveCharacterTextSplitter` on `["\n\n", "\n", " ", ""]`

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter`.

#### 7b HeaderAwareChunker

**File:** `services/header_aware_chunker.py:75-250`
**Config:** `chunk_size_words=375`, `overlap_words=90`, `min_chunk_words=37`

```
Input:  ParsedDocument
Output: list[DocumentChunk] with enriched metadata
ID fmt: {source}:{url_sha1}:hdr:{index:04d}
```

**Algorithm:**
1. **Parse headers** (line 96-145): Regex `^(#{1,6})\s+(.*)` extracts heading hierarchy. Maintains a stack to track `heading_path` (e.g., `("Overview", "Installation")`).
2. **Merge by topic** (line 151-240): Flushes when parent boundary changes or word count exceeds `chunk_size_words`. Adds overlap from previous chunk's tail.

**Output fields populated:**
- `section_header` — e.g. `"Requirements"`
- `chunk_type` — `"text"`, `"code"`, or `"mixed"`
- `word_count` — actual word count
- `heading_path` — tuple of all ancestor headers

#### 7c SemanticChunker

**File:** `services/semantic_chunker.py:89-348`
**Config:** `min_semantic_similarity=0.5`, `chunk_size_words=375`, `overlap_words=90`

**Special path in ingestion** (`async_ingestion.py:119-124`):
```python
sentences = self.chunker.extract_sentences(parsed.text)      # NLTK sent_tokenize
embeddings = await self.embeddings.embed_texts(sentences)     # Embed ALL sentences first
chunks = await self.chunker.chunk(parsed, precomputed_embeddings=embeddings)
```

**Algorithm:**
1. **Sentence extraction** via NLTK `sent_tokenize`
2. **Embed each sentence** (precomputed or on-the-fly)
3. **Greedy clustering** (line 147-204): For each sentence, compute cosine similarity to cluster centers. Add to most similar cluster if ≥ 0.5, else start new cluster.
4. **Merge clusters** (line 206-304): Merge respecting word-count limits, adding overlap at boundaries.

ID format: `{source}:{url_sha1}:semantic:{index:04d}`

### Step 8 Enrichment (`async_ingestion.py:200-207`)

After chunking, three synchronous enrichers run in a thread executor:

1. **ChunkFilter** (`services/text_filter.py`): Drops low-quality chunks (too short, boilerplate text, noise patterns like log lines, excessive brackets).
2. **ApiDocExtractor** (`services/api_extractor.py`): Identifies API documentation chunks via regex patterns (`function_name(`, `class Name:`, `def method`) and tags them with `chunk_type="api"`. Prepends structured metadata: `[API: Module: X | Method: Y | Params: ... | Returns: ...]`.
3. **CodeBlockParser** (`services/code_block_parser.py`): Identifies fenced code blocks, tags with `chunk_type="code"`, optionally splits large blocks at function/class boundaries.

Also runs **ContextualChunkEnricher** (`services/contextual_chunk_enricher.py:83-158`) — uses an LLM to generate a document-level summary and prepends it to each chunk. Config: `contextual_enrichment_enabled=True` (`settings.py:217`).

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

### Step 10 Embedding (`infrastructure/async_embeddings.py:42-124`)

```
Input:  list[str] — chunk texts (up to 256 per batch)
Output: list[list[float]] — dense vectors, 768-dim (nomic-embed-text)
```

**Provider:** Ollama via `POST /api/embed`:
```python
json={"model": "nomic-embed-text", "input": texts}
```

**Batch slicing:** Texts split into `embedding_batch_size=256` sub-batches (`settings.py:109`). Processed sequentially.

**Retry:** 3 attempts, exponential backoff (1-10s), retries on `TimeoutException`, `ConnectError`, `OSError`.

**Dimension validation** (line 95): Checks each embedding length against `embedding_model_dimensions` lookup (`settings.py:117-124`):

| Model | Dimensions |
|---|---|
| `nomic-embed-text` | 768 |
| `mxbai-embed-large` | 1024 |
| `snowflake-arctic-embed2` | 1024 |
| `llama3.2:3b` | 3072 |
| `nvidia/nemotron-3-embed-1b` | 2048 |

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

**Sub-batching:** 256 points per batch (`line 179`) to avoid Qdrant's 32MB payload limit.

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
  │ CachedEmbedder → Ollama API  │
  │ 768-dim vector               │
  └──────────┬───────────────────┘
             │ query_embedding (list[float])
             ▼
  ┌──────────────────────────────┐
  │ 4. VECTOR SEARCH (Qdrant)    │
  │ Dense cosine (top_k * 2)     │
  │   + BM25 sparse (top_k * 2)  │
  │   → RRF fusion (k=60)        │
  └──────────┬───────────────────┘
             │ list[RetrievedChunk]
             ▼
  ┌──────────────────────────────┐
  │ 5. RERANKING                 │
  │ Cross-encoder sigmoid        │
  │ → MMR diversity (λ=0.5)     │
  └──────────┬───────────────────┘
             │ top-5 chunks
             ▼
  ┌──────────────────────────────┐
  │ 6. CONTEXT COMPRESSION       │
  │ Jaccard dedup (0.85)         │
  │ → Relevance re-ranking       │
  └──────────┬───────────────────┘
             │
             ▼
  ┌──────────────────────────────┐
  │ 7. CONTEXT ASSEMBLY          │
  │ Lost-in-middle mitigation    │
  │ → Chunk boundary truncation  │
  └──────────┬───────────────────┘
             │ context_str (4000 max chars)
             ▼
  ┌──────────────────────────────┐
  │ 8. PROMPT BUILDING           │
  │ Density tag + intent-based   │
  │ output format                │
  └──────────┬───────────────────┘
             │ full prompt
             ▼
  ┌──────────────────────────────┐
  │ 9. LLM GENERATION            │
  │ POST /v1/chat/completions    │
  │ Temperature 0.05, 1 attempt  │
  └──────────┬───────────────────┘
             │ answer_text (raw str)
             ▼
  ┌──────────────────────────────┐
  │10. POST-PROCESSING           │
  │ Code validation → Guardrails │
  │ → PII redact → Citations     │
  │ → Groundedness → Cache store │
  └──────────┬───────────────────┘
             │ Answer(text, sources, confidence, ...)
             ▼
  ┌──────────────────────────────┐
  │11. RESPONSE                  │
  │ AskResponse with metrics     │
  │ and stage_times              │
  └──────────────────────────────┘
```

### Step 1 Cache Check (`async_rag.py:83-96`)

```
Input:  question (str)
Lookup: aget(question, query_embedding)
Output: str | None (cached answer)
```

**Three-layer cascade:**
1. **L1 Exact** (`query_cache.py:66-70`): SHA-256 of normalized query. LRU cache (`OrderedDict`, max 1024 entries).
   - Normalization: lowercase, strip, remove non-word chars, collapse whitespace.
2. **L2 Semantic** (`query_cache.py:82-109`): NumPy batch dot product against cached unit vectors. Threshold = 0.95 (config: `semantic_cache_threshold`, `settings.py:207`). LRU cache (`deque`, max 512 entries). TTL = 3600s.
3. **L3 Redis** (`query_cache.py:127-179`): Async Redis client. Exact: `GET rag:cache:exact:{sha256}`. Semantic: `SCAN rag:cache:semantic:*` → `HGETALL` → cosine similarity. On hit, backfills L1/L2.

On cache hit, returns immediately with `Answer(text=cached, confidence=1.0)` — skips the entire pipeline.

### Step 2 Query Rewriting (`query_rewriting.py:203-231`, `async_rag.py:107-128`)

```
Input:  question (str)
Output: RewrittenQuery(intent, decomposed_steps, hyde_query)
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

If no regex matches and `intent_classification_llm_enabled=True` (`settings.py:212`), falls back to LLM:
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

#### 2d HyDE Generation (`query_rewriting.py:264-348`)

HyDE = Hypothetical Document Embedding. Generate a hypothetical answer, then embed THAT instead of the question.

**Prompt:**
```
Write a short, authoritative paragraph that would perfectly answer
the following question. Do not address the user directly.

Question: {query}
```

On success: the generated text is embedded and used for retrieval.
On failure (or streaming path): falls back to embedding the original question.

#### 2e Multi-Query Expansion (`query_rewriting.py:279-301`)

**Prompt:**
```
Generate {max_variations} different search queries that would find
the same information as this question. Return ONLY the queries,
one per line, no numbering.

Original question: {query}

Variations:
```

Called with `max_variations=2`. Results are appended to `all_queries`.

**Final `all_queries` list (`async_rag.py:108-128`):**
```python
all_queries = [question]          # always includes original
+ decomposed_steps                # 1-3 sub-queries from intent
+ expanded_variations             # up to 2 LLM-generated variants
```

Each query in `all_queries` gets its own embedding + vector store query.

### Step 3 Query Embedding (`async_rag.py:136`, `infrastructure/embedding_cache.py:43-66`)

```
Input:  effective_query (str)
Output: query_embedding (list[float], 768-dim)
```

The embedder is wrapped in `CachedEmbedder` (`factory.py:456-459`):
```python
embedder = CachedEmbedder(embedder)  # LRU cache, max 1024 entries
```

**Cache key:** SHA-256 of lowercased+stripped text.

**Cache hit:** Returns cached `list[float]` immediately.
**Cache miss:** Delegates to the inner embedder:
- **Ollama** (`async_embeddings.py:113-118`): `POST /api/embed` with `{"model": "nomic-embed-text", "input": [text]}`
- **OpenRouter/NVIDIA** (`async_openai_compatible_embeddings.py:188-192`): `POST /embeddings` with `{"model": "...", "input": [text]}`, truncating to 3800 tokens.

### Step 4 Vector Search (`infrastructure/async_qdrant_store.py:206-329`)

```
Input:  query_embedding (list[float])
        query_text (str) — for BM25 sparse
        source_filter (list[str] | None)
        chunk_type_filter (str | None)
        top_k (int)
Output: list[RetrievedChunk]
```

#### 4a Filter Construction (lines 240-257)

```python
filter_conditions = []
if source_filter:
    filter_conditions.append(FieldCondition(key="source_name", match=MatchAny(any=source_filter)))
if chunk_type_filter:
    filter_conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter)))
query_filter = Filter(must=filter_conditions) if filter_conditions else None
```

#### 4b Hybrid Search Path (lines 267-286) — default

When `hybrid_search_enabled=True` AND BM25 is fitted:

```python
prefetch = [
    Prefetch(query=query_embedding, using="dense", limit=top_k * 2, filter=query_filter),
    Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2, filter=query_filter),
]
query = RrfQuery(rrf=Rrf(k=60))  # RRF fusion
```

**RRF scoring:** Each result gets score = `1 / (k + rank_dense) + 1 / (k + rank_sparse)`. k=60 softens the rank contribution. Top `top_k` results returned.

**BM25 sparse vector** (line 270): `self._bm25.tokenize_query(query_text)` — Porter-stemmed tokens from the query extracted against the fitted vocabulary. Tokens not in the vocabulary are silently dropped (see `bm25_tokenizer.py:128`).

#### 4c Dense-Only Path (lines 293-298) — fallback

When BM25 is not fitted or hybrid is disabled:
```python
query = query_embedding
using = "dense"
```

#### 4d Result Construction (lines 301-322)

Each Qdrant hit becomes:
```python
RetrievedChunk(
    chunk=DocumentChunk(chunk_id, source_name, title, url, text, ...),
    distance=1.0 - confidence,
    confidence=min(1.0, max(0.0, score)),
)
```

Score is clamped to [0, 1].

### Step 5 Reranking (`async_rag.py:253-268`, `services/reranker.py:84-206`)

#### 5a Cross-Encoder Reranking (lines 84-151)

```
Input:  query (str), chunks (list[RetrievedChunk]), top_k=5
Output: list[RetrievedChunk] — reranked by cross-encoder score
```

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~450MB, downloaded on first use).

**Scoring:**
1. Prepare `(query, chunk_text)` pairs (line 111-112).
2. `self.model.predict(pairs)` — raw logits from cross-encoder (line 115).
3. Sigmoid normalization: `score = 1.0 / (1.0 + math.exp(-logit))` (line 118).
4. Sort descending by score. Return top `reranker_top_k` (default 5).

**Important:** The reranker uses the `effective_query` (first decomposed step or rewritten query), NOT the original question. This is intentional — the rewritten query is more search-optimized.

#### 5b MMR Diversity (`async_rag.py:259-262`, `reranker.py:171-206`)

```
Input:  query_embedding (list[float]), chunks (list[RetrievedChunk]), top_k, lambda=0.5
Output: list[RetrievedChunk] — diverse subset
```

**MMR formula:** `MMR = λ * relevance - (1-λ) * max_similarity_to_selected`

- `relevance` = chunk confidence score from cross-encoder
- `max_similarity_to_selected` = Jaccard cosine of token sets (word-level overlap, NOT embedding cosine)
- `λ=0.5`: equal tradeoff between relevance and diversity

### Step 6 Context Compression (`async_rag.py:266-268`, `context_compression.py:46-75`)

```
Input:  chunks (list[RetrievedChunk]), query (str)
Output: list[RetrievedChunk] — deduplicated + relevance scored
```

1. **Jaccard dedup** (line 56): If two chunks have token-set Jaccard similarity ≥ 0.85, keep only the first. Tokenization: `re.findall(r"[a-z0-9_]+", text.lower())`.
2. **Relevance scoring** (lines 59-73):
   - `cosine = |A∩B| / √(|A|·|B|)` — token-set cosine
   - `overlap = |query_tokens ∩ chunk_tokens| / |query_tokens|`
   - `score = cosine × 0.6 + overlap × 0.4`
3. Return top `max_chunks` by score.

**Note:** `context_compression_enabled` defaults to `False` (`settings.py:214`). When enabled, the `ContextAssembler`'s internal dedup is skipped to avoid double work.

### Step 7 Context Assembly (`async_rag.py:273-276`, `context_assembler.py:31-111`)

```
Input:  chunks (list[RetrievedChunk]), max_context_chars=4000
Output: context_str (str), source_names (list[str])
```

#### 7a Dedup (if compressor didn't run)

Jaccard-like word overlap with 70% threshold. Employs a 12-word filler list to avoid false positives on common words.

#### 7b Lost-in-the-Middle Mitigation (line 58-73)

Reorders chunks so the most relevant appear at BOTH ends of the context:

```
Original (by confidence): [A, B, C, D, E]
Rearranged:               [A, E, B, D, C]
```

Algorithm: alternate picking from left (highest confidence) and right (lowest confidence). Only activates when > 3 chunks remain after dedup.

#### 7c Formatting & Truncation (lines 80-102)

Each chunk formatted:
```
[source_name > section_header] chunk text
```
or (no section header):
```
[source_name] chunk text
```

Truncation happens at chunk boundaries — if adding the next chunk would exceed `max_context_chars`, it's dropped entirely (unless it's the first chunk).

### Step 8 Prompt Building (`async_rag.py:281-286`, `prompt_builder.py:84-139`)

```
Input:  context_str (str), safe_question (str), intent (str)
Output: full_prompt (str)
```

#### 8a Pre-processing

```python
safe_question = PromptBuilder.sanitize_query(question)
# 1. Strip triple backticks
# 2. Convert "## " → "# " (prevent heading injection)
# 3. Truncate to 2000 chars
```

If PII redaction is enabled (default), the question is also PII-redacted before prompt building.

#### 8b Context Density Tag (line 141-153)

```python
density_tag = "HIGH"   if word_count > 100 and alpha_ratio > 0.7
density_tag = "MEDIUM" if word_count > 30  and alpha_ratio > 0.5
density_tag = "LOW"    otherwise
```

Injected into context as: `<chunk>\n[DENSITY: {density_tag}]\n{context}\n</chunk>`

#### 8c Intent-Based Prompt Selection

| Intent | Instructions | Output Format |
|---|---|---|
| `code_example`, `api_lookup` | Code-focused: explanation + complete example | Code block format |
| Documentation + code keywords in query | Documentation + allow code | Code block format |
| Everything else | Documentation only | JSON format |

#### 8d Full Prompt Template

```
## SYSTEM
You are DataEngineeringCopilot, an expert data engineering assistant.
Your role is to answer questions using ONLY the provided documentation context.

## CONSTRAINTS
1. Base your answer strictly on the provided context.
2. Do NOT invent, assume, or use external knowledge.
3. If information is missing or unclear, explicitly state the limitation.
4. Cite specific documentation sources when possible.
5. Use precise technical terminology from the context.
6. Sparse/Low-Signal Text: If the context contains only raw code snippets,
   log lines, boilerplate, or insufficient material — do NOT fabricate.
   Set status to INSUFFICIENT_CONTEXT and list missing information.
7. Ignore API Boilerplate: Discard standard package imports, memory addresses,
   and log timestamps when evaluating the context.

## OUTPUT FORMAT
{varies by intent — see below}

## INSTRUCTIONS
{varies by intent — see below}

## USER QUESTION AND CONTEXT
Context:
<chunk>
[DENSITY: {HIGH|MEDIUM|LOW}]
{context}
</chunk>

Question: {question}

## YOUR ANSWER
```

**JSON output format** (documentation intents):
```
Return ONLY valid JSON with this exact structure (no markdown, no code fences):
{
  "status": "SUCCESS" or "INSUFFICIENT_CONTEXT",
  "answer": "Your detailed answer here, 2-4 sentences, or null if context is insufficient.",
  "missing_info": "Description of missing details if context is low-density and status is
  INSUFFICIENT_CONTEXT, otherwise null."
}
```

**Code output format** (code intents):
```
Return your answer as:
1. A brief explanation (1-3 sentences)
2. A fenced code block with the implementation
3. Source citations

Example:
Brief explanation of the approach.

```scala
// implementation code here
```

Sources: [list of source names]
```

### Step 9 LLM Generation (`async_rag.py:291-292`, `llm_client.py:105-174`)

#### 9a Client Selection

```python
def _select_llm_client(self, intent: str) -> LLMClientProtocol:
    if self.code_llm_client and intent in CODE_INTENTS:
        return self.code_llm_client  # qwen2.5-coder:7b or configured code model
    return self.llm_client           # llama3.2:3b or configured global model
```

#### 9b API Payload

```python
payload = {
    "model": self.model,                        # "llama3.2:3b"
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.05,                        # Very low — nearly deterministic
    **self._extra_body,                         # Ollama: {options: {num_ctx: 4096, num_predict: 512}}
}
```

Note: There is no system message slot — all instructions are embedded in the prompt by `PromptBuilder`.

#### 9c HTTP Request (`_http_post`, lines 182-203)

- **URL:** `POST {base_url}/v1/chat/completions`
- **Retry:** Single attempt per call — no client-side retry, no circuit breaker. On failure the adaptive router marks a category-based cooldown and fails over to the next provider in `llm_fallback_order` (ending at Ollama).
- **Rate limiter:** Non-blocking pre-flight gate before the request — an over-limit provider (OpenRouter/NVIDIA) is skipped without a paid call.
- **Auth:** Bearer token in `Authorization` header (empty string for Ollama).

#### 9d Response Processing

```python
content = body["choices"][0]["message"]["content"]
# Strip <think>...</think> tags (Ollama raw mode)
# Record token usage: LLMUsage(prompt_tokens, completion_tokens, model)
```

### Step 10 Post-Processing (in order)

#### 10a Code Syntax Validation (`async_rag.py`: code intents only)

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

#### 10b Output Guardrails (`async_rag.py:311-324`, `output_guardrails.py:29-117`)

`OutputGuardrails.verify(raw_answer, source_count)`:

1. **Try JSON parse** (lines 68-77): Strip markdown fences, `json.loads()`, validate via Pydantic.
2. **Fallback plain text** (lines 80-93): Accept text with code blocks. Extract citations from `Sources: ...` / `Cited: ...` lines.
3. **Quality checks** (lines 96-108):
   - Reject if empty.
   - If `source_count > 0`: reject if < 20 chars.
   - Reject boilerplate: "I cannot answer", "outside my knowledge", "I don't have enough", "I am not able to", "beyond my knowledge".
   - `INSUFFICIENT_CONTEXT` passes through regardless.
4. **Fallback on rejection** (line 320): If guardrails reject, the raw output is used unchanged (fail-open).

#### 10c PII Redaction (`async_rag.py:328-332`, `pii_redactor.py:99-130`)

**Patterns:** email, phone (US), SSN, credit card, IP address

**Modes:**
- `"full"` (default): Replace with `[REDACTED_EMAIL]`, `[REDACTED_SSN]`, etc.
- `"masked"`: Partial mask (e.g., `j***@***.com`)
- `"none"`: Passthrough

**Applied both:** pre-LLM (on question) and post-LLM (on answer).

#### 10d Citation Verification (`async_rag.py:333-340`, `structured_output.py:16-56`)

```python
parsed = parse_rag_response(answer_text)       # Extract answer + citations from JSON/text
source_names = [c.chunk.source_name for c in retrieved_chunks]
verified = verify_citations(parsed.citations, source_names)  # Keep if source matches
```

#### 10e Groundedness Verification (`async_rag.py:358-385`, `groundedness.py:153-216`)

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

**Scoring:** `score = supported_claims / total_claims`. Threshold = 0.5.

**Fail-open:** If LLM is unavailable, falls back to text-overlap heuristic (Jaccard-ish token overlap with `min_support_score=0.3`). On guardrails failure, appends warning note: `"[Note: Some claims may not be fully supported by the documentation.]"`.

### Step 11 Response Assembly

**Service layer** (`async_rag.py:350-393`):
```python
result = Answer(
    text=answer_text,
    sources=tuple(c.chunk for c in retrieved_chunks),
    confidence=retrieved_chunks[0].confidence,
    stage_times={"rewrite": 45.2, "retrieval": 120.5, "rerank": 89.3, ...},
)
```

**API layer** (`routes.py:262-273`):
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

### Endpoint: `POST /api/v1/ask/stream` (`routes.py:282-314`)

Returns `StreamingResponse` with `media_type="text/event-stream"`.

### SSE Event Sequence

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
| LLM rewriting | ✅ `async_rewrite()` | ❌ Rule-based `rewrite()` only |
| HyDE | ✅ Generated and embedded | ❌ Skipped |
| Query decomposition | ✅ Multiple sub-queries | ❌ Single query |
| Code syntax validation | ✅ Post-generation | ❌ Not performed |
| Groundedness | ✅ LLM NLI check | ❌ Not performed |
| Output guardrails | ✅ Pydantic validation | ✅ Applied (after streaming) |
| PII redaction | ✅ Pre+post | ✅ Pre+post |
| Cache storage | ✅ After generation | ✅ After generation |

### Token Streaming Mechanism (`llm_client.py:220-266`)

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

**Fallback:** On any HTTP error, falls back to non-streaming `generate()` and yields the complete answer as a single token.

**SSE Formatter** (`async_rag.py:32-34`):
```python
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
```

---

## 5. Configuration Reference

### 5.1 Infrastructure Settings

| Setting | Default | File:Line | Description |
|---|---|---|---|
| `qdrant_url` | `http://localhost:6333` | `settings.py:90` | Qdrant HTTP API |
| `redis_url` | `redis://:local_secure_password_123@localhost:6379/0` | `settings.py:94` | Redis (cache, queue, URL registry) |
| `ollama_base_url` | `http://localhost:11434` | `settings.py:91` | Ollama container |
| `langfuse_url` | `http://langfuse:3000` | `settings.py:95` | Langfuse observability |
| `collection_name` | `data_engineering_docs` | `settings.py:87` | Qdrant collection name |

### 5.2 Provider Selection

| Setting | Default | Options | File:Line |
|---|---|---|---|
| `llm_provider` | `ollama` | `ollama`, `openrouter`, `nvidia` | `settings.py:126` |
| `llm_model` | `llama3.2:3b` | Any model name | `settings.py:127` |
| `embedding_provider` | `ollama` | `ollama`, `openrouter`, `nvidia` | `settings.py:128` |
| `embedding_model_name` | `nomic-embed-text` | Any Ollama model | `settings.py:114` |

### 5.3 Per-Purpose LLM Overrides

All default to empty string (fall back to global `llm_provider`/`llm_model`):

| Purpose | Settings Key | Used For |
|---|---|---|
| Answer | `answer_llm_provider/model` | Main answer generation |
| Rewrite | `rewrite_llm_provider/model` | Query rewriting |
| Groundedness | `groundedness_llm_provider/model` | Claim verification |
| Intent | `intent_llm_provider/model` | Intent classification |
| Enrichment | `enrichment_llm_provider/model` | Contextual chunk enrichment |
| Evaluation | `evaluation_llm_provider/model` | Faithfulness evaluation |
| Code | `code_llm_provider/model` | Code-specific answers |

### 5.4 RAG Pipeline Settings

| Setting | Default | Range | File:Line |
|---|---|---|---|
| `retrieval_top_k` | 15 | 1-100 | `settings.py:171` |
| `reranker_enabled` | True | — | `settings.py:172` |
| `reranker_top_k` | 5 | 1-20 | `settings.py:174` |
| `max_context_chars` | 4000 | 500-10000 | `settings.py:175` |
| `confidence_threshold` | 0.18 | 0.0-1.0 | `settings.py:176` |
| `hybrid_search_enabled` | True | — | `settings.py:204` |
| `hybrid_rrf_k` | 60 | 10-200 | `settings.py:205` |
| `semantic_cache_threshold` | 0.95 | 0.5-1.0 | `settings.py:207` |
| `semantic_cache_ttl` | 3600 | seconds | `settings.py:208` |
| `query_rewrite_enabled` | True | — | `settings.py:210` |
| `groundedness_enabled` | True | — | `settings.py:211` |
| `context_compression_enabled` | False | — | `settings.py:214` |
| `temperature` | 0.05 | 0.0-1.0 | `llm_client.py:71` |

### 5.5 Chunking Settings

| Setting | Default | File:Line |
|---|---|---|
| `chunking_strategy` | `sentence_preserving` | `settings.py:163` |
| `chunk_size_words` | 375 | `settings.py:164` |
| `chunk_overlap_words` | 90 | `settings.py:165` |
| `min_semantic_similarity` | 0.5 | `settings.py:167` |
| `embedding_batch_size` | 256 | `settings.py:109` |

### 5.6 Crawl Settings

| Setting | Default | File:Line |
|---|---|---|
| `max_pages_per_source` | 100000 | `settings.py:305` |
| `crawl_delay_seconds` | 0.2 | `settings.py:184` |
| `crawl_async_concurrency` | 20 | `settings.py:196` |
| `crawl_async_per_domain_concurrency` | 3 | `settings.py:198` |
| `crawl_async_conditional_get` | True | `settings.py:199` |

### 5.7 Ollama Settings

| Setting | Default | File:Line |
|---|---|---|
| `ollama_timeout_seconds` | 300 | `settings.py:178` |
| `ollama_num_ctx` | 4096 | `settings.py:179` |
| `ollama_num_predict` | 512 | `settings.py:180` |
| `ollama_retry_context_ratio` | 0.5 | `settings.py:181` |
| `ollama_retry_extra_num_predict` | 512 | `settings.py:182` |

### 5.8 Docker Services

| Service | Image | Port(s) |
|---|---|---|
| redis | `redis:7-alpine` | 6379 |
| qdrant | `qdrant/qdrant:v1.18.3` | 6333, 6334 |
| ollama | `ollama/ollama:0.32.4` | 11434 |
| minio | `minio/minio:RELEASE.2025-09-07` | 9001, 9002 |
| clickhouse | `clickhouse/clickhouse-server:24-alpine` | 8123, 9000 |
| langfuse | `langfuse/langfuse:3` | 3000 |
| postgres (app) | `postgres:16-alpine` | 5433 |
| backend-api | Custom image | 8000 |
| celery_worker | Custom image | (none) |

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
  ├── span: "retrieval"         (embedding + vector DB calls)
  └── generation: "ollama-generation"  (LLM API call)
```

**Span attributes:** `app.input` (truncated 2000), `app.model`, `app.output` (truncated 5000), `app.span_type`.

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

### 6.3 Cache Architecture (`query_cache.py:36-206`)

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

**Read path** (`aget`, query_cache.py:127-179):
1. L1 exact match (in-memory `OrderedDict`)
2. L2 semantic match (in-memory `deque` + NumPy dot product)
3. Redis exact match (`GET`)
4. Redis semantic match (`SCAN` + `HGETALL` + cosine similarity)
5. On hit: backfill in-memory caches from Redis

**Write path** (`aset_exact` + `aset_semantic`, lines 181-206):
1. Write to L1/L2
2. Write to Redis with TTL

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

---

> **Document version:** 2026-07-30  
> **Codebase ref:** commit `006aabb`  
> **Generated from:** deep architectural audit of the DataEngineeringCopilot codebase
