# Project: DataEngineeringCopilot

## 1. Purpose
- Offline RAG assistant over data engineering documentation.
- Crawls configured docs → chunks + embeds into local QdrantDB → answers questions via local Ollama/Qwen.

## 2. Stack
- FE: Streamlit
- BE: Python CLI + service layer
- DB: QdrantDB persistent local vector store
- Infra: urllib HTTP crawling, BeautifulSoup HTML parsing, Ollama embeddings, Ollama local LLM

## 3. Architecture
- Pattern: layered architecture / ports-adapters style
- Entry points: `main.py` CLI; `data_engineering_copilot/ui/streamlit_app.py`
- Layers: config → domain dataclasses → infrastructure adapters → services/workflows → CLI/UI
- RAG design: no LangChain/LlamaIndex; direct crawler/parser/chunker/embed/vector/query/generate pipeline
- Persistence: local `qdrant_db/`; embedding cache under `data/embedding_models`

## 4. Folder Map (Compressed)
- ALWAYS use python virtual environmet located at `/home/rahul/PythonVenvs/data_eng_copilot_env`
- ALWAYS activate python virtual environmet before running any test case or app `/home/rahul/PythonVenvs/data_eng_copilot_env`
- `main.py` → CLI commands: `ingest`, `ask`, `reset-index`, `ui`
- `data_engineering_copilot/config/` → runtime settings + documentation source JSON
- `data_engineering_copilot/domain/` → shared dataclasses
- `data_engineering_copilot/infrastructure/` → HTTP, HTML, embedding, vector DB, Ollama adapters
- `data_engineering_copilot/services/` → ingestion, chunking, RAG orchestration
- `data_engineering_copilot/ui/` → Streamlit app
- `data_engineering_copilot/utils/` → text normalization helpers
- `scripts/` → one-time embedding model download (deprecated)
- `tests/` → focused unit/regression tests for chunking, crawling, parsing, ingestion selection, settings
- `logs/` → ingestion refresh logs
- `qdrant_db/` → local vector index; generated/runtime data (deprecated)
- `data/` → local model/data cache

## 5. Modules (CRITICAL)

### CLI: `main.py`
- Role:
  - Thin command dispatcher for ingestion, Q&A, index reset, and Streamlit command hint.
- Functions:
  - `ingest(max_pages, source_names)` → build ingestion service; crawl selected/all sources; print chunk total
  - `ask(question)` → build RAG service; print answer, sources, confidence
  - `reset_index()` → delete/recreate configured QdrantDB directory
  - `build_parser()` → argparse command tree
  - `main()` → route command to function
- Uses:
  - `factory.build_ingestion_service`
  - `factory.build_rag_service`
  - `config.settings`
  - `shutil`
- CLI:
  - `python main.py ingest --max-pages N`
  - `python main.py ingest --source "Apache Spark Documentation" --source "Delta Lake Documentation"`
  - `python main.py ask "question"`
  - `python main.py reset-index`
- `python main.py ui` (prints a platform-neutral Streamlit launch command)
- Functions:
  - `build_ingestion_service(app_settings=settings)` → `IngestionService`
  - `build_rag_service(app_settings=settings)` → `RagAnswerService`
- Uses:
  - `DocumentationCrawler`
  - `DocumentationHtmlParser`
  - `DocumentChunker`
  - `SentenceTransformerEmbeddings`
   - `QdrantVectorStore`
  - `OllamaClient`
  - `AppSettings`

### Settings: `config/settings.py`
- Role:
  - Defines source config schema, loads JSON sources, centralizes tunables.
- Types:
  - `DocumentationSource` → source name, start URLs, allowed domains, URL prefixes
  - `AppSettings` → paths, model names, retrieval/generation/crawl settings, loaded sources
- Functions:
  - `load_documentation_sources(config_path)` → tuple of validated `DocumentationSource`
  - `_required_string`, `_required_string_tuple`, `_optional_string_tuple` → JSON validation helpers
- Key defaults:
  - `collection_name=data_engineering_docs`
  - `embedding_model_name=nomic-embed-text`
  - `embedding_local_files_only=True`
  - `ollama_base_url=http://localhost:11434`
  - `ollama_model=qwen3.5:9b`
  - `chunk_size_words=420`
  - `chunk_overlap_words=80`
  - `retrieval_top_k=2`
  - `max_context_chars=2200`
  - `confidence_threshold=0.35`
  - `crawl_delay_seconds=0.25`
  - `max_pages_per_source=80`
- Uses:
  - `documentation_sources.json`
  - `pathlib.Path`
  - `json`

### Source Config: `config/documentation_sources.json`
- Role:
  - Configures documentation crawl roots and URL allowlists.
- Sources:
  - Apache Spark Documentation → `https://spark.apache.org/docs/latest/`
  - Apache Airflow Documentation → `https://airflow.apache.org/docs/apache-airflow/stable/`
  - Databricks Documentation → `https://docs.databricks.com/aws/en/`
  - Delta Lake Documentation → `https://docs.delta.io/latest/`
- Constraints:
  - `allowed_domains` restricts host
  - `url_prefixes` restricts path subtree

### Domain Models: `domain/models.py`
- Role:
  - Immutable-ish data contracts across layers.
- Entities:
  - `RawDocument` → crawled HTML page
  - `ParsedDocument` → readable title/text extracted from raw HTML
  - `DocumentChunk` → chunk persisted/retrieved from vector store
  - `RetrievedChunk` → chunk + vector distance + confidence
  - `Answer` → final answer + cited chunks + confidence
  - `IngestionEvent` → progress/logging event for CLI/UI/log file
- Uses:
  - Python `dataclass`

### Crawler: `infrastructure/crawler.py`
- Role:
  - Breadth-first HTML crawler constrained by source domain/prefix.
  - Emits ingestion progress events; yields `RawDocument`.
- Classes:
  - `LinkExtractor(HTMLParser)` → collect `<a href>` values
  - `DocumentationCrawler` → URL queue, download, extraction, filtering
- Functions:
  - `crawl(source, max_pages, on_event=None)` → BFS over allowed HTML pages
  - `_download(url)` → `urllib.request.urlopen`; require `Content-Type` contains `text/html`
  - `_extract_links(html, base_url)` → parse anchors; skip mailto/tel/javascript; `urljoin`; defrag
  - `_clean_url(url)` → remove URL fragments, preserve trailing slash for correct relative link resolution
  - `_dedupe_key(url)` → normalize slash variants and `/index.html` for visited/queued sets
  - `_is_allowed(url, source)` → scheme http/https + exact host + prefix check
  - `_emit(on_event, event)` → optional callback
- Important behavior:
  - Preserves directory trailing slash when fetching; required for Spark relative links like `quick-start.html`
  - Tracks `queued` and `visited` by dedupe key to avoid duplicate slash/index pages
  - Counts failed URLs in crawler event `pages_fetched`; service counts only yielded documents
- Uses:
  - `DocumentationSource`
  - `RawDocument`
  - `IngestionEvent`
  - `urllib.parse`, `urllib.request`, `html.parser`, `deque`

### HTML Parser: `infrastructure/html_parser.py`
- Role:
  - Converts raw HTML to normalized documentation text.
- Class:
  - `DocumentationHtmlParser`
- Functions:
  - `parse(raw)` → `ParsedDocument | None`
  - `_title(soup, fallback)` → prefer first `h1`; else `<title>`; else URL
- Logic:
  - Removes `script`, `style`, `noscript`, `nav`, `footer`, `header`, `aside`
  - Content root priority: `<main>` → `<article>` → `<body>` → full soup
  - Normalizes whitespace
  - Skips pages with fewer than 40 words
- Uses:
  - BeautifulSoup
  - `normalize_whitespace`
  - `RawDocument`, `ParsedDocument`

### Embeddings: `infrastructure/embeddings.py`
- Role:
  - Ollama-only embedding provider using the `/api/embed` endpoint.
- Class:
  - `SentenceTransformerEmbeddings`
- Functions:
  - `__init__(model_name, cache_dir, local_files_only)` → configure Ollama endpoint
  - `embed_texts(texts)` → normalized embedding vectors as `list[list[float]]`
  - `embed_query(text)` → first vector for query
  - `_ollama_embed(texts)` → batch embedding via Ollama `/api/embed` endpoint
  - `_validate_embedding_dimensions(embeddings, texts)` → validate embedding dimensions match config
  - `_slice_texts_into_batches(texts, batch_size)` → split texts into batches for Ollama
  - `_ollama_embed_single_batch(texts)` → single batch embedding via Ollama
- Uses:
  - `urllib.request` for Ollama API calls
- Constraint:
  - Uses Ollama's `nomic-embed-text` model via HTTP API
- Behavior:
  - Uses Ollama `/api/embed` endpoint for all embeddings
  - Batches embeddings to prevent OOM on resource-constrained machines
- Note:
  - The class name `SentenceTransformerEmbeddings` is historical; it's now Ollama-only.
  - The `sentence_transformers` library is no longer used.
  - Default configuration uses Ollama's `nomic-embed-text` model via HTTP API.

### Vector Store: `infrastructure/vector_store.py`
- Role:
  - QdrantDB persistent vector index adapter (compatibility layer for legacy ChromaVectorStore).
- Classes:
  - `QdrantVectorStore`
  - `VectorStoreReadError`
- Functions:
  - `__init__(persist_directory, collection_name)` → persistent client + collection with cosine space
  - `upsert_chunks(chunks, embeddings)` → write ids/docs/embeddings/metadata
  - `query(query_embedding, top_k)` → retrieve `RetrievedChunk` list with confidence = `1 - cosine distance` clamped `[0,1]`
  - `count()` → collection count
  - `hybrid_query(query, top_k)` → embed query and delegate to `query`
- Error behavior:
  - Converts Chroma `InternalError` containing `Nothing found on disk` to `VectorStoreReadError`
  - Validates chunks/embeddings length equality
- Uses:
  - `QdrantVectorStore` (delegates to)
  - `DocumentChunk`, `RetrievedChunk`

### Qdrant Store: `infrastructure/qdrant_store.py`
- Role:
  - Qdrant vector store implementation (primary storage backend).
- Class:
  - `QdrantVectorStore`
- Functions:
  - `__init__(url, collection_name)` → Qdrant client + collection with cosine space
  - `upsert_chunks(chunks, embeddings)` → write ids/docs/embeddings/metadata
  - `query(query_embedding, top_k)` → retrieve `RetrievedChunk` list with confidence = `1 - cosine distance`
  - `count()` → collection count
  - `hybrid_query(query, top_k)` → embed query and delegate to `query`
  - `_embedding_dim()` → returns configurable embedding_dimension from AppSettings
  - `_chunk_to_payload(chunk)` → convert DocumentChunk to Qdrant payload dict
  - `_chunk_id_to_uuid(chunk_id)` → convert chunk ID to deterministic UUID5
- Uses:
  - `qdrant_client.QdrantClient`
  - `DocumentChunk`, `RetrievedChunk`
  - `SentenceTransformerEmbeddings`

### Ollama Client: `infrastructure/ollama_client.py`
- Role:
  - Local LLM generation adapter via Ollama HTTP API.
- Classes:
  - `OllamaClient`
  - `OllamaError`
- Functions:
  - `generate(prompt, num_predict=None, num_ctx=None)` → POST `/api/generate`; non-streaming; raw prompt; return response text
  - `_format_raw_chat_prompt(user_prompt)` → DataEngineeringCopilot chat template in raw mode
  - `_extract_final_response(response)` → clean up <think> tags and whitespace
- Options:
  - `temperature=0.05`
  - `top_p=0.8`
  - `num_ctx=settings.ollama_num_ctx`
  - `num_predict=settings.ollama_num_predict`
  - `raw=True`
  - `stream=False`
- Error behavior:
  - Timeout → actionable `OllamaError`
  - Connection failure → tells user to start Ollama and pull `deepseek-coder:6.7b`
  - Empty response → suggests increasing `ollama_num_predict` or reducing `max_context_chars`
- Uses:
  - `urllib.request`
  - `json`
  - `socket`
  - `re`

### Chunker: `services/chunker.py`
- Role:
  - Splits parsed docs into overlapping word chunks with stable ids.
- Class:
  - `DocumentChunker`
- Functions:
  - `__init__(chunk_size_words, overlap_words)` → validates positive size and overlap `< size`
  - `chunk(document)` → list of `DocumentChunk`
  - `_chunk_id(document, index)` → `<slug(source)>:<sha1(url)[:10]>:<0000-index>`
- Logic:
  - Step = `chunk_size_words - overlap_words`
  - Preserves source name/title/url in each chunk
- Uses:
  - `ParsedDocument`, `DocumentChunk`
  - `slugify`
  - `hashlib.sha1`

### Ingestion Service: `services/ingestion.py`
- Role:
  - Orchestrates crawl → parse → chunk → embed → vector upsert.
- Class:
  - `IngestionService`
- Functions:
  - `ingest(max_pages_per_source=None, source_names=None, on_event=None)` → total chunks indexed/upserted
  - `_selected_sources(source_names)` → filter configured sources by exact name; validate unknown/empty selection
  - `_emit(on_event, event)` → optional progress callback
- Flow:
  - source selection → crawler per source → parser skip if unreadable → chunk → embed texts → Chroma upsert → emit events
- Events:
  - `source_start`
  - `fetch_start`, `fetch_success`, `fetch_error` from crawler
  - `page_skipped`
  - `page_indexed`
  - `source_complete`
- Uses:
  - `AppSettings`
  - `DocumentationCrawler`
  - `DocumentationHtmlParser`
  - `DocumentChunker`
  - `SentenceTransformerEmbeddings`
   - `QdrantVectorStore`
  - `IngestionEvent`

### RAG Service: `services/rag.py`
- Role:
  - Orchestrates question answering from local vector context and Ollama.
- Classes:
  - `RagAnswerService` → base RAG service with `answer()` method
  - `ProductionRagService` → extended RAG service with Langfuse tracing and `answer_question()` method
- Functions:
  - `answer(question)` → embed query → vector query → confidence gate → prompt → Ollama → `Answer`
  - `answer_question(user_id, session_id, question, top_k)` → Langfuse-traced RAG pipeline with retrieval and generation
- Logic:
  - If vector store unreadable → outside-repository answer
  - If no match or top confidence below threshold → outside-repository answer
  - If Ollama fails → return error text with retrieved sources
  - Prompt asks concise answer using only provided context
- Uses:
  - `SentenceTransformerEmbeddings`
  - `QdrantVectorStore`
  - `OllamaClient`
  - `Answer`, `RetrievedChunk`
  - `Langfuse` (optional, for ProductionRagService)

### Streamlit UI: `ui/streamlit_app.py`
- Role:
  - Interactive local UI for index status, ingestion refresh, and Q&A.
- Functions:
  - `rag_service()` → cached RAG service
   - `vector_store()` → cached Qdrant vector store
  - `run_ingestion_refresh(max_pages_per_source, source_names, on_event=None)` → build ingestion service and ingest selected sources
  - `ingestion_log_path()` → `logs/ingestion_refresh.log`
  - `append_ingestion_log(log_path, event)` → append pipe-delimited event line
  - `main()` → Streamlit page/sidebar/query flow
- UI behavior:
  - Sidebar shows chunk count, Ollama config, embedding model, confidence threshold
  - Ingestion sidebar has multiselect `Sources to ingest`, max pages input, source details expander
  - Refresh button disabled when zero sources selected
  - Refresh displays status, page/chunk metrics, recent URLs, log path
  - Successful refresh clears Streamlit caches for RAG/vector store then reruns
  - Main panel accepts question textarea and displays answer/confidence/sources
- Uses:
  - `settings`
  - `build_ingestion_service`, `build_rag_service`
   - `QdrantVectorStore`, `VectorStoreReadError`
  - `IngestionEvent`
  - Streamlit cache/resources/widgets

### API: `api/app.py`
- Role:
  - FastAPI application for async ingestion and RAG service endpoints.
- Classes:
  - `app` → FastAPI app with routes
- Uses:
  - `fastapi.FastAPI`
  - `api.routes`

### API Routes: `api/routes.py`
- Role:
  - API endpoints for ingestion and task status.
- Classes:
  - `IngestRequest` → request body for ingestion endpoint
  - `TaskStatus` → response body for task status endpoint
- Functions:
  - `ingest_documents(request)` → POST `/api/v1/ingest`; start async ingestion task
  - `get_task_status(task_id)` → GET `/api/v1/task/{task_id}`; get task status
- Uses:
  - `fastapi.APIRouter`
  - `pydantic.BaseModel`
  - `celery.result.AsyncResult`
  - `workers.tasks.execute_background_ingestion`

### Workers: `workers/tasks.py`
- Role:
  - Async ingestion task using Crawl4AI and Qdrant.
- Classes:
  - `app` → Celery app with Redis broker
- Functions:
  - `_run_async_crawl(urls)` → crawl URLs concurrently with AsyncWebCrawler
  - `execute_background_ingestion(urls)` → Celery task entry point
- Uses:
  - `Celery`
  - `AsyncWebCrawler`
  - `SentenceTransformerEmbeddings`
  - `QdrantVectorStore`
  - `DocumentChunker`

### Text Utils: `utils/text.py`
- Role:
  - Tiny text normalization helpers used across parser/chunker.
- Functions:
  - `normalize_whitespace(value)` → collapse whitespace to single spaces
  - `slugify(value)` → lowercase alphanumeric dash slug; fallback `document`
- Uses:
  - `re`

### Model Download Script: `scripts/download_embedding_model.py`
- Role:
  - One-time setup script to cache embedding model locally.
- Behavior:
  - Imports `settings`
  - Downloads `settings.embedding_model_name` into `settings.embedding_cache_dir`
  - Note: No longer needed as the system uses Ollama's `nomic-embed-text` model via HTTP API
- Uses:
  - `sentence_transformers.SentenceTransformer` (deprecated, kept for reference)

### Tests
- Role:
  - Fast focused regression coverage.
- Files:
  - `test_chunker.py` → chunk metadata + chunk id prefix
  - `test_crawler.py` → Spark trailing-slash relative link regression; duplicate slash/index dedupe
  - `test_html_parser.py` → title extraction, main text extraction, nav removal
  - `test_ingestion.py` → selected source ingestion and unknown source validation
  - `test_settings.py` → JSON source loading

## 6. Data Models
- `DocumentationSource`:
  - `name: str`
  - `start_urls: tuple[str,...]`
  - `allowed_domains: tuple[str,...]`
  - `url_prefixes: tuple[str,...]`
- `AppSettings`:
  - `project_root`
  - `data_dir`
  - `chroma_dir`
  - `documentation_sources_path`
  - `embedding_cache_dir`
  - `collection_name`
  - `embedding_model_name`
  - `embedding_local_files_only`
  - `ollama_base_url`
  - `ollama_model`
  - `chunk_size_words`
  - `chunk_overlap_words`
  - `retrieval_top_k`
  - `max_context_chars`
  - `confidence_threshold`
  - `request_timeout_seconds`
  - `ollama_timeout_seconds`
  - `ollama_num_ctx`
  - `ollama_num_predict`
  - `crawl_delay_seconds`
  - `max_pages_per_source`
  - `sources`
- `RawDocument`:
  - `source_name`
  - `url`
  - `html`
- `ParsedDocument`:
  - `source_name`
  - `title`
  - `url`
  - `text`
- `DocumentChunk`:
  - `chunk_id`
  - `source_name`
  - `title`
  - `url`
  - `text`
- `RetrievedChunk`:
  - `chunk`
  - `distance`
  - `confidence`
- `Answer`:
  - `text`
  - `sources: tuple[DocumentChunk,...]`
  - `confidence`
- `IngestionEvent`:
  - `event_type`
  - `source_name`
  - `message`
  - `url?`
  - `title?`
  - `chunks_indexed=0`
  - `pages_fetched=0`
  - `error?`

## 7. Core Flows (VERY IMPORTANT)
- CLI Ingest All:
  - `python main.py ingest` → `main.ingest` → `build_ingestion_service` → `IngestionService.ingest(source_names=None)` → all settings sources → crawl → parse → chunk → embed → Chroma upsert → print total
- CLI Ingest Selected:
  - `python main.py ingest --source X` → argparse append source names → `IngestionService._selected_sources` → exact-name match → selected source crawl/index only
- UI Ingest:
  - Streamlit sidebar source multiselect + max pages → `Refresh Documentation` → `run_ingestion_refresh` → `IngestionService.ingest(selected_source_names)` → event callback → log/status/metrics/recent URLs → clear caches → rerun
- Crawl:
  - `DocumentationSource` → seed queue → `_download(text/html)` → yield `RawDocument` → `_extract_links` → `_is_allowed` → enqueue allowed links until `max_pages`
- HTML Parse:
  - `RawDocument.html` → BeautifulSoup → remove chrome tags → select main/article/body → normalize text → word-count gate → `ParsedDocument`
- Chunk + Embed + Store:
  - `ParsedDocument.text` → `DocumentChunker.chunk` → chunk ids by source/url/index → `SentenceTransformerEmbeddings.embed_texts` → `QdrantVectorStore.upsert_chunks`
- Ask CLI:
  - `python main.py ask Q` → `build_rag_service` → `RagAnswerService.answer(Q)` → print answer + sources + confidence
- Ask UI:
  - question textarea → Ask button → `rag_service().answer(question)` → answer block → confidence → sources list
- RAG Answer:
  - question → `embed_query` → `QdrantVectorStore.query(top_k)` → confidence threshold → `_build_prompt` → `OllamaClient.generate` → `Answer`
- Vector Retrieval:
  - query vector → Chroma cosine query → docs/metadatas/distances → confidence=`clamp(1-distance)` → `RetrievedChunk[]`
- Ollama Generation:
  - repository context + question → raw Qwen chat prompt → POST `OLLAMA/api/generate` → response text
- Reset Index:
  - `python main.py reset-index` → delete `settings.chroma_dir` if exists → recreate empty directory
- Setup Embeddings:
  - Note: No longer needed as the system uses Ollama's `nomic-embed-text` model via HTTP API

## 8. API Summary
- HTTP API exposed via FastAPI for async ingestion and task status.
- CLI commands:
  - `ingest` → crawl docs and build/update Chroma index
  - `ingest --source <source name>` → ingest only selected source; repeat for multiple
  - `ingest --max-pages <N>` → cap pages per selected source
  - `ask <question>` → answer against local Chroma/Ollama
  - `reset-index` → recreate Chroma persistence directory
  - `ui` → print Streamlit launch command
- HTTP API endpoints:
  - `POST /api/v1/ingest` → start async ingestion task
  - `GET /api/v1/task/{task_id}` → get task status
- External/local API calls:
  - `POST {ollama_base_url}/api/generate` → local Ollama generation
  - `POST {ollama_base_url}/api/embed` → local Ollama embeddings
  - HTTP GET documentation source pages via `urllib.request.urlopen`

## 9. Rules & Constraints
- Ingestion:
  - Requires internet access for documentation pages
  - Source selection uses exact configured source names
  - Unknown or empty selected source list raises `ValueError`
  - Crawls only `http`/`https`
  - Crawls only exact `allowed_domains`
  - Crawls only configured `url_prefixes` if present
  - Downloads only responses whose `Content-Type` contains `text/html`
  - Skips `mailto:`, `tel:`, `javascript:` links
  - Removes URL fragments
  - Preserves trailing slash for directory URL relative-link correctness
  - Dedupe treats trailing slash variants and `/index.html` as same logical page
  - Sleeps `crawl_delay_seconds` after successful page processing
  - Parser skips pages below 40 words
- Chunking:
  - `chunk_size_words` must be positive
  - `overlap_words` must be `>=0` and `< chunk_size_words`
  - Chunk IDs are deterministic by source slug + URL digest + chunk index
- Embeddings:
  - Default runtime assumes embedding model already exists locally
  - Embeddings normalized before storage/query
- Vector store:
  - Chroma collection uses cosine distance
  - Upsert requires embeddings count equals chunk count
  - Incomplete/corrupt Chroma index maps to reset-index guidance
- RAG:
  - Answers only when top retrieval confidence ≥ `confidence_threshold`
  - Low confidence/unreadable vector store returns fixed outside-repository message
  - Prompt context capped by `max_context_chars`
  - Prompt asks no hidden reasoning, no invented sources, concise answer
- Ollama:
  - Requires local Ollama server
  - Requires `qwen3:4b` pulled
  - Uses raw chat prompt to avoid Qwen thinking-only empty responses
  - Timeout and output/context limits from settings
- UI:
  - Refresh disabled if zero sources selected
  - Refresh logs event lines to `logs/ingestion_refresh.log`
  - Streamlit caches RAG/vector resources and clears them after successful refresh
- Security:
  - No auth layer
  - No remote API server
  - Trusts configured documentation source URLs
  - Local-only answer generation intended after ingestion

## 10. Integrations
- Apache Spark docs:
  - `https://spark.apache.org/docs/latest/`
- Apache Airflow docs:
  - `https://airflow.apache.org/docs/apache-airflow/stable/`
- Databricks docs:
  - `https://docs.databricks.com/aws/en/`
- Delta Lake docs:
  - `https://docs.delta.io/latest/`
- Ollama:
  - local `qwen3.5:9b` via `/api/generate`
- Streamlit:
  - local web UI

## 11. Known Gaps
- No deletion/pruning for removed docs; upsert updates existing chunk IDs but stale chunks may remain if source page/chunk count changes.
- No sitemap support; crawl breadth and ordering depend on linked pages from seed documents.
- No robots.txt handling.
- No retry/backoff for transient HTTP errors.
- Allowed domain matching is exact netloc; subdomains require explicit config.
- No canonical URL normalization beyond fragments, slash/index dedupe; query-string variants may duplicate pages.
- Ingestion can be slow; embeddings and Chroma upserts happen page-by-page rather than batched across pages.
- `pages_fetched` semantics differ between crawler events and service indexed/skipped document count when fetch errors occur.
- Chroma corruption recovery is manual via `reset-index`.
- RAG uses only top-k chunks; no reranking or source diversity enforcement.
- Confidence is simple `1 - cosine distance`; threshold tuning may be corpus/model dependent.
- UI refresh is synchronous; long crawls block the Streamlit session.
- No automated end-to-end test against live docs/Ollama/Chroma.
- No structured API service; CLI/UI only (FastAPI async endpoints available in `api/`).

### SYSTEM ROLE
You are an elite, pragmatic AI Coding Agent specializing in production-grade, maintainable software. Your goal is to deliver working, syntactically correct, and fully optimized code on the first attempt.

### OPERATIONAL CONSTRAINTS
1. ONLY write code for the requested feature. Do not refactor unrelated files unless explicitly asked.
2. NO PLACEHOLDERS. Do not use `// TODO`, `...`, or leave code "for the user to implement." Every loop, edge case, and error handler must be fully written out.
3. DRY & KISS. Favor readability and explicit logic over overly clever or nested abstractions.
4. SINGLE RESPONSIBILITY. Keep functions small, testable, and focused.

### WORKFLOW PROTOCOL (Think Step-by-Step)
Before writing a single line of code, output a brief 3-line execution plan:
1. Target: (What you are building)
2. State: (Current dependencies or state assumptions)
3. Step-by-Step: (The logical sequence of implementation)

### ERROR & EDGE-CASE GUARDRAILS
- Every external API call, database query, or I/O operation MUST be wrapped in explicit try-catch blocks with meaningful logging.
- Validate all function inputs (null checks, empty strings, type checks) before processing.