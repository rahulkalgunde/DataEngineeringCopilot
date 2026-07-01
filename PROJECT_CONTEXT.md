# Project: DataEngineeringCopilot

## 1. Purpose
- Offline RAG assistant over data engineering documentation.
- Crawls configured docs → chunks + embeds into local ChromaDB → answers questions via local Ollama/Qwen.

## 2. Stack
- FE: Streamlit
- BE: Python CLI + service layer
- DB: ChromaDB persistent local vector store
- Infra: urllib HTTP crawling, BeautifulSoup HTML parsing, sentence-transformers embeddings, Ollama local LLM

## 3. Architecture
- Pattern: layered architecture / ports-adapters style
- Entry points: `main.py` CLI; `data_engineering_copilot/ui/streamlit_app.py`
- Layers: config → domain dataclasses → infrastructure adapters → services/workflows → CLI/UI
- RAG design: no LangChain/LlamaIndex; direct crawler/parser/chunker/embed/vector/query/generate pipeline
- Persistence: local `chroma_db/`; embedding cache under `data/embedding_models`

## 4. Folder Map (Compressed)
- `main.py` → CLI commands: `ingest`, `ask`, `reset-index`, `ui`
- `data_engineering_copilot/config/` → runtime settings + documentation source JSON
- `data_engineering_copilot/domain/` → shared dataclasses
- `data_engineering_copilot/infrastructure/` → HTTP, HTML, embedding, vector DB, Ollama adapters
- `data_engineering_copilot/services/` → ingestion, chunking, RAG orchestration
- `data_engineering_copilot/ui/` → Streamlit app
- `data_engineering_copilot/utils/` → text normalization helpers
- `scripts/` → one-time embedding model download
- `tests/` → focused unit/regression tests for chunking, crawling, parsing, ingestion selection, settings
- `logs/` → ingestion refresh logs
- `data/` → local model/data cache

## 5. Modules (CRITICAL)

### CLI: `main.py`
- Role:
  - Thin command dispatcher for ingestion, Q&A, index reset, and Streamlit command hint.
- Functions:
  - `ingest(max_pages, source_names)` → build ingestion service; crawl selected/all sources; print chunk total
  - `ask(question)` → build RAG service; print answer, sources, confidence
  - `reset_index()` → delete/recreate configured ChromaDB directory
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
  - `ChromaVectorStore`
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
  - `embedding_model_name=sentence-transformers/all-MiniLM-L6-v2`
  - `embedding_local_files_only=True`
  - `ollama_base_url=http://localhost:11434`
  - `ollama_model=qwen3:4b`
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
  - Adapter around local/cached sentence-transformers model.
- Class:
  - `SentenceTransformerEmbeddings`
- Functions:
  - `__init__(model_name, cache_dir, local_files_only)` → load model
  - `embed_texts(texts)` → normalized embedding vectors as `list[list[float]]`
  - `embed_query(text)` → first vector for query
- Uses:
  - `sentence_transformers.SentenceTransformer`
- Constraint:
  - Default `local_files_only=True`; setup must pre-download model.

### Vector Store: `infrastructure/vector_store.py`
- Role:
  - ChromaDB persistent vector index adapter.
- Classes:
  - `ChromaVectorStore`
  - `VectorStoreReadError`
- Functions:
  - `__init__(persist_directory, collection_name)` → persistent client + collection with cosine space
  - `upsert_chunks(chunks, embeddings)` → write ids/docs/embeddings/metadata
  - `query(query_embedding, top_k)` → retrieve `RetrievedChunk` list with confidence = `1 - cosine distance` clamped `[0,1]`
  - `count()` → collection count
- Error behavior:
  - Converts Chroma `InternalError` containing `Nothing found on disk` to `VectorStoreReadError`
  - Validates chunks/embeddings length equality
- Uses:
  - `chromadb.PersistentClient`
  - `DocumentChunk`, `RetrievedChunk`

### Ollama Client: `infrastructure/ollama_client.py`
- Role:
  - Local LLM generation adapter via Ollama HTTP API.
- Classes:
  - `OllamaClient`
  - `OllamaError`
- Functions:
  - `generate(prompt)` → POST `/api/generate`; non-streaming; raw prompt; return response text
  - `_format_raw_chat_prompt(user_prompt)` → Qwen chat template in raw mode
- Options:
  - `temperature=0.1`
  - `top_p=0.9`
  - `num_ctx=settings.ollama_num_ctx`
  - `num_predict=settings.ollama_num_predict`
  - `raw=True`
  - `stream=False`
- Error behavior:
  - Timeout → actionable `OllamaError`
  - Connection failure → tells user to start Ollama and pull `qwen3:4b`
  - Empty response → suggests increasing `ollama_num_predict` or reducing `max_context_chars`
- Uses:
  - `urllib.request`
  - `json`
  - `socket`

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
  - `ChromaVectorStore`
  - `IngestionEvent`

### RAG Service: `services/rag.py`
- Role:
  - Orchestrates question answering from local vector context and Ollama.
- Constants:
  - `OUTSIDE_REPOSITORY_MESSAGE="I cannot answer this question because it is outside my knowledge repository."`
- Class:
  - `RagAnswerService`
- Functions:
  - `answer(question)` → embed query → vector query → confidence gate → prompt → Ollama → `Answer`
  - `_build_prompt(question, matches)` → compact source-labeled context with max char budget
  - `_unique_sources(matches)` → de-duplicate returned source chunks by `(title,url)`
- Logic:
  - If vector store unreadable → outside-repository answer
  - If no match or top confidence below threshold → outside-repository answer
  - If Ollama fails → return error text with retrieved sources
  - Prompt forbids hidden reasoning and invention; asks concise answer
- Uses:
  - `SentenceTransformerEmbeddings`
  - `ChromaVectorStore`
  - `OllamaClient`
  - `Answer`, `RetrievedChunk`

### Streamlit UI: `ui/streamlit_app.py`
- Role:
  - Interactive local UI for index status, ingestion refresh, and Q&A.
- Functions:
  - `rag_service()` → cached RAG service
  - `vector_store()` → cached Chroma vector store
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
  - `ChromaVectorStore`, `VectorStoreReadError`
  - `IngestionEvent`
  - Streamlit cache/resources/widgets

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
  - Needed because runtime defaults to local-files-only embedding loading
- Uses:
  - `sentence_transformers.SentenceTransformer`

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
  - `ollama_retry_context_ratio`
  - `ollama_retry_extra_num_predict`
  - `ollama_retry_max_num_predict`
  - `crawl_delay_seconds`
  - `max_pages_per_source`
  - `ingestion_batch_chunk_size`
  - `logging_enabled`
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
  - `ParsedDocument.text` → `DocumentChunker.chunk` → chunk ids by source/url/index → `SentenceTransformerEmbeddings.embed_texts` → `ChromaVectorStore.upsert_chunks`
- Ask CLI:
  - `python main.py ask Q` → `build_rag_service` → `RagAnswerService.answer(Q)` → print answer + sources + confidence
- Ask UI:
  - question textarea → Ask button → `rag_service().answer(question)` → answer block → confidence → sources list
- RAG Answer:
  - question → `embed_query` → `ChromaVectorStore.query(top_k)` → confidence threshold → `_build_prompt` → `OllamaClient.generate` → `Answer`
- Vector Retrieval:
  - query vector → Chroma cosine query → docs/metadatas/distances → confidence=`clamp(1-distance)` → `RetrievedChunk[]`
- Ollama Generation:
  - repository context + question → raw Qwen chat prompt → POST `OLLAMA/api/generate` → response text
- Reset Index:
  - `python main.py reset-index` → delete `settings.chroma_dir` if exists → recreate empty directory
- Setup Embeddings:
  - `python scripts/download_embedding_model.py` → cache sentence-transformers model → runtime embedding load succeeds with `local_files_only=True`

## 8. API Summary
- No public HTTP API exposed by this project.
- CLI commands:
  - `ingest` → crawl docs and build/update Chroma index
  - `ingest --source <source name>` → ingest only selected source; repeat for multiple
  - `ingest --max-pages <N>` → cap pages per selected source
  - `ask <question>` → answer against local Chroma/Ollama
  - `reset-index` → recreate Chroma persistence directory
  - `ui` → print Streamlit launch command
- External/local API calls:
  - `POST {ollama_base_url}/api/generate` → local Ollama generation
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
- ChromaDB:
  - local persistent vector storage
- sentence-transformers:
  - `sentence-transformers/all-MiniLM-L6-v2`
- Ollama:
  - local `qwen3:4b` via `/api/generate`
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
- No structured API service; CLI/UI only.

---
## File Contents
---

### `main.py`
```python
from __future__ import annotations

import argparse
import logging
import shutil

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.logging_config import configure_logging


logger = logging.getLogger("data_engineering_copilot.main")


def ingest(max_pages: int | None, source_names: tuple[str, ...] | None) -> None:
    from data_engineering_copilot.factory import build_ingestion_service

    logger.info("CLI ingest started max_pages=%s sources=%s", max_pages, source_names or "all")
    service = build_ingestion_service()
    total_chunks = service.ingest(max_pages_per_source=max_pages, source_names=source_names)
    logger.info("CLI ingest completed chunks=%s", total_chunks)
    print(f"Indexed {total_chunks} chunks.")


def ask(question: str) -> None:
    from data_engineering_copilot.factory import build_rag_service

    logger.info("CLI ask started question=%r", question[:200])
    service = build_rag_service()
    answer = service.answer(question)
    logger.info("CLI ask completed confidence=%.4f sources=%s", answer.confidence, len(answer.sources))
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"- {source.title}: {source.url}")
    print(f"\nConfidence: {answer.confidence:.2f}")


def reset_index() -> None:
    logger.warning("Resetting ChromaDB index path=%s", settings.chroma_dir)
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ChromaDB index reset path=%s", settings.chroma_dir)
    print(f"Reset ChromaDB index at: {settings.chroma_dir}")


def export_index(output_path: str | None) -> None:
    import shutil
    from pathlib import Path

    out = Path(output_path) if output_path else Path("chroma_db_export.zip")
    logger.info("Exporting ChromaDB index from %s to %s", settings.chroma_dir, out)
    if not settings.chroma_dir.exists():
        print(f"ChromaDB directory does not exist: {settings.chroma_dir}")
        return
    # shutil.make_archive expects a base name without extension
    base = out.with_suffix("")
    archive = shutil.make_archive(str(base), 'zip', root_dir=str(settings.chroma_dir))
    print(f"Exported ChromaDB to: {archive}")


def import_index(archive_path: str) -> None:
    from pathlib import Path
    import zipfile

    archive = Path(archive_path)
    if not archive.exists():
        print(f"Archive not found: {archive}")
        return
    logger.info("Importing ChromaDB index from %s to %s", archive, settings.chroma_dir)
    # Remove existing dir first
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(archive), 'r') as zf:
        zf.extractall(path=str(settings.chroma_dir))
    print(f"Imported ChromaDB to: {settings.chroma_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG assistant for data engineering documentation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Crawl documentation and build the ChromaDB index.")
    ingest_parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl per source.")
    ingest_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Documentation source name to ingest. Repeat to ingest multiple sources. Defaults to all sources.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the local repository.")
    ask_parser.add_argument("question", help="Question to answer.")

    subparsers.add_parser("reset-index", help="Delete the local ChromaDB index so ingestion can rebuild it.")
    export_parser = subparsers.add_parser("export-index", help="Export the ChromaDB directory to a zip archive.")
    export_parser.add_argument("--output", help="Output zip path. Defaults to chroma_db_export.zip.")
    import_parser = subparsers.add_parser("import-index", help="Import a ChromaDB zip archive into local chroma_db.")
    import_parser.add_argument("archive", help="Path to the chroma_db zip archive to import.")
    subparsers.add_parser("ui", help="Print the Streamlit command.")
    return parser


def main() -> None:
    if settings.logging_enabled:
        configure_logging(settings.project_root)
    parser = build_parser()
    args = parser.parse_args()
    logger.info("CLI command received command=%s", args.command)

    try:
        if args.command == "ingest":
            ingest(max_pages=args.max_pages, source_names=tuple(args.source) if args.source else None)
        elif args.command == "ask":
            ask(question=args.question)
        elif args.command == "reset-index":
            reset_index()
        elif args.command == "export-index":
            export_index(output_path=args.output)
        elif args.command == "import-index":
            import_index(args.archive)
        elif args.command == "ui":
            logger.info("CLI ui command displayed Streamlit launch command")
            print("Run: python -m streamlit run data_engineering_copilot/ui/streamlit_app.py")
    except Exception:
        logger.exception("CLI command failed command=%s", args.command)
        raise


if __name__ == "__main__":
    main()
```

### `requirements.txt`
```
beautifulsoup4>=4.12,<5
chromadb>=1.1,<2
sentence-transformers>=5,<6
streamlit>=1.50,<2
torch>=2.12
pytest>=8.0
```

### `setup.py`
```python
from __future__ import annotations

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


class InstallWithModels(_install):
    def run(self):
        # Run the standard install first
        _install.run(self)

        # Attempt to download embedding model into local cache
        script = os.path.join(os.path.dirname(__file__), "scripts", "download_embedding_model.py")
        if os.path.exists(script):
            try:
                print("Running embedding model cache script:", script)
                subprocess.check_call([sys.executable, script])
            except Exception as exc:  # pragma: no cover - best-effort at install time
                print("Warning: failed to cache embedding model during install:", exc)
        else:
            print("Embedding download script not found; skipping model cache step.")


setup(
    name="data-engineering-copilot",
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    cmdclass={"install": InstallWithModels},
)
```

### `data_engineering_copilot/factory.py`
```python
from __future__ import annotations

import logging

from data_engineering_copilot.config.settings import AppSettings, settings
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService
from data_engineering_copilot.services.rag import RagAnswerService


logger = logging.getLogger(__name__)


def build_ingestion_service(app_settings: AppSettings = settings) -> IngestionService:
    logger.info(
        "Building ingestion service sources=%s chroma_dir=%s collection=%s",
        len(app_settings.sources),
        app_settings.chroma_dir,
        app_settings.collection_name,
    )
    return IngestionService(
        settings=app_settings,
        crawler=DocumentationCrawler(
            timeout_seconds=app_settings.request_timeout_seconds,
            delay_seconds=app_settings.crawl_delay_seconds,
        ),
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=app_settings.chunk_size_words,
            overlap_words=app_settings.chunk_overlap_words,
        ),
        embeddings=SentenceTransformerEmbeddings(
            model_name=app_settings.embedding_model_name,
            cache_dir=app_settings.embedding_cache_dir,
            local_files_only=app_settings.embedding_local_files_only,
        ),
        vector_store=ChromaVectorStore(
            persist_directory=str(app_settings.chroma_dir),
            collection_name=app_settings.collection_name,
        ),
    )


def build_rag_service(app_settings: AppSettings = settings) -> RagAnswerService:
    logger.info(
        "Building RAG service model=%s top_k=%s max_context_chars=%s threshold=%.4f",
        app_settings.ollama_model,
        app_settings.retrieval_top_k,
        app_settings.max_context_chars,
        app_settings.confidence_threshold,
    )
    return RagAnswerService(
        embeddings=SentenceTransformerEmbeddings(
            model_name=app_settings.embedding_model_name,
            cache_dir=app_settings.embedding_cache_dir,
            local_files_only=app_settings.embedding_local_files_only,
        ),
        vector_store=ChromaVectorStore(
            persist_directory=str(app_settings.chroma_dir),
            collection_name=app_settings.collection_name,
        ),
        ollama_client=OllamaClient(
            base_url=app_settings.ollama_base_url,
            model=app_settings.ollama_model,
            timeout_seconds=app_settings.ollama_timeout_seconds,
            num_ctx=app_settings.ollama_num_ctx,
            num_predict=app_settings.ollama_num_predict,
        ),
        top_k=app_settings.retrieval_top_k,
        max_context_chars=app_settings.max_context_chars,
        confidence_threshold=app_settings.confidence_threshold,
        retry_context_ratio=app_settings.ollama_retry_context_ratio,
        retry_extra_num_predict=app_settings.ollama_retry_extra_num_predict,
        retry_max_num_predict=app_settings.ollama_retry_max_num_predict,
    )
```

### `data_engineering_copilot/logging_config.py`
```python
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "data_engineering_copilot"
DEFAULT_LOG_FILENAME = "application.log"


def configure_logging(project_root: Path, level: int = logging.INFO) -> Path:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / DEFAULT_LOG_FILENAME

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    resolved_log_path = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == resolved_log_path:
            return log_path

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.info("Logging configured path=%s level=%s", log_path, logging.getLevelName(level))
    return log_path
```

### `data_engineering_copilot/config/__init__.py`
```python
"""Application configuration."""
```

### `data_engineering_copilot/config/settings.py`
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DocumentationSource:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    url_prefixes: tuple[str, ...] = ()


def load_documentation_sources(config_path: Path) -> tuple[DocumentationSource, ...]:
    with config_path.open("r", encoding="utf-8") as file:
        raw_sources = json.load(file)

    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Documentation source config must contain a non-empty list: {config_path}")

    sources: list[DocumentationSource] = []
    for index, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"Documentation source #{index} must be an object.")

        name = _required_string(raw_source, "name", index)
        start_urls = _required_string_tuple(raw_source, "start_urls", index)
        allowed_domains = _required_string_tuple(raw_source, "allowed_domains", index)
        url_prefixes = _optional_string_tuple(raw_source, "url_prefixes", index)

        sources.append(
            DocumentationSource(
                name=name,
                start_urls=start_urls,
                allowed_domains=allowed_domains,
                url_prefixes=url_prefixes,
            )
        )

    return tuple(sources)


def _required_string(raw_source: dict, field_name: str, index: int) -> str:
    value = raw_source.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Documentation source #{index} must define a non-empty `{field_name}` string.")
    return value.strip()


def _required_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = _optional_string_tuple(raw_source, field_name, index)
    if not value:
        raise ValueError(f"Documentation source #{index} must define at least one `{field_name}` value.")
    return value


def _optional_string_tuple(raw_source: dict, field_name: str, index: int) -> tuple[str, ...]:
    value = raw_source.get(field_name, [])
    if not isinstance(value, list):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Documentation source #{index} field `{field_name}` must contain only non-empty strings.")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    documentation_sources_path: Path = PROJECT_ROOT / "data_engineering_copilot" / "config" / "documentation_sources.json"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "embedding_models"
    collection_name: str = "data_engineering_docs"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    chunk_size_words: int = 350
    chunk_overlap_words: int = 70
    retrieval_top_k: int = 3
    max_context_chars: int = 1500
    confidence_threshold: float = 0.35
    request_timeout_seconds: int = 15
    ollama_timeout_seconds: int = 240
    ollama_num_ctx: int = 2048
    ollama_num_predict: int = 512
    ollama_retry_context_ratio: float = 0.5
    ollama_retry_extra_num_predict: int = 512
    ollama_retry_max_num_predict: int = 1024
    crawl_delay_seconds: float = 0.2
    max_pages_per_source: int = 80
    ingestion_batch_chunk_size: int = 256
    logging_enabled: bool = False
    sources: tuple[DocumentationSource, ...] = load_documentation_sources(documentation_sources_path)


settings = AppSettings()
```

### `data_engineering_copilot/config/documentation_sources.json`
```json
[
  {
    "name": "Apache Spark Documentation",
    "start_urls": [
      "https://spark.apache.org/docs/latest/"
    ],
    "allowed_domains": [
      "spark.apache.org"
    ],
    "url_prefixes": [
      "https://spark.apache.org/docs/latest/"
    ]
  },
  {
    "name": "Apache Airflow Documentation",
    "start_urls": [
      "https://airflow.apache.org/docs/apache-airflow/stable/"
    ],
    "allowed_domains": [
      "airflow.apache.org"
    ],
    "url_prefixes": [
      "https://airflow.apache.org/docs/apache-airflow/stable/"
    ]
  },
  {
    "name": "Databricks Documentation",
    "start_urls": [
      "https://docs.databricks.com/aws/en/"
    ],
    "allowed_domains": [
      "docs.databricks.com"
    ],
    "url_prefixes": [
      "https://docs.databricks.com/aws/en/"
    ]
  },
  {
    "name": "Delta Lake Documentation",
    "start_urls": [
      "https://docs.delta.io/latest/"
    ],
    "allowed_domains": [
      "docs.delta.io"
    ],
    "url_prefixes": [
      "https://docs.delta.io/latest/"
    ]
  }
]
```

### `data_engineering_copilot/domain/models.py`
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDocument:
    source_name: str
    url: str
    html: str


@dataclass(frozen=True)
class IngestionEvent:
    event_type: str
    source_name: str
    message: str
    url: str | None = None
    title: str | None = None
    chunks_indexed: int = 0
    pages_fetched: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    source_name: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source_name: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    distance: float
    confidence: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[DocumentChunk, ...]
    confidence: float
```

### `data_engineering_copilot/domain/__init__.py`
```python
"""Domain objects used across the application."""
```

### `data_engineering_copilot/infrastructure/crawler.py`
```python
from __future__ import annotations

import logging
import time
from collections import deque
from html.parser import HTMLParser
from typing import Callable, Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.domain.models import IngestionEvent, RawDocument


logger = logging.getLogger(__name__)


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


class DocumentationCrawler:
    def __init__(self, timeout_seconds: int, delay_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds

    def crawl(
        self,
        source: DocumentationSource,
        max_pages: int,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> Iterable[RawDocument]:
        visited: set[str] = set()
        queued: set[str] = set()
        queue: deque[str] = deque(self._clean_url(url) for url in source.start_urls)
        queued.update(self._dedupe_key(url) for url in queue)
        logger.info(
            "Crawler started source=%s max_pages=%s start_urls=%s",
            source.name,
            max_pages,
            len(source.start_urls),
        )

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            url_key = self._dedupe_key(url)
            if url_key in visited or not self._is_allowed(url, source):
                continue

            self._emit(
                on_event,
                IngestionEvent(
                    event_type="fetch_start",
                    source_name=source.name,
                    url=url,
                    message=f"Fetching HTML page: {url}",
                ),
            )
            try:
                html = self._download(url)
            except Exception as exc:
                visited.add(url_key)
                message = f"Skipping {url}: {exc}"
                print(message)
                logger.warning(
                    "Crawler fetch failed source=%s url=%s pages_fetched=%s error=%s",
                    source.name,
                    url,
                    len(visited),
                    exc,
                )
                self._emit(
                    on_event,
                    IngestionEvent(
                        event_type="fetch_error",
                        source_name=source.name,
                        url=url,
                        message=message,
                        pages_fetched=len(visited),
                        error=str(exc),
                    ),
                )
                continue

            visited.add(url_key)
            logger.info("Crawler fetch succeeded source=%s url=%s pages_fetched=%s", source.name, url, len(visited))
            self._emit(
                on_event,
                IngestionEvent(
                    event_type="fetch_success",
                    source_name=source.name,
                    url=url,
                    message=f"Fetched HTML page: {url}",
                    pages_fetched=len(visited),
                ),
            )
            yield RawDocument(source_name=source.name, url=url, html=html)

            for link in self._extract_links(html, url):
                link_key = self._dedupe_key(link)
                if link_key not in visited and link_key not in queued and self._is_allowed(link, source):
                    queue.append(link)
                    queued.add(link_key)

            time.sleep(self.delay_seconds)

        logger.info(
            "Crawler completed source=%s pages_visited=%s queued_remaining=%s",
            source.name,
            len(visited),
            len(queue),
        )

    def _download(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "DataEngineeringCopilot/1.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError(f"unsupported content type: {content_type}")
            return response.read().decode("utf-8", errors="replace")

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        parser = LinkExtractor()
        parser.feed(html)
        links: list[str] = []
        for href in parser.links:
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            links.append(self._clean_url(urljoin(base_url, href)))
        return links

    def _clean_url(self, url: str) -> str:
        return urldefrag(url)[0]

    def _dedupe_key(self, url: str) -> str:
        clean_url = self._clean_url(url)
        parsed = urlparse(clean_url)
        if parsed.path.endswith("/index.html"):
            clean_url = clean_url[: -len("index.html")]
            parsed = urlparse(clean_url)
        if parsed.path == "/":
            return clean_url
        return clean_url.rstrip("/")

    def _is_allowed(self, url: str, source: DocumentationSource) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.netloc not in source.allowed_domains:
            return False
        if source.url_prefixes and not any(url.startswith(prefix.rstrip("/")) for prefix in source.url_prefixes):
            return False
        return True

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
```

### `data_engineering_copilot/infrastructure/embeddings.py`
```python
from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, cache_dir: Path, local_files_only: bool) -> None:
        logger.info(
            "Loading embedding model model=%s cache_dir=%s local_files_only=%s",
            model_name,
            cache_dir,
            local_files_only,
        )
        # transformers deprecates the `cache_dir`/`cache_folder` argument; pass via kwargs
        try:
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={"cache_dir": str(cache_dir)},
                config_kwargs={"cache_dir": str(cache_dir)},
                processor_kwargs={"cache_dir": str(cache_dir)},
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - runtime HF errors depend on environment
            msg = str(exc)
            if local_files_only and (
                "outgoing traffic has been disabled" in msg
                or "Cannot find the requested files in the disk cache" in msg
                or "LocalEntryNotFoundError" in msg
            ):
                raise RuntimeError(
                    "Embedding model not found in local cache and downloads are disabled. "
                    "Either run `python scripts/download_embedding_model.py` to cache the model locally, "
                    "or set `embedding_local_files_only=False` in your AppSettings to allow downloads."
                ) from exc
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("Embedded texts count=%s", len(texts))
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
```

### `data_engineering_copilot/infrastructure/html_parser.py`
```python
from __future__ import annotations

import logging

from bs4 import BeautifulSoup

from data_engineering_copilot.domain.models import ParsedDocument, RawDocument
from data_engineering_copilot.utils.text import normalize_whitespace


logger = logging.getLogger(__name__)


class DocumentationHtmlParser:
    def parse(self, raw: RawDocument) -> ParsedDocument | None:
        soup = BeautifulSoup(raw.html, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = self._title(soup, raw.url)
        content = soup.find("main") or soup.find("article") or soup.body or soup
        text = normalize_whitespace(content.get_text(" "))

        if len(text.split()) < 40:
            logger.info("HTML parser skipped short page source=%s url=%s title=%r", raw.source_name, raw.url, title)
            return None

        logger.info(
            "HTML parser extracted document source=%s url=%s title=%r words=%s",
            raw.source_name,
            raw.url,
            title,
            len(text.split()),
        )
        return ParsedDocument(
            source_name=raw.source_name,
            title=title,
            url=raw.url,
            text=text,
        )

    def _title(self, soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find("h1")
        if heading:
            return normalize_whitespace(heading.get_text(" "))
        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string)
        return fallback
```

### `data_engineering_copilot/infrastructure/ollama_client.py`
```python
from __future__ import annotations

import json
import logging
import re
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when local Ollama cannot return an answer."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        num_ctx: int,
        num_predict: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, prompt: str, num_predict: int | None = None, num_ctx: int | None = None) -> str:
        logger.info(
            "Ollama generation started model=%s prompt_chars=%s num_ctx=%s num_predict=%s",
            self.model,
            len(prompt),
            self.num_ctx,
            self.num_predict,
        )
        if num_predict is None:
            num_predict = self.num_predict
        if num_ctx is None:
            num_ctx = self.num_ctx

        payload = {
            "model": self.model,
            "prompt": self._format_raw_chat_prompt(prompt),
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.8,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            logger.exception("Ollama generation timed out timeout_seconds=%s", self.timeout_seconds)
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except socket.timeout as exc:
            logger.exception("Ollama generation socket timeout timeout_seconds=%s", self.timeout_seconds)
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds} seconds. "
                "Try again after Ollama finishes loading the model, or reduce the configured context/output limits."
            ) from exc
        except URLError as exc:
            logger.exception("Ollama connection failed base_url=%s", self.base_url)
            raise OllamaError(
                "Could not reach Ollama. Start Ollama and run: ollama pull deepseek-coder:6.7b"
            ) from exc

        response = self._extract_final_response(str(body.get("response", "")))
        done_reason = body.get("done_reason", "unknown")
        logger.info(
            "Ollama generation completed done_reason=%s raw_response_chars=%s final_response_chars=%s",
            done_reason,
            len(str(body.get("response", ""))),
            len(response),
        )
        if not response:
            logger.warning("Ollama returned no final answer done_reason=%s body_keys=%s", done_reason, sorted(body))
            raise OllamaError(
                "Ollama returned no final answer. "
                f"Generation stopped with reason `{done_reason}`. "
                "The model likely spent its output budget on reasoning. "
                "Try again, or increase `ollama_num_predict` in settings.py."
            )
        return response

    def _extract_final_response(self, response: str) -> str:
        response = response.strip()
        if not response:
            return ""

        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        if response.lower().startswith("<think>"):
            return ""
        return response

    def _format_raw_chat_prompt(self, user_prompt: str) -> str:
        return "\n".join(
            [
                "You are DataEngineeringCopilot. Answer concisely using only provided context.",
                "- No invented information.",
                "- Brief and practical.",
                "",
                user_prompt,
                "",
                "Answer:",
            ]
        )
```

### `data_engineering_copilot/infrastructure/vector_store.py`
```python
from __future__ import annotations

import logging

import chromadb
from chromadb.errors import InternalError

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk


logger = logging.getLogger(__name__)


class VectorStoreReadError(RuntimeError):
    """Raised when the persisted Chroma index cannot be read."""


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        logger.info("Opening Chroma vector store path=%s collection=%s", persist_directory, collection_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            logger.info("Skipping vector upsert because chunk list is empty")
            return
        if len(chunks) != len(embeddings):
            logger.error("Vector upsert length mismatch chunks=%s embeddings=%s", len(chunks), len(embeddings))
            raise ValueError("chunks and embeddings must have the same length")

        logger.info("Upserting chunks count=%s first_chunk_id=%s", len(chunks), chunks[0].chunk_id)
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_name": chunk.source_name,
                    "title": chunk.title,
                    "url": chunk.url,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        logger.info("Vector query started top_k=%s embedding_dimensions=%s", top_k, len(query_embedding))
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except InternalError as exc:
            if "Nothing found on disk" in str(exc):
                logger.exception("Vector query failed because Chroma index is incomplete")
                raise VectorStoreReadError(
                    "The ChromaDB index is incomplete or corrupted. Run `python main.py reset-index`, then ingest again."
                ) from exc
            logger.exception("Vector query failed with Chroma internal error")
            raise

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            confidence = max(0.0, min(1.0, 1.0 - float(distance)))
            chunk = DocumentChunk(
                chunk_id=str(metadata["chunk_id"]),
                source_name=str(metadata["source_name"]),
                title=str(metadata["title"]),
                url=str(metadata["url"]),
                text=text,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, distance=float(distance), confidence=confidence))
        logger.info(
            "Vector query completed results=%s top_confidence=%.4f",
            len(retrieved),
            retrieved[0].confidence if retrieved else 0.0,
        )
        return retrieved

    def count(self) -> int:
        try:
            count = self.collection.count()
            logger.info("Vector store count completed count=%s", count)
            return count
        except InternalError as exc:
            if "Nothing found on disk" in str(exc):
                logger.exception("Vector count failed because Chroma index is incomplete")
                raise VectorStoreReadError(
                    "The ChromaDB index is incomplete or corrupted. Run `python main.py reset-index`, then ingest again."
                ) from exc
            logger.exception("Vector count failed with Chroma internal error")
            raise
```

### `data_engineering_copilot/services/rag.py`
```python
from __future__ import annotations

import logging

from data_engineering_copilot.domain.models import Answer, RetrievedChunk
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore, VectorStoreReadError


OUTSIDE_REPOSITORY_MESSAGE = "I cannot answer this question because it is outside my knowledge repository."
logger = logging.getLogger(__name__)


class RagAnswerService:
    def __init__(
        self,
        embeddings: SentenceTransformerEmbeddings,
        vector_store: ChromaVectorStore,
        ollama_client: OllamaClient,
        top_k: int,
        max_context_chars: int,
        confidence_threshold: float,
        retry_context_ratio: float,
        retry_extra_num_predict: int,
        retry_max_num_predict: int,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.confidence_threshold = confidence_threshold
        self.retry_context_ratio = retry_context_ratio
        self.retry_extra_num_predict = retry_extra_num_predict
        self.retry_max_num_predict = retry_max_num_predict

    def answer(self, question: str) -> Answer:
        logger.info("RAG answer started question=%r", question[:200])
        query_embedding = self.embeddings.embed_query(question)
        try:
            matches = self.vector_store.query(query_embedding, top_k=self.top_k)
        except VectorStoreReadError:
            logger.exception("Vector store unreadable during RAG answer")
            return Answer(text=OUTSIDE_REPOSITORY_MESSAGE, sources=(), confidence=0.0)
        confidence = matches[0].confidence if matches else 0.0
        logger.info(
            "RAG retrieval completed matches=%s top_confidence=%.4f threshold=%.4f",
            len(matches),
            confidence,
            self.confidence_threshold,
        )

        if confidence < self.confidence_threshold:
            logger.info("RAG answer rejected by confidence gate confidence=%.4f", confidence)
            return Answer(text=OUTSIDE_REPOSITORY_MESSAGE, sources=(), confidence=confidence)

        prompt = self._build_prompt(question, matches)
        logger.info("RAG sending prompt prompt_chars=%s", len(prompt))
        try:
            generated = self.ollama_client.generate(prompt)
        except OllamaError as exc:
            logger.warning("Ollama generation failed first attempt: %s", exc)
            if "length" in str(exc).lower():
                reduced_chars = max(200, int(self.max_context_chars * self.retry_context_ratio))
                logger.info("Retrying Ollama with reduced context max_context_chars=%s", reduced_chars)
                prompt_retry = self._build_prompt(question, matches, max_context_chars=reduced_chars)
                logger.info("RAG retry sending prompt prompt_chars=%s", len(prompt_retry))
                try:
                    generated = self.ollama_client.generate(prompt_retry)
                except OllamaError as exc2:
                    logger.warning("Ollama retry with reduced context failed: %s", exc2)
                    try:
                        original_np = getattr(self.ollama_client, "num_predict", None) or 0
                        increased_np = min(original_np + self.retry_extra_num_predict, self.retry_max_num_predict)
                        logger.info("Final attempt: increasing num_predict to %s and retrying", increased_np)
                        generated = self.ollama_client.generate(prompt_retry, num_predict=increased_np)
                    except OllamaError:
                        logger.exception("Ollama failed during RAG answer on final retry")
                        return Answer(text=str(exc), sources=self._unique_sources(matches), confidence=confidence)
            else:
                logger.exception("Ollama failed during RAG answer")
                return Answer(text=str(exc), sources=self._unique_sources(matches), confidence=confidence)
        unique_sources = self._unique_sources(matches)
        logger.info(
            "RAG answer completed answer_chars=%s sources=%s confidence=%.4f",
            len(generated),
            len(unique_sources),
            confidence,
        )
        return Answer(text=generated, sources=unique_sources, confidence=confidence)

    def _build_prompt(self, question: str, matches: list[RetrievedChunk], max_context_chars: int = None) -> str:
        if max_context_chars is None:
            max_context_chars = self.max_context_chars
        context_blocks: list[str] = []
        used_chars = 0
        for index, match in enumerate(matches, start=1):
            chunk = match.chunk
            remaining_chars = max(0, max_context_chars - used_chars)
            if remaining_chars <= 0:
                break
            headroom = 200
            chunk_text = chunk.text[: max(0, remaining_chars - headroom)]
            used_chars += len(chunk_text)
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Source: {chunk.source_name}",
                        f"Title: {chunk.title}",
                        f"URL: {chunk.url}",
                        f"Chunk ID: {chunk.chunk_id}",
                        f"Text: {chunk_text}",
                    ]
                )
            )

        context = "\n\n".join(context_blocks)
        logger.info(
            "RAG prompt built context_chars=%s blocks=%s max_context_chars=%s",
            len(context),
            len(context_blocks),
            max_context_chars,
        )
        return f"""You are DataEngineeringCopilot, an offline assistant for data engineering documentation.
    Answer only from the provided repository context.
    If the context does not contain the answer, reply exactly:
    {OUTSIDE_REPOSITORY_MESSAGE}

    Provide a concise, practical answer using at most 3 bullet points or a single short paragraph.
    Limit the answer to approximately 150 words (or ~800 characters). If the answer exceeds this, summarize the key points.
    Prefer direct facts, commands, and caveats from the context. Do not show hidden reasoning. Do not invent sources.

    Repository context:
    {context}

    Question:
    {question}

    Answer:"""

    def _unique_sources(self, matches: list[RetrievedChunk]) -> tuple:
        seen: set[tuple[str, str]] = set()
        sources = []
        for match in matches:
            chunk = match.chunk
            key = (chunk.title, chunk.url)
            if key in seen:
                continue
            seen.add(key)
            sources.append(chunk)
        return tuple(sources)
```

### `data_engineering_copilot/services/chunker.py`
```python
from __future__ import annotations

import hashlib
import logging

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify


logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size_words: int, overlap_words: int) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be >= 0 and less than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        words = document.text.split()
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        step = self.chunk_size_words - self.overlap_words

        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            text = " ".join(words[start:end])
            chunk_id = self._chunk_id(document, index)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=text,
                )
            )
            if end == len(words):
                break
            start += step
            index += 1

        logger.info(
            "Chunked document source=%s url=%s title=%r words=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            len(words),
            len(chunks),
        )
        return chunks

    def _chunk_id(self, document: ParsedDocument, index: int) -> str:
        digest = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
        source = slugify(document.source_name)
        return f"{source}:{digest}:{index:04d}"
```

### `data_engineering_copilot/services/__init__.py`
```python
"""Application services."""
```

### `data_engineering_copilot/services/ingestion.py`
```python
from __future__ import annotations

import logging
from typing import Callable, Iterable

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker


logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: AppSettings,
        crawler: DocumentationCrawler,
        parser: DocumentationHtmlParser,
        chunker: DocumentChunker,
        embeddings: SentenceTransformerEmbeddings,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.settings = settings
        self.crawler = crawler
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store

    def ingest(
        self,
        max_pages_per_source: int | None = None,
        source_names: Iterable[str] | None = None,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> int:
        page_limit = max_pages_per_source or self.settings.max_pages_per_source
        total_chunks = 0
        selected_sources = self._selected_sources(source_names)
        logger.info(
            "Ingestion started page_limit=%s sources=%s",
            page_limit,
            [source.name for source in selected_sources],
        )

        batch_chunks: list[DocumentChunk] = []

        def flush_batch() -> None:
            if not batch_chunks:
                return
            batch_vectors = self.embeddings.embed_texts([chunk.text for chunk in batch_chunks])
            self.vector_store.upsert_chunks(batch_chunks, batch_vectors)
            batch_chunks.clear()

        for source in selected_sources:
            print(f"Crawling {source.name}")
            logger.info("Ingestion source started source=%s page_limit=%s", source.name, page_limit)
            source_pages_fetched = 0
            source_chunks_indexed = 0
            self._emit(
                on_event,
                IngestionEvent(
                    event_type="source_start",
                    source_name=source.name,
                    message=f"Crawling {source.name}",
                ),
            )
            for raw_document in self.crawler.crawl(source, max_pages=page_limit, on_event=on_event):
                source_pages_fetched += 1
                parsed = self.parser.parse(raw_document)
                if parsed is None:
                    logger.info("Ingestion skipped unreadable page source=%s url=%s", raw_document.source_name, raw_document.url)
                    self._emit(
                        on_event,
                        IngestionEvent(
                            event_type="page_skipped",
                            source_name=raw_document.source_name,
                            url=raw_document.url,
                            message=f"Skipped page with no readable documentation content: {raw_document.url}",
                            pages_fetched=source_pages_fetched,
                        ),
                    )
                    continue
                chunks = self.chunker.chunk(parsed)
                batch_chunks.extend(chunks)

                if len(batch_chunks) >= self.settings.ingestion_batch_chunk_size:
                    flush_batch()

                total_chunks += len(chunks)
                source_chunks_indexed += len(chunks)
                print(f"Indexed {len(chunks):>3} chunks from {parsed.title}")
                logger.info(
                    "Ingestion indexed page source=%s url=%s title=%r chunks=%s total_chunks=%s",
                    parsed.source_name,
                    parsed.url,
                    parsed.title,
                    len(chunks),
                    total_chunks,
                )
                self._emit(
                    on_event,
                    IngestionEvent(
                        event_type="page_indexed",
                        source_name=parsed.source_name,
                        url=parsed.url,
                        title=parsed.title,
                        chunks_indexed=len(chunks),
                        pages_fetched=source_pages_fetched,
                        message=f"Indexed {len(chunks)} chunks from {parsed.title}",
                    ),
                )

            self._emit(
                on_event,
                IngestionEvent(
                    event_type="source_complete",
                    source_name=source.name,
                    chunks_indexed=source_chunks_indexed,
                    pages_fetched=source_pages_fetched,
                    message=(
                        f"Completed {source.name}: fetched {source_pages_fetched} HTML pages, "
                        f"indexed {source_chunks_indexed} chunks."
                    ),
                ),
            )
            logger.info(
                "Ingestion source completed source=%s pages=%s chunks=%s",
                source.name,
                source_pages_fetched,
                source_chunks_indexed,
            )

        flush_batch()

        logger.info("Ingestion completed total_chunks=%s", total_chunks)
        return total_chunks

    def _selected_sources(self, source_names: Iterable[str] | None):
        if source_names is None:
            return self.settings.sources

        requested_names = tuple(name.strip() for name in source_names if name.strip())
        if not requested_names:
            logger.error("Ingestion source selection failed because no source names were selected")
            raise ValueError("At least one documentation source must be selected.")

        sources_by_name = {source.name: source for source in self.settings.sources}
        unknown_names = sorted(set(requested_names) - set(sources_by_name))
        if unknown_names:
            available_names = ", ".join(sources_by_name)
            logger.error("Ingestion source selection failed unknown=%s available=%s", unknown_names, available_names)
            raise ValueError(
                f"Unknown documentation source(s): {', '.join(unknown_names)}. "
                f"Available sources: {available_names}"
            )

        return tuple(sources_by_name[name] for name in requested_names)

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
```

### `data_engineering_copilot/ui/__init__.py`
```python
"""Streamlit UI package."""
```

### `data_engineering_copilot/ui/streamlit_app.py`
```python
from __future__ import annotations

from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Callable

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.factory import build_ingestion_service, build_rag_service
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore, VectorStoreReadError
from data_engineering_copilot.logging_config import configure_logging


if settings.logging_enabled:
    configure_logging(settings.project_root)
logger = logging.getLogger(__name__)


@st.cache_resource
def rag_service():
    logger.info("Streamlit cached RAG service requested")
    return build_rag_service()


@st.cache_resource
def vector_store():
    logger.info("Streamlit cached vector store requested")
    return ChromaVectorStore(str(settings.chroma_dir), settings.collection_name)


def run_ingestion_refresh(
    max_pages_per_source: int,
    source_names: tuple[str, ...],
    on_event: Callable[[IngestionEvent], None] | None = None,
) -> int:
    logger.info("Streamlit ingestion refresh started max_pages=%s sources=%s", max_pages_per_source, source_names)
    service = build_ingestion_service()
    total_chunks = service.ingest(
        max_pages_per_source=max_pages_per_source,
        source_names=source_names,
        on_event=on_event,
    )
    logger.info("Streamlit ingestion refresh completed chunks=%s", total_chunks)
    return total_chunks


def ingestion_log_path() -> Path:
    return settings.project_root / "logs" / "ingestion_refresh.log"


def append_ingestion_log(log_path: Path, event: IngestionEvent) -> None:
    if not settings.logging_enabled:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    parts = [
        timestamp,
        f"event={event.event_type}",
        f"source={event.source_name}",
    ]
    if event.url:
        parts.append(f"url={event.url}")
    if event.title:
        parts.append(f"title={event.title}")
    if event.pages_fetched:
        parts.append(f"pages_fetched={event.pages_fetched}")
    if event.chunks_indexed:
        parts.append(f"chunks_indexed={event.chunks_indexed}")
    if event.error:
        parts.append(f"error={event.error}")
    parts.append(f"message={event.message}")
    with log_path.open("a", encoding="utf-8") as file:
        file.write(" | ".join(parts) + "\n")
    logger.info("Ingestion UI event logged event=%s source=%s url=%s", event.event_type, event.source_name, event.url)


def main() -> None:
    logger.info("Streamlit app render started")
    st.set_page_config(page_title="DataEngineeringCopilot", layout="wide")
    st.title("DataEngineeringCopilot")
    st.caption("Offline RAG over Spark, Airflow, Databricks, and Delta Lake documentation.")

    with st.sidebar:
        st.subheader("Repository")
        try:
            chunk_count = vector_store().count()
        except VectorStoreReadError as exc:
            chunk_count = 0
            logger.exception("Streamlit sidebar vector count failed")
            st.warning(str(exc))
        st.write(f"Chunks indexed: {chunk_count}")
        st.write(f"Ollama model: `{settings.ollama_model}`")
        st.write(f"Ollama timeout: `{settings.ollama_timeout_seconds}s`")
        st.write(f"Ollama output limit: `{settings.ollama_num_predict}` tokens")
        st.write(f"Embedding model: `{settings.embedding_model_name}`")
        st.write(f"Confidence threshold: `{settings.confidence_threshold}`")
        if settings.logging_enabled:
            st.write(f"Application log: `{settings.project_root / 'logs' / 'application.log'}`")
        else:
            st.write("Application logging is disabled.")
        st.divider()
        st.subheader("Ingestion")
        source_names = tuple(source.name for source in settings.sources)
        selected_source_names = tuple(
            st.multiselect(
                "Sources to ingest",
                options=source_names,
                default=source_names,
            )
        )
        max_pages = st.number_input(
            "Max pages per source",
            min_value=1,
            max_value=5000,
            value=min(settings.max_pages_per_source, 5000),
            step=10,
        )
        with st.expander("Documentation sources"):
            for source in settings.sources:
                st.markdown(f"**{source.name}**")
                st.write(f"Start URLs: {len(source.start_urls)}")
                for url in source.start_urls:
                    st.markdown(f"- [{url}]({url})")
                st.write(f"Allowed domains: `{', '.join(source.allowed_domains)}`")
                if source.url_prefixes:
                    st.write("URL prefixes:")
                    for prefix in source.url_prefixes:
                        st.markdown(f"- `{prefix}`")
                else:
                    st.write("URL prefixes: all paths on allowed domains")
                st.caption(f"Refresh limit: up to {int(max_pages)} HTML pages for this source")
        refresh_disabled = not selected_source_names
        if refresh_disabled:
            st.warning("Select at least one documentation source to refresh.")
        if st.button(
            "Refresh Documentation",
            type="secondary",
            use_container_width=True,
            disabled=refresh_disabled,
        ):
            log_path = ingestion_log_path()
            status_box = st.empty()
            metrics_box = st.empty()
            urls_box = st.empty()
            log_box = st.empty()
            recent_urls: list[str] = []
            pages_by_source: dict[str, int] = {}
            chunks_by_source: dict[str, int] = {}

            def handle_ingestion_event(event: IngestionEvent) -> None:
                append_ingestion_log(log_path, event)
                if event.event_type == "fetch_success":
                    pages_by_source[event.source_name] = event.pages_fetched
                if event.event_type == "page_indexed":
                    chunks_by_source[event.source_name] = chunks_by_source.get(event.source_name, 0) + event.chunks_indexed
                if event.url and event.event_type in {"fetch_start", "fetch_success", "fetch_error"}:
                    label = "fetching" if event.event_type == "fetch_start" else event.event_type.replace("_", " ")
                    recent_urls.insert(0, f"- `{label}` [{event.url}]({event.url})")
                    del recent_urls[25:]

                total_pages = sum(pages_by_source.values())
                total_chunks = sum(chunks_by_source.values())
                status_box.info(event.message)
                metrics_box.write(f"HTML pages fetched: `{total_pages}` | Chunks indexed: `{total_chunks}`")
                if recent_urls:
                    urls_box.markdown("**Recent HTML page URLs**\n\n" + "\n".join(recent_urls))
                log_box.caption(f"Refresh log: `{log_path}`")

            with st.spinner("Crawling documentation and updating ChromaDB..."):
                try:
                    indexed_chunks = run_ingestion_refresh(
                        max_pages_per_source=int(max_pages),
                        source_names=selected_source_names,
                        on_event=handle_ingestion_event,
                    )
                except Exception as exc:
                    logger.exception("Streamlit ingestion refresh failed")
                    st.error(f"Ingestion failed: {exc}")
                    log_box.caption(f"Refresh log: `{log_path}`")
                else:
                    rag_service.clear()
                    vector_store.clear()
                    st.success(f"Refresh complete. Indexed or updated {indexed_chunks} chunks.")
                    st.caption(f"Refresh log saved to: `{log_path}`")
                    st.rerun()

    question = st.text_area("Question", placeholder="How do I configure Spark dynamic allocation?", height=120)
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            logger.info("Streamlit ask ignored because question was empty")
            st.warning("Enter a question.")
            return

        logger.info("Streamlit ask started question=%r", question.strip()[:200])
        with st.spinner("Searching local repository and asking Ollama..."):
            answer = rag_service().answer(question.strip())
        logger.info(
            "Streamlit ask completed confidence=%.4f sources=%s answer_chars=%s",
            answer.confidence,
            len(answer.sources),
            len(answer.text),
        )

        st.subheader("Answer")
        st.write(answer.text)
        st.caption(f"Confidence: {answer.confidence:.2f}")

        if answer.sources:
            st.subheader("Sources")
            for source in answer.sources:
                st.markdown(f"- [{source.title}]({source.url})")


if __name__ == "__main__":
    main()
```

### `data_engineering_copilot/utils/text.py`
```python
from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "document"
```

### `data_engineering_copilot/utils/__init__.py`
```python
"""Utility helpers."""
```

### `scripts/download_embedding_model.py`
```python
from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings


def main() -> None:
    settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    # Use the newer kwargs to pass cache dir to transformers internals
    SentenceTransformer(
        settings.embedding_model_name,
        model_kwargs={"cache_dir": str(settings.embedding_cache_dir)},
        config_kwargs={"cache_dir": str(settings.embedding_cache_dir)},
        processor_kwargs={"cache_dir": str(settings.embedding_cache_dir)},
        local_files_only=False,
    )
    print(f"Cached embedding model: {settings.embedding_model_name}")
    print(f"Cache directory: {settings.embedding_cache_dir}")


if __name__ == "__main__":
    main()
```

### `tests/test_chunker.py`
```python
from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker


def test_chunker_preserves_required_metadata():
    document = ParsedDocument(
        source_name="Apache Spark Documentation",
        title="Spark SQL",
        url="https://spark.apache.org/docs/latest/sql-programming-guide.html",
        text=" ".join(f"word{i}" for i in range(30)),
    )

    chunks = DocumentChunker(chunk_size_words=10, overlap_words=2).chunk(document)

    assert len(chunks) == 4
    assert chunks[0].source_name == "Apache Spark Documentation"
    assert chunks[0].title == "Spark SQL"
    assert chunks[0].url == document.url
    assert chunks[0].chunk_id.startswith("apache-spark-documentation:")
```

### `tests/test_crawler.py`
```python
from data_engineering_copilot.config.settings import DocumentationSource
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler


class FakeCrawler(DocumentationCrawler):
    def __init__(self, pages: dict[str, str]) -> None:
        super().__init__(timeout_seconds=1, delay_seconds=0)
        self.pages = pages

    def _download(self, url: str) -> str:
        return self.pages[url]


def test_crawler_resolves_relative_links_from_directory_url():
    source = DocumentationSource(
        name="Apache Spark Documentation",
        start_urls=("https://spark.apache.org/docs/latest/",),
        allowed_domains=("spark.apache.org",),
        url_prefixes=("https://spark.apache.org/docs/latest/",),
    )
    crawler = FakeCrawler(
        {
            "https://spark.apache.org/docs/latest/": """
                <html><body>
                    <main><p>{}</p></main>
                    <a href="index.html">Overview</a>
                    <a href="quick-start.html">Quick Start</a>
                </body></html>
            """.format("overview " * 40),
            "https://spark.apache.org/docs/latest/quick-start.html": """
                <html><body><main><p>{}</p></main></body></html>
            """.format("quickstart " * 40),
        }
    )

    documents = list(crawler.crawl(source, max_pages=10))

    assert [document.url for document in documents] == [
        "https://spark.apache.org/docs/latest/",
        "https://spark.apache.org/docs/latest/quick-start.html",
    ]


def test_crawler_deduplicates_trailing_slash_variants():
    source = DocumentationSource(
        name="Example Docs",
        start_urls=("https://example.com/docs/", "https://example.com/docs"),
        allowed_domains=("example.com",),
        url_prefixes=("https://example.com/docs/",),
    )
    crawler = FakeCrawler(
        {
            "https://example.com/docs/": """
                <html><body><main><p>{}</p></main></body></html>
            """.format("content " * 40),
        }
    )

    documents = list(crawler.crawl(source, max_pages=10))

    assert [document.url for document in documents] == ["https://example.com/docs/"]
```

### `tests/test_settings.py`
```python
import json

from data_engineering_copilot.config.settings import AppSettings, load_documentation_sources


def test_load_documentation_sources_from_json(tmp_path):
    config_path = tmp_path / "documentation_sources.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "Example Docs",
                    "start_urls": ["https://example.com/docs/"],
                    "allowed_domains": ["example.com"],
                    "url_prefixes": ["https://example.com/docs/"],
                }
            ]
        ),
        encoding="utf-8",
    )

    sources = load_documentation_sources(config_path)

    assert len(sources) == 1
    assert sources[0].name == "Example Docs"
    assert sources[0].start_urls == ("https://example.com/docs/",)
    assert sources[0].allowed_domains == ("example.com",)


def test_app_settings_default_logging_enabled() -> None:
    settings = AppSettings()

    assert settings.logging_enabled is False
```

### `tests/test_ingestion.py`
```python
import pytest

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument
from data_engineering_copilot.services.ingestion import IngestionService


class RecordingCrawler:
    def __init__(self) -> None:
        self.source_names: list[str] = []

    def crawl(self, source: DocumentationSource, max_pages: int, on_event=None):
        self.source_names.append(source.name)
        yield RawDocument(source_name=source.name, url=source.start_urls[0], html="<html></html>")


class SkippingParser:
    def parse(self, raw: RawDocument):
        return None


class UnusedChunker:
    def chunk(self, document):
        raise AssertionError("chunker should not be called for skipped documents")


class UnusedEmbeddings:
    def embed_texts(self, texts):
        raise AssertionError("embeddings should not be called for skipped documents")


class UnusedVectorStore:
    def upsert_chunks(self, chunks, vectors) -> None:
        raise AssertionError("vector store should not be called for skipped documents")


class BatchRecordingEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts):
        self.batches.append(list(texts))
        return [[0.0] * 384 for _ in texts]


class BatchRecordingVectorStore:
    def __init__(self) -> None:
        self.upserted_chunks: list[list[str]] = []

    def upsert_chunks(self, chunks, vectors) -> None:
        self.upserted_chunks.append([chunk.chunk_id for chunk in chunks])


class SimpleParser:
    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source_name=raw.source_name,
            title="Test Page",
            url=raw.url,
            text="This is a test document with enough words to be chunked.",
        )


class SimpleChunker:
    def chunk(self, document: ParsedDocument):
        return [
            DocumentChunk(
                chunk_id=f"test:{document.url}",
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=document.text,
            )
        ]


def build_service(
    crawler: RecordingCrawler,
    parser=None,
    chunker=None,
    embeddings=None,
    vector_store=None,
) -> IngestionService:
    settings = AppSettings(
        sources=(
            DocumentationSource(
                name="Apache Spark Documentation",
                start_urls=("https://spark.apache.org/docs/latest/",),
                allowed_domains=("spark.apache.org",),
                url_prefixes=("https://spark.apache.org/docs/latest/",),
            ),
            DocumentationSource(
                name="Delta Lake Documentation",
                start_urls=("https://docs.delta.io/latest/",),
                allowed_domains=("docs.delta.io",),
                url_prefixes=("https://docs.delta.io/latest/",),
            ),
        )
    )
    return IngestionService(
        settings=settings,
        crawler=crawler,
        parser=parser or SkippingParser(),
        chunker=chunker or UnusedChunker(),
        embeddings=embeddings or UnusedEmbeddings(),
        vector_store=vector_store or UnusedVectorStore(),
    )


def test_ingest_only_selected_sources():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    total_chunks = service.ingest(source_names=("Delta Lake Documentation",))

    assert total_chunks == 0
    assert crawler.source_names == ["Delta Lake Documentation"]


def test_ingest_batches_chunks_and_flushes_at_end():
    class TwoPageCrawler(RecordingCrawler):
        def crawl(self, source, max_pages, on_event=None):
            self.source_names.append(source.name)
            yield RawDocument(source_name=source.name, url=source.start_urls[0], html="<html></html>")

    crawler = TwoPageCrawler()
    embeddings = BatchRecordingEmbeddings()
    vector_store = BatchRecordingVectorStore()
    service = build_service(
        crawler,
        parser=SimpleParser(),
        chunker=SimpleChunker(),
        embeddings=embeddings,
        vector_store=vector_store,
    )

    total_chunks = service.ingest(source_names=("Apache Spark Documentation",))

    assert total_chunks == 1
    assert len(embeddings.batches) == 1
    assert len(vector_store.upserted_chunks) == 1
    assert vector_store.upserted_chunks[0] == ["test:https://spark.apache.org/docs/latest/"]


def test_ingest_rejects_unknown_source_name():
    crawler = RecordingCrawler()
    service = build_service(crawler)

    with pytest.raises(ValueError, match="Unknown documentation source"):
        service.ingest(source_names=("Missing Docs",))

    assert crawler.source_names == []
```

### `tests/test_html_parser.py`
```python
from data_engineering_copilot.domain.models import RawDocument
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser


def test_html_parser_extracts_title_and_main_text():
    html = """
    <html>
      <head><title>Fallback</title></head>
      <body>
        <nav>ignore me</nav>
        <main>
          <h1>Useful Page</h1>
          <p>{}</p>
        </main>
      </body>
    </html>
    """.format(" ".join(["content"] * 50))

    parsed = DocumentationHtmlParser().parse(
        RawDocument(source_name="Delta Lake Documentation", url="https://docs.delta.io/latest/", html=html)
    )

    assert parsed is not None
    assert parsed.title == "Useful Page"
    assert "ignore me" not in parsed.text
    assert "content" in parsed.text
```

### `tests/test_ollama_client.py`
```python
from __future__ import annotations

import json

import pytest

from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def _client() -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434",
        model="deepseek-coder:6.7b",
        timeout_seconds=5,
        num_ctx=4096,
        num_predict=768,
    )


def test_prompt_format_for_deepseek_coder() -> None:
    prompt = _client()._format_raw_chat_prompt("What is Delta Lake?")

    assert "DataEngineeringCopilot" in prompt
    assert "provided context" in prompt
    assert "What is Delta Lake?" in prompt


def test_generate_strips_thinking_block(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            {
                "response": "<think>I should reason privately.</think>\nDelta Lake supports ACID table storage.",
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("data_engineering_copilot.infrastructure.ollama_client.urlopen", fake_urlopen)

    assert _client().generate("Answer from context") == "Delta Lake supports ACID table storage."


def test_generate_reports_reasoning_only_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            {
                "response": "<think>I ran out before the final answer.",
                "done_reason": "length",
            }
        )

    monkeypatch.setattr("data_engineering_copilot.infrastructure.ollama_client.urlopen", fake_urlopen)

    with pytest.raises(OllamaError, match="spent its output budget on reasoning"):
        _client().generate("Answer from context")
```

### `tests/test_logging_config.py`
```python
from __future__ import annotations

import logging

from data_engineering_copilot.logging_config import LOGGER_NAME, configure_logging


def test_configure_logging_creates_application_log_without_duplicate_handlers(tmp_path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = list(logger.handlers)
    logger.handlers.clear()

    try:
        first_path = configure_logging(tmp_path)
        second_path = configure_logging(tmp_path)
        logger.info("test log entry")

        assert first_path == tmp_path / "logs" / "application.log"
        assert second_path == first_path
        assert first_path.exists()
        assert len(logger.handlers) == 1
        assert "test log entry" in first_path.read_text(encoding="utf-8")
    finally:
        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
```