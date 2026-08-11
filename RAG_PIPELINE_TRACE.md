# RAG Ingestion Pipeline — End-to-End Trace

This document traces a single documentation page from raw HTML through every
stage of the ingestion pipeline implemented in `data_engineering_copilot`, up
to the vector point payload written to Qdrant. Every attribute name,
threshold, and dimension cited below is taken verbatim from the code.

---

## 1. Executive Architecture Overview

The pipeline converts crawled documentation HTML into dense + sparse vectors in
Qdrant using a strict, stage-gated, async pipeline. Its design principles:

| Principle | Implementation |
|---|---|
| **Async-first execution** | `AsyncIngestionService` runs one worker pool per source; parse/chunk CPU-bound work is offloaded to dedicated `ThreadPoolExecutor`s (`parse_executor` max 4, `chunk_executor` max 4); embedding calls use native `async`/`await` (`AsyncOllamaEmbeddings`, `httpx.AsyncClient`). |
| **Provider fallback everywhere** | All LLM and embedding calls route through `ProviderFallbackChain` (`infrastructure/provider_fallback.py`). Embedding is built per-purpose via `build_embedding_fallback_chain()` (adaptive NVIDIA → OpenRouter → Ollama) or the single-provider `build_embedder()`; the offline Spark builder wraps the chain with `FallbackEmbedder`. |
| **Quality filtering** | Pages that parse to too little content are dropped at parse time (`min_words=40`); chunks that are sparse/repetitive/noisy are dropped by `ChunkFilter` (`min_word_count=15`, `min_alpha_ratio=0.5`, `max_repetition_ratio=0.3`); content-hash dedup skips unchanged pages. |
| **Structured enrichment** | Post-chunking processors tag chunks with API metadata (`ApiDocExtractor`), reclassify code-dominant chunks (`CodeBlockParser`), and inject an Anthropic-style document summary prefix (`ContextualChunkEnricher`). |
| **Reproducible, verified indexing** | The Spark offline path (`SparkIndexBuilder`) performs lossless token-budget segmentation (3,800 tokens / 6,000 chars), persists `chunks.jsonl`, and validates Qdrant point count and per-segment reconstruction before activation. |

Pipeline stage order (as wired in `factory.build_async_ingestion_service()`):

```
crawl → parse → dedup (sha256 content hash) → chunk → [filter → api → code-block] enrich
      → embed (ingestion batches of 512 chunks, provider batches of 128)
      → upsert to Qdrant (sub-batches of 256) → fit BM25 → mark frontier
```

The offline Spark build (`dec spark-build`) runs the same logical stages from a
pinned git commit instead of a crawler, using native source files and locally
rendered HTML, and writes directly through the frozen-BM25 upsert path.

---

## 2. Step-by-Step Deep Dive

### Step 1 — Raw HTML → Markdown

#### Component & Module

- `data_engineering_copilot/infrastructure/html_to_markdown.py`
  - `html_to_markdown(html: str, min_words: int = 40) -> str | None` (line 12)
  - `MarkdownParser` (line 45) — implements `ParserProtocol`, `parse(RawDocument) -> ParsedDocument | None`
  - `_clean_markdown(text)` (line 39), `MarkdownParser._title` (line 66)
- Related parsers selected by `factory._build_content_aware_parser()`:
  - `data_engineering_copilot/infrastructure/spark_html_parser.py` — `SparkHtmlParser` (rendered docs, `min_words=10`)
  - `data_engineering_copilot/infrastructure/rst_parser.py` — `RstParser` (RST content)
  - `data_engineering_copilot/infrastructure/native_document_parser.py` — `NativeDocumentParser` (raw Spark repo files)

#### Core Concepts & Best Practices

This stage exists because LLM retrieval quality is bounded by what enters the
index. Boilerplate, navigation, scripts, and page furniture are noise that
correlate with *every* query and pollute similarity scores (a common RAG
failure mode). The parser therefore:

- **Removes chrome before conversion**: `script`, `style`, `noscript`, `nav`,
  `footer`, `header`, `aside` are decomposed from the DOM before any text
  extraction (lines 19–20).
- **Selects the semantic content root** in priority order `<main>` → `<article>`
  → `<body>` → whole soup (line 22), so page chrome in sibling elements is never
  converted.
- **Rejects near-empty pages**: conversion returns `None` when the result has
  fewer than `min_words` (default 40) words — the caller treats `None` as
  "no_content" and permanently skips the page (prevents index pollution).
- **Normalizes whitespace** (3+ newlines → 2, runs of spaces → 1) so downstream
  token counting and chunk merging see clean text.

`SparkHtmlParser` extends this idea for locally rendered Sphinx/Jekyll output:
Jekyll guide pages put content in `div#content` with navigation in
`div.left-menu-wrapper` (not a `<nav>` tag), so the generic `<body>` fallback
would drag the sidebar in. `SparkHtmlParser` selects `<main>` → `<article>` →
configured content root, strips `[source]` links and `### [source]` heading
artifacts, and rejects pages under `_MIN_CONTENT_WORDS = 10` as
navigation-only.

#### Input Data Schema

`RawDocument` (`domain/models.py:80`) — frozen dataclass:

```python
@dataclass(frozen=True)
class RawDocument:
    source_name: str
    url: str
    html: str
    content_type: str = "text/html"
```

Sample input payload (as produced by the crawler for a rendered Spark page):

```json
{
  "source_name": "Apache Spark 4.0.0",
  "url": "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "content_type": "text/html",
  "html": "<!DOCTYPE html><html><body><nav>...</nav><main id=\"main-content\"><article class=\"bd-article\"><section id=\"pyspark-sql-functions-to-timestamp\"><h1>pyspark.sql.functions.to_timestamp...</h1>...</section></article></main></body></html>"
}
```

#### Transformation Logic

1. `BeautifulSoup(html, "html.parser")`.
2. Decompose chrome tags.
3. Select content root; render via `markdownify(str(content), heading_style="ATX", strip=[...])`.
4. `_clean_markdown`: collapse `\n{3,}` → `\n\n`, collapse `[ \t]+` → single space, strip.
5. `word_count = len(markdown_text.split())`; return `None` if `< min_words`.
6. `MarkdownParser.parse` derives the title from `<h1>` → `<title>` →
   URL fallback (`normalize_whitespace` applied), and returns a `ParsedDocument`.

#### Output Data Schema

`ParsedDocument` (`domain/models.py:117`):

```python
@dataclass(frozen=True)
class ParsedDocument:
    source_name: str
    title: str
    url: str
    text: str
    sections: tuple[DocumentSection, ...] = ()
    doc_type: str = ""
    language: str = ""
    spark_version: str = ""
    module: str = ""
    source_commit: str = ""
    file_path: str = ""
    license: str = ""
```

Sample output payload (see Section 3 for the full concrete trace):

```json
{
  "source_name": "Apache Spark 4.0.0",
  "title": "pyspark.sql.functions.to_timestamp",
  "url": "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "text": "pyspark.sql.functions.to_timestamp(*col*, *format=None*)\n: Converts a [`Column`](...) into [`pyspark.sql.types.TimestampType`](...)\n using the optionally specified format. ...\n\n Added in version 2.2.0.\n\n Changed in version 3.4.0: Supports Spark Connect.\n\n Parameters:\n ...",
  "sections": [],
  "doc_type": "api_reference",
  "language": "python",
  "spark_version": "4.0.0",
  "module": "",
  "source_commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
  "file_path": "reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "license": "Apache-2.0"
}
```

---

### Step 2 — Quality Filtering & Cleaning

#### Component & Module

- `data_engineering_copilot/services/text_filter.py`
  - `ChunkFilter` (line 12)
    - `extract(chunks: list[DocumentChunk]) -> list[DocumentChunk]` (line 32)
    - `_is_sparse(raw_text) -> tuple[bool, dict]` (line 70)
    - `_clean_text(text)` (line 62)
    - `is_sparse`, `process_chunk` convenience wrappers (lines 104, 108)

Defaults (lines 15–19):

```python
ChunkFilter(
    enabled: bool = True,
    min_word_count: int = 15,
    min_alpha_ratio: float = 0.5,
    max_repetition_ratio: float = 0.3,
)
```

Wired in via `factory.build_async_ingestion_service()` as
`ChunkFilter(enabled=getattr(app_settings, "chunk_filtering_enabled", True))`
and run inside `AsyncIngestionService._apply_enrichers()`
(`services/async_ingestion.py:375`).

#### Core Concepts & Best Practices

Retrieval is only as good as the chunks in the index; a single noisy chunk can
outrank relevant content or inject wrong context into the prompt. `ChunkFilter`
is a cheap, deterministic gate that runs *before* embedding so wasted compute
and index pollution are avoided. It drops a chunk when any of:

- **`low_word_count`** — `word_count < min_word_count` (15). Too-short chunks
  rarely carry enough signal for semantic retrieval.
- **`low_alpha_density`** — `alpha_ratio < min_alpha_ratio` (0.5), where
  `alpha_ratio = alphanumeric chars / total chars`. Text dominated by
  punctuation, logs, or code noise is near-meaningless as prose.
- **`high_repetition`** — `repetition_ratio > max_repetition_ratio` (0.3), where
  `repetition_ratio = 1 - (unique_lines / total_lines)`. Repeated boilerplate
  bloats chunks and correlates with every query.
- **`empty` / `no_words`** — blank or symbol-only content.

It also *cleans* surviving chunks, removing noise patterns that embed well but
add no meaning:

- Timestamped log lines (`MM/DD/YY HH:MM:SS INFO/WARN/DEBUG/ERROR ...`)
- `org.apache.spark.<...>` fully-qualified class names
- Lone brackets/parens/chevrons
- 3+ consecutive newlines

#### Input Data Schema

`DocumentChunk` (`domain/models.py:140`) — the same schema as the chunker
output (Step 3): `chunk_id`, `source_name`, `title`, `url`, `text`,
`content_hash`, `section_header`, `chunk_type`, `word_count`,
`heading_path: tuple[str, ...]`, `chunk_index`, `total_chunks` (required),
then optional provenance (`doc_type`, `language`, `spark_version`, `module`,
`source_commit`, `file_path`, `license`, `representation`, `parser_version`,
`chunker_version`, `index_generation`, `crawled_at`, `source_url`,
`chunk_index_in_doc`, `total_chunks_in_doc`, `parent_content_hash`,
`segment_index`, `segment_total`, `token_count`, `character_count`). The
filter consumes `chunk.text`, `chunk.chunk_id`, and produces a `replace()`d
copy with cleaned `text` and refreshed `word_count`.

#### Transformation Logic

For each chunk (`_process_one`, line 47):

1. `_is_sparse(chunk.text)`:
   - `cleaned = _clean_text(raw_text)`; `word_count = len(cleaned.split())`.
   - `char_count = len(raw_text)`; `alpha_count = sum(c.isalnum() ...)`;
     `alpha_ratio = alpha_count / char_count`.
   - lines = non-empty stripped lines;
     `repetition_ratio = 1 - len(set(lines)) / len(lines)`.
   - Evaluate thresholds in order; return
     `(True, {reason, word_count, alpha_ratio, repetition_ratio})` on the
     first violation.
2. If sparse → drop (`None`), logging
   `chunk_filter.dropped chunk_id=... reason=... word_count=... alpha_ratio=... repetition_ratio=...`.
3. Otherwise → `replace(chunk, text=cleaned, word_count=len(cleaned.split()))`.

#### Output Data Schema

Same `DocumentChunk` schema, with cleaned text. Dropped chunks are simply
absent from the returned list; `extract` logs `chunk_filter dropped=N chunks`.

---

### Step 3 — Header-Aware / Semantic Chunking

Chunker selection is config-driven in `factory.build_chunker()`
(`factory.py:786`):

```
chunking_strategy
 ├─ "semantic" (+ enable_semantic_chunking)  → SemanticChunker
 ├─ "header_aware"                           → HeaderAwareChunker
 └─ default "sentence_preserving"            → DocumentChunker (langchain)
```

For Spark-sourced content the flow is richer: `AsyncIngestionService._process_raw`
routes to `SparkChunker` (when `self._spark_chunker is not None and parsed.doc_type`),
and the offline builder uses `SparkChunker` (native files) or `SparkRenderedChunker`
(rendered HTML). The full set is documented below.

#### 3a. HeaderAwareChunker

##### Component & Module

- `data_engineering_copilot/services/header_aware_chunker.py`
  - `HeaderAwareChunker` (line 34)
  - `_RawSection` intermediate dataclass (line 24)
  - `chunk(document, precomputed_embeddings=None) async` (line 66)
  - `_split_into_sections(text)` (line 98), `_merge_sections(...)` (line 153),
    `_number_chunks(...)` (line 244), `_chunk_id(...)` (line 256)

Defaults:

```python
HeaderAwareChunker(
    chunk_size_words: int = 375,
    overlap_words: int = 90,
    min_chunk_words: int = 10,
)
```

Factory wiring uses `min_chunk_words=int(chunk_size_words * 0.1)`.

##### Core Concepts & Best Practices

Fixed-size splitting cuts arbitrarily across semantic boundaries, producing
chunks whose topics drift mid-chunk and whose edges bleed into each other.
Markdown documentation is already hierarchically structured, so splitting along
headers (`#`…`######`) preserves topical boundaries:

- **Heading paths as chunk context**: every section tracks its `heading_path`
  (e.g. `("Migration Guide: SQL", "Upgrading from 3.0 to 3.1")`), which is
  stored in the payload and searchable.
- **Parent-boundary flush**: small sibling sections merge with *following*
  sections under the same parent header, but a change of parent flushes first —
  two different topics never share a chunk (lines 209–212).
- **Bounded chunk size with overlap**: sections are accumulated until the next
  one would exceed `chunk_size_words` (375); a new chunk then opens with the
  last `overlap_words` (90) words of the flushed chunk to preserve continuity
  across boundaries.
- **Code-block preservation**: fenced code blocks are extracted per section and
  carried alongside prose, so a chunk containing code is classified `code`
  (code only) or `mixed` (code + prose) — not mangled by splitting.
- **Deterministic IDs**: `uuid5(url-namespace, "source:hdr:NNNN")` makes chunk
  IDs stable across runs for the same document.

##### Input / Output

Input: `ParsedDocument.text` (markdown). Output: `list[DocumentChunk]` with
`chunk_type` ∈ `{"text", "code", "mixed"}`, `heading_path`, `section_header`,
`chunk_index`/`total_chunks`, `word_count`. `extract_sentences()` returns
`None` (this chunker has no sentence pre-extraction).

#### 3b. SparkChunker

##### Component & Module

- `data_engineering_copilot/services/spark_chunker.py`
  - `SparkChunker` (line 33), `chunk(document, metadata)` (line 39)
  - `_chunk_guide` (line 74) → delegates to the injected header chunker
  - `_chunk_api` (line 81) / `_chunk_code` (line 85) → `_split_into_chunks`
  - `_chunk_sql_functions` (line 419), `parse_function_registry` (line 338),
    `_find_annotation_spans` (line 369), `_resolve_annotation_owner` (line 398)
  - `chunk_spark_document(...)` synchronous offline variant (line 158)
  - `_deterministic_chunk_id` (line 327), `_split_python_top_level` (line 277),
    `_split_on_blank_lines` (line 293), `_split_markdown_sections` (line 308)

`SparkMetadata` lives in `data_engineering_copilot/services/spark_metadata.py:16`:

```python
@dataclass(frozen=True)
class SparkMetadata:
    doc_type: str       # guide | api_reference | code_example | sql_function_ref
    language: str
    spark_version: str
    module: str
    source_commit: str  # must match ^[0-9a-fA-F]{40}$
    file_path: str
    license: str
```

Derivation: `derive_spark_metadata()` (spark_metadata.py:29) — version from
`source.ref` tag (empty for `master/latest/main/snapshot`), module derived for
Python files from the path segment after `pyspark/` (e.g.
`python/pyspark/sql/functions/builtin.py` → `pyspark.sql.functions.builtin`).

##### Core Concepts & Best Practices

Spark documentation is not homogeneous prose — it mixes conceptual guides,
Python API references, code examples, and Scala `@ExpressionDescription`
function sources. A single chunking strategy mis-serves at least one of these.
`SparkChunker` **routes by `doc_type`**:

- **`guide`** → header-aware chunking (topical, prose-preserving).
- **`api_reference`** → split Python at top-level `def`/`class` boundaries
  (`_split_python_top_level`); non-Python splits on blank lines capped at 200
  lines (`_split_on_blank_lines`). Each signature becomes its own chunk.
- **`code_example`** → same structural split, tagged `chunk_type="code"`.
- **`sql_function_ref`** → one chunk **per `@ExpressionDescription` annotation
  block** (`_find_annotation_spans` walks balanced parens). `_FUNC_` placeholders
  in the description are resolved to the real SQL function names by parsing
  `FunctionRegistry.scala` via `parse_function_registry` (regexes for
  `expression[...]("name")`, builder-object registrations, and
  `registerInternalExpression`). Each registered alias emits its own chunk; an
  unresolvable owner keeps `_FUNC_` literal — **content is never dropped**.
- Deterministic IDs: `spark-<commit[:8]>-<index>-<sha256[:12]>` over
  `commit|path|index|text`, so re-running the same commit yields byte-identical
  chunk IDs.
- Guides get a header path; all non-guide chunks carry the full provenance
  metadata (`doc_type`, `language`, `spark_version`, `module`, `source_commit`,
  `file_path`, `license`).

##### Input / Output

Input: `ParsedDocument` + `SparkMetadata`. Output: `list[DocumentChunk]` with
`chunk_type` ∈ `{"text", "api", "code"}`, full Spark provenance, deterministic
`chunk_id`, `content_hash` (sha256 of piece text).

#### 3c. SparkRenderedChunker

`data_engineering_copilot/services/spark_rendered_chunker.py` (line 27) chunks
locally rendered Sphinx/Jekyll HTML main content. Heading-bounded split with
fenced-code preservation; `_classify()` returns `code` when a chunk is pure
fence, `mixed` when code + prose, `api` when the header mentions
`api|class|method|function`. Falls back to paragraph splitting when a page has
no headings (`_MIN_CHUNK_WORDS = 10`). Chunk IDs:
`spark-rendered-<commit[:8]>-<index>-<sha256[:12]>` (representation included so
native and rendered chunks never collide).

#### 3d. Config-driven alternatives (non-Spark sources)

- **`DocumentChunker`** (`services/chunker.py:15`, default
  `chunking_strategy="sentence_preserving"`) — langchain
  `RecursiveCharacterTextSplitter`, char-based
  `chunk_size_chars = chunk_size_words * 5` = 1875, overlap ×5 = 450; detects
  language from URL (`/api/python/`, `/pyspark`, `/api/scala/`, `/api/java/`,
  `/api/r/`) to use language-aware splitters.
- **`SemanticChunker`** (`services/semantic_chunker.py`, `strategy="semantic"`)
  — embeds candidate sentences, merges by `min_semantic_similarity=0.5`,
  `min_chunk_words=int(chunk_size_words*0.1)`, `max_chunk_words` 1.5× target.

#### Sample output (Step 3)

Real chunk from the offline Spark corpus (`chunks.jsonl`, `text` truncated;
values verified against the persisted record):

```json
{
  "chunk_id": "spark-rendered-fa33ea00-0-27f2518a4bdd:seg:0",
  "chunk_type": "text",
  "chunker_version": "spark-chunker-v1",
  "content_hash": "7803ccdadd637f3ee5bc5c33b993fc94ffe41e7211d1f130f616ddb0aaea8a10",
  "doc_type": "api_reference",
  "file_path": "reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "heading_path": ["pyspark.sql.functions.to\\_timestamp"],
  "index_generation": "spark-v4.0.0-fa33ea00-hybrid-20260807",
  "language": "python",
  "license": "Apache-2.0",
  "module": "",
  "representation": "rendered",
  "section_header": "pyspark.sql.functions.to\\_timestamp",
  "source_commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
  "source_name": "Apache Spark 4.0.0",
  "spark_version": "4.0.0",
  "text": "pyspark.sql.functions.to\\_timestamp(*col*, *format=None*)[](../../../_modules/pyspark/sql/functions/builtin.html#to_timestamp) ...",
  "url": "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "word_count": 171,
  "character_count": 3365,
  "token_count": 824,
  "parent_content_hash": "7803ccdadd637f3ee5bc5c33b993fc94ffe41e7211d1f130f616ddb0aaea8a10",
  "segment_index": 0,
  "segment_total": 1,
  "chunk_index": 0,
  "total_chunks": 1,
  "crawled_at": ""
}
```

---

### Step 4 — Metadata & API Enrichment

Three post-processors run after chunking, before embedding, inside
`AsyncIngestionService._apply_enrichers()` (`async_ingestion.py:375`):

```python
def _apply_enrichers(self, chunks):
    if self._chunk_filter is not None:
        chunks = self._chunk_filter.extract(chunks)      # Step 2
    if self._api_extractor is not None:
        chunks = self._api_extractor.extract(chunks)     # 4a
    if self._code_block_parser is not None:
        chunks = self._code_block_parser.extract(chunks) # 4c
    return chunks
```

Contextual LLM enrichment (4b) is applied per-document in `_process_raw` and
decoupled through a Redis queue on the Celery path.

#### 4a. ApiDocExtractor

##### Component & Module

- `data_engineering_copilot/services/api_extractor.py`
  - `ApiDocExtractor(enabled=True)` (line 38), `extract(chunks)` (line 48),
    `_enrich_one(chunk)` (line 57)
  - `_SIG_PATTERNS` (line 15): `def name(`, `Class.method(` at line start,
    module chains like `spark.read.parquet(`, backticked signatures
    `` `spark.read.parquet(path)` ``
  - `_PARAM_PATTERNS` (line 27): `:param name:`, markdown table rows
    `| name | Type | ... |`
  - `_RETURN_PATTERN` (line 35): `-> ReturnType`

##### Core Concepts & Best Practices

Pure prose chunking mangles API signatures: a function definition is the
highest-value retrieval target in a docs corpus, yet a generic chunker scatters
its signature, params, and return type across unrelated chunks. `ApiDocExtractor`
runs *after* the main chunker to detect API-style content and prepend a
structured `[API: ...]` header that improves both lexical and semantic
retrieval and gives the LLM structured context. Chunks already tagged
`chunk_type="api"` are skipped (the Spark chunker already did this job).

##### Transformation Logic

1. Detect the first matching signature; split `module.method` into
   `module`/`method` on the last dot.
2. Collect params from `:param` directives or markdown table rows (deduped,
   order-preserving).
3. Extract the `-> Type` return annotation.
4. Build `[API: Module: <m> | Method: <n> | Params: a, b | Returns: T]`
   (only present components) and prepend it to `chunk.text`;
   set `chunk_type="api"`.

#### 4b. ContextualChunkEnricher / LLMContextSummarizer

##### Component & Module

- `data_engineering_copilot/services/contextual_chunk_enricher.py`
  - `LLMContextSummarizer` (line 67)
    - `__init__(llm_client, max_summary_words=50, max_retries=2,
      retry_backoff_seconds=2.0, failure_recorder=None, telemetry=None)` (line 77)
    - `summarize(document) -> str` (line 103), `_is_transient(exc)` (line 94),
      `_clean_summary(raw)` (line 143)
  - `ContextualChunkEnricher` (line 161)
    - `__init__(summarizer=None, enabled=False, batch_size=20, telemetry=None)` (line 168)
    - `enrich(document, chunks)` (line 184), `_is_blacklisted_url` (line 180)
  - `_SUMMARY_PROMPT` (line 22) — offline fallback for the Langfuse-managed
    `chunk-enrichment-summary` prompt, registered via `register_fallback`
  - `_INDEX_URL_BLACKLIST` (line 37): `index-all.html`, `deprecated-list.html`,
    `package-summary.html`, `allclasses-index.html`, `allpackages-index.html`,
    `constant-values.html`, `serialized-form.html`, `overview-tree.html`,
    `help-doc.html`
  - `_MIN_CONTENT_WORDS = 40` (line 51)

Wired as `ContextualChunkEnricher(enabled=contextual_enrichment_enabled,
batch_size=enrichment_batch_size)` with a per-purpose `enrichment` LLM chain
(`build_llm_fallback_chain(purpose="enrichment", purpose_model=enrichment_llm_model)`).

##### Core Concepts & Best Practices

Anthropic-style contextual retrieval. A chunk that is semantically perfect in
isolation loses its document-level meaning during similarity search (a fragment
about `date_add` is meaningless without "part of `pyspark.sql.functions`").
This stage:

- Generates a **1–2 sentence summary of the whole page** (≤ `max_summary_words`
  = 50) and **prepends it to every chunk** as `[Document Context: <summary>]`
  followed by the chunk text.
- **Fail-open**: if summarization fails, chunks are indexed *without* context
  rather than blocking ingestion. Transient provider errors
  (`RETRYABLE`, `RATE_LIMITED`, `TEMPORARY_UNAVAILABLE`, `QUOTA_EXCEEDED`, or
  HTTP 408/429/5xx) retry up to `max_retries` (2) with exponential backoff;
  permanent errors skip to the failure path. On final failure the URL is handed
  to `failure_recorder` so a later re-enrich pass can pick it up.
- **Skip guards**: blacklisted index/listing URLs and documents under
  `_MIN_CONTENT_WORDS` (40) are never summarized (they carry no per-document
  meaning).
- **Output hygiene**: `_clean_summary` strips code fences, quotes, and intro
  phrases ("Here is the summary:", etc.) and truncates to 400 chars.
- **Batch decoupling** (Celery path): documents are queued to
  `ingestion:<task_id>:enrichment_queue` in Redis and enriched by a background
  worker, so slow LLM summarization does not block the crawl/embed path.

#### 4c. CodeBlockParser

##### Component & Module

- `data_engineering_copilot/services/code_block_parser.py`
  - `CodeBlockParser(enabled=True, max_code_lines=500)` (line 36)
  - `extract(chunks)` (line 40), `_process_one(chunk)` (line 51),
    `_scope_prefix(chunk)` (line 108), `_split_at_boundaries` (line 118),
    `_split_with_ast` (line 145)

##### Core Concepts & Best Practices

Code embedded in prose chunks is poorly retrieved and poorly served to an LLM.
This pass:

- Reclassifies a chunk as `chunk_type="code"` when ≥ **60%** of its characters
  are inside fenced code blocks.
- Prepends a **scope header** (`# Source: <source_name> / # Document: <title>`
  / `# Section: <section_header>`) so an isolated code chunk still identifies
  its origin.
- Splits oversized code blocks (> `max_code_lines` = 500) at `def`/`class`
  boundaries using Python AST first (`_split_with_ast`), falling back to a
  regex boundary split; each part becomes `chunk_id:<chunk_id>:code:<i>:<j>`.

#### Sample enriched output

Illustrative **online-path** result (not the offline `chunks.jsonl` record —
the offline build persists the unenriched chunk, `chunk_type="text"`):

```json
{
  "chunk_id": "spark-rendered-fa33ea00-0-27f2518a4bdd:seg:0",
  "chunk_type": "api",
  "text": "[Document Context: Documentation for the pyspark.sql.functions.to_timestamp function: converts a Column to TimestampType using an optional datetime format; added in Spark 2.2.0 and supports Spark Connect since 3.4.0.]\n[API: Module: pyspark.sql.functions | Method: to_timestamp | Params: col, format | Returns: Column]\n\npyspark.sql.functions.to_timestamp(*col*, *format=None*) ...",
  "word_count": 341
}
```

---

### Step 5 — Vector Embedding Generation

#### Component & Module

- `data_engineering_copilot/infrastructure/async_embeddings.py`
  - `AsyncOllamaEmbeddings` (line 34) — native async Ollama provider
    - `__init__(model_name, base_url=None, retry_wait=None, batch_size=128,
      timeout_seconds=180, max_concurrency=1, keep_alive="10m",
      connect_timeout_seconds=5, pool_timeout_seconds=5)` (line 37)
    - `embed_texts(texts)` (line 170), `embed_query(text)` (line 176),
      `call(request)` (line 184, `ProviderClientProtocol`)
    - `_aollama_embed` (line 127) — concurrent batch submission capped at 2 in
      flight via `_request_semaphore`
    - `_aollama_embed_single_batch` (line 74) — `tenacity` retry:
      `stop_after_attempt(3)`, `wait_exponential(multiplier=1, min=1, max=10)`,
      retry on `httpx.TimeoutException`/`ConnectError`/`OSError` or HTTP 503
    - `_validate_embedding_dimensions` (line 158)
- `data_engineering_copilot/infrastructure/fallback_embedder.py`
  - `FallbackEmbedder(chain)` (line 17) — adapts a
    `ProviderFallbackChain[list[str], list[list[float]]]` or a bare
    `EmbedderProtocol` into the uniform `EmbedderProtocol` interface
    (`embed_texts` / `embed_query` / `close`)
- `data_engineering_copilot/factory.py`
  - `build_embedder(app_settings)` (line 686) — single-provider path
    (`openrouter`/`nvidia`/`gemini` → OpenAI-compatible embeddings; `ollama` →
    `AsyncOllamaEmbeddings`)
  - `build_embedding_fallback_chain(purpose="global", ...)` (line 626) —
    returns a `ProviderFallbackChain` when ≥2 providers are configured, else
    the bare embedder

#### Model dimensions

`config/settings.py` (`embedding_model_dimensions`, line 406) — dimension is
**model-dependent, not provider-dependent**:

| Model | Dimension |
|---|---|
| `nomic-embed-text` (default, Ollama) | 768 |
| `mxbai-embed-large` | 1024 |
| `snowflake-arctic-embed2` | 1024 |
| `llama3.2:3b` | 3072 |
| `nvidia/nemotron-3-embed-1b` / `:free` | 2048 |
| `text-embedding-004` (Gemini) | 768 |
| unknown model → `default_embedding_dimension` | 768 |

`settings.get_embedding_dimension()` (`settings.py:733`) resolves the active
provider's model and looks it up. The Qdrant store is constructed with
`embedding_dimension=settings.get_embedding_dimension()` so the collection
schema always matches the active model. Related defaults:
`embedding_batch_size` = 128, `embed_concurrency` = 1, `ollama_keep_alive` =
`"10m"`. `MAX_SAFE_TOKENS = 3800` in
`infrastructure/async_openai_compatible_embeddings.py:28` (safe buffer below
OpenRouter's 4096 limit).

#### Core Concepts & Best Practices

Embedding is the only stage where *every* chunk hits a network service, so it
is engineered for resilience and correctness:

- **Dimension validation**: `_validate_embedding_dimensions` raises
  `EmbeddingError` if an embedding is not a list, is empty, or has a different
  dimension than the configured model expects — a silently mismatched dimension
  would break cosine similarity and Qdrant upserts.
- **Transient retry**: 3 attempts with exponential backoff on timeouts,
  connect errors, and Ollama 503 overload (`_RETRYABLE_ERRORS`).
- **Batching + bounded concurrency**: inputs slice into `batch_size` (128)
  batches; multiple batches run concurrently (≤ 2 in flight) under a semaphore.
- **Adaptive backpressure upstream**: `AsyncIngestionService` counts
  consecutive Ollama embedding failures and emits backpressure warnings at ≥3.
- **Provider fallback**: `build_embedding_fallback_chain` builds an ordered
  adaptive chain (e.g. NVIDIA → OpenRouter → degraded Ollama) from
  `EMBEDDING_FALLBACK_ORDER`, with per-provider sliding-window rate limiters
  and a shared `ProviderHealthRegistry`; `FallbackEmbedder` exposes it through
  the standard `EmbedderProtocol` used by the offline Spark builder.
- **Lossless pre-embedding segmentation (Spark path)**:
  `infrastructure/token_budget.py` guarantees the embedded text never exceeds
  the provider budget via truncation. `split_text_losslessly` (defaults
  `DEFAULT_MAX_TOKENS = 3800`, `DEFAULT_MAX_CHARS = 6000`) splits chunks into
  atomic segments (fenced code blocks are never split; only a fence that alone
  exceeds the budget is split by lines) such that `"".join(segments)`
  reproduces the source exactly. Every segment carries `parent_content_hash`,
  `segment_index`, `segment_total`, `token_count`, `character_count`.
  Boundaries tried in order: headings → paragraphs → lists → sentences.

#### Input / Output

- Input: `list[str]` of chunk texts.
- Output: `list[list[float]]`, each vector of the configured model dimension
  (768 for the default `nomic-embed-text`).

```json
{
  "text": "pyspark.sql.functions.to_timestamp(*col*, *format=None*) ...",
  "embedding": [0.0421, -0.0513, 0.0188, 0.0772, "... 763 more ...", 0.0039]
}
```

---

### Step 6 — Vector Database Storage Payload

#### Component & Module

- `data_engineering_copilot/infrastructure/async_qdrant_store.py`
  - `AsyncQdrantVectorStore` (line 47)
    - `__init__(url, collection_name, hybrid_search=True, hybrid_rrf_k=60,
      embedding_dimension=None)` (line 61)
    - `initialize()` (line 107) — creates collection + payload indexes
    - `upsert_chunks(chunks, vectors, _sub_batch_size=256)` (line 229)
    - `upsert_frozen_chunks(chunks, vectors, _sub_batch_size=256)` (line 627) — Spark path
    - `_chunk_to_payload(chunk)` (line 199)
    - `_chunk_id_to_uuid(chunk_id)` (line 226)
    - `fit_bm25(texts)` (line 568), `fit_bm25_corpus(texts)` (line 604)
    - `query(...)` (line 315), `_build_query_filter` (line 277),
      `bm25_status` (line 504), `validate_index_generation` (line 670)
- `data_engineering_copilot/infrastructure/bm25_tokenizer.py`
  - `BM25Tokenizer(k1=1.2, b=0.75)` (line 73) — regex word extraction
    `[a-zA-Z0-9_\-]{2,}`, 35-word stopword list, Porter stemming, Qdrant
    `SparseVector` output; `fit()` freezes the vocabulary
- Config: `collection_name = "data_engineering_docs"` (`settings.py:360`),
  `hybrid_search_enabled = True`, `hybrid_rrf_k = 60` (`settings.py:621`)

#### Collection schema (`initialize`, line 107)

- Dense vector: `{"dense": VectorParams(size=<768>, distance=COSINE)}`.
- Sparse vector: `{"sparse": SparseVectorParams(index=SparseIndexParams())}`.
- `on_disk_payload=True`, `HnswConfigDiff(m=16, ef_construct=150,
  full_scan_threshold=10000)`.
- Payload keyword indexes: `url`, `source_name`, `chunk_type`,
  `section_header`, `doc_type`, `language`, `spark_version`, `module`
  (`CreateIndex`); DATETIME index on `crawled_at`.

#### Core Concepts & Best Practices

The Qdrant store is both the persistence layer and the dedup source of truth:

- **Point identity**: each point ID is
  `str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))` — deterministic and
  idempotent, so re-upserting the same chunk overwrites rather than duplicates.
- **Hybrid payload**: in hybrid mode every point carries both a dense vector
  and a **BM25 sparse vector** (`indices`/`values` from
  `BM25Tokenizer.tokenize(text)`), enabling Qdrant-native RRF fusion at query
  time (`Rrf(k=60)` over dense + sparse prefetches). Sparse ids are stable
  only if the tokenizer vocabulary is *frozen* before writes — the store warns
  on unfrozen writes into a non-empty collection (desync hazard) and the Spark
  path enforces `fit_bm25_corpus` before `upsert_frozen_chunks`.
- **Full payload provenance**: every chunk field is stored, making retrieved
  points self-describing (source, URL, section path, version, commit,
  generation) without a second lookup.
- **Request-size safety**: upserts are sub-batched at 256 points to stay under
  Qdrant's `max_request_size_mb` (32 MB).
- **BM25 lifecycle**: after a crawler ingestion, `fit_bm25(corpus_texts)`
  accumulates corpus stats and persists the tokenizer to
  `.bm25_cache/<collection>.json` (resolved to the active generation
  collection when a Spark alias is active). A tokenizer loaded from disk is
  never re-fitted (`fit_bm25`) — re-fitting on a partial corpus would reassign
  vocabulary ids and silently break hybrid search.
- **Dedup via payload**: `get_content_hash_for_url(url)` scrolls points for a
  URL and returns the stored `content_hash`; `AsyncIngestionService` skips a
  page when the stored hash equals the freshly computed sha256 of the parsed
  text. Qdrant, not Redis, is the dedup authority.

#### Payload schema (`_chunk_to_payload`, line 199)

```json
{
  "chunk_id": "spark-rendered-fa33ea00-0-27f2518a4bdd:seg:0",
  "chunk_type": "text",
  "content_hash": "7803ccdadd637f3ee5bc5c33b993fc94ffe41e7211d1f130f616ddb0aaea8a10",
  "source_name": "Apache Spark 4.0.0",
  "title": "pyspark.sql.functions.to\\_timestamp",
  "url": "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "text": "...",
  "section_header": "pyspark.sql.functions.to\\_timestamp",
  "word_count": 171,
  "heading_path": ["pyspark.sql.functions.to\\_timestamp"],
  "chunk_index": 0,
  "total_chunks": 1,
  "crawled_at": "",
  "doc_type": "api_reference",
  "language": "python",
  "spark_version": "4.0.0",
  "module": "",
  "source_commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
  "file_path": "reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
  "license": "Apache-2.0",
  "parser_version": "",
  "chunker_version": "spark-chunker-v1",
  "index_generation": "spark-v4.0.0-fa33ea00-hybrid-20260807"
}
```

(`heading_path` is stored as a list for JSON compatibility; `chunk_index`,
`word_count`, `total_chunks`, `token_count`, `character_count`, `segment_index`,
`segment_total` are ints.)

Note on `crawled_at` and `chunk_type` — **offline vs online divergence**:

- This payload is what the **offline Spark build** (`spark_index_builder.py`)
  writes: it persists each `DocumentChunk` field verbatim, so the Qdrant point
  payload is byte-for-byte identical to the `chunks.jsonl` record — **no
  desync**. Because the offline builder skips the enrichment pass, `chunk_type`
  stays `"text"` and `crawled_at` stays `""`.
- The **online crawler path** differs in two fields only. `AsyncIngestionService`
  stamps `crawled_at = datetime.now(UTC).isoformat()` on every chunk before
  upsert (`async_ingestion.py:203-204`), and `ApiDocExtractor` may reclassify
  the chunk to `chunk_type="api"` (`_apply_enrichers`, `async_ingestion.py:375`).
  The point schema itself is identical; only those values change.

---

## 3. End-to-End Concrete Example

Trace one concrete page: **`pyspark.sql.functions.to_timestamp`** from the
Spark 4.0.0 rendered corpus (commit `fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4`).
All artifacts are grounded under
`data/spark_corpus/spark-v4.0.0-fa33ea00-hybrid-20260807/`.

### 3.1 Raw HTML input

File: `pyspark_api/output/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html`
(~394 KB). Verified DOM structure:

```html
<main id="main-content" class="bd-main" role="main">
  <div class="bd-content">
    <div class="bd-article-container">
      <article class="bd-article">
        <section id="pyspark-sql-functions-to-timestamp">
          <h1>pyspark.sql.functions.to_timestamp<a class="headerlink" href="#pyspark-sql-functions-to-timestamp" title="Link to this heading">#</a></h1>
          <dl class="py function">
            <dt class="sig sig-object py" id="pyspark.sql.functions.to_timestamp">
              <span class="sig-prename descclassname"><span class="pre">pyspark.sql.functions.</span></span>
              <span class="sig-name descname"><span class="pre">to_timestamp</span></span>
              <span class="sig-paren">(</span>
              <em class="sig-param"><span class="n"><span class="pre">col</span></span></em>,
              <em class="sig-param"><span class="n"><span class="pre">format</span></span>
                <span class="o"><span class="pre">=</span></span>
                <span class="default_value"><span class="pre">None</span></span></em>
              <span class="sig-paren">)</span>
            </dt>
            <dd><p>Converts a <code>Column</code> into <code>TimestampType</code> ...</p></dd>
          </dl>
        </section>
      </article>
    </div>
  </div>
</main>
```

### 3.2 Stage 1 — Parse (`SparkHtmlParser`)

- Selects `<main id="main-content">` → `<article class="bd-article">`;
  strips breadcrumb `<nav>`, page-TOC `<nav>`, header/footer, and the
  `[source]` link.
- Converts to ATX markdown; drops the `### [source]` artifact heading.
- Verified output = 3,365 characters, 171 words (see `character_count` /
  `word_count` in chunks.jsonl). The page passes `SparkHtmlParser`'s
  `min_words` gate (`_MIN_CONTENT_WORDS = 10`,
  `spark_html_parser.py:29,62,101`) — note this is the *parse* gate; the
  chunker's `_MIN_CHUNK_WORDS = 10` (`spark_rendered_chunker.py:23`) is a
  separate per-chunk floor.

Output `ParsedDocument`: title = `pyspark.sql.functions.to_timestamp`
(Sphinx-escaped as `to\_timestamp` in the persisted record), `doc_type="api_reference"`,
`language="python"`, `spark_version="4.0.0"`,
`file_path="reference/pyspark.sql/api/...to_timestamp.html"`,
`source_commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"`,
`license="Apache-2.0"`.

### 3.3 Stage 2 — Chunk (`SparkRenderedChunker`)

- Single `## pyspark.sql.functions.to_timestamp` heading → one heading-bounded
  chunk; `_classify` sees a pure-prose section → `chunk_type="text"`.
- Versioned ID: `spark-rendered-fa33ea00-0-27f2518a4bdd:seg:0`.
- Content hash: `7803ccdadd637f3ee5bc5c33b993fc94ffe41e7211d1f130f616ddb0aaea8a10`
  (matches the persisted `chunks.jsonl` record).

### 3.4 Stage 2b — Quality filter + enrichment (online path only)

The **offline build applies none of the Step 2/4 enrichment stages** — it goes
straight from `SparkRenderedChunker` output to `fit_bm25_corpus` +
`upsert_frozen_chunks` (`spark_index_builder.py:196,200`). That is why the
persisted record keeps `chunk_type="text"`, `word_count=171`, and
`crawled_at=""`. Had this page gone through the **online crawler path**
(`AsyncIngestionService._apply_enrichers`, `async_ingestion.py:375`), the
following would apply:

- `ChunkFilter` (`text_filter.py:12`): 171 words ≫ `min_word_count` (15);
  alpha density ≥ 0.5 and repetition ratio ≤ 0.3 → kept.
- `ApiDocExtractor`: detects the signature
  `pyspark.sql.functions.to_timestamp(col, format=None)` and the `:param`
  table; prepends `[API: Module: pyspark.sql.functions | Method: to_timestamp |
  Params: col, format | Returns: Column]` and reclassifies the chunk
  `chunk_type="api"`.
- `ContextualChunkEnricher`: the page passes `_MIN_CONTENT_WORDS` (40) and is
  not blacklisted → an LLM summary is prepended as
  `[Document Context: ...]`.
- `crawled_at` is stamped with `datetime.now(UTC).isoformat()`
  (`async_ingestion.py:203-204`).

### 3.5 Stage 3 — Embedding

- Default provider `ollama`, model `nomic-embed-text` → **768-dim** dense
  vector.
- Spark path also builds the **BM25 sparse vector** (stemmed terms, e.g.
  `to_timestamp`, `column`, `format`, `timestamp`, `connect`), with
  vocabulary ids fixed by the frozen tokenizer (`fit_bm25_corpus` before
  writes).

### 3.6 Stage 4 — Qdrant point payload

- Collection: `data_engineering_docs` (dense `size=768`, `COSINE`; sparse
  BM25 index; RRF `k=60`).
- Point ID: `uuid5(NAMESPACE_DNS, chunk_id)` — deterministic.

```json
{
  "id": "a8f1c3...",
  "vector": {
    "dense": [0.0421, -0.0513, 0.0188, 0.0772, "... 763 more ...", 0.0039],
    "sparse": { "indices": [241, 512, 890, 1203], "values": [2.14, 1.87, 1.55, 1.02] }
  },
  "payload": {
    "chunk_id": "spark-rendered-fa33ea00-0-27f2518a4bdd:seg:0",
    "chunk_type": "text",
    "content_hash": "7803ccdadd637f3ee5bc5c33b993fc94ffe41e7211d1f130f616ddb0aaea8a10",
    "source_name": "Apache Spark 4.0.0",
    "title": "pyspark.sql.functions.to\\_timestamp",
    "url": "https://spark.apache.org/docs/4.0.0/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
    "section_header": "pyspark.sql.functions.to\\_timestamp",
    "word_count": 171,
    "heading_path": ["pyspark.sql.functions.to\\_timestamp"],
    "chunk_index": 0,
    "total_chunks": 1,
    "crawled_at": "",
    "doc_type": "api_reference",
    "language": "python",
    "spark_version": "4.0.0",
    "module": "",
    "source_commit": "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
    "file_path": "reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html",
    "license": "Apache-2.0",
    "parser_version": "",
    "chunker_version": "spark-chunker-v1",
    "index_generation": "spark-v4.0.0-fa33ea00-hybrid-20260807"
  }
}
```

At query time, this point participates in hybrid retrieval: dense (cosine) +
sparse (BM25) scores are fused with `Rrf(k=60)`, filtered by payload
constraints (e.g. `doc_type`, `spark_version`, `module`), then reranked by the
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker before the top context is
assembled for the LLM.

---

_Appendix: pipeline stages are wired in `factory.build_async_ingestion_service()`
(`factory.py:944`). Ingestion orchestration: `services/async_ingestion.py`
(`AsyncIngestionService`). Offline Spark build: `services/spark_index_builder.py`
(`SparkIndexBuilder`). Guardrails against retrieved-document injection:
`services/input_guardrails.py` (`InputGuardrails`, `INJECTION_THRESHOLD`),
with shared patterns in `services/prompt_injection.py`._
