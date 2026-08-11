---
name: qdrant
description: Use for ANY task involving the DataEngineeringCopilot Qdrant vector store — collection schema, payload fields, dense+sparse/hybrid search, BM25, filter predicates, Spark index generations, alias activation/rollback, reset/rebuild, inspection, or the AsyncQdrantVectorStore. Triggers: qdrant, vector store, collection, alias, spark-build/validate/activate/rollback, reset-index/qdrant/crawler-db, inspect-db, BM25, sparse vectors, hybrid search, chunk payload, content_hash dedup, generation.
---

# DataEngineeringCopilot Qdrant

Project-specific guide for the Qdrant vector store. The stack's Qdrant is a
Docker service `qdrant/qdrant:v1.18.3` on host port `6333` (HTTP) / `6334`
(GRPC); the app talks HTTP only. In-app access goes exclusively through
`AsyncQdrantVectorStore` in
`data_engineering_copilot/infrastructure/async_qdrant_store.py`.

## Core configuration

- `settings.qdrant_url` → default `http://localhost:6333` (in-app URL is
  `http://qdrant:6333`).
- `settings.collection_name` → `data_engineering_docs` (legacy default).
- `settings.active_collection_alias` → `data_engineering_docs`. This is the
  **logical alias** that Spark activation repoints.
- `settings.active_index_generation` / `active_collection_name` → used when a
  Spark generation is active; overrides the legacy collection name.
- `settings.hybrid_search_enabled` (default true), `hybrid_rrf_k` (default 60),
  `settings.get_embedding_dimension()`.

## Collection schema (AsyncQdrantVectorStore.initialize)

- **Dense**: named vector `dense`, size = embedding dim, `Cosine` distance.
- **Sparse**: named vector `sparse` (BM25), created only when hybrid search is on.
- **Collection params**: `on_disk_payload=True`, HNSW `m=16`, `ef_construct=150`,
  `full_scan_threshold=10000`.
- **Payload indexes** (KEYWORD): `url`, `source_name`, `chunk_type`; also
  `module` where relevant.
- Point IDs are `uuid5(NAMESPACE_DNS, chunk_id)`.

## Point payload (chunk_to_payload)

Every point carries the full `DocumentChunk` fields:
`chunk_id, source_name, title, url, text, content_hash, section_header,
chunk_type, word_count, heading_path, chunk_index, total_chunks, crawled_at,
doc_type, language, spark_version, module, source_commit, file_path, license,
parser_version, chunker_version, index_generation`.

- `content_hash` is the ingestion dedup mechanism: re-crawled pages with
  unchanged content are skipped (see `get_content_hash_for_url`,
  `delete_by_url`, `scroll_urls`).
- `url` is indexed so `delete_by_url` / URL-scoped filters are fast.

## Query & filtering

`AsyncQdrantVectorStore.query(query_embedding, top_k, query_text, source_filter,
chunk_type_filter, metadata_filters, fused_limit)`:

- **Hybrid**: when hybrid search is on and the BM25 tokenizer is fitted, uses
  Qdrant native prefetch + **RRF fusion** over dense + sparse vectors. When
  `query_text` is provided the sparse vector is computed internally.
- **Fallback**: pure dense cosine when BM25 is unavailable.
- **`fused_limit`**: RRF suppresses single-modality hits, so the reranker needs
  a wider pool than `top_k`. Defaults to `max(top_k * 4, 40)`. The RAG service
  passes a larger value when it reranks.
- **Filter predicates** (`_build_query_filter`): maps `RetrievalFilters` →
  Qdrant `FieldCondition`/`MatchAny`:
  - `source_names` → `source_name`, `doc_types` → `doc_type`,
    `languages` → `language`, `versions` → `spark_version`,
    `chunk_types` → `chunk_type`.
  - `modules` matches **`module` OR `title`** (rendered API pages carry an
    empty `module` but store the dotted identifier in `title`) — never filter
    on `module` alone.
- Empty filters return `None` (callers skip the `must` condition).

## BM25 / hybrid state

- BM25 tokenizer persists to `.bm25_cache/{collection_name}.json` (CLI path) —
  but the store resolves the cache to the **active generation** collection when
  the alias is targeted, so hybrid stays enabled after Spark activation
  (`_resolve_bm25_cache_path`). Do not bypass this.
- `bm25_status()` / `is_hybrid_ready()` report whether hybrid is usable.
- `_warn_unfrozen_bm25_desync` guards against upserting sparse vectors from an
  unfitted tokenizer.

## Spark index generations & aliases

Generation lifecycle (all in `data_engineering_copilot/cli.py`):

1. **`dec spark-build --generation <gen>`** — builds collection
   `data_engineering_docs__{generation}` via `SparkIndexBuilder` + unified
   embedding fallback chain (never a bare provider). **Does not activate.**
   Default generation = `spark-{ref}-{commit8}-{sha256_12}`.
2. **`dec spark-validate --generation <gen>`** — strict artifact checks
   (coverage records, manifest path uniqueness, chunk IDs, per-chunk
   generation/commit metadata, point count vs `chunks.jsonl`) + store checks
   (dense/sparse config, BM25 state, metadata presence, payload text). Writes
   `validation-{generation}.json` under `settings.index_state_dir`
   (`.index_state/`); **`passed: true` is required to activate**.
3. **`dec spark-activate --generation <gen>`** — refuses without a passing
   validation report; requires interactive confirmation (or `FORCE=1`);
   atomically repoints the alias `data_engineering_docs` → the generation
   collection via a single `POST /collections/aliases` (delete + create in one
   action batch); writes `.index_state/active.json` + appends to
   `history.jsonl`.
4. **`dec spark-rollback --generation <gen>`** — only if `<gen>` is the current
   active generation; repoints the alias to the previous entry in
   `history.jsonl`.

State files under `.index_state/`: `active.json`, `history.jsonl`,
`validation-{generation}.json`. `resolve_active_generation()` (in
`config/settings.py`) reads `active.json` so cache scoping and collection
routing follow the active generation without env changes or restarts.

## Reset / rebuild / inspect

| Command | Effect |
|---------|--------|
| `dec reset-qdrant` | Delete + recreate `data_engineering_docs` (dim/hybrid from settings) and delete its persisted BM25 cache. Qdrant only. |
| `dec reset-index` | Full clean rebuild: recreate Qdrant + BM25 cache, clear Redis `crawl:*` and `rag:cache:*` keys, drop PG frontier tables (`crawl_frontier`, `sitemap_edges`). Qdrant is recreated **first** so failure aborts before frontier history is dropped. |
| `dec reset-crawler-db` | Reset Redis `crawl:*` + PG frontier **only**; Qdrant preserved so `content_hash` dedup still works. |
| `dec inspect-db` | Collection overview: vector config, mode (hybrid vs dense), points, sources, chunk types, sample payloads. |

## Guardrails

- **Never** call a provider directly for embeddings — always the unified
  fallback chain (`build_embedding_fallback_chain`, `factory.py`).
- `reset-index` and `reset-qdrant` destroy index data; `reset-index` also
  clears crawl state. Get explicit user approval before running.
- After Spark activation, the store must resolve the BM25 cache to the
  generation collection, or hybrid silently degrades to dense-only.
- Never drop `on_disk_payload`/HNSW params when recreating a collection — the
  reset path (`_recreate_qdrant_collection`) intentionally uses the same
  shape the store creates.

## Reference

- Store: `data_engineering_copilot/infrastructure/async_qdrant_store.py`
- CLI: `data_engineering_copilot/cli.py` (spark_* / reset_* / inspect_db /
  `_qdrant_change_alias` / `_spark_generation_collection`)
- Settings: `data_engineering_copilot/config/settings.py`
- Builder: `data_engineering_copilot/services/spark_index_builder.py`
- CLI guide: `docs/cli_guide.md`
