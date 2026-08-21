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
- **Payload indexes** (KEYWORD unless noted): `url`, `source_name`,
  `chunk_type`, `section_header`, `crawled_at` (**DATETIME**), `doc_type`,
  `language`, `spark_version`, `module`, plus `index_generation`,
  `source_commit`, and `parent_chunk_id`.
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
- **Source-scoped dedup** (Phase 1): `delete_by_url`/`get_content_hash_for_url`
  take an optional `source_name`; when provided the `must` filter adds
  `source_name == source_name` so two sources may each host the same URL and
  dedup is per-source. Sourced writes (`upsert_chunks` from crawler tasks) carry
  `source_name`; legacy/unsourced paths keep URL-only filters (pre-Phase-1
  behavior).
- `url` is indexed so `delete_by_url` / URL-scoped filters are fast.

## Query & filtering

`AsyncQdrantVectorStore.query(query_embedding, top_k, query_text, source_filter,
chunk_type_filter, metadata_filters, fused_limit, rrf_profile, search_mode)`:

- **Hybrid**: when hybrid search is on and the BM25 tokenizer is fitted, uses
  Qdrant native prefetch + **RRF fusion** over dense + sparse vectors. When
  `query_text` is provided the sparse vector is computed internally.
- **Fallback**: pure dense cosine when BM25 is unavailable.
- **`search_mode`** (`services/query_signals.py` `SearchMode`, passed by the
  RAG service from intent + signals): `bm25_only` / `dense_only` short-circuit
  to a single modality; `hybrid_equal` / `hybrid_sparse_bias` /
  `hybrid_dense_bias` set the RRF weights (sparse bias = sparse 1.25/dense
  1.0, dense bias the reverse).
- **RRF profiles** (`rrf_profile`): `equal_rrf` (default) vs
  `identifier_sparse_rrf` (dense 1.0 / sparse 1.25) for technical queries —
  gated OFF via `identifier_sparse_rrf_enabled=False` until its benchmark gate
  passes. `settings.hybrid_rrf_k` (default 60) is the RRF depth constant.
- **`fused_limit`**: RRF suppresses single-modality hits, so the reranker needs
  a wider pool than `top_k`. Defaults to `max(top_k * 4, 40)`; the RAG service
  passes `_rerank_pool_size()` (= rerank pool size: `max(top_k*4,
  reranker_top_k*8)`) when it reranks.
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
- **Tokenizer versioning**: caches record `tokenizer_version` — `legacy`
  (pre-namespace, word-only) vs `namespace-v1` (adds identifier preservation).
  `load()` rejects any other stored version; the store fails fast on a cache/
  tokenizer mismatch (`_require_no_bm25_version_mismatch`) rather than serving
  sparse vectors from an incompatible vocabulary. The namespace-aware tokenizer
  itself stays off by default (`namespace_bm25_enabled=False` — rejected
  experiment per ADR-005).
- `bm25_status()` / `is_hybrid_ready()` report whether hybrid is usable.
- `_warn_unfrozen_bm25_desync` guards against upserting sparse vectors from an
  unfitted tokenizer.
- **Combined pinned generation** (`PinnedIndexBuilder.build`): `fit_bm25_corpus`
  is called **once** over the concatenated corpus of every prepared source
  (spark + airflow + delta + claude url-index) into the single
  `data_engineering_docs__{gen}` collection, then `upsert_frozen_chunks` +
  `validate_index_generation`. The interim Claude crawler path uses an
  accumulating `fit_bm25` instead; it is superseded by `dec gen-build`.

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

**Combined pinned generation (all 5 sources, one alias)** — `dec gen-build`
prepares Spark (SparkChunker full fidelity), Airflow/Delta (HeaderAwareChunker
with RST/MDX heading conversion), and Claude url-index pages, then
`PinnedIndexBuilder` stamps `index_generation`, dedups by content hash across
sources, fits **one combined BM25 corpus**, and freezes everything into a single
`data_engineering_docs__{gen}`. Lifecycle mirrors spark-*:
`dec gen-manifest` → `dec gen-build` → `dec gen-validate --generation <gen>` →
`dec gen-activate --generation <gen>` (same validation-report gate + alias
change) → `dec gen-rollback`. `dec gen-reset` drops the alias, deletes every
`data_engineering_docs__*` collection, purges `.index_state` + generation BM25
caches, then runs the `reset-index` crawl purge. `dec gen-stale` classifies
collections as active/stale/orphan (see `services/pin_maintenance.py`).

**Naming contract** (`config/naming.py`): every name derives from the
generation ID via `resolve_naming()` — collection
`data_engineering_docs__<generation>` **must equal** the artifact dir name,
and `active_alias` is always `data_engineering_docs`. `validate_naming()`
fail-fasts before any artifact I/O or upsert. The frozen write path is
`upsert_frozen_chunks()` (requires a fitted+frozen BM25 tokenizer; rejects
unfitted/desynced state) so immutable generations are written exactly once.

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
- CLI: `data_engineering_copilot/cli.py` (spark_* / gen_* / reset_* / inspect_db /
  `_qdrant_change_alias` / `_spark_generation_collection`)
- Settings: `data_engineering_copilot/config/settings.py`
- Naming contract: `data_engineering_copilot/config/naming.py`
- Builder: `data_engineering_copilot/services/spark_index_builder.py` +
  `data_engineering_copilot/services/pinned_index_builder.py`
- CLI guide: `docs/cli_guide.md`
