# ADR-012: api_lookup → BM25_ONLY Routing on Original Query

## Status

Accepted — 2026-09-02

## Context

`baseline_inscope.json` per-intent `api_lookup n=14 recall@k 0.0` — exact API lookups (e.g. `spark.sql.functions.col`, `dense_rank()`) retrieve nothing with the shipped hybrid path. Hybrid `RRF k=20 L100` fuses dense + sparse with equal weights `(1,1)`; dense embeddings dilute exact token matches for dotted identifiers and SQL/code syntax, and the rewrite stage can drift user-typed terms (`dense_rank()` → `"dense ranking"`), moving the sparse signal away from the BM25 index.

Store-level hybrid is proven overall (`R@10 0.273 n=220`, ADR-009) but per-intent `api_lookup` needs exact lexical matching to match reference pages whose payload carries the dotted identifier in `title`/`text`. The routing layer `select_search_mode(intent, signals)` already intends `api_lookup`/`code_example → BM25_ONLY`, yet two gaps allowed drift:

1. `AsyncRagService._compute_search_mode` could be (and a streaming path was) fed the rewritten `effective_query` instead of the original `question` — rewrite drift then changes `classify_query_signals` output.
2. `select_search_mode` only accepted `QuerySignals`, not a raw `query: str`, so callers that already drifted the text could not be hardened at the signal boundary.
3. `AsyncQdrantVectorStore.query(fused_limit=100)` documentation did not state that `Prefetch(limit=...)` is **per-leg** (100 dense + 100 sparse → RRF fusion of up to 200 → `limit` 10), inviting `top_k*2` misconfiguration.

Plan `plans/2026-09-02_rag_pipeline_simplification_plan.md:Task 3` requires proving `BM25_ONLY` on the held `api_lookup` subset (14q) with `recall ≥ hybrid`.

## Decision

### 1. Hard gate `api_lookup`/`code_example → BM25_ONLY` on intent

`data_engineering_copilot/services/query_signals.py:109 select_search_mode`

- Intent `api_lookup` and `code_example` **always** return `SearchMode.BM25_ONLY`, ignoring signals/query content — even when rewrite drifts `dense_rank()` → `"dense ranking"`.
- Signature widened to `select_search_mode(intent, signals|str|None, *, query: str|None)` for backward compatibility: production path passes `QuerySignals` positional; new callers/tests may pass `query="..."` keyword (classified internally via `classify_query_signals`). Fallback signal-based routing uses the resolved `effective_signals`; unknown intent + no signals → `HYBRID_EQUAL`.
- Docstring carries ADR-012 invariant: callers MUST use the **original** question, not `effective_query`.

### 2. Routing on original question

`data_engineering_copilot/services/async_rag.py:562 _compute_search_mode`

```python
def _compute_search_mode(self, rewritten, question: str) -> SearchMode | None:
    # ADR-012: uses original question, not rewritten effective_query
    return select_search_mode(rewritten.intent, classify_query_signals(question))
```

- `answer()` already did this; docstring and inline comment now state the ADR.
- `answer_stream` single-query streaming path fixed: `select_search_mode(intent, classify_query_signals(safe_question))` instead of `classify_query_signals(effective_query)`. Multi-query chat path already used `safe_question`.

`AsyncQdrantVectorStore.query(search_mode=BM25_ONLY, fused_limit=100)` is then called with a BM25-only sparse lookup (frozen tokenizer required).

### 3. Prefetch limit per-leg documentation

`data_engineering_copilot/infrastructure/async_qdrant_store.py:762 query(fused_limit, rrf_profile)`

- Docstring: `fused_limit` is the per-leg `Prefetch(limit=...)` value (`retrieval_prefetch_limit=100`, not `top_k*2`) — hybrid issues two prefetches each with this limit → up to `2*fused_limit` before RRF.
- Inline comments at `prefetch_limit = effective_fused_limit` and both `Prefetch(limit=prefetch_limit)` state `per-leg: 100 dense / 100 sparse → 200 before RRF`.

### 4. Chunk-type filter unchanged

`async_rag.py:948 chunk_type_filter == "api"` for `api_lookup` when no hard `modules` filter remains; combining with `BM25_ONLY` preserves the exact-API-page path (rendered reference pages; `module`/`title` filter).

## Consequences

- `api_lookup` (14q held, also 14q overall) retrieval uses pure BM25 against the frozen namespace-aware tokenizer; dense dilution removed for this intent.
- `how_to`/`factual` etc. unchanged (`DENSE_ONLY`/`HYBRID_*`); per-intent gate `how_to recall ≥ baseline -0.02` enforced in retrieval eval.
- BM25 tokenizer must be frozen/ready for `BM25_ONLY`; otherwise `VectorStoreError("BM25 tokenizer is not ready")` — same invariant as hybrid.
- New provenance still records `search_mode` via `_compute_search_mode`.

## Verification

- Unit gate: `tests/unit/test_query_signals_bm25_only.py`
  - `test_api_lookup_routes_bm25_only` — `query="spark.sql.functions.col"` → `BM25_ONLY`
  - `test_rewrite_drift_still_bm25_only` — `query="dense_rank() over window"` (drifted surface) → `BM25_ONLY`
  - `test_api_lookup_with_signals_still_bm25_only` — `QuerySignals` path also `BM25_ONLY`
  - `test_code_example_routes_bm25_only` and `test_non_api_intents_not_forced_bm25` guard regression
- Integration expectation: on held `api_lookup` subset (14q) `BM25_ONLY recall@10 ≥ hybrid recall@10` (baseline `0.0`; any gain counts, target `≥0.05` on 14q, e.g. `0.07 vs 0.03`). Validate via `dec_venv/bin/python scripts/eval_pipeline_ablation.py --subset api_lookup --k 10` or `dec eval-retrieval --split held` per-intent table — same as Task 3 Step 4.

## Alternatives Considered

- Signal-based routing for `api_lookup` (e.g. `identifier_heavy → HYBRID_SPARSE_BIAS`): rejected — intent is the stronger, LLM-classified signal; signals derived from rewritten text reintroduce drift.
- Per-query BM25 for all intents: rejected — `how_to` benefits from dense semantics (`R@10 0.42`); only exact-lookup intents want pure lexical.
- Keep `top_k*2` fused_limit: rejected — shipped `retrieval_prefetch_limit=100` (ADR-009) is the per-leg RRF depth that holds the `k=20` win; halving it to `top_k*2 ≈ 20` collapses recall.

## Provenance

- Plan: `plans/2026-09-02_rag_pipeline_simplification_plan.md:Task 3`
- Base win: `74236d9 RRF k=20 L100 (1,1)`
- Baseline: `tests/evaluation/benchmarks/baseline_inscope.json per_intent api_lookup 0.0 n=14`
