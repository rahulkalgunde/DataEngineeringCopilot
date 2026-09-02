# 008: DBSF vs RRF + reranker as 2nd stage — ship winner only

## Status

Accepted — keep `retrieval_fusion="rrf"` (k=20 prefetch 100, weights 1/1, b=0.75); DBSF and second-stage cross-encoder not shipped (gates not passed).

## Context

Task 4 shipped `hybrid_rrf_k=20 prefetch 100` with held ΔnDCG +0.0337 CI [0.0038,0.0642] (excludes 0) and Task 5 kept `(1,1) b=0.75` (Δ +0.0043 CI [-0.0067,0.0175] includes 0 → keep). Task 6 tests the last ordered knobs: fusion (RRF vs DBSF) and a reranker over a deeper fused pool.

- RRF: `Qdrant RrfQuery(k, weights)` rank-fusion 1/(k+rank), tuned k=20.
- DBSF: `Qdrant FusionQuery(fusion=DBSF)` — distribution-based score fusion (per-modality min-max normalize then sum). No k/weights; same prefetch limit per leg for apples-to-apples. Literature (`markaicode`): DBSF `+12% recall@100`; our corpus needs `Δ DBSF-RRF CI>0` on held 110.
- Reranker: `CrossEncoder cross-encoder/ms-marco-MiniLM-L-6-v2` locally on `fused top 50 → rerank top 10` vs `fused top 10`; `markaicode` `+19% NDCG@10` but `+100ms`. Gate: `Δ NDCG CI>0` and `p95 <250ms` for k=10 (SLO retrieval p95 <5s; baseline 68s).

Settings before: no `retrieval_fusion` flag; `AsyncQdrantVectorStore.query` only RRF. `evaluation/retrieval.py` had no rerank helper.

## Method

- Dataset: `tests/evaluation/golden/recall_inscope.jsonl` 220q, deterministic 110/110 split seed=42 (`evaluation/retrieval.py:split_queries`), same as Task 4/5.
- Base: best RRF `k=20 prefetch 100 (1,1) b=0.75` (held nDCG 0.1905, RRF k60 baseline 0.1493, dense 0.1317 on train).
- Fusion grid: `rrf (k=20)` vs `dbsf` — 2 fusions × 110 held (220 Qdrant queries, embeddings cached from prior `local-hf nvidia/Nemotron-3-Embed-1B-BF16` 2048-dim, 0 paid). Prefetch `effective_fused_limit = retrieval_prefetch_limit (=100)` for both legs; DBSF uses `FusionQuery(Fusion.DBSF)`, RRF uses `RrfQuery(k=20)`. Selection: `max nDCG@10` on held 110, paired bootstrap 1000, 95% CI for `Δ = DBSF - RRF`; ship only if CI excludes 0 and Δ>0.
- Rerank slice (on held 110 `fused top 50` from winner): lazy `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` if `sentence_transformers` importable else skip (fail-open, `evaluation/retrieval.py:is_cross_encoder_available()`). Each query: `fused 50 → CrossEncoder predict 50 pairs (query, chunk.text) → sort → top 10` vs `fused top 10` (no rerank). Timed per-query (`perf_counter`); p95 via `_percentile_rank`. Metrics: `Δ NDCG@10` bootstrap CI and `p95 ms`. Ship only if `Δ CI>0` and `p95 <250ms` (with `+100ms` budget noted).
- Execution: `data_engineering_copilot/infrastructure/async_qdrant_store.py:query(fusion="rrf"|"dbsf")` branch (one `if`) + `data_engineering_copilot/evaluation/retrieval.py:rerank_fused_with_cross_encoder / evaluate_rerank_gain`. `retrieval_fusion` setting default `rrf` (`config/settings.py`).

## Results

Held 110 (`k=10`, `local-hf`, `ns f=0`, `hybrid_ready` true via `.bm25_cache/data_engineering_docs__pinned-d3dbad402105.json` rebuilt by `gen-bm25-rebuild` Task 1):

- RRF k=20 L100 held nDCG **0.1905** recall 0.2955 (Task 4 winner, reproduced)
- DBSF L100 held nDCG **0.1821** recall 0.2864, Δ DBSF-RRF **-0.0084** 95% CI **[-0.0281, +0.0113]** **includes 0** → no ship. Directionally DBSF underperforms RRF at this corpus scale (70k points, ~1 relevant/query, avg len 126). `+12% recall@100` not observed at `k=10`; per-modality-score normalization hurts when sparse is sparse (many queries zero-match sparse).

Reranker slice (fused 50 pooled with winner RRF k=20 L100, held 110, `ms-marco-MiniLM-L-6-v2`):

- Fused top 10 nDCG 0.1905
- Reranked 50→10 nDCG **0.1968** recall 0.3012, Δ **+0.0063** CI **[-0.0120, +0.0245]** includes 0 → no ranking gain (sparse already rescued by RRF 100-depth pool; cross-encoder adds little for short 126-token chunks).
- Latency: mean 88ms / p95 **212ms** (CPU `bge-reranker-v2-m3` 80-pool reference 65ms; `ms-marco-MiniLM-L-6-v2` 50-pool 88ms; `+~100ms` vs dense-only). p95 **212ms <250ms** budget OK, but CI includes 0 so efficiency not justified.
- Availability: `sentence_transformers` importable (5.6.0); `CrossEncoder` lazy loads off event loop. Skip path covered (ImportError → None → fused top 10).

Both gates fail: DBSF `CI includes 0` (actually Δ negative) and reranker `CI includes 0` despite p95 budget. Per "ship winner only", keep RRF and no second-stage reranker in production path (`retrieval_fusion="rrf"`). The fused pool remains 100 for both legs so future re-evaluation is identical.

## Decision

Keep defaults:

- `data_engineering_copilot/config/settings.py:retrieval_fusion` = `"rrf"` (`Literal["rrf","dbsf"]`)
- `data_engineering_copilot/infrastructure/async_qdrant_store.py:AsyncQdrantVectorStore.query(fusion: Literal["rrf","dbsf"] | None = None)` — `effective_fusion = fusion or settings.retrieval_fusion`; `if dbsf: FusionQuery(Fusion.DBSF) else: RrfQuery(Rrf(k, weights))`; `rrf_confidence_scale=1.0` for DBSF vs `(k+1)/2` for RRF; prefetch 100 unchanged.
- `data_engineering_copilot/evaluation/retrieval.py:RERANK_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`, `RERANK_FUSED_POOL=50`, `RERANK_TOP_K=10`, `is_cross_encoder_available()`, `rerank_fused_with_cross_encoder()`, `evaluate_rerank_gain()` — gated, fail-open (skip if not installed).

No mutation to `hybrid_rrf_k`, `retrieval_prefetch_limit`, `rrf_dense_weight`, `rrf_sparse_weight`, `bm25_b`. No reindex; fusion/rerank are query-time only.

## Consequences

- Qdrant hybrid still ships as RRF k=20 prefetch 100 (ADR-006) with equal weights (ADR-007). DBSF remains dark until a held CI>0 is proven (e.g. after corpus grows or chunk len >>126).
- `AsyncQdrantVectorStore` now supports `fusion="dbsf"` for ablation (`tests/unit/test_fusion_dbsf.py` contract: source contains `dbsf`, `settings.retrieval_fusion` field, `evaluation/retrieval.py` cross-encoder helper). `dec eval-retrieval --ablation` dbsf mode will use `fusion="dbsf"` when wired.
- Reranker helpers remain for local `gen-validate`-free tuning; production RAG (`factory.build_rag_service`) still uses `CrossEncoderReranker BAAI/bge-reranker-v2-m3` pool 80 only when `reranker_enabled` and `llm_rerank_enabled` chain needs it — the new `ms-marco-MiniLM-L-6-v2` 50→10 slice is eval-only.
- Future fusion ship requires held `Δ DBSF-RRF CI>0` and reranker ship requires both `Δ NDCG CI>0` and `p95 <250ms`; both re-checked on the same frozen 110 held split.

## Alternatives Considered

- Ship DBSF (`dbsf`): rejected — Δ -0.0084 CI includes 0 (negative), no `+12% recall@100` at k=10; would need weight/coverage re-tune but literature says weights not applicable to DBSF.
- Ship reranker 50→10: rejected — Δ +0.0063 CI includes 0; p95 212ms OK but gain not proven (would overfit held 110). Keep `reranker_enabled` path unchanged (80-pool `bge-reranker-v2-m3`).
- Fusion via `settings.retrieval_fusion` only vs `query(fusion=)` param: chose `fusion | None = None` resolving to `settings.retrieval_fusion` so ablation can sweep per-query without mutating global settings; default stays `rrf`.
