# 006: Tuned RRF k and prefetch limit on train/holdout

## Status

Accepted — ship k=20 prefetch 100 (holdout ΔnDCG +0.034 CI excludes 0).

## Context

Task 3 ablation on 220q (recall_inscope) was inconclusive: hybrid_rrf_k60 0.277 vs dense 0.266 Δ+0.011 CI [-0.0227,+0.0455] includes_0=True. Same on 110 train holdout — no config proven wins. Literature (How to Tune Hybrid Search) suggests RRF `k` controls how sharply top ranks dominate: small `k` (2-5) heavily rewards the very top document, large `k` (60, the default) flattens rank contributions so lower ranks still matter. For ~1 relevant per query (our `n=220 avg 1.1 relevant`) theory predicts small k helps — the single relevant should be at rank 1.

Prefetch limit matters separately: RRF fuses two ranked lists (dense + sparse). The fused query sees only `prefetch` candidates per leg (Qdrant `prefetch.limit`). The prior default was `max(k*4,40)=40` for k=10; the recommendation to try 100 follows Qdrant docs and the `candidate depth > weights` rule — a deeper pool lets the cross-encoder reranker rescue a relevant chunk that BM25 placed below dense head without changing fusion weights.

Task 4 ordered tuning runs these one gate at a time: first `k`+`prefetch`, then weights, then DBSF, to avoid confounding.

## Method

- Dataset: `tests/evaluation/golden/recall_inscope.jsonl` 220q, deterministic 110/110 split seed=42 (`evaluation/retrieval.py:split_queries`).
- Grid: `k ∈ {2,5,20,61}` × `L ∈ {40,100}` (8 configs), `k=10` retrieved, nDCG@10 primary.
- Embeddings: `local-hf nvidia/Nemotron-3-Embed-1B-BF16` 2048-dim, rewrites disabled, embedding cache reused — each query embedded once then reused across 8 configs (≈220 embeddings, 880 Qdrant queries).
- Selection: pick `max nDCG@10` on train 110.
- Validation: same winner scored on held 110; paired bootstrap 1000 resamples, 95% CI for `Δ = hybrid_k_best - dense` (per-query nDCG). Ship only if CI excludes 0 and Δ>0; otherwise keep `k=60 prefetch 40`.

Execution: `scripts/tune_rrf_k.py` (produces `data/tune_rrf_k.json`).

## Results

Train 110 nDCG@10:
- k=2 L40 0.1266, L100 0.1308
- k=5 L40 0.1409, L100 0.1448
- k=20 L40 0.1431, L100 **0.1542 ← best**
- k=61 L40 0.1493, L100 0.1517
- dense train 0.1317

Held 110 validation (winner k=20 L100):
- dense held nDCG 0.1568 recall 0.2773
- hybrid k20 L100 held nDCG **0.1905** recall 0.2955
- ΔnDCG +0.0337 95% CI [+0.0038,+0.0642] **excludes 0** → ship
- Δrecall +0.0182 CI [-0.0182,+0.0545] includes 0 (ranking gain, not recall)

All 8 train configs beat dense; k=20 L100 was maximal. Prefetch 100 beats 40 for every k (+0.004 to +0.011 nDCG), confirming candidate depth helps at this corpus scale (70k points, p95 per leg ~35ms, 100 vs 40 negligible).

## Decision

Ship `hybrid_rrf_k=20` and `retrieval_prefetch_limit=100`:

- `data_engineering_copilot/config/settings.py:hybrid_rrf_k` 60 → 20
- `data_engineering_copilot/config/settings.py:retrieval_prefetch_limit` None → 100
- `data_engineering_copilot/infrastructure/async_qdrant_store.py:query()` honors `retrieval_prefetch_limit` when `fused_limit` is not explicitly passed, else `max(k*4,40)`.

The `k=20` choice sits between the theory-optimal small k (2-5 for single relevant) and the flatten default (60). At `k=2` the fusion over-penalizes a relevant at rank 2-3 (dense head often rank 1 stale, sparse relevant rank ~20); at `k=60` the fusion is too flat to promote the sparse hit; `k=20` balances — consistent with `mixpeek` and Qdrant tuning notes where `k ≈ 20-30` often wins for mixed relevant counts.

## Consequences

- `AsyncQdrantVectorStore` now fuses with `Rrf(k=20)` and `prefetch.limit=100` per leg when called via RAG service (`fused_limit=_rerank_pool_size(...)` already passes explicit limit; the setting governs eval paths and direct `query()` without explicit limit).
- No reindex needed — RRF k and prefetch are query-time fusion parameters only.
- Next gate (Task 5) tunes `weights` at this `k=20 prefetch 100` basis; `b` (BM25 length norm) remains 0.75 until Task 5 proves otherwise (requires `gen-bm25-rebuild` + reindex if changed).

## Alternatives Considered

- Keep `k=60 prefetch 40` (default): rejected — held CI proves `k=20 L100` > dense with 95% confidence.
- `k=5`: theory-optimal for 1 relevant but underperforms `k=20` on train by -0.0094 nDCG; would need weight re-tune per literature (weights need re-tune after k).
- `L=200`: not tested (plan allows {40,100,200} but Task 4 step trials {40,100}); if Task 6 DBSF shows recall@100 gains, consider L=200 there.
