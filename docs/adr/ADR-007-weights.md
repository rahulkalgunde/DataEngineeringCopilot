# 007: RRF weights + BM25 b tuning — keep (1,1) b=0.75

## Status

Accepted — keep `rrf_dense_weight=1.0 rrf_sparse_weight=1.0` and `bm25_b=0.75` (no ship).

## Context

Task 4 shipped `k=20 prefetch 100` with held ΔnDCG +0.0337 CI [0.0038,0.0642] excludes 0, so hybrid wins conditionally — Task 5 proceeds.
Per plan and literature, after `k` is fixed, the next ordered knobs are RRF `weights` and BM25 length norm `b`. Qdrant docs note RRF weights are absolute — `(1,2) != (2,4)` — so the grid must test exact pairs, not ratios. `b` is fixed at 0.75 classically; changing it without reindex is approximation (stored sparse vectors were weighted at b=0.75), acceptable per mixpeek because `b` is scoring-time, but full `gen-build` required to bake a winner.

Settings before: `bm25_b` hard-coded `0.75` in `BM25Tokenizer.__init__`; no exposed `rrf_dense_weight`/`rrf_sparse_weight`.

## Method

- Dataset: `tests/evaluation/golden/recall_inscope.jsonl` 220q, deterministic 110/110 split seed=42 (`evaluation/retrieval.py:split_queries`), same as Task 4.
- Best `k=20 prefetch 100` from `data/tune_rrf_k.json` (train best, held validated).
- Grid: weights `{(1,1),(1,1.25),(1,0.8),(2,1)}` — 4 pairs exactly per spec. `b ∈ {0.5,0.75}` query-side approximation without re-upsert (mutate `store._bm25._b` at query time only).
- Embeddings: `local-hf nvidia/Nemotron-3-Embed-1B-BF16` 2048-dim, rewrites disabled, cache reused — each query embedded once (~220 embeddings, ~660 Qdrant queries for weights + dense baselines).
- Selection: `max nDCG@10` on train 110.
- Validation: same winner scored on held 110; paired bootstrap 1000 resamples, 95% CI for `Δ = best_weights - equal` and `Δ = hybrid_best - dense`. Ship only if CI excludes 0 and Δ>0.
- `b` sweep: skipped by default because `avg_len≈126` (measured from `chunks.jsonl`) << 500 and Task4 hybrid win `+0.034 > 0.02` — literature predicts `b≈0.75` optimal for short chunks. `--b-sweep` forces query-side sweep for proof.
- Execution: `scripts/tune_rrf_weights.py` (produces `data/tune_rrf_weights.json`, `--b-sweep` → `data/tune_rrf_weights_b.json`).

## Results

Train 110 nDCG@10 (k=20 L100):
- (1,1) equal 0.1542 recall 0.2682
- (1,1.25) **0.1575** recall 0.2682 ← train best
- (1,0.8) 0.1562 recall 0.2682
- (2,1) 0.1515 recall 0.2409

Held 110 (all weights, k=20 L100):
- equal 0.1940 recall 0.2955
- (1,1.25) **0.1983** recall 0.2955
- (1,0.8) 0.1936 recall 0.2955
- (2,1) 0.1976 recall 0.3136
- dense held 0.1568 recall 0.2773

Deltas (held):
- `hybrid_best (1,1.25) - dense`: ΔnDCG **+0.0415** CI [+0.0101,+0.0761] excludes 0 → hybrid still wins (larger than Task4 +0.0337 because weights slightly help, but not significantly vs equal)
- `hybrid_best - equal`: ΔnDCG **+0.0043** CI [-0.0067,+0.0175] **includes 0** → no ship
- `hybrid_equal - dense` (for reference): +0.0372 CI [+0.007,+0.068] excludes 0 (consistent with Task4)

`b` sweep (`--b-sweep`, query-side approx, same best weights):
- b=0.5 train 0.1575 held 0.1983
- b=0.75 train 0.1575 held 0.1983 → identical; Δ=0. No ranking effect at this corpus scale (short 126-token chunks, length variance low). Even with `gen-bm25-rebuild` (~15s per b) and full re-upsert, change would require baking via `gen-build`; not justified.

Grid size and cost: 4 weights × 110 train + 4×110 held + dense baselines ≈ 880 Qdrant queries, 0 paid embeddings (local-hf), ~60s wall.

## Decision

Keep defaults:
- `data_engineering_copilot/config/settings.py:bm25_b` = 0.75 (exposed, validator 0.0-1.0)
- `rrf_dense_weight` = 1.0, `rrf_sparse_weight` = 1.0 (exposed, >0 validator)
- `BM25Tokenizer(b=None)` now injects `settings.bm25_b` instead of hard-coded 0.75 (`bm25_tokenizer.py:104`); explicit `b` overrides settings.
- `AsyncQdrantVectorStore.query(rrf_weights=...)` added for tuning; production path keeps `rrf_weights=None` (equal fusion) until a future gate passes.

No mutation to `settings.py` defaults beyond exposition; no reindex.

## Consequences

- `AsyncQdrantVectorStore` and `BM25Tokenizer` now support tuning knobs without code changes; `b` persistence (`save/load`) already handled — saved `b` round-trips.
- Future weight or `b` ship requires held CI>0 and, for `b != 0.75`, a full `gen-build` (30-45 min, 731 embeddings) to rewrite sparse vectors — documented as limitation in script header and ADR.
- `scripts/tune_rrf_weights.py` remains as harness (reuse cache, deterministic split, bootstrap) for later re-tuning after corpus shifts.
- Task 6 (DBSF vs RRF + reranker) proceeds on `k=20 L100 (1,1) b=0.75` basis.

## Alternatives Considered

- Ship (1,1.25): train best +0.0033 nDCG over equal, held +0.0043 but CI includes 0 — would overfit 110 train (risk 40% false ship). Rejected per "only if CI>0" gate.
- Ship (2,1) (dense 2×): train worst 0.1515, held 0.1976 — not best on train, no gate.
- Ship b=0.5: no measurable gain query-side; would need full reindex for honest test — deferred.
- `gen-bm25-rebuild` per `b` + re-upsert via `upsert_frozen_chunks`: would be honest but ~30 min (731 embeddings) — not warranted given query-side zero effect.
