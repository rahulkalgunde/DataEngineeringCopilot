# 009: Hybrid vs dense-only decision — ship tuned hybrid (k=20 prefetch 100)

## Status

Accepted — ship hybrid `RRF k=20 prefetch 100 (1,1) b=0.75 rrf` (ADR-006/007/008). Dense-only branch rejected (held ΔnDCG +0.034 CI excludes 0).

## Context

Tasks 1-2 fixed BM25 fragility (`gen-bm25-rebuild` 15s local, alias-atomic `gen-activate` copy, startup warning — no silent dense fallback). Task 3 ablation on `220q recall_inscope` (110/110 seed=42) was inconclusive at default `k=60 prefetch 40`: `hybrid 0.277 vs dense 0.266 Δ+0.011 CI [-0.023,+0.046] includes 0` — no gate. Literature sweep therefore ran ordered: `k+prefetch` → `weights+b` → `fusion+reranker`, each gated on held-out bootstrap 95% CI excluding 0.

Inputs: `tests/evaluation/golden/recall_inscope.jsonl` 220q, `local-hf nvidia/Nemotron-3-Embed-1B-BF16` 2048-dim, Qdrant `data_engineering_docs` 70082 pts sparse+dense, `BM25Tokenizer(0.75) PorterStemmer`, `RrfQuery(k, weights)` / `FusionQuery(DBSF)`.

## Results (held 110, k=10, nDCG primary)

| Gate | Config | Train | Held | Δ held | 95% CI | Verdict |
|------|--------|-------|------|--------|--------|---------|
| T3 ablation | k60 L40 | dense 0.127 dense-only view | dense 0.157 hybrid 0.168 | +0.011 | [-0.023,+0.046] | **inconclusive** → tune |
| T4 k+prefetch (ADR-006) | **k20 L100** | 0.1542 best of 8 | **0.1905** vs dense 0.1568 | **+0.0337** | **[+0.0038,+0.0642]** | **ship k=20 L100** |
| T5 weights (ADR-007) | (1,1.25) best train 0.1575 vs (1,1) 0.1542 | held 0.1983 vs 0.1940 | +0.0043 vs equal | [-0.0067,+0.0175] | keep (1,1) CI includes 0 |
| T5 b | 0.5 vs 0.75 | identical 0.1575 | identical 0.1983 | 0 | — | keep b=0.75 query-side approxnil; reindex needed to bake |
| T6 DBSF (ADR-008) | DBSF L100 | — | 0.1821 vs RRF 0.1905 | -0.0084 | [-0.0281,+0.0113] | keep rrf CI includes 0, negative |
| T6 reranker 50→10 ms-marco-MiniLM-L-6-v2 | pooled 50 | — | 0.1968 vs 0.1905 | +0.0063 | [-0.0120,+0.0245] | keep no rerank CI includes 0, p95 212ms <250ms OK but no gain |

Baseline gate: `baseline_inscope.json R@10 0.273 n=220`, tuned hybrid held `R@10 0.2955` vs dense `0.277` — global gate `R@10 ≥ 0.25 (baseline-0.02) and per-intent ≥ baseline-0.05` passes on `make eval-retrieval-gate` (see `data_engineering_copilot/evaluation/gates/retrieval_gate.py`).

## Decision

**Ship tuned hybrid** (no dense-only flip):

- `data_engineering_copilot/config/settings.py:hybrid_search_enabled=True` (unchanged)
- `hybrid_rrf_k=20` (was 60) — ADR-006 `k 2..5 for ~1 rel/query` theory vs `k=60 flatten`; k=20 balances rank 1 vs rank 2-3 sparse rescue.
- `retrieval_prefetch_limit=100` (was None→40) — ADR-006 candidate depth > weights; +0.004-0.011 nDCG every k, p95 ~35ms negligible.
- `rrf_dense_weight=1.0 rrf_sparse_weight=1.0` — ADR-007 keep equal (absolute weights).
- `bm25_b=0.75` — ADR-007 keep; no reindex.
- `retrieval_fusion="rrf"` — ADR-008 keep; DBSF underperforms at 126-token avg len.
- No second-stage cross-encoder — ADR-008 p95 OK but Δ CI includes 0.

Settings provenance annotated `ADR-006/007/008` at each field (`settings.py:1371-1405`). `.env.example` documents `HYBRID_RRF_K` / `RETRIEVAL_PREFETCH_LIMIT` overrides. No `hybrid_search_enabled=False` (would need `dec reset-index` + full rebuild without sparse, ~10% storage save — not warranted).

## Consequences

- Query path stays hybrid `dense+BM25` fused `Rrf(k=20)` with 100 per-leg prefetch; `AsyncQdrantVectorStore.query(fusion)` already branches RRF/DBSF, future DBSF ship only on held CI>0.
- `gen-bm25-rebuild` (Task 1) + alias-atomic (Task 2) keep file↔Qdrant id-space single-truth; `gen-validate` now ✅ after rebuild (`vocab 93276`, `~2.5M`).
- Tier2 `make eval-fast` (10/10 rows, 70082 pts, MRR 0.54) and `make eval-retrieval-gate` green on chosen config; `pytest tests/unit/ -n 6` ~3300 pass.
- Future tuning: re-run `scripts/tune_rrf_k.py` / `tune_rrf_weights.py` on corpus shift; b ship requires full `gen-build` (731 embeddings).

## Alternatives Considered

- Dense-only (`hybrid_search_enabled=False`): rejected — hybrid wins with CI>0, so storage/interference saving not justified.
- Keep k=60 L40: rejected — held CI proves k20 L100 > dense.
- Ship (1,1.25) weights or b=0.5 or DBSF or reranker 50→10: each rejected — held CI includes 0 (would overfit 110 train).
