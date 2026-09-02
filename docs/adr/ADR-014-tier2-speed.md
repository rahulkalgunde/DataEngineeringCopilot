# ADR-014: Tier2 Speed — retrieval_top_k 50→25 + mrl_multistage

## Status
Deferred — keep `retrieval_top_k=50` `mrl_multistage_enabled=False` — 2026-09-02

## Context
Tier2 speed proposal cuts retrieval cost by halving the candidate pool
`retrieval_top_k 50→25` and adding Matryoshka (MRL) multistage rescoring
(`mrl_multistage_enabled`, `mrl_small_dim 256`, `mrl_oversample_factor 4`,
store `dense_small` cosine index with `m=0` rescore-only vectors). Gate per
`settings.py:1365` (MRL dark until `Recall@10 within -0.01` + `p95 ≥20%`)
and plan `Task 1: pick p95 -20% && recall Δ CI> -0.01 on held 110`
(110/110 seed 42, `local-hf 2048d`, zero-LLM `retrieval_only=True`).

Existing serving pool is already decoupled from `top_k` via
`retrieval_prefetch_limit=100` / `fused_limit 100` per-leg
(`async_qdrant_store.py:581 dense+ sparse RRF`): the fused candidate pool
exposed to the reranker is `2×100` before `limit=fused_limit`, not
`top_k*4`. Cutting `top_k` trims only the final context slice, not the
prefetch that dominates latency. MRL requires a `dense_small` named vector
space; the active generation `pinned-d3dbad402105` has no `dense_small`
(`qdrant info vectors={'dense': 2048}`), so MRL queries fail with
`Wrong input: Not existing vector name error: dense_small` (Qdrant 400).

## Decision
Keep `retrieval_top_k=50` and `mrl_multistage_enabled=False` on held 110.
No change to `data_engineering_copilot/config/settings.py:1220` /
`settings.py:1370` or `.env.example`.

## Evaluation
`scripts/tune_tier2_speed.py` — grid `top_k {25,50} × mrl {off,on}` (4 configs),
train 110 then held gate, `k=10`, `local-hf`, `batch 55` parity,
`bootstrap_delta_ci n_boot=1000 seed=42`.

Train 110 (seed 42):

| cfg | recall@10 | p95 ms | p50 ms | p95 vs 50/off |
|---|---|---|---|---|
| 50/off (baseline) | 0.1773 | 909 | 478 | — |
| 25/off | 0.1773 | 887 | 463 | +2.4% |
| 50/on | 0.0000 | 13 | 10 | fails — 0 recall, dense_small missing |
| 25/on | 0.0000 | 14 | 10 | fails — 0 recall, dense_small missing |

`25/off` p95 improves only `+2.4%` (<20% gate). No candidate qualifies for
train winner, so train decision is `keep 50/off`.

Held 110 verification for the would-be winner `25/off`:

- `held 50/off: recall 0.1182 p95 879ms`
- `held 25/off: recall 0.1182 p95 879ms` (identical recall, same p95 band)
- `Δ recall 25-50 on held: +0.0000 CI [0.0,0.0]` but re-checked with `25/off`
  as winner the delta is `0.0` with `CI lower 0.0 > -0.01` passing recall,
  but `p95 improvement 0%` fails the `≥20%` latency gate → `keep 50`.
- MRL configs remain `recall 0` (`dense_small` not in collection) → never ship
  without a new generation built with `dense_small m=0` vectors.

Output `data/tune_tier2_speed.json`:
`{"best_top_k":50,"mrl":false,"held_recall":0.1182,"held_p95":879,"delta_recall_ci":[0,0],"decision":"keep 50"}`.

## Consequences
- `retrieval_top_k` stays `50` (frozen until store `recall@10 ≥0.35` per ADR-010).
  The `fused_limit 100` already dodges `top_k` — shrinking `top_k` cannot
  deliver `≥20%` retrieval p95 on this index shape; future speed wins must
  target prefetch / oversample or model dim, not the final slice.
- `mrl_multistage_enabled` stays `False` until a new index generation is built
  with `dense_small 256 m=0` (`async_qdrant_store.py:279 vectors_config`).
  Re-running `tune_tier2_speed` after that generation can re-gate `Δ recall CI> -0.01`
  and `p95 ≥20%` with real `dense_small` rescoring.
- No `.env.example` change (`RETRIEVAL_TOP_K 25 (ADR-014)` not shipped).
- Baseline remains `tests/evaluation/benchmarks/baseline_inscope.json R@10 0.273`
  frozen per plan Global Constraint — captures to `/tmp/new_baseline2/` only.

## Verification
- `dec_venv/bin/python scripts/tune_tier2_speed.py --k 10 --split held` → `data/tune_tier2_speed.json` `decision keep 50`.
- `grep -c dense_small` on Qdrant collection info shows absent — expected failure mode for MRL.
- Tier1: `ruff check/format/pyright` on `scripts/tune_tier2_speed.py` +
  `pytest tests/unit/test_tier2_speed.py -v -n 0` PASS.
