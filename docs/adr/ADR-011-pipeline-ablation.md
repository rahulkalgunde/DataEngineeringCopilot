# ADR-011: Pipeline Ablation — guardrails / sibling / dedup

## Status
Accepted — 2026-09-02

## Context
The 16-stage pipeline (async_rag.py:363) carries downstream stages that were
added without a holdout ablation: input guardrails (scan_chunks 1056),
sibling rejoin (994 max_blocks 3), and assembly dedup (ContextAssembler
content_hash_dedup / Jaccard). Store recall on held 110 is ~0.27 (R@10
baseline_inscope). Downstream stages can dilute recall by dropping or
merging candidates before context assembly. No stage should stay on unless
it demonstrably helps.

## Decision
Ablate each stage on **held 110 only** (seed 42, 110/110 split) via
`dec eval-retrieval --pipeline-ablation {guardrails,sibling,dedup,all}`
and `data_engineering_copilot/evaluation/gates/pipeline_ablation.py`. For each stage compare **with vs
without**:

- `guardrails`: `input_guardrails_enabled True` vs `False` (InputGuardrails
  disabled → `service.input_guardrails = None`, i.e. scan_chunks skipped).
- `sibling`: `assembly_enable_sibling_merge True` vs `False` plus
  `_rejoin_sibling_chunks(max_sibling_blocks 3→0)` patched to no-op when
  disabled.
- `dedup`: `assembly_content_hash_dedup True` vs `False` plus
  `context_compression_enabled True` vs `False` (ContextAssembler
  `content_hash_dedup` + `deduplicate` toggled).

Metrics: mean `recall@10` and `nDCG@10` per stage with paired bootstrap
95% CI (1000 resamples, same as `scripts/tune_rrf_k.py`). Reuses
`local-hf` embedding cache, `batch_size 55`, `retrieval_only=True` (no LLM).

Emit `data/pipeline_ablation.json`:

```json
{
  "guardrails": {"with": 0.143, "without": 0.19, "delta": -0.047, "ci": [-0.08, -0.02]},
  "sibling": {"with": 0.21, "without": 0.22, "delta": -0.01, "ci": [-0.03, 0.01]},
  "dedup": {"with": 0.20, "without": 0.19, "delta": 0.01, "ci": [-0.02, 0.04]}
}
```

Decision rule per stage (mirrors `evaluation/retrieval.py:pipeline_stage_decision`):

> `ship = CI excludes 0 and delta>0 else keep_off`

I.e. keep a stage **on** only when its CI excludes 0 and the delta is
positive (stage helps). Otherwise keep it **off** until gold-aware tuning
proves otherwise. Bootstrap uses `bootstrap_delta_ci` (paired) with seed 13.

Holdout is the gate: train is not evaluated. Run on held only
(`--split held`) per `make eval-pipeline-ablation --k 10 --split held`.

## Consequences
- If guardrails hurts (negative delta with CI excluding 0), recommend
  `input_guardrails_enabled=False` until gold-aware injection thresholds are
  tuned.
- Sibling and dedup similarly stay off when neutral or harmful.
- Provenance: `74236d9 k20 L100` base unchanged; no LLM for eval.
- Flags stay dark until ablation proves `+Δ CI>0` on held.

## Verification
- `dec eval-retrieval --pipeline-ablation all --k 10 --split held` emits table
  and `data/pipeline_ablation.json` with per-stage CI and decision.
- `dec_venv/bin/python data_engineering_copilot/evaluation/gates/pipeline_ablation.py --k 10 --split held`
  grid 110/110, `Δ recall/nDCG` + 95% CI.
- Unit gate: `tests/unit/test_eval_pipeline_ablation.py` asserts
  `--pipeline-ablation` in `--help` and choices, and
  `PIPELINE_ABLATION_STAGES` set.
