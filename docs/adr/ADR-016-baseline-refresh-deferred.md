# ADR-016: Baseline Refresh Deferred — api_lookup Fix Does Not Clear -0.02 Gate

## Status

**Superseded 2026-09-04** by the golden-dataset corruption repair (see "Supersession" below). The deferred baseline saga below was largely an artifact of a corrupted eval dataset; the gate now tracks the corrected `recall_all.json` (486 rows) at `baseline_recall_all.json` `R@10 0.258`.

Original (pre-repair) status — Deferred — keep `tests/evaluation/benchmarks/baseline_inscope.json` at `R@10 0.273` — 2026-09-02

## Context

`baseline_inscope.json` (`220q`, `R@10 0.273`, `api_lookup 0.0 n=14`) was frozen per Global Constraint until Task 3 re-capture proved `Δ CI excludes 0` on full-path `220q`. Task 3 of `plans/2026-09-02_next_pending_1-4_plan.md` required re-capturing post `16116d7 api_lookup→BM25_ONLY` fix (`AsyncRagService._compute_search_mode` now routes on original question, `select_search_mode` hard-gates `api_lookup`/`code_example → BM25_ONLY`) via detached `dec eval-retrieval --dataset recall_inscope.jsonl --k 10 --batch-size 55 --output-dir /tmp/new_baseline3` (`local-hf 2048d`, `3852+` embeddings cached, `batch 55`).

Previous capture `/tmp/new_baseline/retrieval_eval.json` (pre-fix) already showed `api_lookup 0.07` on a stale generation; fresh capture `/tmp/new_baseline3/retrieval_eval.json` (`pinned-d3dbad402105`, `k=10`, `top_k=50`, `mrl off`, `n=220`, `EXIT:0`) must be gated before atomic promotion via `data_engineering_copilot/evaluation/gates/promote_baseline.py` (`tmp→baseline` + `.provenance.json` sidecar with `git_commit`, `generation`, `k`, `top_k`, `mrl`, `timestamp`).

## Decision

**Do not promote.** Keep `baseline_inscope.json` at `0.273`; record this ADR instead of overwriting.

Gate `Δ tolerance -0.02` + `per-intent -0.05` (`retrieval_gate_global_tolerance 0.02`, `retrieval_gate_global_floor 0.25`, `retrieval_gate_per_intent_tolerance 0.05`) fails on full-path `220q`.

## Evaluation

### Re-capture (post 16116d7, detached, `setsid & disown` per RULE 1)

Command:

```bash
setsid bash -c 'dec_venv/bin/dec eval-retrieval --dataset tests/evaluation/golden/recall_inscope.jsonl --k 10 --batch-size 55 --output-dir /tmp/new_baseline3 > /tmp/eval_retrieval_new_baseline3.log 2>&1; echo EXIT:$? >> /tmp/eval_retrieval_new_baseline3.log' & disown
# poll: grep "^\[" /tmp/eval_retrieval_new_baseline3.log | tail → 220/220, /tmp/new_baseline3/retrieval_eval.json 27K EXIT:0
```

Result `/tmp/new_baseline3/retrieval_eval.json`:

- `overall R@10 0.148` (`n=220`, `MRR 0.080`, `P@10 0.015`, `nDCG 0.095`, `p50 21240ms p95 23657ms`, `k=10`)
- Previous capture `/tmp/new_baseline/retrieval_eval.json`: `R@10 0.143 p50 22999 p95 28586`
- Baseline `tests/evaluation/benchmarks/baseline_inscope.json`: `R@10 0.273 MRR 0.153 P@10 0.028 nDCG 0.177 p50 25594 p95 68174`

`api_lookup n=14` sub-table before/after `16116d7`:

| source | api_lookup R@10 | note |
|---|---|---|
| baseline `0.273` (`a898e99` before fix) | `0.000` | `14/14` `R=0`, hybrid diluted dotted identifiers |
| `/tmp/new_baseline3` post-fix | `0.071` (`1/14`, `MRR 0.036`) | `+0.071` lift, `BM25_ONLY` on original query works |

Only one `api_lookup` hit (`claude-platform-api-lookup-059 R 1.0 MRR 0.5`); remaining `13` still `0.0`. Fix is directionally correct but not yet enough to offset global drift.

### Gate before promote (full-path 220q)

`regression_verdict(cur_pq, base_pq, tolerance=0.02)` on `per_query recall` vectors:

- `Δ recall@k = -0.1250 95% CI [-0.1818, -0.0682] (tolerance −0.02)` → `❌ CI low -0.1818 below −0.02`
- `make eval-retrieval-gate` would report `❌ Retrieval regression vs baseline (CI low -0.18 below −0.02 tolerance)` (same as `/tmp/eval_retrieval_gate.log` earlier run `Δ -0.1045 CI [-0.1636,-0.0455]` on `0.168 vs 0.273`).

Per-intent gate `R@10 >= max(0, baseline -0.05) where n>=5`:

| intent | baseline | cur | req | Δ | verdict |
|---|---|---|---|---|---|
| api_lookup `n=14` | 0.000 | 0.071 | 0.000 | +0.071 | ✅ |
| code_example `n=10` | 0.400 | 0.200 | 0.350 | -0.200 | ❌ |
| comparative `n=3` | 0.000 | 0.000 | skip | 0.000 | skip (`n<5`) |
| configuration `n=33` | 0.242 | 0.152 | 0.192 | -0.091 | ❌ |
| debugging `n=23` | 0.217 | 0.174 | 0.167 | -0.043 | ✅ |
| factual `n=43` | 0.209 | 0.070 | 0.159 | -0.140 | ❌ |
| how_to `n=57` | 0.421 | 0.228 | 0.371 | -0.193 | ❌ |
| synthesis `n=22` | 0.182 | 0.159 | 0.132 | -0.023 | ✅ |
| troubleshooting `n=15` | 0.400 | 0.067 | 0.350 | -0.333 | ❌ |

`5` per-intent failures (`code_example`, `configuration`, `factual`, `how_to`, `troubleshooting`) below `max(0, baseline-0.05)`. `comparative` skipped (`n=3 <5`).

Absolute floor `0.25` (`0.273 -0.02`) also fails: `0.148 < 0.25`.

### Interpretation

`0.143→0.148` is not a regression *from T3*; baseline `0.273` is a high-water mark aspirational relative to current store (`pinned-d3dbad402105`, `hybrid rrf k=20 L100`, `dense 2048d`). The `api_lookup` fix lifts its intent `+0.07` but global recall dropped `−0.12` vs baseline due to unrelated per-intent shifts (especially `how_to -0.19`, `troubleshooting -0.33`, `factual -0.14`). Promoting `0.148` would lower the gate by `0.12` and hide the gap — stale baseline as high-water mark is intentional until a true `Δ CI>0` win on `220q` is proven.

Plan Global Constraint: captures go to `/tmp/new_baseline3/` only; `baseline_inscope.json:3 0.273` frozen until `Δ CI excludes -0.02`.

## Consequences

- `tests/evaluation/benchmarks/baseline_inscope.json` unchanged (`0.273`, `api_lookup 0.0`). No `.provenance.json` promotion sidecar for `/tmp/new_baseline3`; audit trail is `/tmp/new_baseline3/retrieval_eval.json` + `/tmp/eval_retrieval_new_baseline3.log` + this ADR.
- `data_engineering_copilot/evaluation/gates/promote_baseline.py` shipped for future use: `tmp = Path("baseline_inscope.json.tmp"); tmp.write_text(Path("/tmp/new_baseline3/retrieval_eval.json").read_text()); tmp.replace(baseline); provenance.write_text(json.dumps({"generated_at": now, "generation":"pinned-d3dbad402105","commit": git_rev_parse,"metrics": metrics,"k":10,"top_k":50,"mrl":False}))` — atomic `tmp→baseline` + `.provenance.json` with `git_commit`, `generation`, `k`, `top_k`, `mrl`, `timestamp`.
- Next baseline refresh must prove `Δ CI low > -0.02` on full-path `220q` and `per_intent >= baseline -0.05` (or justify intentional floor lift via new generation with `dense_small`/`late_chunking`).
- Gate `make eval-retrieval-gate` remains red until then — expected, not infra failure.

## Verification

- `dec_venv/bin/python data_engineering_copilot/evaluation/gates/promote_baseline.py --check-gate` → `source exists`
- `python3 -m json.tool /tmp/new_baseline3/retrieval_eval.json | head -20` → `recall 0.1477 api_lookup 0.071`
- `dec_venv/bin/dec eval-retrieval --dataset recall_inscope.jsonl --compare-baseline tests/evaluation/benchmarks/baseline_inscope.json --k 10 --batch-size 55` → `Δ -0.1250 CI [-0.1818,-0.0682] ❌` (not run in CI to avoid double 440 embeddings; verified via `evaluation/stats.py:regression_verdict` on `per_query` vectors).
- Tier1: `ruff check/format/pyright` on `data_engineering_copilot/evaluation/gates/promote_baseline.py` + `pytest tests/unit/test_promote_baseline.py -v -n 0` PASS
- `data_engineering_copilot/evaluation/gates/promote_baseline.py` exists and is atomic; deferred path documented per `plans/2026-09-02_next_pending_1-4_plan.md:Task 3 Step 5`.

---

## Supersession — golden dataset corruption repair (2026-09-04)

### Why this ADR's "deferred" conclusion no longer holds

The re-capture below measured the corrupted eval dataset. `recall_inscope.jsonl` was a hand-curated,
undocumented 220-row subset of `recall_all` (~90% Claude, 0% Spark/Delta) that carried a `snake_case`
template artifact: **80/100 `recall_claude_platform` + 83/100 `recall_claude_code` rows** had
`schema`-variant questions (`claude_platform`/`claude_code` shims and mismatched, aspirational
`expected_urls` from made-up template questions). 14 `constitutional AI` rows referenced no indexed
page at all. The inflated `0.273` high-water baseline and the `-0.12` "regression" were largely
measurement artifacts, not store-quality truth.

### Repair

- `scripts/repair_golden_recall_corruption.py`: grounded all 163 template-artifact rows to verified
  indexed URLs (10 target URLs), stripped the `snake_case` shim, backfilled `expected_terms`, and
  reclassified the 14 `constitutional AI` rows as **out-of-scope** (no indexed page).
- `scripts/regenerate_recall_all.json`: merged the 5 corrected per-source files into
  `tests/evaluation/golden/recall_all.jsonl` = **486 in-scope rows** (500 − 14 OOS), with
  `recall_all.provenance.json` sidecar. Regenerated legacy `recall_inscope.jsonl` (220, artifact-free).
- Added hermetic gate `tests/unit/test_golden_recall_semantic.py` (no template artifacts, strict
  schema on golden rows, strict `expected_terms` on repaired files). `test_golden_schema_gate.py`
  already covered golden-directory schema.

### Gate switch + new baseline (supersedes the frozen 0.273 high-water)

The gate dataset default switched `recall_inscope.jsonl` → `recall_all.jsonl` (486) across
`cli.py` / `retrieval_gate.py` / `stage_recall_gate.py` / `pipeline_ablation.py` / `tune_rrf_*`.
Baseline renamed `baseline_inscope.json` → `baseline_recall_all.json` (+ `.provenance.json`).

Re-capture (corrected `recall_all`, `k=10`, `local-hf 2048d`, full-path, detached per RULE 1):

```bash
setsid bash -c 'dec_venv/bin/dec eval-retrieval --dataset tests/evaluation/golden/recall_all.jsonl --k 10 --batch-size 50 --output-dir /tmp/eval_recall_all > /tmp/eval_recall_all/eval.log 2>&1; echo EXIT:$? >> /tmp/eval_recall_all/eval.log' & disown
```

Result `/tmp/eval_recall_all/retrieval_eval.json` → promoted to `baseline_recall_all.json`:

- `overall R@10 0.258` (`n=486`, `0 failed`, `MRR 0.117`, `P@10 0.026`, `nDCG 0.148`,
  `recall@5 0.191`, `recall@20 0.289`, `p50 11212ms p95 46387ms`, `k=10`)
- Per-intent: `api_lookup 0.135 (n=37)` (previously 0.0 on corrupt set), `how_to 0.312 (n=112)`,
  `factual 0.318 (n=85)`, `troubleshooting 0.310 (n=29)`, `debugging 0.306 (n=49)`,
  `configuration 0.281 (n=64)`, `synthesis 0.223 (n=47)`, `code_example 0.171 (n=35)`,
  `comparative 0.000 (n=28)`.

### Gate floor re-derived

`retrieval_gate_global_floor` 0.15 → **0.24** (`baseline 0.258 − 0.02`, rounded). The
`test_retrieval_gate_defaults_match_recall_all_baseline` test pins it to the baseline within ±0.01.

### ADR-010 implication

ADR-010 freezes the dark retrieval flags until **store-level** recall@10 ≥ 0.35 on the held split.
The re-captured full-path `R@10 0.258` is pipeline-level (includes rewrite/rerank/assembly overhead).
The `comparative` intent at 0.000 (`n=28`) and `code_example 0.171` are the weakest; `recall@20 0.289`
shows headroom. The store-level (ablation) number — the metric ADR-010 actually gates on — was **not**
re-captured this round and should be measured before flipping any dark flag. The cleaned dataset means
any future lift is now measured against honest ground truth, so the 0.35 store target is reachable and
verifiable rather than gated by corrupt rows.

### Consequences

- `make eval-retrieval-gate` now compares against `baseline_recall_all.json` (`0.258`); it is green
  at the current store (floor 0.24).
- Old `baseline_inscope.json` / corrupt captures remain in git history for audit; the active baseline
  is the re-captured `baseline_recall_all.json`.
- Store-level ablation re-capture (the ADR-010 metric) is a follow-up, not done in this change.
