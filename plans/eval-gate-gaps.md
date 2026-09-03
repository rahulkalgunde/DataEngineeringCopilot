# Evaluation Gates: Gaps & Fix Plan

## Stage-by-Stage Evaluation Gate Review

| RAG Stage | Gate / Evaluation Module Found | Status & Description |
|---|---|---|
| **1. Golden Schema & Dataset** | `test_golden_schema_gate.py`, `eval_schema.py` | **Present & Active:** Validates that incoming dataset files (`eval_dataset.jsonl`, `baseline.json`) adhere strictly to schema requirements. |
| **2. Ingestion & Chunking** | `run_chunking_eval.py`, `chunking_eval.py` | **Present:** Measures boundary preservation, structural integrity, and token distribution limits across documentation sources (e.g., Spark, Airflow, Delta Lake). |
| **3. Retrieval & Reranking** | `scripts/eval_retrieval_gate.py`, `retrieval_metrics.py`, `tune_rrf_weights.py` | **Present:** Evaluates search precision, recall, and NDCG against configured floors before allowing index updates or RRF weight tuning. |
| **4. Pipeline Ablation & Routing** | `scripts/eval_pipeline_ablation.py`, `test_eval_pipeline_ablation.py` | **Present:** Quantifies the marginal impact of architectural upgrades (like query rewriting, HyDE, or ColBERT reranking) against performance baselines. |
| **5. Generation & Groundedness** | `generation_eval.py`, `groundedness.py`, `ragas_evaluation.py` | **Present:** Assesses faithfulness, answer relevance, and hallucination rates using calibrated LLM judges and Langfuse integrations. |

## Why This Architecture Causes Bottlenecks

While having a gate at *every* stage is architecturally thorough, running them sequentially or synchronously during development creates the exact friction you are experiencing:

1. **The Cascade Effect:** If your chunking or ingestion gate fails, or if a minor change shifts a retrieval metric by a fraction of a percent, the entire downstream pipeline halts before you even get generation feedback.
2. **Heavy Compute Overhead:** Running LLM-backed generation evaluators alongside retrieval metric calculations on every code push forces long build-gen waits.
3. **Rigid Thresholds:** If these gates block merges based on absolute, unyielding scores rather than relative deltas, normal iterations trigger false alarms that require manual debugging.

### Recommended Adjustments to Streamline Your Pipeline

* **Decouple the Gates:** Ensure that lower-level gates (Schema and Chunking) run as quick unit-level checks, completely separated from heavy retrieval and LLM generation evaluations.
* **Make Mid-Pipeline Gates Warning-Only (Soft Gates):** Turn exploratory ablation gates into non-blocking warnings during local development, keeping only the final generation/groundedness checks as hard blocking gates for main branch merges.

## Confirmed Gaps

1. **Brittle absolute thresholds in generation gates**
   - `generation_faithfulness_gate`, `generation_relevance_gate`, `generation_rubric_gate`, `judge_kappa_gate`, `judge_raw_gate` are static absolutes.
   - `promote_baseline.py --check-gate` only verifies source-file existence, not regression delta.

2. **CI cost / latency bottleneck**
   - Single `test-eval` job runs schema + chunking + retrieval + LLM-backed eval harnesses together.
   - No tiered strategy: fast sanity vs deep eval.

3. **OOS refusal regression not hard-gated**
   - `out_of_scope_refusal_rate` is computed in `cli.py` but never asserted in a blocking test/gate.
   - `recall_oos.jsonl` exists but is not enforced as a CI failure condition.

4. **No closed-loop trigger from drift/telemetry to re-run gates**
   - `drift_detector.py` and `feedback_telemetry.py` log/compare passively.
   - Nothing automatically re-runs evaluation or alerts when drift exceeds tolerance.

## Implementation-Ready Fix Plan

### Phase 1: Relative Regression Tolerance in Baseline Promotion
**Files to modify:**
- `scripts/promote_baseline.py`

**Changes:**
- Extend `--check-gate` to load the active baseline (`tests/evaluation/benchmarks/baseline_inscope.json`) and compare the candidate metrics.
- Enforce relative regression tolerance: candidate overall R@10 must be within `retrieval_gate_global_tolerance` (currently `0.02`) of the baseline.
- Exit non-zero and print delta if regression exceeds tolerance.

**Validation:**
- Run `dec_venv/bin/python scripts/promote_baseline.py --check-gate --source <regressed_baseline>` and confirm it exits `1`.
- Run with current baseline and confirm it exits `0`.

**Status:** ✅ Implemented
- Added `check_gate()` with relative tolerance `0.015`
- Added `--baseline` and `--relative-tolerance` CLI flags
- Verified with current baseline

---

### Phase 2: Tiered CI Evaluation
**Files to modify:**
- `.github/workflows/test.yml`
- `Makefile`

**Changes:**
- Split `test-eval` job into two jobs:
  - `test-eval-fast`: runs `make test-eval-data` + `make eval-retrieval-gate` + `make eval-stage-recall-gate --sample-size 10`.
  - `test-eval-deep`: runs `make test-eval` + `make eval-rerank-smoke` + `make eval-prompt-aug-smoke`.
- Gate configuration:
  - Fast job runs on every PR.
  - Deep job runs on main branch and as a required check before merge.

**Validation:**
- Confirm fast job completes in <10 min.
- Confirm deep job is not required on PRs but runs on main.

**Status:** ✅ Implemented
- Added `test-eval-fast` and `test-eval-deep` jobs to `.github/workflows/test.yml`
- Added `make eval-stage-recall-gate` target to Makefile

---

### Phase 3: OOS Refusal Hard Gate
**Decision: Option A** — deterministic `insufficient_context` rate gate.

**Rationale:**
- Matches existing `out_of_scope_refusal_rate` metric in `cli.py`.
- No LLM cost or flakiness in CI.
- Enforceable with existing retrieval eval infrastructure.

**Files to modify:**
- `tests/unit/test_spark_eval_diagnostics.py`
- `data_engineering_copilot/cli.py`
- `Makefile` (optional: new target `make eval-oos-gate`)

**Changes:**
- Added `gate_oos_refusal_rate()` helper in `cli.py` with threshold parameter
- Added 3 unit tests: pass, fail, and no-OOS-rows cases
- Added `make eval-oos-gate` target to Makefile

**Validation:**
- ✅ Run new unit test and confirm it passes at threshold `0.95` and fails at `0.94`.
- ✅ Run `make eval-oos-gate` against current dataset and confirm it passes.

**Status:** ✅ Implemented

---

### Phase 4: Judge Cache Expansion
**Files to modify:**
- `data_engineering_copilot/evaluation/judge_cache.py`
- `data_engineering_copilot/evaluation/generation_eval.py`

**Changes:**
- Expanded `judge_cache_key()` to accept optional `dataset_hash` parameter
- Updated `score_faithfulness()`, `score_relevance()`, `score_rubric()` to pass `dataset_hash`
- Updated all internal call sites in `generation_eval.py` to compute and forward dataset hash

**Validation:**
- ✅ Cache keys now include dataset hash for cross-dataset isolation
- ✅ Backward compatible: `dataset_hash=""` preserves original behavior

**Status:** ✅ Implemented

---

## Rollout Order
1. Phase 1 (baseline gate) — unblock current calibration workflow.
2. Phase 3 (OOS gate) — add missing hard gate with minimal cost.
3. Phase 2 (CI tiering) — reduce developer friction after fast gate is stable.
4. Phase 4 (judge cache) — cost optimization, lowest priority.

## Risks & Mitigations
- **Risk:** Tiered CI may hide slow deep-eval failures from PR authors.
  - **Mitigation:** Deep eval runs on main and as a required status check before merge; PR authors see status in GitHub checks.
- **Risk:** OOS refusal threshold may be too strict/lenient.
  - **Mitigation:** Set at `0.95` initially; adjust after one week of data. Document in plan if changed.
- **Risk:** Judge cache key expansion may invalidate existing cached verdicts.
  - **Mitigation:** New key is a superset; old keys without dataset hash still work. No forced invalidation needed.
