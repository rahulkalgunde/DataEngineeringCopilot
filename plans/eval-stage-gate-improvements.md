# Stage-by-Stage Evaluation Gate Improvements

## Current State

All five major RAG stages already have dedicated evaluation modules and gates. The missing piece is **orchestration**: gates are not tiered, not gracefully degraded, and not wired into a closed-loop re-evaluation trigger.

## Proposed Changes

### 1. Soft-Gate Wrapper for Mid-Pipeline Checks
**Files to add/modify:**
- `scripts/eval_soft_gate.py` (new)
- `Makefile`

**Behavior:**
- Wrap `eval_pipeline_ablation.py`, `tune_rrf_k.py`, `tune_rrf_weights.py` so they emit non-blocking warnings locally but fail CI when soft-gate conditions are met.
- Hard failures remain for `test-eval-fast`/`test-eval-deep`; exploratory ablations emit warnings instead of failing the job.
- **Decision:** Soft-gate warnings **fail the CI job** when executed through the wrapper in CI. Locally, they remain non-blocking.

### 2. Decoupled Fast/Slow Gate Execution
**Files to modify:**
- `.github/workflows/test.yml`
- `Makefile`

**Behavior:**
- `test-eval-fast` runs on every PR: schema, chunking invariants, retrieval regression, stage-recall attribution, OOS refusal.
- `test-eval-deep` runs on main or manual trigger: full generation/groundedness, reranker smoke, prompt-aug, ablation sweeps.

### 3. Drift-Triggered Re-Evaluation Hook
**Files to add/modify:**
- `scripts/drift_gate_hook.py` (new)
- `data_engineering_copilot/services/drift_detector.py`

**Behavior:**
- After `record()`, if `compare()` reports drift beyond threshold, emit a machine-readable alert file (`/tmp/drift_alert.json`).
- **Decision:** Drift hook triggers **automatically** after eval runs when drift is detected. No manual approval required.

## Implementation Status

### Phase 1: Soft-Gate Wrapper
- ✅ Created `scripts/eval_soft_gate.py`
- ✅ Added `make eval-soft-gate` target to Makefile
- ✅ Wired into `.github/workflows/test.yml` under `test-eval-deep`
- ✅ Decision implemented: soft-gate warnings fail CI

### Phase 2: Drift-Triggered Re-Evaluation Hook
- ✅ Created `scripts/drift_gate_hook.py`
- ✅ Modified `drift_detector.py` to call hook after `record()` when drift detected
- ✅ Hook emits `/tmp/drift_alert.json` and schedules background re-eval
- ✅ Decision implemented: drift hook triggers automatically

### Phase 3: CI/Makefile Wiring
- ✅ Added `test-eval-fast` and `test-eval-deep` jobs to `.github/workflows/test.yml`
- ✅ Added `make eval-soft-gate` and `make drift-gate-hook` targets
- ✅ Soft-gate wrapped ablation script added to `test-eval-deep`

## Open Decisions
- Resolved: soft-gate warnings fail CI
- Resolved: drift hook triggers automatically

## Verification
- ✅ `ruff check` passed for all new/modified files
- ✅ `ruff format` passed
- ✅ `pyright` passed with 0 errors
- ✅ `pytest tests/unit/test_spark_eval_diagnostics.py tests/unit/test_drift_detector.py tests/unit/test_eval_ablation.py` — 44 passed
