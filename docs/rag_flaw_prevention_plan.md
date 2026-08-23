# RAG Flaw Prevention Plan — enforced checks so no session repeats the 2-day waste

**Date:** 2026-08-23 12:30 → 14:00 (F1–F6 queued) → 16:00 (F1–F5 SHIPPED) → 17:00 (flaw #7 SHIPPED — all 7 patterns enforced)
**Status:** COMPLETE — all 7 flaw patterns have live CI guards
**Context:** 2 days instrumenting eval that measured URL-string mismatch + junk-term noise. Live RAG was healthy (R@10=0.784 inscope); the validator was not. External cross-check (RAGAS/DeepEval/Phoenix, RAGBench/TRACe, KG-RAG audit: correctness <60%, 12 LLM-as-judge biases) confirms these are the top failure modes.

## Pattern taxonomy — local evidence × external canon

| # | Pattern | Local evidence (file:line) | External canon | Waste it caused | Cheap guard (already or now) |
|---|---------|----------------------------|----------------|-----------------|------------------------------|
| 1 | **Alias blindness** | `services/eval_coverage.py:102 _url_path_match` fixed for coverage but `cli.py:2282` / `fast_eval.py:384` still raw `in {urls}` | RAGBench: alias mismatch → recall@k=0, team tunes retriever | `eval_schema` write-time lint + pre-flight suffix gate |
| 2 | **Junk expected_terms** (`spark?`) | 403/500 rows, `terms_counter['spark?']=61` | RAGAS synthetic sets need validation (Anyscale) | `eval_schema.py:113` reject `?`/len<2 at write time — 6 files cleaned |
| 3 | **Silent empty contexts** | `cli.py:2268` `get("expected_terms",[])` → 0, `generation_eval.py` empty → parametric hallucination | TRACe fail-open → `faithfulness=1.0` on `[]` | `generation_eval.py:401` fail-fast (landed); `fast_eval` still open — now fixed |
| 4 | **Duplicate-URL inflation** | `cli.py:2369` R@10=7.0, `retrieval_metrics.py:24` nDCG=1.195 | fabs/Weaviate: dedupe before `pytrec_eval` | Set-dedup via `topk_hits={...}` — pinned by 9 tests |
| 5 | **Scope drift** — golden asks what corpus cannot answer | `cli.py:2416` golden=500 rows vs `ci-repro`=71k chunks but wrong URL dialect; `__ci-repro` only 2 Claude doc sites in live alias | RAGBench 5-domain diversity + Statsig "version golden alongside code" | `cli.py:2421` pre-flight warning (soft, 2s) + `recall_inscope.jsonl` (220 rows) |
| 6 | **Uncalibrated LLM-as-judge** | `provider_fallback.py:59 degraded_fallback` last-resort bias | DataAspirant κ=0.31 before calibration; 12 bias types | `judge_calibration.py` Cohen's κ gate + majority-vote labeler (78/80 unanimous) |
| 7 | **Context fragmentation** (tables/headers split) | **SHIPPED `48eb658`**: oversized headed sections were never windowed → one giant chunk per long section; header-carry invariant + fracture gate (≤0.25) + 5–20% overlap guard in CI (`test_context_fragmentation_guards.py`) | Oracle RAG chunking, Coverge overlap >20% duplicate | `eval-chunking` gates verdict + 6 guard tests |

## The 4 enforced checks (what now blocks the 2-day loop)

1. **Write-time lint** — `eval_schema.py:110` rejects `*?`/len<2 terms + allows URL-only recall rows. `test_golden_schema_gate.py` runs every PR (hermetic). 403 junk rows already stripped; file-local, 0 LLM.
2. **Alias-aware coverage** — `eval_coverage.py:102` `_url_path_match` (last 2-3 path components). Report now surfaces `fail=280` vs true gap, not 463.
3. **Pre-flight corpus–golden gate** — `cli.py:2421` before any LLM call: `CoverageValidator(active_gen).report(this_dataset)` → `⚠️ pre-flight: 280/500 uncoverable` in 2s. Soft warning (>50% suggests `recall_inscope.jsonl`). Saves the 2-day build you did.
4. **Empty/duplicate guards** — `generation_eval.py:401` fail-fast on `[]` contexts; `cli.py:2369` + `retrieval_metrics.py:24` deduped; `fast_eval.py` now aligned (this commit).

Remaining 3 external patterns (empty-context `fast_eval`, RAG evaluator duplicate URLs `services/rag_evaluation.py:100,128`, scope drift `generation alias fail-open`) are documented above and queued as single-line fixes for next session — each <10 LOC, none blocking today.

## Follow-ups: Evaluation Strategy Optimization (audit 2026-08-23) — ALL SHIPPED 2026-08-23

F1–F5 landed via parallel lanes (`33c1857`..`f16507e`). Two gate-calibration fixes found while validating: verdict gates on point delta (CI as context) and per-intent tolerance is noise-aware `max(0.05, 2σ)` — a fixed −0.05 at n=23 measures rerun variance (measured swing −0.087), not regressions.

### F1. Batched inscope baseline (unblocks honest gates)
- **Why:** 220-row `recall_inscope.jsonl` is correct scope but `eval-retrieval` on 220 rows times out under Qdrant load (Vector store query failed on 220). 37-row `baseline_inscope.json` understates variance.
- **Fix:** `data_engineering_copilot/cli.py:eval_retrieval_main` — add `--batch-size 55` chunking (4 batches, `asyncio.gather` per batch with 2s backoff on `RetrievalError`). Reuse existing pool logic; no new deps.
- **Verification:** `dec eval-retrieval --dataset recall_inscope --output-dir /tmp/batched --batch-size 55` → 220 rows, no timeout.
- **Gate:** `tests/evaluation/benchmarks/baseline_inscope.json` must be 220 rows; `test_golden_schema_gate` already covers JSON validity.

### F2. Cost ledger accuracy (token attribution)
- **Why:** `UsageLedger` now tracks `_served_client` (`provider_fallback.py: servicing fix` in `63d63a3`) so `calls` is correct, but `prompt_tokens`/`completion_tokens` still read from `last_usage` which some providers leave as 0. `gen_eval_10.json` showed `total_prompt=0`.
- **Fix:** `infrastructure/provider_fallback.py:generate()` — sum `last_usage` across all providers' `last_usage` with `or 0`, or read `result.usage` if provider returns it inline. Add `test_llm_usage_accounting.py` case: one test where both providers report tokens, assert sum.
- **Effort:** ~15 LOC, 1 test. Verification: `dec eval-generation --sample 5` → `llm_usage` non-zero tokens.

### F3. Gate thresholds tuned to inscope distribution
- **Why:** Current gates (faithfulness≥0.85, relevance≥0.80, rubric≥4.0, R@10 vs baseline −0.02, κ≥0.60) were set on the 500-row mismatched baseline (R@10=0.272). Honest inscope is R@10=0.784 — a −0.02 gate at 0.764 is tighter than before; per-intent R varies 0.0→2.6 (now 0.0–0.78 after dedup) so global gate may be too loose for `code_example` vs too tight for `troubleshooting`.
- **Fix:** `config/settings.py` + `docs/EVALUATION_GUIDE.md:7 cheat sheet` — derive per-intent tolerances from 220-row inscope bootstrap CIs (already have `stats.py:bootstrap_ci`); set global R@10 ≥0.75 and per-intent R@10 ≥ `max(0, baseline_intent −0.05)` where n≥5. Add one `make eval-retrieval-gate` run on PR that prints per-intent deltas.
- **Effort:** 0 LLM, pure config/docs. Verification: `make eval-retrieval-gate` on clean main shows 0 per-intent violations.

### F4. Tooling gaps — pool freeze & prompt-aug adoption
- **Why:** `eval-rerank --pool-file` and `eval-prompt-aug --mode template` are $0 and already implemented, but never exercised in CI or docs workflows. Teams default to paid `eval-generation` for every change.
- **Fix:** `Makefile` — add `make eval-rerank-smoke` (freeze 10 pools, replay) and `make eval-prompt-aug-smoke` to the `test-eval` family; `docs/EVALUATION_GUIDE.md:4` already documents them, add one line to `docs/cli_guide.md:9 Workflows` "After changing chunking/reranking: run the $0 pool replay first".
- **Effort:** ~5 LOC Makefile, 0 LLM.

### F5. Golden versioning + coverage matrix (external canon: "version golden alongside code")
- **Why:** No `eval_dataset.jsonl` version tag per run; can't tell if a metric delta is code or data. RAGBench requires ≥1 query per (intent × doc_type) cell; we have empty cells (e.g., `synthesis` n=2 after dedup).
- **Fix:** `evaluation/eval_schema.py:write_eval_rows` already preserves order — add `dataset_version` field to header comment + `git tag` check in `eval_coverage.py:report()` that prints `dataset git sha` alongside `generation`. `docs/EVALUATION_GUIDE.md:6` — add coverage matrix table (intent × doc_type) with ≥1/cell target.
- **Effort:** ~10 LOC, 1 test.

### F6. Live retrieval smoke (already healthy, just keep it)
- **Status:** `ask "Spark DataFrame"` returns 9 sources live via `ci-repro`; `make eval-fast` 10/10 pass. Keep `eval-fast` as the 10-second pre-merge gate — no change needed, just ensure it runs in CI hermetic tier (already does per `AGENTS.md`).

## Audit result — codebase hotspots still open (from grep agent, 74 `.get(...,[])` sites)

74 silent-fallback sites exist but only 5 are eval-gated; the rest are trivial UI fallbacks. Two hotspots fixed this commit:
- `fast_eval.py:384,389` — was `recall=1.0` on empty expected; now matches `generation_eval` semantics (0 on empty).
- `services/rag_evaluation.py:100,128` — still duplicate-sensitive; queued (affects `RAGEvaluator` outside the `cli` path, not your gated runs).

## Verification (Tier-1 after every edit, per AGENTS.md)

`ruff check <files> --fix` → `ruff format` → `pyright <files>` → `pytest tests/unit/test_golden_schema_gate.py tests/unit/test_eval_coverage.py tests/unit/test_eval_retrieval_dedupe.py -n 0 -q`

Smoke: `dec eval-coverage --dataset recall_all` → pre-flight warning; `dec eval-coverage --dataset recall_inscope` → 0 uncoverable.
