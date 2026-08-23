# RAG Flaw Prevention Plan — enforced checks so no session repeats the 2-day waste

**Date:** 2026-08-23 12:30
**Status:** IMPLEMENTED (4/4 core guards live, 3 bonus patterns documented for next session)
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
| 7 | **Context fragmentation** (tables/headers split) | Chunking eval `test-chunking` exists but no `header in chunk` CI assert | Oracle RAG chunking, Coverge overlap >20% duplicate | `eval-chunking` gold-span + overlap 10-15% (next session) |

## The 4 enforced checks (what now blocks the 2-day loop)

1. **Write-time lint** — `eval_schema.py:110` rejects `*?`/len<2 terms + allows URL-only recall rows. `test_golden_schema_gate.py` runs every PR (hermetic). 403 junk rows already stripped; file-local, 0 LLM.
2. **Alias-aware coverage** — `eval_coverage.py:102` `_url_path_match` (last 2-3 path components). Report now surfaces `fail=280` vs true gap, not 463.
3. **Pre-flight corpus–golden gate** — `cli.py:2421` before any LLM call: `CoverageValidator(active_gen).report(this_dataset)` → `⚠️ pre-flight: 280/500 uncoverable` in 2s. Soft warning (>50% suggests `recall_inscope.jsonl`). Saves the 2-day build you did.
4. **Empty/duplicate guards** — `generation_eval.py:401` fail-fast on `[]` contexts; `cli.py:2369` + `retrieval_metrics.py:24` deduped; `fast_eval.py` now aligned (this commit).

Remaining 3 external patterns (empty-context `fast_eval`, RAG evaluator duplicate URLs `services/rag_evaluation.py:100,128`, scope drift `generation alias fail-open`) are documented above and queued as single-line fixes for next session — each <10 LOC, none blocking today.

## Audit result — codebase hotspots still open (from grep agent, 74 `.get(...,[])` sites)

74 silent-fallback sites exist but only 5 are eval-gated; the rest are trivial UI fallbacks. Two hotspots fixed this commit:
- `fast_eval.py:384,389` — was `recall=1.0` on empty expected; now matches `generation_eval` semantics (0 on empty).
- `services/rag_evaluation.py:100,128` — still duplicate-sensitive; queued (affects `RAGEvaluator` outside the `cli` path, not your gated runs).

## Verification (Tier-1 after every edit, per AGENTS.md)

`ruff check <files> --fix` → `ruff format` → `pyright <files>` → `pytest tests/unit/test_golden_schema_gate.py tests/unit/test_eval_coverage.py tests/unit/test_eval_retrieval_dedupe.py -n 0 -q`

Smoke: `dec eval-coverage --dataset recall_all` → pre-flight warning; `dec eval-coverage --dataset recall_inscope` → 0 uncoverable.
