# ADR-010: Freeze Dark Flags Until Store Recall ≥0.35 — Docs == Code

## Status

Accepted — freeze all experimental retrieval/pipeline dark flags at `False`/`rrf`
until store `recall@10 ≥0.35` on held 110 (seed 42) with bootstrap 95% CI excludes 0.
Provenance: shipped base `RRF k=20 L100 (1,1) b=0.75 rrf` at `74236d9` (ADR-009).

## Context

- The 16-stage RAG pipeline (`.com/docs/RAG_SYSTEM_LEARNER_GUIDE.md:44`) had
  `16 stages` with several unproven retriever-adjacent knobs shipping `False`
  in code but `true` in local `.env` and ambiguous docs.
- Overengineering audit (2026-09-02 retrospective) flagged the gap:
  "strange codebase talks fancy tech but not implement it correctly" —
  e.g. `ColBERT` is a char-3gram lexical proxy (ADR-011), not neural
  late-interaction; `identifier_sparse_rrf` and `namespace_bm25` never passed
  their benchmark gates yet were documented as if active.
- Store recall on held 110 is `~0.27–0.29` (baseline `tests/evaluation/benchmarks/baseline_inscope.json: R@10 0.273 n=220`);
  pipeline ablation shows downstream stages can dilute `0.273→0.143` (full-path).
  No dark flag should ship until the store itself clears `≥0.35` — i.e. the
  retriever wins before the pipeline embellishes.
- Plan `plans/2026-09-02_rag_pipeline_simplification_plan.md` freezes dark flags
  (Task 1), then ablates `guardrails/sibling/dedup` (Task 2), then fixes
  `api_lookup→BM25_ONLY` routing (Task 3), then renames the proxy (Task 4).

## Decision

Freeze — docs == code — until held 110 bootstrap gate passes:

| Flag (settings.py) | Current | Frozen Until | Ship Gate (held 110, seed 42, local-hf 2048d) | Source |
|---|---|---|---|---|
| `identifier_sparse_rrf_enabled` | `False` | `False` | identifier recall ≥ +0.05 with all global recall/MRR thresholds satisfied, CI excludes 0 | `settings.py:1391` |
| `namespace_bm25_enabled` | `False` | `False` | identifier recall ≥ +0.05, generic recall ≤ -0.01, MRR ≤ -0.02, CI excludes 0; new generation required | `settings.py:1401` |
| `retrieval_fusion` | `"rrf"` | `"rrf"` | `dbsf` Δ nDCG vs `rrf` CI excludes 0 and >0 (ADR-008 failed: Δ -0.008 CI includes 0) | `settings.py:1405` |
| `llm_rerank_enabled` | `False` | `False` | store recall@10 ≥0.35 and rerank Δ nDCG +0.02 CI excludes 0, p95 ≤2× cross_encoder | `settings.py:1255` |
| `context_compression_enabled` | `False` | `False` | store recall@10 ≥0.35 and compression proves +Δ without needle-loss on held | `settings.py:1458` |
| `retrieval_prefetch_limit` | `100` | `100` | shipped `k=20 L100` at `74236d9` stays — no new flags until recall ≥0.35 | `settings.py:1374` |
| `hybrid_rrf_k` | `20` | `20` | shipped `k=20` (ADR-006 held ΔnDCG +0.034 CI [0.004,0.064] ship) — frozen | `settings.py:1372` |
| CRAG corrective gate | off in `retrieval_only` | off | downstream ablation must prove +Δ recall/nDCG CI>0 before re-enable | `async_rag.py:1153` |
| DBSF (`retrieval_fusion=dbsf`) | `rrf` | `rrf` | see `retrieval_fusion` row above | — |
| ColBERT proxy (`reranker_type=colbert`) | `cross_encoder` default, colbert = lexical char-3gram proxy (dark) | dark | rename to `lexical_ngram` in ADR-011 — not neural | `settings.py:1267` |

- `.env.example` adds `RETRIEVAL TUNING — FROZEN — ADR-010` block documenting each dark flag as `false`/`rrf` with `Frozen ADR-010` comment — env overrides must not flip them.
- `settings.py` annotates each dark flag with `# Frozen until store recall@10 ≥0.35 on held 110 — ADR-010` at definition and a section header `RETRIEVAL TUNING — FROZEN — ADR-010` at `mrl_multistage`/`hybrid_search_enabled`.
- `docs/RAG_SYSTEM_LEARNER_GUIDE.md` annotates `namespace / identifier_rrf / DBSF / LLM rerank / CRAG / compression` as `dark, frozen ADR-010` in gated-off profiles, reranker table, Step 8/9 headers, and config reference.
- Hermetic gate: `tests/unit/test_settings_frozen_flags.py::test_dark_flags_frozen_until_recall_035` asserts `make_settings()` defaults above stay frozen.

## Consequences

- New captures go to `/tmp/new_baseline/` only — do not overwrite `baseline_inscope.json` until pipeline ablation (Task 2) proves `+Δ CI>0`.
- Flipping any dark flag without the held 110 bootstrap gate is a docs-vs-code violation caught by `test_settings_frozen_flags`.
- After `eval-retrieval --split held` clears `recall@10 ≥0.35` with CI excludes 0, flip is gated per-row in the table above (identifier recall, DBSF nDCG, rerank, compression) — not a single global flip.
- `.env` local overrides that set `NAMESPACE_BM25_ENABLED=true` / `IDENTIFIER_SPARSE_RRF_ENABLED=true` remain for the pre-existing `ns-aware-001` generation but are not the shipped default; `make_settings()` hermetic gate ignores `.env`.

## Provenance

- Base win: `74236d9` ships `hybrid RRF k=20 L100 (1,1) b=0.75 rrf` — ADR-006/007/008/009.
- Audit: `docs/RAG_SYSTEM_LEARNER_GUIDE.md:44` 16-stage diagram.
- Plan: `plans/2026-09-02_rag_pipeline_simplification_plan.md:Task 1`.

## Alternatives Considered

- Ship dark flags now with `True` defaults: rejected — A/B with correct NVIDIA embeddings shows `identifier_sparse_rrf` hurts (`Δ -0.066`), `namespace_bm25` needs new generation and never passed generic recall gate, `DBSF` underperforms (`Δ -0.008`), `llm_rerank` and `compression` dilute without store headroom.
- Keep docs vague: rejected — audit requires docs == code, each `settings.py` field carries frozen provenance.
