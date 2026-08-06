# General RAG Improvement Execution Plan

This plan is intended for automated FLASH executors. It is deliberately
corpus- and query-general: no phase may add a rule that recognizes one golden
question, one API name, or one document URL.

## Executor Contract

```yaml
name: rag-general-improvement
version: 1
workspace: /home/rahul/workspace/DataEngineeringCopilot
python: dec_venv/bin/python
network_policy: no paid provider probes; use configured evaluation calls only
infra:
  qdrant: localhost:6333
  redis: redis://:local_secure_password_123@localhost:6379/0
state_dir: .rag_eval
required_checks:
  - ruff
  - pyright
  - unit_tests
  - retrieval_evaluation
```

Executors must run from the repository root. They must not commit, reset the
worktree, delete an active index, or run `dec probe-llm`. Every phase writes a
JSON result under `.rag_eval/runs/<run_id>/`; a failed gate stops downstream
phases.

## Executor Driver (`dec rag-plan`)

The plan ships an executor driver, `data_engineering_copilot/plan_executor.py`,
wired to the CLI as `dec rag-plan`. It makes the plan self-contained for FLASH
executors:

1. **Active-generation discovery** is automatic (same resolution the RAG service
   uses: `.index_state/active.json`, falling back to settings).
2. **Machine-readable output**: `--json` prints the run summary; every run writes
   `result.json`, per-step logs under `logs/`, and artifacts under `artifacts/`.
3. **Artifacts**: each run creates `.rag_eval/runs/<run_id>/` with
   `artifacts/`, `logs/`, and `trials/` subdirectories.
4. **Dry-run**: `--dry-run` prints the exact commands each phase would run and
   executes nothing (no checkpoint is persisted).
5. **Checkpoint/resume**: completed phases are recorded in `checkpoint.json`;
   re-running the same `--run-id` skips completed phases unless `--force` is
   given.
6. **Failure schema**: any failed/blocked/halted phase writes `failure.json`
   with `schema_version`, `run_id`, phase, step, command, exit code, output
   excerpt, artifacts dir, and a resume command.

```bash
dec rag-plan --dry-run                          # preview every planned command
dec rag-plan --phase 0                          # reproducibility gate
dec rag-plan --phase 1 --run-id baseline-01     # baseline evaluation (resume-safe)
dec rag-plan --phase 3 --candidate-generation <gen> --force   # build + validate
dec rag-plan --json                             # full run, JSON summary
```

Exit codes: `0` success; `2` usage; `4` gate failure; `5` command failure;
`10` halted (phases 4-6 require a code change); `11` blocked (phase 3/7 needs
`--force`). Phases 3 and 7 require `--candidate-generation`.

Commands are invoked through the `dec_venv/bin/dec` console script (the
`python -m dec` form used below is shown for human convenience only).

## Reference Architecture

The target pipeline follows the common production pattern documented by
Anthropic Contextual Retrieval, Qdrant Hybrid Queries, and OpenAI Retrieval:

```text
document parsing
  -> contextual parent/child chunks
  -> dense + BM25 indexes
query
  -> intent/rewrite/subquery generation
  -> broad per-query retrieval
  -> generic RRF fusion and deduplication
  -> cross-encoder reranking
  -> coverage-aware context assembly
  -> grounded generation and citation verification
```

The first stage optimizes recall. The reranker performs the relevance decision.
Generation must never be used as a substitute for retrieval evaluation.

## Phase 0: Reproducibility Gate

### Actions

```bash
git status --short
dec_venv/bin/dec status
dec_venv/bin/dec spark-config-check
dec_venv/bin/dec spark-validate --generation <active-generation>
```

### Artifacts

- `.rag_eval/runs/<run_id>/preflight.json`
- Active generation identifier and validation report
- Git status snapshot

### Gate

- Qdrant reachable and active generation validated.
- No active generation mutation is performed by evaluation.
- If a command fails, stop and report the exact command and output.

## Phase 1: Baseline Retrieval Evaluation

### Actions

```bash
dec_venv/bin/dec evaluate --spark --output-dir .rag_eval/runs/<run_id>/artifacts/eval
dec_venv/bin/dec evaluate --dataset tests/evaluation/eval_dataset.jsonl
```

Run both the Spark dataset and the general golden dataset. Clear only the
generation-scoped query cache before a baseline run; record whether the run was
cold or warm. The `--output-dir` form (used by `dec rag-plan` Phase 1) writes
`retrieval_provenance.json` and `retrieval_metrics.json`, which the later
diagnostics and phases 4-6 consume.

### Required Metrics

- Expected-term recall in assembled context
- Expected-source recall
- Recall@5, Recall@10, Recall@20 where golden chunk IDs are available
- MRR and nDCG where graded relevance is available
- `INSUFFICIENT_CONTEXT` rate
- Median and p95 retrieval latency
- Context character/token budget usage

### Gate

Baseline is accepted only if all metrics are written, even when thresholds
fail. No ranking change is accepted without a before/after comparison.

## Phase 2: Chunk Quality Audit

### Actions

1. Materialize a manifest for the active generation.
2. Calculate chunk distributions by source, module, doc type, word count, and
   duplicate content hash.
3. Detect short code stubs, duplicate overloads, missing headings, and chunks
   whose text is only a forwarding assignment.
4. Sample at least 20 chunks from each problem category and store them for
   review.

### Artifacts

- `.rag_eval/runs/<run_id>/chunk_quality.json`
- `.rag_eval/runs/<run_id>/chunk_samples.jsonl`

### Gate

Do not change chunking until the audit identifies a corpus-wide defect and the
sample demonstrates it. Any chunking change requires a new index generation;
the active alias remains unchanged until validation passes.

## Phase 3: Contextual Indexing

### Actions

1. Add deterministic context to each child chunk: source, module, file path,
   heading path, symbol/API name, and a short parent-section summary.
2. Keep original text separate from contextual retrieval text.
3. Embed contextual text and fit BM25 on the same contextual representation.
4. Store context version and parser/chunker versions in metadata.
5. Build a candidate generation without activating it.

### Gate

- Content hashes remain stable for unchanged source content.
- Dense and BM25 indexes use the same generation and context version.
- Validation reports point counts, dimensions, sparse readiness, and metadata
  completeness.

## Phase 4: Generic Multi-Query Fusion

### Actions

1. Keep the literal user query.
2. Add concise rewrite and independently generated subqueries.
3. Retrieve a broad pool per query variant.
4. Fuse by generic RRF, deduplicate by chunk ID/content hash, and preserve
   query provenance for diagnostics.
5. Do not add document-text boosts, function-name boosts, URL boosts, or
   golden-query exceptions.

### Gate

Every candidate must be explainable by at least one retrieval variant. The
diagnostic must show per-variant rank and fused rank for failed golden rows.

## Phase 5: Reranking and Context Selection

### Actions

1. Rerank a broad fused candidate set, normally 5-10 times the final context
   size.
2. Compare original-query and concise-rewrite reranking on the evaluation set.
3. Select the final context directly from reranker scores.
4. Do not enable lexical MMR by default unless held-out evaluation proves a
   benefit.
5. Add generic parent/neighbor recovery only when a selected child requires
   surrounding context.

### Gate

- Recall must not regress at the candidate-pool stage.
- Final context recall and answer completeness must improve or remain within
  the configured tolerance.
- Latency and context size must remain within operational limits.

## Phase 6: Evaluation-Driven Tuning

### Actions

Tune only one variable at a time using a train/validation split:

- per-query retrieval `top_k`
- fused candidate pool size
- final reranker `top_k`
- context budget
- dense/sparse fusion weights, if supported
- contextual chunk version

Never tune on the same rows used for the final report. Store every trial in:

```text
.rag_eval/runs/<run_id>/trials/<trial_id>.json
```

### Gate

Promote a setting only when it improves the primary metric without violating
latency, cost, or groundedness limits on the held-out split.

## Phase 7: Generation Rollout

### Actions

```bash
dec_venv/bin/dec spark-build --generation <new-generation>
dec_venv/bin/dec spark-validate --generation <new-generation>
FORCE=1 dec_venv/bin/dec spark-activate --generation <new-generation>
```

Run the full evaluation after activation, then clear only caches scoped to the
new generation if required. Keep the prior generation for rollback.

### Rollback

```bash
FORCE=1 dec_venv/bin/dec spark-rollback --generation <prior-generation>
```

Rollback is mandatory if validation fails, source recall drops below the gate,
or production error/latency thresholds are exceeded.

## Current Implementation Status

Completed in the current worktree:

- Broad production retrieval defaults increased to `30`.
- Final reranked context increased to `20`.
- Candidate rerank pool widened to five times the configured retrieval/final
  size.
- Spark-specific post-rerank function injection removed.
- Default MMR selection removed from the RAG path.
- Reranking uses the concise retrieval rewrite while generation retains the
  original question.
- Full unit suite passes.
- FLASH executor driver `dec rag-plan` ships the plan: active-generation
  discovery, machine-readable `--json` output, `.rag_eval/runs/<run_id>/`
  artifacts, `--dry-run`, checkpoint/resume, and a pinned failure JSON schema.
  Phases 0-3 and 7 orchestrate the existing CLI; phases 4-6 halt with exit code
  `10` until their code change lands.
- Structured candidate provenance and retrieval-stage metrics are instrumented:
  `AsyncRagService.answer(..., provenance=[])` captures per-variant retrieval,
  fused ranks, rerank pool/order, final context, and stage timings;
  `dec evaluate --spark --output-dir <dir>` writes `retrieval_provenance.json`
  and `retrieval_metrics.json` (candidate-vs-final source recall, expected
  sources dropped by rerank/truncation, insufficient-context rate, median/p95
  retrieval latency). Phase 1 writes these into the run artifacts.

Next executor-safe implementation target:

- Use the baseline provenance/recall gap (candidate vs final context) to decide
  whether any corpus-wide chunking or context change is warranted (Phases 2-3),
  then implement the code changes that unblock phases 4-6.

## Stop Conditions

Stop and request human review when:

- The active alias or generation would be deleted or overwritten.
- A provider probe or paid API call is required.
- Evaluation data needs to be changed to make a regression pass.
- A query-specific lexical/ranking exception is proposed.
- A new generation fails point-count, dimension, BM25, or metadata validation.
