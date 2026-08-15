---
name: evaluation
description: Use for ANY task involving DataEngineeringCopilot evaluation — dec evaluate (golden dataset, --spark retrieval recall), ragas, LLM-as-a-judge, langfuse datasets/experiments/metrics/review queue, faithfulness/relevance scoring, or drift detection. Triggers: evaluate, eval, golden dataset, eval_dataset.jsonl, ragas, retrieval recall, faithfulness, relevance, langfuse-evaluate, langfuse-metrics, drift, experiment, LLM-as-judge.
---

# DataEngineeringCopilot Evaluation

Several distinct eval surfaces exist. Choose by what you are measuring and
which infra is available.

## `dec evaluate` (golden-dataset QA; mocked embedder, no infra needed)

Default dataset: `tests/evaluation/eval_dataset.jsonl`; per-source:
`tests/evaluation/eval_dataset_{source}.jsonl`
(spark / airflow / delta_lake / databricks).

Flags: `--dataset <path>`, `--source <name>`, `--verbose`,
`--experiment-name <name>` (upload rows to a Langfuse dataset + run a RAG
experiment), `--dataset-name`, `--spark`, `--output-dir <dir>` (write
machine-readable provenance + metrics JSON).

Metrics come from `services/rag_evaluation.py`:
- `RetrievalEvaluator` (term overlap / recall on retrieved chunks)
- `AnswerEvaluator.score_relevance` (embedding cosine to context)
- `FaithfulnessEvaluator.evaluate` (LLM-as-judge on answer↔context)
- `RAGEvaluator` (aggregate run + report).

`--spark` mode = **retrieval-recall** evaluation with expected terms/sources
per query + provenance JSON — good for validating a generation's retrieval
quality without a full LLM answer pass.

## Ragas (optional, ragas==0.3.9)

`services/ragas_evaluation.py` `RagasEvaluator` + `services/ragas_adapters.py`
adapters wire the project's own fallback chains into ragas (never raw ragas
clients). `_build_runtime` uses the `evaluation` LLM chain and the evaluation
embedding chain. Requires infra for real runs; unit tests mock the adapters.

## Langfuse-driven

- `dec langfuse-seed-prompts` — idempotently seed managed prompts.
- `dec langfuse-evaluate` — LLM-as-a-judge (faithfulness/relevance/
  out-of-scope) over production `rag-query-pipeline` traces
  (`evaluation/langfuse_evaluators.py`, `--filter`, review-queue support).
- `dec langfuse-metrics` — Metrics API v2 queries (cost/tokens/latency/score).
- `dec langfuse-seed-score-configs` — score-config seeding.
- Datasets/experiments: `dec evaluate --experiment-name` / `--dataset-name`
  drive `evaluation/` dataset upload + `run_rag_experiment`.

## Drift detection

`services/drift_detector.py` tracks eval metrics over time
(`settings.drift_eval_history_path`, `drift_window_days`); flagged regressions
surface in `dec langfuse-metrics` / review tooling. Run drift checks when an
eval metric moves across releases.

## Operating notes

- Basic `dec evaluate` needs **no infra** (mocked embedder) — the `evaluation`
  marker tests are hermetic. `--spark` / experiment modes need Qdrant +
  embedder + LLM.
- The `evaluation` LLM chain is its own per-purpose chain
  (`build_llm_fallback_chain(purpose="evaluation")`) — do not reuse `answer`.
- Golden datasets are the source of truth for regression gates; when behavior
  intentionally changes, update the dataset + expect metric deltas, don't
  silently accept drift.
- Eval runs emit structured metrics; use `--output-dir` for reproducible
  provenance that can be diffed across generations/commits.
