# Langfuse Integration

This document covers how the Data Engineering Copilot uses Langfuse v4 for
observability, evaluation, and quality analytics — the architecture, trace
taxonomy, prompt/experiment workflows, evaluators, dashboards, monitors, S3
export, and the migration/rollback runbook.

See also `docs/langfuse_evaluators.md` (evaluators + score configs) and
`docs/langfuse-v4-sdk-surface.md` (SDK v4 API factsheet).

## Architecture

- **SDK**: `langfuse` Python v4.14.3. Everything goes through the compat
  wrapper in `data_engineering_copilot/observability/langfuse_client.py`
  (`LangfuseCompat`, `_ObservationCompat`, `get_langfuse_instance`) and the
  tracer factory in `observability/telemetry.py`
  (`build_telemetry_tracer` → `LangfuseTelemetryTracer` / `NoOpTelemetryTracer`).
- **Server**: `langfuse/langfuse:4` + `langfuse/langfuse-worker:4` with
  ClickHouse 26.4. Currently in **dual-write** migration mode
  (`LANGFUSE_MIGRATION_V4_WRITE_MODE=dual`) — v4 data is written alongside the
  legacy pipeline. Cutover to `events_only` is deferred (Task 1.5).
- **Scoring**: `telemetry.score(trace_id, name, value, data_type, ...)`
  supports NUMERIC / BOOLEAN / CATEGORICAL values. Categorical scores must be
  config-bound (pass `config_id` + the string label) to render labels in the
  UI; otherwise the value is coerced to a number.
- **Trace vs span ids (v4)**: `_ObservationCompat.id` is the 16-hex span id;
  `.trace_id` is the 32-hex trace id. **Always score against `.trace_id`.**

## Trace taxonomy

| Trace name | When | Notes |
|---|---|---|
| `rag-query-pipeline` | Full RAG answer (non-streaming) | Scored: `confidence`, `groundedness`, `relevance`, `completeness`, `cache_hit`(false), `intent`, `cost_usd` |
| `rag-query-pipeline-stream` | Streaming RAG answer | Same scores |
| `rag-query-pipeline-cache-hit` | Answer served from cache | `cache_hit=true` (BOOLEAN); distinct name so eval filters can exclude it |
| `experiment-item-run` | `run_rag_experiment` per dataset item | Phase 6 |
| `sess-integration` | Integration-test traces | Tests only |

- **Tags**: `app:data-engineering-copilot`.
- **Observation types**: root span (trace-level), `retrieval`, `query-rewriting`,
  `llm-generation`, `llm-json-retry`, `groundedness-verification`, `reranking`.
  The `retrieval` observation's output is the list of retrieved chunk texts —
  used by the faithfulness judge.
- **Environment / release**: trace metadata carries `app_env`, `git_sha`,
  `token_usage`, `cost_usd`, `intent`, `num_sources`, `stage_times_ms`.
- **Users / sessions**: `user_id`/`session_id` propagate from the API
  (`X-User-ID`/`X-Session-ID` headers or `user_id`/`session_id` query params),
  `dec ask --user-id/--session-id`, and Streamlit.

## Score types

Seeded by `dec langfuse-seed-score-configs` (idempotent, reconciles drift):

| Name | Type | Range / values |
|---|---|---|
| `confidence`, `groundedness`, `relevance`, `faithfulness`, `user_feedback`, `completeness` | NUMERIC | 0–1 |
| `cache_hit`, `out_of_scope` | BOOLEAN | true/false |
| `intent` | CATEGORICAL | factual, code_example, api_lookup, comparative, debugging, how_to |
| `ragas_context_recall`, `ragas_context_precision`, `ragas_faithfulness`, `ragas_answer_relevancy` | NUMERIC | 0–1 |

`user_feedback` is written by `POST /api/v1/feedback` (thumbs up/down) and by
the Streamlit feedback buttons.

## Prompt management workflow

1. Every prompt has a hardcoded fallback registered under the same name used to
   seed Langfuse (`register_fallback` in `observability/langfuse_prompts.py`).
   Runtime code compiles via `get_langfuse_prompt(name).compile(**kwargs)` —
   byte-identical in degraded mode.
2. Seed/update prompts: `dec langfuse-seed-prompts` (creates a new version
   under `production` label; runtime resolves the newest).
3. Version, label, and edit in the Langfuse UI; re-seed to bump a version.

Managed prompts: `rag-answer`, `query-intent-classify`, `query-rewrite`,
`query-expand`, `query-hyde`, `groundedness-nli`, `chunk-enrichment-summary`,
`eval-faithfulness`, `judge-faithfulness`, `judge-relevance`,
`judge-out-of-scope`, `rag-json-retry-suffix`.

## Experiment workflow

- `dec evaluate` runs the RAG eval on a golden dataset and uploads rows to a
  Langfuse dataset `dec-evaluate-{source}-{timestamp}` (`--dataset-name`
  overrides).
- `--experiment-name <name>` runs a RAG experiment via `dataset.run_experiment`
  (synchronous; the RAG task answers each item, term-overlap `faithfulness`
  evaluator runs, and offline RAGAS scores are pushed onto item traces).
- `--experiment-name <name> --dataset-name <existing>` runs directly against an
  existing dataset.
- Low-confidence answers are queued into the `low-confidence-review` dataset
  (`source_trace_id` links back to the production trace) via the injectable
  `review_dataset_hook` (Phase 6, Task 6.3).

## Evaluators (LLM-as-a-judge)

Three judges run over production traces via `dec langfuse-evaluate`
(`run_batched_evaluation`, `scope="traces"`):

| Evaluator | Score | Type |
|---|---|---|
| Faithfulness | `faithfulness` | NUMERIC 0–1 |
| Relevance | `relevance` | NUMERIC 0–1 |
| Out-of-scope | `out_of_scope` | BOOLEAN |

- Filter (default): `[{"type": "string", "column": "name", "operator": "=",
  "value": "rag-query-pipeline"}]` — the public-API filter array format
  (NOT a bare object).
- Judges run through the purpose-`evaluation` LLM fallback chain (no pinned
  provider; local Ollama last).
- **Cost gate**: without `--max-items`, the run is skipped with probability
  `1 - LANGFUSE_SAMPLE_RATE`. `--max-items N` always runs (operator intent).
- Evaluators are configured in the **UI** (the 4.6 server exposes no public
  evaluators API — `POST /api/public/evaluators` → 404). Use the seeded
  `judge-*` prompts. See `docs/langfuse_evaluators.md`.

## Metrics API (Task 8.2)

`dec langfuse-metrics <preset> [--days N] [--json]` queries the v2 Metrics API:

| Preset | What |
|---|---|
| `cost-by-model` | Total cost by `providedModelName` |
| `daily-volume-latency` | Daily `count_count` + `p95_latency` |
| `score-summary [--score-name X]` | Average numeric score by name |

Implemented in `data_engineering_copilot/evaluation/langfuse_metrics.py`.
The v2 endpoint requires `fromTimestamp`/`toTimestamp`; grouping by high
cardinality fields (`id`, `traceId`, `userId`, `sessionId`) is not allowed.

## Dashboards (Task 8.1)

Create dashboards in the Langfuse UI (Analytics → Dashboards), using curated
widgets as a starting point and filtering by environment/tags:

- **Production Monitoring**: latency p50/p95/p99, throughput, error rate, model
  usage (group by model), cache hit rate (`cache_hit` boolean).
- **Cost**: `cost_usd` over time, cost per user, tokens per query.
- **Quality**: confidence/groundedness/relevance distributions,
  `user_feedback` trend, `ragas_*` trend.
- **Usage**: traces by user, by session, by `intent` (categorical).

## Annotation queues (Task 8.3)

Create an annotation queue in the Langfuse UI: **`low-quality-review`**
filtering traces where `groundedness < 0.5 OR user_feedback = 0`. Reviewers
annotate in the queue; annotations attach as scores and feed quality analytics.

## Monitors & alerts (Task 8.4)

Monitors are a v4 feature available after the Phase 1 cutover. Define alerts in
the Langfuse UI:
- `groundedness` average < 0.6 over 1h
- `user_feedback` = 0 spike
- error rate > 5%

Wire a webhook to Slack/email. Validate by injecting bad data and confirming the
monitor fires.

## S3 event upload & export (Task 8.5)

- Langfuse events are uploaded to MinIO (`local-dev` bucket) — verified live
  (observation, dataset_run_item, etc. objects appear). Configured via
  `LANGFUSE_S3_EVENT_UPLOAD_*` in `docker-compose.yml`.
- **Before the Phase 1.5 cutover**, switch the export source in the Langfuse
  UI from **"Traces and observations (legacy)"** → **"Enriched observations"**.
  After cutover the legacy export stops producing data, so the switch must
  happen first.

## Migration runbook (Phase 1) & rollback

- Phase 1 upgraded ClickHouse to 26.4 and Langfuse server/worker to `:4`, and
  put the stack in **dual-write** mode (`LANGFUSE_MIGRATION_V4_WRITE_MODE=dual`).
- Monitor the UI background-migrations page until the v4 backfill completes.
- **Cutover (Task 1.5, deferred)**: flip `LANGFUSE_MIGRATION_V4_WRITE_MODE` to
  `events_only` and restart. Before doing so, complete the export source switch
  (Task 8.5).
- **Rollback**: flip back to `dual` (or `none`) and restart the stack. No data
  is lost while in dual mode. If the SDK is incompatible, pin back the `langfuse`
  package in `pyproject.toml` and rebuild the image (`make rebuild`).
