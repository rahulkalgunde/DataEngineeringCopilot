# Langfuse Evaluators & Score Configs

Phase 7 of the Langfuse utilization plan. Covers LLM-as-a-judge evaluators,
score configs, and the richer score types emitted by the RAG pipeline.

## Evaluators

Three LLM-as-a-judge evaluators are defined. Their judge prompts are
Langfuse-managed (`judge-faithfulness`, `judge-relevance`,
`judge-out-of-scope`) and are seeded by `dec langfuse-seed-prompts`; hardcoded
fallbacks keep behavior byte-identical when Langfuse is unavailable.

| Evaluator | Score name | Type | Range | Prompt |
|---|---|---|---|---|
| Faithfulness judge | `faithfulness` | NUMERIC | 0–1 | "Is {{output}} supported by {{context}}? Score 0-1" |
| Relevance judge | `relevance` | NUMERIC | 0–1 | "Does {{output}} answer {{input}}? Score 0-1" |
| Out-of-scope detector | `out_of_scope` | BOOLEAN | true/false | "Is {{input}} answerable from the docs? TRUE/FALSE" |

The judges run through the purpose-`evaluation` LLM fallback chain (no pinned
provider — each call uses the first currently-available provider, local Ollama
last). Retrieved context for the faithfulness judge is pulled from the
`retrieval` observation of each trace.

### Creating evaluators in the UI

The Langfuse server (4.6.0) exposes no public `evaluators` API
(`POST /api/public/evaluators` returns 404), so evaluators are created in the
**UI** (recommended by the plan):

1. Open `http://localhost:3000/evaluators`.
2. Create a new LLM evaluator per row in the table above, selecting the
   seeded `judge-*` prompt and the matching score type.
3. Save. The evaluator is now available to attach to traces/experiments.

### Running production trace evaluation

```bash
dec langfuse-evaluate                 # all rag-query-pipeline traces (sampled)
dec langfuse-evaluate --max-items 10  # explicit run, sampling bypassed
dec langfuse-evaluate --filter '[{"type": "string", "column": "name", "operator": "=", "value": "rag-query-pipeline"}]' --verbose
```

- Uses the v4 `run_batched_evaluation` API with `scope="traces"`.
- Cost is gated by `LANGFUSE_SAMPLE_RATE`: without `--max-items`, the run is
  skipped entirely with probability `1 - sample_rate`. Passing `--max-items`
  always runs (explicit operator intent).
- Judged scores are written back onto each trace and show up in the trace
  details and score analytics.

## Score Configs

Create the following score configs in the Langfuse UI (Settings → Score
Configs) so scores display with proper types/ranges in analytics:

| Name | Type | Range / values |
|---|---|---|
| `confidence` | NUMERIC | 0–1 |
| `groundedness` | NUMERIC | 0–1 |
| `relevance` | NUMERIC | 0–1 |
| `faithfulness` | NUMERIC | 0–1 |
| `user_feedback` | NUMERIC | 0 or 1 |
| `cache_hit` | BOOLEAN | true/false |
| `out_of_scope` | BOOLEAN | true/false |
| `intent` | CATEGORICAL | `factual`, `code_example`, `api_lookup` |
| `ragas_context_recall` | NUMERIC | 0–1 |
| `ragas_context_precision` | NUMERIC | 0–1 |
| `ragas_faithfulness` | NUMERIC | 0–1 |
| `ragas_answer_relevancy` | NUMERIC | 0–1 |

The `intent` categorical score is emitted as `intent_label` by the RAG
pipeline so it remains unambiguous next to the numeric `confidence` score.

## Score types emitted by the RAG pipeline

See `services/async_rag.py` scoring block. Per answered query:

- `confidence` (NUMERIC 0–1)
- `groundedness` (NUMERIC 0–1)
- `relevance` (NUMERIC 0–1)
- `intent_label` (CATEGORICAL: `factual` | `code_example` | `api_lookup`)
- `cache_hit` (BOOLEAN) — emitted on the cache-hit trace as `true`

Cache-hit answers return early before the full pipeline trace is created, so a
lightweight `rag-query-pipeline-cache-hit` trace records the hit and scores
`cache_hit = true`; the regular trace scores `cache_hit = false`.
