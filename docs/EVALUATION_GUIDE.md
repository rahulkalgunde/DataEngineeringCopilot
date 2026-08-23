# Evaluation Guide — Every Eval Surface Explained

A complete reference for the DataEngineeringCopilot evaluation system, written for
engineers new to RAG pipelines. It covers **what** each evaluation measures,
**why** it matters, **what it costs** (LLM calls vs. free/local), **which
question datasets** feed it, **how to run it**, and **what values count as
good**.

Companion reading: `docs/cli_guide.md` (per-command usage), `docs/makefile_guide.md`
(make targets), `docs/RAG_SYSTEM_LEARNER_GUIDE.md` (pipeline internals).

---

## Table of contents

1. [RAG evaluation in 5 minutes](#1-rag-evaluation-in-5-minutes)
2. [The cost map — free vs. paid](#2-the-cost-map--free-vs-paid)
3. [The layered evaluation contract](#3-the-layered-evaluation-contract)
4. [Command-by-command reference](#4-command-by-command-reference)
   - [dec eval-fast](#dec-eval-fast) · [dec eval-coverage](#dec-eval-coverage) ·
     [dec eval-retrieval](#dec-eval-retrieval) · [dec evaluate](#dec-evaluate) ·
     [dec evaluate --spark](#dec-evaluate---spark) · [dec eval-generation](#dec-eval-generation) ·
     [dec eval-rerank](#dec-eval-rerank) · [dec eval-assembly](#dec-eval-assembly) ·
     [dec eval-prompt-aug](#dec-eval-prompt-aug) · [dec eval-chunking](#dec-eval-chunking) ·
     [dec gen-synthetic-eval](#dec-gen-synthetic-eval) · [Ragas](#ragas-integration) ·
     [Langfuse judges & metrics](#langfuse-judges--metrics) ·
     [Drift detection & provenance](#drift-detection--provenance)
5. [Metrics glossary for RAG newcomers](#5-metrics-glossary-for-rag-newcomers)
6. [Question datasets reference](#6-question-datasets-reference)
7. [Recommended values cheat sheet](#7-recommended-values-cheat-sheet)
8. [How to run — make targets, CI vs. local, exit codes](#8-how-to-run--make-targets-ci-vs-local-exit-codes)
9. [Workflows — what to run when](#9-workflows--what-to-run-when)
10. [Best practices & gotchas](#10-best-practices--gotchas)

---

## 1. RAG evaluation in 5 minutes

A RAG pipeline is a chain of stages; a bad answer can be caused by any one of them:

```
question → query rewrite/HyDE → retrieval (dense+sparse fusion)
        → rerank → context assembly → LLM generation → answer
```

If you only measure "is the final answer good?", you can't tell *which stage*
broke when quality drops. So this repo evaluates **stage by stage**, and each
harness freezes every stage except the one under test ("frozen inputs"): a metric
delta then attributes to that stage alone.

| Stage | Question it answers | Harness |
|---|---|---|
| Corpus / index | Is the indexed content intact and deduplicated? | `eval-fast` layers 1–2 |
| Embeddings | Are vectors sane (right dims, no NaNs, semantics preserved)? | `eval-fast` layer 3 |
| Vector DB | Does Qdrant return what was stored? | `eval-fast` layer 4 |
| Retrieval | Do we fetch the *right documents* at all? | `eval-fast` layer 5, `eval-retrieval`, `evaluate --spark` |
| Rerank | Do we push the right documents to the top? | `eval-rerank` |
| Context assembly | Is the assembled prompt clean (no dupes, nothing dropped)? | `eval-assembly` |
| Chunking | Were documents split at sensible boundaries? | `eval-chunking`, `make test-chunking` |
| Prompt augmentation | Do prompts enforce format/citations/injection defense? | `eval-prompt-aug` |
| Generation | Is the answer faithful to the evidence and on-topic? | `eval-generation`, `evaluate` (QA mode), Ragas |
| Production | What is actually happening to real users? | Langfuse judges + metrics, drift detection |

Two families of measurement are used deliberately:

- **Deterministic lexical/IR metrics** (Recall@K, MRR, nDCG, token-F1, IoU,
  duplicate rate…) — computed with plain code against labeled expectations.
  Zero-cost, CI-stable, reproducible.
- **LLM-as-a-judge metrics** (faithfulness, answer relevance, rubric scores) —
  an LLM reads `(question, context, answer)` and scores it. Nuanced but
  stochastic and paid; always pinned to temperature 0.0 and a dedicated judge
  model (`evaluation` purpose chain), never the production answer chain.

---

## 2. The cost map — free vs. paid

This is the single most important operational distinction.

### $0 — no LLM calls (safe to run anytime)

| Command | Infra needed | What it still needs |
|---|---|---|
| `make test-eval`, `make test-eval-data` | none | mocked embedder, fully hermetic — runs in CI |
| `dec eval-chunking` | none | gold span files on disk only |
| `dec eval-prompt-aug --mode template` (default) | none | dataset file only |
| `dec gen-synthetic-eval` | none | generation corpus on disk (`chunks.jsonl`) |
| `dec eval-coverage` | none | corpus chunks.jsonl on disk |
| `dec eval-fast` | Qdrant up | active generation + local embedder (in-process HF sentence-transformers; embeddings never go through Ollama or paid APIs) |
| `dec eval-assembly` | Qdrant + Redis | embedder |
| `dec eval-rerank` | Qdrant + Redis | embedder (+ reranker model; optional — passthrough if unconfigured) |

### 💰 — makes paid LLM API calls (treat as approved, expensive actions)

| Command | LLM usage |
|---|---|
| `dec eval-generation` | per row: 1 generator call (purpose-`answer` chain) + `--n-trials` × judge calls (purpose-`evaluation` chain). Default `--n-trials 3`. Use `--sample N` to evaluate a deterministic stratified subset (dev loop). `--compare baseline.jsonl` runs position-swapped pairwise A/B. |
| `dec evaluate` (QA mode) | full RAG answers per query; **RAGAS is opt-in** — pass `--ragas` to enable (~18–20 extra calls/query). Without `--ragas`, no RAGAS calls. |
| `dec evaluate --experiment-name` | same as QA mode + Langfuse experiment replay |
| `dec eval-retrieval` | mostly free (`retrieval_only=True` short-circuits answer generation/groundedness/scope), **but** query rewrite/HyDE may still call the LLM unless disabled |
| `dec evaluate --spark` | free for pure recall rows (retrieval-only short-circuit); only rows needing answer text (out-of-scope refusals, forbidden-term checks) generate |
| `dec eval-prompt-aug --mode llm` | live generation per sample through the provider pinned by `--provider` |
| `dec langfuse-evaluate` | 3 judge calls per sampled production trace (faithfulness, relevance, out-of-scope) |
| `dec eval-judge-calibrate` | judges every calibration row (~1× per row) to compute Cohen's κ vs human labels. Run only after labeling. |
| `dec eval-proxy-validate` | LLM-judges a deterministic sample of chunks (~sample×k judge calls) to validate proxy-recall vs ground truth. |

Rules that follow from this map:

- Layers 1–5 (see next section) must stay LLM-free. Never add LLM calls to them.
- Run `dec eval-fast` after code changes to prove the index is intact **before**
  paying for a full `dec evaluate`.
- The judge LLM is injected into `AsyncRagService` but is **never called in the
  live answer path** — judging happens offline only. Changing
  `EVALUATION_LLM_PROVIDER/MODEL` never affects production answers.

---

## 3. The layered evaluation contract

Evaluation is layered; each layer has a gate and must stay in its lane:

| Layer | Gate | Costs LLM? |
|---|---|---|
| 1. Corpus integrity (counts, dupes, coverage) | `dec eval-fast` | no |
| 2. Chunk quality (size/boundary heuristics) | `dec eval-fast` | no |
| 3. Embedding sanity (dims/NaN/pairs) | `dec eval-fast` | no |
| 4. Vector DB integrity (count/metadata/self-retrieval) | `dec eval-fast` | no |
| 5. Retrieval quality | `dec eval-fast` + `dec evaluate --spark` / `eval-retrieval` | no* |
| 6. Generation quality | `dec evaluate` (QA), `dec eval-generation` | **yes** |

\* Retrieval-only rows use the `retrieval_only` short-circuit in
`AsyncRagService.answer()` — no answer-generation, groundedness, or scope-check
calls. Only rewrite/HyDE (and rerank, which isn't an LLM chat call) remain.

Additional standing rules:

- Never rebuild the eval framework inside the live RAG path.
- Never auto-edit golden datasets; a changed row must pass the schema gate
  (`make test-eval-data`) and `dec eval-coverage` before landing.
- Retrieval flags ship dark until their benchmark gate passes (see §9).

---

## 4. Command-by-command reference

### `dec eval-fast`

> **Cost: $0** · Infra: Qdrant + active generation + local embedder

The workhorse. A five-layer, zero-LLM integrity gate to run after any pipeline
or indexing change, *before* paying for a full evaluation. If this fails, fix
the index first — everything downstream would produce garbage numbers anyway.

```
usage: dec eval-fast [--dataset DATASET] [--generation GEN] [--output-dir DIR]
```

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `tests/evaluation/recall_fast.jsonl` | retrieval-recall rows for layer 5 |
| `--generation` | active generation | which corpus/index to validate |
| `--output-dir` | stdout summary | writes machine-readable `fast_eval.json` here |

Layers executed:

1. **Corpus integrity** — chunk counts, duplicate detection, coverage.
2. **Chunk quality** — size/boundary heuristics: flags chunks `oversized`
   (>6000 chars), `over_token_budget` (>3800 tokens), `under_char_budget`
   (<50 chars).
3. **Embedding sanity** — vector dims, NaN check, consistency, semantic pair
   sanity (related texts should embed closer than unrelated ones).
4. **Vector DB integrity** — point count matches corpus, ID↔metadata agreement,
   self-retrieval (a chunk retrieves itself).
5. **Retrieval recall** — URL-recall + MRR over the fast golden set.

```bash
make eval-fast          # after every RAG-pipeline change
dec eval-fast --output-dir .rag_eval/   # keep the JSON report
```

Exit codes: `0` pass; nonzero = a layer failed (fix before proceeding).

---

### `dec eval-coverage`

> **Cost: $0** · Infra: none (reads corpus `chunks.jsonl` from disk)

Dataset-hygiene gate: proves every in-scope **recall** row's evidence actually
exists in the indexed corpus. For each row, every `expected_url` must resolve
to an indexed chunk (exact normalized match, tolerant suffix matching across
hosts) and every `expected_term` must literally occur in the corpus text. A row
whose "expected" evidence isn't even in the index would make all downstream
recall numbers meaningless — this catches that at authoring time.

Out-of-scope rows pass by design (they carry no evidence).

```
usage: dec eval-coverage [--dataset DATASET] [--generation GEN] [--json]
```

- Default validates **all** recall-format files in `tests/evaluation/`.
- Exit codes: `0` all pass · `1` any row fails · `2` bad input/no corpus.

**Provenance & coverage matrix:** the report prints the dataset **git sha**
(`git rev-parse --short HEAD`) next to the generation and each file's
`# version:` header value (when present). It also prints an **intent ×
doc_type coverage matrix**, flagging cells with 0 rows — target ≥1 query per
cell (RAGBench-style completeness; an empty cell means that intent/doc_type
combination is never measured).

```bash
make eval-coverage                                        # merge prerequisite
dec eval-coverage --dataset tests/evaluation/recall_claude.jsonl --json
```

---

### `dec eval-retrieval`

> **Cost: ~$0*** · Infra: Qdrant + Redis + embedder · *rewrite/HyDE may call the LLM unless disabled*

Retrieval-only benchmark over a golden recall set: **Recall@K, MRR@K,
Precision@K** overall and grouped by intent. Each query runs through the RAG
service with `retrieval_only=True` (no answer generation — GraphRAG/CRAG
augmentations skipped) and `bypass_cache=True` (stale cached answers never skew
a benchmark).

```
usage: dec eval-retrieval [--dataset DATASET] [--k K] [--output-dir DIR]
                          [--compare-baseline BASELINE]
```

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `tests/evaluation/golden/recall_all.jsonl` | recall-format rows (`question`, optional `intent`/`expected_urls`) |
| `--k` | `10` | cutoff for Recall/MRR/Precision |
| `--output-dir` | summary only | writes `retrieval_eval.json` (consumed by `make eval-set-baseline`) |
| `--compare-baseline` | none | regression gate: exit `1` when Recall@K < baseline − 0.02 |

Baseline workflow (the repo's retrieval regression gate):

```bash
make eval-set-baseline OUTPUT=tests/evaluation/benchmarks/baseline.json  # write baseline
make eval-retrieval-gate                                                 # compare against it
```

A missing/unparseable baseline file is a warning — the gate is skipped and exit
stays `0`.

Significance example: you switch the reranker or flip a hybrid-search flag.
Unit tests stay green, but `eval-retrieval-gate` shows Recall@10 dropped from
0.91 → 0.86 — a silent retrieval regression no unit test would catch.

Exit codes: `0` pass (or gate skipped) · `1` regression vs baseline ·
`2` bad input · `5` operational failure (no results produced).

---

### `dec evaluate`

> **Cost: 💰 paid LLM per query** · Infra: Qdrant + Redis + embedder + LLM

End-to-end golden-dataset QA evaluation: runs every question through the real
pipeline and scores the answers. Default dataset:
`tests/evaluation/eval_dataset.jsonl`.

```
usage: dec evaluate [-h] [--verbose] [--dataset DATASET] [--source SOURCE]
                    [--experiment-name NAME] [--dataset-name NAME]
                    [--spark] [--output-dir DIR]
```

What it reports:

- Per-query answer snippet, confidence, retrieved-context count.
- Summary: average confidence, **INSUFFICIENT_CONTEXT rate**, average answer
  **correctness** (token-F1 vs `ground_truth`; see glossary).
- Built-in evaluators (`services/rag_evaluation.py`):
  `RetrievalEvaluator` (term overlap/recall on retrieved chunks),
  `AnswerEvaluator.score_relevance` (embedding cosine to context),
  `FaithfulnessEvaluator` (LLM-as-judge on answer↔context),
  aggregated by `RAGEvaluator` (blend weights retrieval 0.6 / answer 0.4).
- **RAGAS** metrics if installed (`context_recall`, `context_precision`,
  `faithfulness`, `answer_relevancy`, blended `overall`). ⚠️ Adds many paid
  judge calls per query (~18–20). Omit unless needed.
- `--output-dir`: writes `per_question_results.jsonl` (id, question, answer,
  confidence, correctness, contexts) for drift analysis and bisection.
- **Drift snapshot**: records an `EvalSnapshot` into `data/eval_history.jsonl`
  and compares against the trailing window (`DRIFT_WINDOW_DAYS`, default 7).
- **Langfuse upload**: rows go to dataset `dec-evaluate-{source}-{timestamp}`;
  `--experiment-name` additionally replays the dataset as a Langfuse
  experiment scored by term-overlap faithfulness + offline RAGAS;
  `--dataset-name` runs directly against an existing Langfuse dataset.

Exit codes: `0` complete · `1` missing dataset, `--source` matched no rows, or
(Spark mode) below threshold.

---

### `dec evaluate --spark`

> **Cost: ~$0 for recall rows** · Infra: Qdrant + Redis + embedder (answer path only where required)

Retrieval-recall evaluation over `tests/evaluation/eval_dataset_spark.jsonl`
(51 in-scope queries + 2 out-of-scope traps). Measures whether retrieval finds
the expected evidence — without paying for answer generation on most rows.

Diagnostics per query:

- `term_recall` / `source_recall` — expected terms / URLs present in the
  **final** context.
- `candidate_source_recall` — same but on the **fused candidate pool**
  (pre-rerank). *(Candidate − final)* gap isolates rerank/truncation loss.
- `expected_fused_ranks` / `dropped_expected_urls` — which expected sources were
  retrieved but dropped.
- `forbidden_term_hits` — terms that must never surface (e.g. Delta/Airflow
  terms in a Spark row); any hit fails the eval.
- `out_of_scope` — OOS rows **must** produce a refusal; answering fails the eval.
- Stage latencies (`retrieval_ms`, `rerank_ms`, `total_ms`).

Gates (**exit 1** if violated): avg term or source recall over in-scope rows
below **0.9** · any forbidden term surfaced · any OOS row not refused.

Artifacts via `--output-dir`: `retrieval_provenance.json` (per-query candidate
records) and `retrieval_metrics.json` (aggregates incl. median/p95 latency).

```bash
dec evaluate --spark --output-dir .rag_eval/baseline-01
```

---

### `dec eval-generation`

> **Cost: 💰 ~1 generator + n-trials × judge calls per row** · Infra: none beyond LLM keys (no Qdrant/Redis)

Generation-quality evaluation with **retrieval frozen**: each dataset row ships
its own gold `contexts`; those are fed straight to the answer LLM. Any score
delta therefore attributes to generation (prompt/model/params), not retrieval.

```
usage: dec eval-generation [-h] [--dataset DATASET] [--n-trials N] [--output PATH]
```

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `tests/evaluation/eval_dataset.jsonl` | qa rows: `question`, `contexts`, `ground_truth` |
| `--n-trials` | `3` | judge trials averaged per row (dampens judge variance) |
| `--output` | stdout | write full JSON report |

Metrics and hard gates:

| Metric | Gate | Meaning |
|---|---|---|
| Faithfulness | ≥ **0.85** | fraction of answer claims backed by the frozen context |
| Answer relevance | ≥ **0.80** | how directly the answer addresses the question |
| Rubric correctness | ≥ **4.0** (of 5) | completeness/accuracy/tone vs the gold answer |

Design details worth knowing:

- Generator = purpose-`answer` fallback chain; judge = separate
  purpose-`evaluation` chain at `evaluation_temperature=0.0`, ideally a
  different model family than the generator to reduce self-judging bias.
- Latency is deliberately not measured here (isolated harness).

```bash
dec eval-generation
dec eval-generation --n-trials 5 --output .rag_eval/gen_eval.json
```

Exit codes: ⚠️ note the inverted convention — `0` gates passed · **`2` any gate
failed** · `1` dataset missing.

---

### `dec eval-rerank`

> **Cost: $0** (embedder/reranker compute only, no chat LLM) · Infra: Qdrant + Redis (+ reranker)

Isolated reranker A/B on frozen candidate pools — no pipeline coupling. For each
query: retrieve top `k×4` candidates → rerank down to top-k → report **post − pre
rerank gains** at K: `ndcg_gain`, `mrr_gain`, `precision_gain`, `recall_gain`.

```
usage: dec eval-rerank [--dataset DATASET] [--k K] [--pool-file PATH]
```

- `--dataset` default `tests/evaluation/golden/rerank_eval_sample.jsonl`
  (rows: `query`, `source_urls`, binary `relevance_labels`).
- `--pool-file` freezes/reuses candidate pools: newly retrieved pools are merged
  back into the JSON so later runs replay identical candidates — making A/B
  comparisons purely about the reranker.

No reranker configured → passthrough (gains ≈ 0), not an error.

```bash
dec eval-rerank --pool-file .rag_eval/rerank_pool.json
```

Acceptance criterion for the experimental `pylate_colbert` reranker:
nDCG@10 ≥ cross_encoder + **0.02** AND p95 pool latency ≤ **2×** cross-encoder.

Exit codes: `0` success · `2` failure (missing dataset, unreachable infra).

---

### `dec eval-assembly`

> **Cost: $0** · Infra: Qdrant + Redis + embedder

Context-assembly evaluation: retrieves top-k candidates per query, runs them
through the production `ContextAssembler` (16,000-char budget), and scores the
frozen result set.

```
usage: dec eval-assembly [--dataset DATASET] [--k K]
```

Per-query metrics:

| Metric | Meaning |
|---|---|
| `duplicate_candidate_rate` | duplicated content in the assembled context |
| `source_coverage_rate` | share of retrieved sources surviving into the context |
| `compression_ratio` | assembled-context chars ÷ raw retrieved chars |
| `needle_loss_rate` | gold facts dropped from the context |

Significance example: compression ratio 0.95 means assembly barely compresses
(wasteful prompts, high token spend); needle_loss > 0 means facts you paid to
retrieve were cut before the LLM ever saw them.

Default dataset `golden/assembly_eval_sample.jsonl`; default `--k 20`.
Exit codes: `0` success · `2` failure.

---

### `dec eval-prompt-aug`

Prompt-template evaluation over frozen `(query, context)` pairs.

```
usage: dec eval-prompt-aug -h --dataset PATH [--mode {template,llm}] [--provider NAME]
```

- **`--mode template` (default): fully hermetic — zero infra, zero cost.** Only
  `PromptBuilder` runs; checks prompt construction itself.
- **`--mode llm`: paid.** Generates real outputs via the answer-purpose chain
  pinned to `--provider` (default `ollama`). Prompt flags come from settings
  (`prompt_salted_xml_tags`, `prompt_trailing_instructions`,
  `prompt_citation_enforcement`) so you can A/B prompt features on identical data.

Metrics: `format_compliance_rate`, `citation_precision`, `citation_recall`,
`injection_defense_rate`, `zero_context_fallback_accuracy`.
Informational — no pass/fail gates.

```bash
dec eval-prompt-aug --dataset tests/evaluation/golden/prompt_aug_eval_sample.jsonl
dec eval-prompt-aug --dataset ... --mode llm --provider groq
```

Exit codes: `0` success · `2` failure (missing dataset, LLM error in llm mode).

---

### `dec eval-chunking`

> **Cost: $0 — fully offline, safe anywhere** · Infra: none

Chunker-quality evaluation against committed **gold spans** (human/derived
annotations of where chunk boundaries *should* be). Loads gold data, chunks
every document with the selected strategy, reports overlap and fracture.

```
usage: dec eval-chunking [--strategy {all,...}] [--gold {synthetic,human,all}] [--output PATH]
```

⚠️ argparse accepts many strategy names, but the evaluator supports exactly
**`recursive`, `sentence`, `header`, `structured`** (and `all` = those four);
anything else exits `2`.

Per-strategy metrics: token **IoU**, excerpt precision, `boundary_similarity`,
`structural_fracture_rate`.

Gold data: `tests/evaluation/golden/chunking/{synthetic_gold,human_slice}.jsonl`.
Report defaults to `/tmp/chunking_eval.json`. Pair with `make test-chunking`
(unit-level invariants/snapshots).

---

### `dec gen-synthetic-eval`

> **Cost: $0 (deterministic mode — the only mode wired)** · Infra: none (corpus on disk)

Generates a corpus-grounded synthetic **recall** set from the active generation
(heading-derived candidates), filtering every row through the coverage
validator before writing. Use it to scale question sets cheaply instead of hand-
writing rows.

```bash
make eval-gen-source SOURCE="Claude Platform Docs" LIMIT=50
# → tests/evaluation/golden/recall_synthetic_<slug>.jsonl
```

Flags: `--source` (required), `--generation`, `--limit` (default 50), `--out`.
Exit codes: `0` written · `1` zero rows survived the coverage gate ·
`2` no corpus found.

---

### Ragas integration

Optional deep-dive metrics via `ragas==0.3.9` (`services/ragas_evaluation.py`,
`services/ragas_adapters.py`):

- Metrics: `context_recall`, `context_precision`, `faithfulness`,
  `answer_relevancy`; aggregate `overall = recall×0.3 + faithfulness×0.4 +
  relevancy×0.3`.
- Uses the project's **own** `evaluation`-purpose LLM/embedding fallback chains
  wired into ragas — never raw ragas clients.
- Triggered inside `dec evaluate` when the package is installed. Requires infra
  for real runs; expensive (~18–20 LLM calls/query). Unit tests mock adapters.
- Prefer the built-in `eval-generation` gates for routine regression checks;
  reach for ragas when you want the finer-grained four-pack.

### Langfuse judges & metrics

Production-side evaluation (offline, on traces — never inline in the request path):

- **`dec langfuse-evaluate`** — LLM-as-a-judge (faithfulness / relevance /
  out-of-scope, 0–1 scale) over production `rag-query-pipeline` traces; scores
  are written back onto traces. Judge contexts come from each trace's
  `retrieval` observation, capped at 12,000 chars. Without `--max-items` the
  whole run samples by `langfuse_sample_rate` (default 1.0). Needs reachable
  Langfuse + LLM keys. Supports `--filter` and review queues.
- **`dec langfuse-metrics`** — read-only analytics via Metrics API v2. Presets:
  `cost-by-model`, `daily-volume-latency`, `score-summary`; `--days N` (default
  7), `--json`. No LLM calls.

### Drift detection & provenance

`services/drift_detector.py` tracks eval metrics over time
(`data/eval_history.jsonl`, trailing window `DRIFT_WINDOW_DAYS`=7) and flags
regressions; deltas surface in `dec langfuse-metrics` / review tooling.

Every `EvalSnapshot` records provenance (`evaluation/provenance.py`):
`git_commit`, `generation`, `embedding_model`, `reranker`, `chunk_size`,
`chunk_overlap`, `retrieval_top_k`, `config_fingerprint`.

> **Rule:** a metric move that coincides with a `config_fingerprint` change is
> an environment/config change, not a regression. Diff the fingerprint first,
> then decide.

Drift alert thresholds (defaults, `services/drift_detector.py`):
faithfulness 0.8 · context_recall 0.7 · context_precision 0.6 ·
answer_relevancy 0.7 · overall 0.7 · confidence 0.5 (+0.1 fallback for unknown
metrics).

Note: don't confuse these **offline eval gates** with the runtime verifier
thresholds in `settings.py` (`groundedness_threshold`=0.6,
`confidence_threshold`=0.18, `reranker_confidence_threshold`=0.10) — the latter
gate live answers (fail-open/fail-closed contract), not evaluations.

---

## 5. Metrics glossary for RAG newcomers

Deterministic IR/lexical metrics (computed by code, not an LLM):

| Metric | Formula / intuition | Example |
|---|---|---|
| **Recall@K** | Of all expected relevant items, what fraction appears in the top K? | Expected URLs = {A,B,C}; top-10 contains A,B → Recall@10 = 2/3 ≈ 0.67 |
| **Precision@K** | Of the top K items, what fraction is relevant? | Top-10 has 4 expected URLs → P@10 = 0.4 |
| **MRR@K** | Mean (over queries) of `1/rank` of the first relevant hit. Rewards getting *something* right quickly. | First hit at rank 1 → 1.0; rank 4 → 0.25; miss → 0 |
| **nDCG@K** | Like MRR but rewards *all* relevant items weighted by log-position. Standard for graded/binary relevance ranking. | Higher than baseline ⇒ better ordering overall |
| **term_recall** | Fraction of `expected_terms` literally present in retrieved/assembled context text. | Expected `{select, filter}`; both appear → 1.0 |
| **token-F1 (correctness)** | Harmonic mean of token-overlap precision & recall between answer and `ground_truth`. Cheap proxy for answer accuracy. | Answer covers half the ground-truth tokens → F1 ≈ 0.5 |
| **token-IoU (chunking)** | Intersection-over-union of predicted chunk tokens vs gold span tokens. | Chunk exactly equals gold span → 1.0 |
| **duplicate_candidate_rate** | Share of assembled-context candidates that are near-duplicates. Wastes budget and biases answers. | 2 dupes among 20 → 0.10 |
| **compression_ratio** | Assembled chars ÷ raw retrieved chars. Lower = tighter prompt. | 12k of 20k chars kept → 0.60 |
| **needle_loss_rate** | Fraction of known gold facts lost during assembly. Should be ~0. | 1 fact dropped of 4 → 0.25 |

LLM-as-a-judge metrics (paid, temperature 0.0, dedicated judge chain):

| Metric | Question the judge answers | Repo gate |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the provided context? (Anti-hallucination.) Scored as fraction of supported claims; also the runtime fallback heuristic uses token-overlap ≥ 0.3 per claim. | ≥ 0.85 |
| **Answer relevance** | Does the answer actually address the question asked? (Topicality, not correctness.) | ≥ 0.80 |
| **Rubric correctness** | 1–5 holistic score vs the gold answer (completeness/accuracy/tone). | ≥ 4.0 |
| **Out-of-scope judgment** | Should this trace have been refused? Used on production traces. | informational |

Why two families? Deterministic metrics are cheap, stable, CI-safe — perfect for
regression gates. Judge metrics capture qualities lexical matching can't ("the
answer contradicts the source") but add cost and variance — hence they're gated
at milestone time with multi-trial averaging.

---

## 6. Question datasets reference

All eval records share one schema (`evaluation/eval_schema.py`): stable `id`
(lowercase-hyphen slug), `question`, optional metadata
(`source_name`, `doc_type`, `intent`, `complexity`, `abstraction`), plus
kind-specific fields. Kind is detected by field presence:

- **qa rows** — `ground_truth`, `contexts`, `source_name`. Feed `dec evaluate`
  (QA mode) and `dec eval-generation`. Ground truth must be copyable from
  corpus chunks.
- **recall rows** — `expected_terms`, `expected_urls`, `expected_doc_types`,
  `forbidden_terms`, `must_not_require`, `out_of_scope`. Feed `evaluate --spark`,
  `eval-fast` (layer 5), `eval-retrieval`, `eval-coverage`. Prefer this format:
  cheaper to author correctly and verifiable against the corpus.
- **out-of-scope (OOS) rows** — `out_of_scope: true`, carry `expected_terms` as
  refusal triggers and must **not** carry `expected_urls`.

Example qa row (`tests/evaluation/eval_dataset.jsonl`):

```json
{"id": "qa-spark-001",
 "question": "What is Apache Spark?",
 "ground_truth": "Apache Spark is a unified analytics engine for large-scale data processing with APIs in Scala, Java, Python, and R.",
 "contexts": ["Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Scala, Java, Python, and R."],
 "source_name": "Apache Spark 4.0.0"}
```

Example recall row (`tests/evaluation/recall_spark_api.jsonl`):

```json
{"id": "spark-api-dataframe-select-101",
 "question": "How do you select and filter columns on a pyspark.sql.DataFrame?",
 "expected_urls": ["https://raw.githubusercontent.com/apache/spark/fa33ea00.../python/pyspark/sql/dataframe.py"],
 "expected_terms": ["select", "filter", "withColumn"],
 "intent": "api_reference", "complexity": "single_hop", "abstraction": "specific",
 "source_name": "Apache Spark 4.0.0", "forbidden_terms": []}
```

Example OOS row (`tests/evaluation/recall_oos.jsonl`):

```json
{"id": "oos-react-101",
 "question": "What is the difference between the useState and useMemo hooks in React?",
 "out_of_scope": true, "expected_terms": ["react"], "expected_urls": []}
```

### Dataset inventory

Root sets (`tests/evaluation/`) — consumed by CLI commands, gated hermetically
in CI by schema checks:

| File | Kind | Rows | Consumed by |
|---|---|---|---|
| `eval_dataset.jsonl` | qa | 12 | `dec evaluate` (default), `dec eval-generation` (default) |
| `eval_dataset_spark.jsonl` | recall | 51 + 2 OOS | `dec evaluate --spark` |
| `eval_dataset_airflow.jsonl` / `_delta_lake.jsonl` | qa | 3 / 3 | `dec evaluate --dataset …` |
| `recall_claude.jsonl` | recall | 27 | `eval-retrieval`/`eval-coverage` |
| `recall_spark_api.jsonl` | recall | 20 | same |
| `recall_airflow.jsonl` / `recall_delta.jsonl` | recall | 7 / 4 | same |
| `recall_multi_hop.jsonl` | recall | 10 | same |
| `recall_oos.jsonl` | recall | 10 | OOS refusal traps |
| `recall_fast.jsonl` | recall | small fast set | `dec eval-fast` layer 5 |
| `recall_synthetic_*.jsonl` | recall | generated | `gen-synthetic-eval` output |

(`eval_dataset_databricks.jsonl` was removed — Databricks is not in the active
pinned generation.)

Golden benchmark sets (`tests/evaluation/golden/`) — the consolidated v1.0 set,
documented in `golden/README.md`:

| File | Contents |
|---|---|
| `recall_{spark,airflow,delta,claude_platform,claude_code}.jsonl` | 100 queries each, pinned to exact upstream commits, corpus-aligned |
| `recall_all.jsonl` | shuffled union (500) — **default dataset for `dec eval-retrieval`** |
| `recall_oos.jsonl` | 20 out-of-scope traps |
| `qa_*.jsonl` (per source) | frozen-context QA for `eval-generation` |
| `rerank_eval_sample.jsonl` | frozen input for `eval-rerank` |
| `assembly_eval_sample.jsonl` | frozen input for `eval-assembly` |
| `prompt_aug_eval_sample.jsonl` | frozen input for `eval-prompt-aug` |
| `chunking/{synthetic_gold,human_slice}.jsonl` | gold spans for `eval-chunking` |

**520-query corpus-aligned set** = 500 in-scope (5 sources × 100: Apache Spark
4.0.0, Airflow 3.3.1, Delta Lake 4.3.1, Claude Platform Docs, Claude Code Docs)
+ 20 OOS. Every `expected_url` is asserted to exist in the pinned corpus
generation (verified via `eval-coverage`).

Baselines: `tests/evaluation/benchmarks/baseline.json` (written by
`make eval-set-baseline`, consumed by `make eval-retrieval-gate`).

Governance: golden datasets are the regression-gate source of truth. When
behavior intentionally changes, update the dataset consciously and expect metric
deltas — don't silently accept drift. Every edit must pass the schema gate and
`eval-coverage`.

**Golden versioning:** golden files may start with a header comment line such
as `# version: 2026-08-23` — parsers skip comment lines and
`evaluation/eval_schema.py:dataset_version_of()` reads it back (`None` when
absent). Rows may also carry an optional `dataset_version` field (schema-valid,
ignored by validators). **Never compare metrics across dataset versions**: a
delta between runs on different versions (or different git shas — printed by
`dec eval-coverage`) attributes to the data, not the code. The coverage matrix
(intent × doc_type, ≥1 query per cell) is surfaced in the same report.

---

## 7. Recommended values cheat sheet

Repo-enforced gates (hardcoded constants or settings defaults — treat these as
"recommended values" validated by benchmarks):

| Metric | Target | Enforced by |
|---|---|---|
| Faithfulness (judge) | ≥ 0.85 | `eval-generation` gate (`settings.generation_faithfulness_gate`) |
| Answer relevance (judge) | ≥ 0.80 | `eval-generation` gate (`settings.generation_relevance_gate`) |
| Rubric correctness | ≥ 4.0 / 5 | `eval-generation` gate (`settings.generation_rubric_gate`) |
| Term/source recall (Spark set, in-scope avg) | ≥ 0.90 | `evaluate --spark` gate |
| Recall@10 vs baseline (honest inscope 220 rows) | ≥ baseline_inscope 0.259 −0.02 (absolute ≥0.24) | `eval-retrieval --compare-baseline tests/evaluation/benchmarks/baseline_inscope.json` — `settings.retrieval_gate_global_tolerance` / `retrieval_gate_global_floor` |
| Recall@10 per-intent (n≥5) | ≥ max(0, baseline_intent −0.05) — e.g. how_to 0.386→0.336, code_example 0.40→0.35, api_lookup 0.071→0.021 | `eval-retrieval --compare-baseline` per-intent deltas (with `evaluation/stats.py:bootstrap_ci` CIs when per_query vectors present) — `settings.retrieval_gate_per_intent_tolerance` / `retrieval_gate_per_intent_min_n` |
| Cohen's κ (judge calibration) | ≥ 0.60 (raw ≥0.80) | `eval-judge-calibrate` gate (`settings.judge_kappa_gate` / `judge_raw_gate`, `evaluation/judge_calibration.py:KAPPA_GATE`) |
| Source recall Δ vs baseline | ≥ −0.01 | optimization benchmark |
| MRR Δ vs baseline | ≥ −0.02 | optimization benchmark |
| Identifier recall Δ | ≥ **+0.05** (required improvement) | optimization benchmark |
| Generic recall Δ | ≥ −0.01 | optimization benchmark |
| Provider calls reduction | ≤ −20% relative | optimization benchmark |
| Duplicate-rate reduction | ≤ −10% relative | diversity benchmark |
| Oversized chunk | > 6000 chars flagged | `eval-fast` heuristic |
| Over-token-budget chunk | > 3800 tokens flagged | `eval-fast` heuristic |
| Drift alerts | faithfulness 0.8, context_recall 0.7, context_precision 0.6, answer_relevancy 0.7, confidence 0.5 | `DriftDetector` defaults |
| Judge determinism | `evaluation_temperature = 0.0` | settings default |

**Honest inscope vs legacy baseline:** the 500-row `recall_all.jsonl` baseline was URL-mismatched (R@10=0.272 undercounts real retrieval; overall Recall@10 is inflated/deflated by out-of-scope rows). The honest gate uses `tests/evaluation/benchmarks/baseline_inscope.json` (220 inscope rows, R@10=0.259 with dedup) — global floor 0.24 and per-intent `max(0, baseline−0.05)` where n≥5 (e.g. `comparative` n=3 is skipped). See `data_engineering_copilot/config/settings.py:retrieval_gate_*` and `docs/rag_flaw_prevention_plan.md:F3`.

General guidance beyond the repo gates: prefer Recall-oriented gates for
retrieval (missing evidence is fatal; a slightly noisy top-k is recoverable by
rerank); require faithfulness strictly higher than relevance (hallucination is
worse than incompleteness); when a metric moves, diff the provenance
`config_fingerprint` before diagnosing.

---

## 8. How to run — make targets, CI vs. local, exit codes

### Make targets

| Target | Runs | Where |
|---|---|---|
| `make test-eval` | `pytest tests/evaluation/ -v` — eval machinery with **mocked embedder**, no infra | CI ✅ |
| `make test-eval-data` | hermetic dataset-quality gates: dataset schema, eval schema, coverage logic, run metrics, synthetic generator | CI ✅ |
| `make eval-fast` | `dec eval-fast` — 5-layer zero-LLM integrity gate | local (needs Qdrant) |
| `make eval-coverage` | `dec eval-coverage` — corpus alignment gate | local |
| `make eval-retrieval` | `dec eval-retrieval --k 10` | local |
| `make eval-retrieval-gate` | `dec eval-retrieval --compare-baseline tests/evaluation/benchmarks/baseline_inscope.json --k 10` (prints global Δ with 95% bootstrap CI + per-intent Δ vs `max(0, baseline−0.05)` where n≥5) | local gate |
| `make eval-rerank-smoke` | `dec eval-rerank --pool-file /tmp/rerank_pool_smoke.json --k 10` freeze + `$0` replay — reranker A/B without LLM | local ($0 replay) |
| `make eval-prompt-aug-smoke` | `dec eval-prompt-aug --dataset tests/evaluation/golden/prompt_aug_eval_sample.jsonl --mode template` — fully hermetic, no infra | local ($0) |
| `make eval-set-baseline OUTPUT=path` | write a fresh baseline | local |
| `make eval-gen-source SOURCE=name [LIMIT=n]` | `dec gen-synthetic-eval` | local |
| `make eval-golden-consolidate` | consolidate golden sets + coverage validation | local |
| `make eval-rag-benchmark` | full optimization benchmark vs gates | local |
| `make test-chunking` / `-serial` | chunker invariant/metric/snapshot suites | local (hermetic) |

CI (`.github/workflows/test.yml`) runs **only** the hermetic subset: lint →
unit → `test-eval` → `test-eval-data`. Everything needing Docker/Ollama/Qdrant
(`eval-fast`, `eval-retrieval-gate`, integration/e2e/smoke) stays local —
hard-gated locally by `make test-real`.

### Exit-code conventions

Not fully uniform across harnesses — check before scripting:

| Code | Meaning |
|---|---|
| `0` | pass (or gate skipped where documented) |
| `1` | gate/regression failure for most harnesses; missing dataset for `eval-generation`; missing-dataset/no-match for `evaluate` |
| `2` | config error / bad input (missing/unparseable dataset) — **but also gate failure for `eval-generation` specifically** |
| `5` | operational failure producing no results (`eval-retrieval`) |

---

## 9. Workflows — what to run when

### After touching RAG-pipeline code (every time)

```bash
make eval-fast          # prove index integrity, $0
```

### After changing datasets / re-crawling / new generation

```bash
make eval-coverage      # expected evidence exists in corpus
make test-eval-data     # schema/slug gates (also in CI)
```

### Before merging anything retrieval-related

```bash
make eval-retrieval-gate    # Recall@10 within −0.02 of baseline
```

### When touching a specific stage

```bash
dec eval-chunking                       # chunkers (+ make test-chunking)
dec eval-rerank --pool-file p.json      # reranker A/B (freeze pools!)
dec eval-assembly                       # context assembler
dec eval-prompt-aug --dataset ...       # prompt templates (template mode first)
```

### Milestone / release gates (paid — deliberate actions)

```bash
dec eval-generation --output .rag_eval/gen_eval.json   # judge gates
dec eval-generation --sample 10 --output /tmp/quick.json  # dev loop: fast judge check
dec evaluate --output-dir .rag_eval/run-$(date +%F)    # end-to-end QA + drift snapshot
```

### Judge calibration & proxy validation (paid — deliberate actions)

```bash
dec eval-judge-calibrate   # validate judge vs human labels (label rows first!)
dec eval-proxy-validate    # validate proxy-recall vs LLM-judge ground truth
```

### Flipping a dark retrieval flag

Flags like `late_chunking_enabled`, `identifier_sparse_rrf_enabled`,
`namespace_bm25_enabled`, `mrl_multistage_enabled` ship `False` with acceptance
criteria documented in `settings.py` comments. Never flip without running the
named harness against baseline, e.g.:

- `late_chunking_enabled`: `make eval-fast` PASS **and** `eval-retrieval`
  Recall@10 ≥ baseline − 0.01.
- `identifier_sparse_rrf_enabled` / `namespace_bm25_enabled`: identifier recall
  ≥ **+0.05** with all global recall/MRR thresholds satisfied.
- `mrl_multistage_enabled`: Recall@10 within −0.01 **and** p95 latency improved
  ≥ 20%.

### Monitoring production

```bash
dec langfuse-evaluate --max-items 10     # judge sampled traces (paid)
dec langfuse-metrics score-summary       # read-only score analytics
```

Run drift checks whenever an eval metric moves across releases; diff
`config_fingerprint` first.

---

## 10. Best practices & gotchas

1. **Respect the layered contract.** Layers 1–5 are free — never add LLM calls
   to them. Generation is the only paid layer; gate new eval work behind a cost
   check.
2. **Always `eval-fast` before any paid eval.** A broken index makes expensive
   numbers meaningless.
3. **Freeze inputs for A/B comparisons.** Use `--pool-file` (rerank), frozen
   gold contexts (generation), committed corpora (chunking/fast). Otherwise you
   measure retrieval noise, not your change.
4. **Judge hygiene.** Judge = purpose-`evaluation` chain at temperature 0.0,
   different model family than the generator; average multiple trials. Don't
   reuse the answer chain for judging.
5. **Prefer recall-format rows** over hand-written `ground_truth` (which must be
   copyable from corpus chunks). Give recall rows real `expected_urls` +
   `expected_terms` that pass `eval-coverage`, and stable unique hyphen-slug
   `id`s.
6. **Never auto-edit golden datasets.** Schema gate + coverage gate must pass
   before an edited row lands. Intentional behavior changes → update dataset +
   expect metric deltas, explicitly.
7. **Provenance before panic.** On any metric movement, diff the EvalSnapshot's
   `config_fingerprint` (git commit, generation, embedding model, reranker,
   chunk params) before calling it a regression.
8. **Watch the exit-code quirks** (`eval-generation` inverts 1/2; missing
   baselines skip rather than fail) when scripting gates.
9. **Beware accidental costs**: installing `ragas` turns every `dec evaluate`
   into dozens of extra paid calls per query; `eval-retrieval` may still hit
   rewrite/HyDE unless disabled; `langfuse-evaluate` without `--max-items`
   samples the whole window.
10. **Scale datasets synthetically, validate deterministically**:
    `make eval-gen-source SOURCE="..." LIMIT=50` produces corpus-grounded rows
    already filtered through the coverage gate.
