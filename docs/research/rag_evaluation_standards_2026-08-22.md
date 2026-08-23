# RAG Evaluation Standards — External Research & Gap Analysis (2026-08-22)

**Scope:** Industry standards for *evaluating* RAG systems — frameworks, metric catalogs,
thresholds, and layer-by-layer/module-by-module isolation methodology — compared against
this repo's evaluation system (`docs/EVALUATION_GUIDE.md`, commit-era 2026-08-22).
**Motivation:** iteration currently burns hundreds of paid LLM calls; the comparison below
prioritizes cost-relevant findings.

**Method:** two web-research passes against primary sources (official vendor docs, arxiv
papers, official repos), August 2026. Companion prior research:
`rag_best_practices_reference_2026-08-21.md` (§8 Evaluation),
`rag_best_practices_comparison_2026-08-21.md`.

---

## 1. Framework landscape

| Framework | Metrics offered | Judge methodology | Published thresholds | Status |
|---|---|---|---|---|
| **RAGAS** (docs.ragas.io, 2025-12) | Context Precision/Recall/Entities Recall, Noise Sensitivity, Faithfulness, Response Relevancy; Factual Correctness, aspect critics, rubrics; agent metrics | claim-decomposition + entailment; LLM judge | ⚠️ none documented | stable, widely adopted |
| **TruLens RAG Triad** (trulens.org) | Groundedness, Context relevance, Answer relevance — evaluated as a **conjunction** (all three edges OK ⇒ hallucination-free w.r.t. the KB) | feedback functions, structured output | ⚠️ none | active |
| **DeepEval** (deepeval.com) | G-Eval, contextual precision/recall/relevancy, faithfulness/hallucination | G-Eval CoT judging, JSON structured labels | **default 0.5 on all metrics (0–1)** — the only broad vendor-published numeric default; `strict_mode` treats faithfulness near-binary | active |
| **Arize Phoenix** (arize.com/docs) | retrieval precision/recall/nDCG; hallucination & QA-correctness classifier templates; tool-calling judges with structured output | single-shot, tool-called labels + CoT | **≥85% F1 of its judge templates vs a golden set** — a gate on the *judge*, not the app | active |
| **LangSmith** (docs.langchain.com) | dataset/experiment model; reference-based & reference-less evaluators; pairwise mode | few-shot judges officially recommended; Boolean/Categorical/Continuous feedback | ⚠️ none (ships prompts, not gates); online evals controlled via sampling + spend limits | active |
| **Vertex AI Gen AI eval** (cloud.google.com, 2025–26) | computation-based (rouge/bleu/exact_match); PointwiseMetric with criteria+rubric templates; PairwiseMetric win-rate 0–1; AutoSxS | rubric-driven; **adaptive rubrics** (`GENERAL_QUALITY` recommended default); static `GROUNDING` rubric "crucial for RAG" | 1–5 anchor-text rubrics published; ⚠️ no pass/fail gates | active |
| **MLflow GenAI** (mlflow.org, 3.x) | predefined judges incl. **RetrievalRelevance / RetrievalGroundedness / RetrievalSufficiency** (trace-coupled, require RETRIEVER spans); guidelines judges; human-aligned judges (`align()`) | single-shot templates; custom via `make_judge`; default judge model **gpt-4o-mini** (explicit cost lever) | ⚠️ none | fast-moving |
| **ARES** (arxiv 2311.09476, NAACL 2024) | context relevance, answer faithfulness, answer relevance | **trained lightweight judges + PPI (prediction-powered inference)** → statistically corrected scores **with confidence intervals** from a few hundred human labels | ⚠️ none (CI-bearing estimates instead) | influential academically |
| **RGB** (arxiv 2309.01431, AAAI 2024) | 4 abilities: **noise robustness, negative rejection, information integration, counterfactual robustness** | benchmark testbeds, EN+ZH | diagnostic, no gates | canonical taxonomy source |
| **RAGBench + TRACe** (arxiv 2407.11005, Galileo) | TRACe: Relevance, **Utilization** (fraction of retrieved text the generator used), **Completeness** (fraction of relevant material used), Adherence (=faithfulness, token-level support sets) | 100k labeled examples; meta-finding: **fine-tuned 400M DeBERTa beat few-shot LLM judges** | ⚠️ none | NeurIPS D&B release |
| **CRAG / KDD Cup 2024** (arxiv 2406.04744) | truthfulness = mean of perfect 1 / acceptable 0.5 / missing 0 / **incorrect −1** (auto-eval 1/0/−1 ⇒ prefers abstention over wrong); macro-avg across domains | **two-judge averaging (GPT-3.5 + Llama-3-70B) explicitly to avoid self-preference**; judged F1 94.7%/98.9% vs humans | fully documented scoring | challenge closed, methodology durable |
| **BEIR/MTEB** (arxiv 2104.08663, 2210.07316) | zero-shot retrieval via **nDCG@10** over fixed datasets; BM25 as mandatory sparse baseline | deterministic IR tooling | rankings, no pass thresholds | de-facto retriever-component standard |
| **RAGChecker** (arxiv 2408.08067, NeurIPS 2024 D&B) | claim-level entailment (RefChecker-based); overall P/R/F1; retriever (claim recall, context precision) + generator (context utilization, noise sensitivity ×2, hallucination, self-knowledge, faithfulness) | claim decomposition + NLI entailment | ⚠️ none | official LlamaIndex integration |
| 2025–26 entrants | Opik (Comet), W&B Weave scorers, OpenEvals (LangChain — now backs LangSmith prebuilt judges), MLflow alignment | trajectory/agentic eval growing across ALL vendors simultaneously (TRAJECT-Bench, TRACE, Agent-as-a-Judge) | — | genuinely adopted |

## 2. Consensus: the canonical minimal suite

Converged independently by TruLens, RAGAS, ARES, TRACe, DeepEval, MLflow, LangSmith, Phoenix:

1. **Context relevance** (query ↔ chunks) — retriever check.
2. **Faithfulness / groundedness / adherence** (answer ↔ chunks; claim-decompose + entailment) — hallucination check.
3. **Answer relevancy** (query ↔ answer) — usefulness check.
4. With references: **answer correctness**, **context recall**, plus classic IR (**recall@k / nDCG@10**) for the pure retrieval component.

Composition doctrine: the triad is a **conjunction, not an average**; CRAG operationalizes failure asymmetry (wrong ≫ abstention).

## 3. What vendors actually publish as thresholds (honest list)

- DeepEval: **0.5 default** (only broad numeric default anywhere).
- Phoenix: **≥85% F1 judge-template agreement vs golden set** (gate the judge, then trust it).
- Vertex: 1–5 rubric anchors (text published); pairwise win-rate 0–1.
- CRAG: 1 / 0.5 / 0 / −1 scoring; two-judge anti-self-preference.
- Everything else: directional guidance only. The blog folklore (">70% context relevance, >90% faithfulness") traces to **no primary vendor doc**.

**Implied correct practice:** validate your judge against a human-labeled golden set, then set application gates **relative to a frozen baseline** — exactly the pattern this repo already implements.

## 4. Methodology consensus (isolated / stage-wise evaluation)

Nearly every primary source prescribes:

1. **Evaluate components separately, attribute regressions per stage** — Qdrant (qrels + 2×2 diagnostic), Haystack/deepset (`add_isolated_node_eval`), LlamaIndex (separate retriever/response evaluators), Weaviate, Vespa (match-phase vs ranking evaluators), Elastic.
2. **TREC-lineage tooling for retrieval** — qrels + trec_eval/pytrec_eval/ranx; don't hand-roll IR math.
3. **Deterministic/code graders first; reserve LLM judges for subjective dimensions** — Anthropic, OpenAI, Langfuse, Elastic cost guidance.
4. **Calibrate any LLM judge against human labels before scale** — MT-Bench (>80% agreement ≈ human-human parity); Vespa pipeline: small human set → confusion matrix → scale.
5. **Position-swap/randomize in pairwise judging** — MT-Bench (2023), independently reconfirmed by GraphRAG bias audits (arxiv 2506.06331: fixed answer positions + length confounds shrank reported gains sharply; tie rates >20%).
6. **Gate changes in CI on golden datasets with per-metric thresholds**; freeze inputs; promote only on measured wins.
7. **Include negative/unanswerable cases** — SQuAD 2.0 design, RGB negative rejection, OWASP adversarial testing.
8. **Structured-output labels + CoT rationales for judges** — now the standard implementation everywhere (Phoenix tool-calling, DeepEval JSON, Vertex templates).
9. **Production:** sample expensive judges over real traffic into reusable judged collections (Vespa: last month of traffic); aggregate real questions back into benchmarks ("benchmarking is effectively a regression activity" — Elastic); anomaly alerting on quality signals; Phoenix-style drift tracking (query-centroid distance, p10/p90 query→top-k cosine, cluster entropy, P@k drift).

**Contested / open (as of Aug 2026):** self-enhancement bias magnitude; judge determinism — *"temperature=0 is deterministic" is falsified* (surviving advice: multi-run variance reporting, which few harnesses implement); Cohen's κ target for judge-human agreement (raw-% vs κ coexist unreconciled); human-review ratio for synthetic sets (qualitative only); GraphRAG gain magnitudes (many published wins were evaluation artifacts); standardized injection-ASR benchmarks, canary/shadow promotion recipes, long-context RAG harnesses, cost-Pareto report formats — **no primary standard exists**.

## 5. Comparison vs our evaluation system

Already meeting or exceeding consensus (do not rebuild):

| Standard | Us |
|---|---|
| Component-wise frozen-input isolation | ✅✅ best-practice form (eval-rerank frozen pools, eval-generation frozen contexts, eval-chunking gold spans) |
| IR metrics vs golden qrels | ✅ Recall@K/MRR/nDCG/P@K, 520-query corpus-aligned golden set, pinned upstream commits |
| Gates vs frozen baseline (not folklore numbers) | ✅ `eval-retrieval-gate`, numeric gates next to code |
| Cost-tiered ladder concept | ✅ layers 1–6 documented; layers 1–5 LLM-free |
| Negative/out-of-scope cases | ✅ OOS refusal traps (root + golden sets) |
| Judge hygiene basics | 🟡 different-family judge, temp 0.0, n-trials averaging — but see gaps |
| Provenance/config fingerprinting before diagnosing drift | ✅✅ rarer in industry than here |
| Dark-flag acceptance criteria tied to named harnesses | ✅✅ |

### Gaps (numbered; cost impact noted)

| # | Gap | Evidence | Cost impact |
|---|---|---|---|
| G1 | **RAGAS auto-runs inside `dec evaluate` when installed** (~18–20 paid calls/query) | internal; vendors treat deep metric packs as opt-in deep-dives | 🔴 largest single burn |
| G2 | **No judge-verdict cache** — identical (judge, prompt-version, question, answer, context) re-paid on every run | internal; Redis caches exist for query/embedding/BM25 but not judges | 🔴 high on iterate-re-run loops |
| G3 | **Point-estimate gates only** — no confidence intervals / significance on deltas; ±0.02 tolerance arbitrary → re-runs "to confirm" | ARES PPI; MT-Bench variance; determinism-falsified studies | 🟠 medium-high |
| G4 | **Judge never calibrated vs human labels** (known since 08-21 comparison) | Phoenix procedure, Vespa pipeline, consensus #4 | 🟠 blocks safe judge-model downgrades |
| G5 | **No position-swapped pairwise A/B mode**; A/B done as two independent full evals | MT-Bench; GraphRAG audits | 🟠 doubles A/B cost + adds bias risk |
| G6 | **Judge calls not enforced to structured output**; parse failures waste calls | consensus #8 | 🟡 medium |
| G7 | **`langfuse_sample_rate` defaults 1.0** — judges every production trace | LangSmith/Vespa sampling practice | 🔴 unbounded prod burn |
| G8 | **No noise-robustness / negative-rejection probes** (RGB taxonomy) — OOS traps exist, noisy/counterfactual contexts don't | RGB; SQuAD 2.0 | 🟡 cheap to add via frozen-context variants |
| G9 | **No context utilization/completeness metrics** (TRACe) — computable $0 from existing provenance captures | RAGBench/TRACe | 🟢 free diagnostic value |
| G10 | **`metrics.py` pseudo-metrics** (confidence-proxy MRR/P@K, hardcoded 0.45) presented like ground truth | known P2 item; violates honesty norm | 🟡 misleading |
| G11 | **`eval-chunking` CLI advertises strategies the evaluator rejects** | known P2 item | 🟢 trivial fix |
| G12 | **Synthetic generation lacks diversity/difficulty controls & review loop**; Ragas TestsetGenerator branch unwired | RAGAS generator docs; qualitative consensus | 🟡 dataset quality ceiling |
| G13 | **Fixed drift alert thresholds** (not baseline-relative); no unified cross-harness trend report | Phoenix/Pinecone practice | 🟡 late detection |
| G14 | **Fixed n-trials** — always max trials even when converged | variance-aware practice | 🟡 direct savings |
| G15 | **Single-judge architecture** — no cheap-first-escalate cascade | CRAG two-judge; MLflow alignment; ARES/RAGBench small-judge finding | 🔴 biggest structural saving once G4 lands |

## 6. Recommended strategy direction (summary)

1. **Stop paying for depth you didn't ask for** (G1, G7): RAGAS opt-in; sampled production judging with spend caps.
2. **Never pay twice for the same judgment** (G2, G14): cache judge verdicts; adaptive trials.
3. **Pay for certainty, not repetition** (G3, G5): CIs/significance so one run suffices; position-swapped pairwise A/B.
4. **Make cheap judges trustworthy, then use them** (G4 → G15): calibrate vs human labels; cascade local-judge-first.
5. **Widen $0 metric coverage** (G8, G9): robustness probes via corrupted frozen contexts; TRACe utilization/completeness from provenance.
6. **Fix honesty bugs** (G10, G11) — mislabeled pseudo-metrics erode trust in the free layers.

Remediation plan with task-level detail: `plans/2026-08-22_20-39_eval_gap_fix_plan.md`.
