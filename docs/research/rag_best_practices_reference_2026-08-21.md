# RAG Pipeline Best Practices — Primary-Source Reference (2025–2026)

**Compiled:** 2026-08-21
**Method:** Every claim below was verified against a primary source fetched on this date (official engineering blogs, original arXiv papers, official product docs). Secondary blog summaries were avoided. Where a source could not be reached, it is not cited.

---

## 1. Reference architectures

### 1.1 The canonical RAG formulation
The original RAG paper (Lewis et al., NeurIPS 2020) frames RAG as combining a parametric memory (seq2seq LLM) with a non-parametric memory (dense vector index accessed by a neural retriever), and distinguishes two formulations: one conditioning the whole generation on the same retrieved passages, one allowing different passages per token. It also established the still-relevant motivation: provenance, knowledge updates, and precise knowledge manipulation are weaknesses of parametric-only models.
- Source: https://arxiv.org/abs/2005.11401 (submitted 2020-05-22, v4 2021-04-12)

### 1.2 Modular RAG: the modern structural taxonomy
Modular RAG (Gao et al.) argues the naive linear "retrieve-then-generate" loop no longer describes real systems. It decomposes RAG into independent modules and operators with **routing, scheduling, and fusion** mechanisms, and identifies four prevalent flow patterns: **linear, conditional, branching, and looping**. This is the accepted vocabulary for describing production pipelines in 2025–2026.
- Source: https://arxiv.org/abs/2407.21059 (2024-07-26)

### 1.3 Anthropic's contextual retrieval stack
Anthropic's engineering post defines a concrete high-performing preprocessing + retrieval stack:
- Chunk → generate a chunk-specific context blurb (50–100 tokens) with an LLM given the whole document → prepend before embedding **and** before BM25 indexing ("Contextual Embeddings" + "Contextual BM25").
- Measured results (top-20 retrieval failure rate): contextual embeddings alone −35% (5.7%→3.7%); + contextual BM25 −49% (→2.9%); + reranking −67% (→1.9%).
- Their runtime shape: retrieve top-150 candidates → rerank → keep top-20 for generation. They found top-20 chunks beat top-10/top-5.
- Cost note: with prompt caching, one-time contextualization cost ≈ $1.02 per million document tokens.
- Explicitly rejected alternatives after evaluation: generic document summaries prepended to chunks ("very limited gains") and summary-based indexing ("low performance"); HyDE listed as a different approach.
- Source: https://www.anthropic.com/news/contextual-retrieval (2024-09-19)

### 1.4 Agentic RAG
The Agentic RAG survey (v4 revised April 2026) positions agentic RAG as embedding autonomous agents into the pipeline using four design patterns — **reflection, planning, tool use, multi-agent collaboration** — to dynamically manage retrieval strategies and iteratively refine context. It provides a taxonomy by agent cardinality, control structure, autonomy, and knowledge representation, and flags open challenges in evaluation, coordination, memory management, efficiency, and governance. Treat as the direction of travel, not a default architecture: static workflows remain correct for most single-domain QA.
- Source: https://arxiv.org/abs/2501.09136 (2025-01-15, v4 2026-04-01)

### 1.5 Production-engine view (Vespa)
Vespa's RAG documentation models RAG as retrieval system + prompt construction + LLM call inside the query path (`RAGSearcher`), i.e., pushing retrieval-aware prompt assembly into the search engine rather than orchestrating purely application-side.
- Source: https://docs.vespa.ai/en/rag/rag.html

---

## 2. Query understanding

### 2.1 HyDE (Hypothetical Document Embeddings)
HyDE zero-shot instructs an LLM to write a hypothetical answer document, embeds *that*, and retrieves neighbors of the hypothetical document; the encoder's dense bottleneck filters hallucinated details. Strong when no relevance labels exist and the query/document vocabulary mismatch is large.
- Source: https://arxiv.org/abs/2212.10496 (2022-12-20)
- **When it hurts:** adds an LLM call per query (latency/cost), and Anthropic reports evaluating context-augmentation alternatives of this family and seeing low performance relative to contextual retrieval (https://www.anthropic.com/news/contextual-retrieval). Use only where evals show gains.

### 2.2 Step-back prompting
Asks the model to derive a high-level concept/principle question from the specific query first, then reason with both. Gains reported on knowledge QA and multi-hop tasks (e.g., TimeQA +27%, MuSiQue +7% on PaLM-2L). Applicable as a query-transformation step before retrieval for abstraction-heavy questions.
- Source: https://arxiv.org/abs/2310.06117 (2023-10-09, ICLR 2024)

### 2.3 Query decomposition / self-ask for multi-hop
Self-Ask has the model explicitly ask and answer follow-up sub-questions before answering the initial question; its structured format makes it trivial to plug a retriever/search engine in to answer each follow-up. Motivated by the compositionality gap: GPT-3-family models improve on single-hop recall without improving compositional ability.
- Source: https://arxiv.org/abs/2210.03350 (2022-10-07, Findings of EMNLP 2023)

### 2.4 Intent classification
Intent routing (which retriever, which namespace, whether retrieval is needed at all) is standard practice in modular/agentic RAG systems as a conditional-routing operator (Modular RAG's "conditional" pattern; Agentic RAG's planning pattern).
- Sources: https://arxiv.org/abs/2407.21059 ; https://arxiv.org/abs/2501.09136

---

## 3. Retrieval

### 3.1 Hybrid search (dense + sparse/BM25) is the baseline
- Anthropic: "Embeddings+BM25 is better than embeddings on their own" across all tested domains; BM25 wins on exact identifiers ("Error code TS-999" example). (https://www.anthropic.com/news/contextual-retrieval, 2024-09-19)
- Weaviate: hybrid = parallel keyword (BM25/BM25F) + dense vector search fused into one ranking; dense handles meaning disambiguation, sparse handles exact terms like product names. (https://weaviate.io/blog/hybrid-search-explained, 2025-01-27)
- Qdrant ships server-side hybrid via its Query API since 1.10 (sparse vectors + dense + fusion in one request). (https://qdrant.tech/articles/hybrid-search/, 2024-07-25)

### 3.2 Fusion: RRF, not linear combination
- **RRF** (`Σ 1/(k + rank)`) is the de-facto standard fusion method; Qdrant calls it exactly that and implements `FusionQuery(fusion=RRF)` natively. Elastic documents RRF with default `rank_constant=60`, requires ≥2 child retrievers, no tuning needed, and notes RRF outperforms either query individually while removing weight tuning. Weaviate uses k=60 too.
- **Do not linearly combine raw scores**: Qdrant shows BM25 and cosine scores plotted in 2D are not linearly separable between relevant/non-relevant — "None of the linear formulas would be able to distinguish between them."
- Weaviate nuance: `rankedFusion` (rank-only) vs `relativeScoreFusion` (min-max normalized scores, default since v1.24) — the latter preserves score magnitude information that rank-only fusion discards.
- Sources: https://qdrant.tech/articles/hybrid-search/ ; https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html (redirects to https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) ; https://weaviate.io/blog/hybrid-search-explained

### 3.3 Chunking strategies
| Strategy | Mechanism | Evidence |
|---|---|---|
| Fixed-size w/ overlap | Token windows | Baseline; boundary choice affects retrieval (Anthropic implementation note #1) |
| Contextual retrieval | LLM-generated chunk-specific context prepended pre-embed & pre-BM25 | −49% retrieval failures; −67% with rerank (Anthropic, 2024-09-19) |
| Late chunking | Run long-context embedding transformer over whole doc, mean-pool token vectors per chunk *afterwards* | Beats naive chunking on BeIR nDCG@10 across SciFact/TRECCOVID/FiQA/NFCorpus; gains grow with document length (Jina AI, 2024-08-22; paper arXiv:2409.04701) |
| Hierarchical / parent-child | Retrieve small chunks, return larger parents | Standard operator in modular frameworks; Vespa supports parent-child document relationships natively |
| Semantic chunking | Embedding-boundary splits | Jina explicitly argues regex/boundary-cue segmentation beats model-based semantic chunking on cost/speed |

Sources: https://jina.ai/news/late-chunking-in-long-context-embedding-models ; https://docs.vespa.ai/en/schemas/parent-child.html (referenced from Vespa docs nav); https://www.anthropic.com/news/contextual-retrieval

### 3.4 Metadata filtering
Filter-first or filter-during-retrieval (Qdrant payload filters, Weaviate where-filters, Elastic query clauses inside retrievers) is table stakes; Qdrant's Query API composes filtered prefetches with fusion/reranking stages server-side. (https://qdrant.tech/articles/hybrid-search/)

### 3.5 Embedding model selection
Anthropic found Voyage and Gemini embeddings best of those tested, and that contextualization improves every embedding model tested — i.e., technique choice dominates marginal model choice, but model quality still matters. Matryoshka embeddings enable multistage retrieve-then-rerank within one model at multiple dimensionalities (Qdrant Query API example: 64d prefetch → 128d rerank → 256d rerank).
- Sources: https://www.anthropic.com/news/contextual-retrieval ; https://qdrant.tech/articles/hybrid-search/

---

## 4. Reranking

### 4.1 Cross-encoder rerankers (the default)
Cohere Rerank is the reference hosted cross-encoder API: query+documents in, relevance-scored ordering out; billed per "search unit"; structured docs should be serialized as YAML strings for best performance; current generation `rerank-v4.0-pro`/`rerank-v4.0-fast`, single multilingual model covering 100+ languages.
- Source: https://docs.cohere.com/docs/rerank-overview

### 4.2 Pool sizing and cost/latency tradeoff
Anthropic's operating point: retrieve **top-150**, rerank down to **top-20**; reranking cut failure rate by a further ~35% relative (2.9%→1.9%) on top of contextual hybrid retrieval. They state the tradeoff explicitly: reranking more chunks improves quality but adds latency/cost; tune on your workload.
- Source: https://www.anthropic.com/news/contextual-retrieval

### 4.3 Real late interaction (ColBERT) vs approximations
- ColBERT-style late interaction stores per-token embeddings; MaxSim scoring allows precomputed document representations (unlike cross-encoders), so it can rerank fast without scanning the corpus. Qdrant supports multivectors with `MultiVectorComparator.MAX_SIM` for this.
- **Real ColBERT at scale = PLAID**: centroid interaction + centroid pruning treats passages as bags of centroids, cutting late-interaction latency up to **7× on GPU / 45× on CPU** vs vanilla ColBERTv2 at 140M-passage scale with no quality loss. Anything else calling itself "ColBERT" without PLAID-class optimizations is an approximation.
- Production footgun (Qdrant): late-interaction vectors used only for reranking don't need HNSW graphs — set `hnsw_config(m=0)` on that vector space or indexing cost explodes (hundreds of embeddings/doc).
- Sources: https://arxiv.org/abs/2205.09707 (2022-05-19) ; https://qdrant.tech/articles/hybrid-search/

### 4.4 LLM rerankers
LLM listwise/pointwise reranking exists but each primary source above stops short of recommending it over dedicated rerankers; Cohere/Qdrant/Anthropic all use dedicated rerank models. Treat LLM-as-reranker as emerging, gated on cost/latency budget.

---

## 5. Context assembly & prompt construction

### 5.1 Lost in the middle
Performance is highest when relevant info sits at the **beginning or end** of the context and degrades significantly in the middle, even for long-context models (multi-document QA and key-value retrieval, TACL 2023). Mitigations: put best-ranked evidence first/last, cap context size rather than stuffing.
- Source: https://arxiv.org/abs/2307.03172 (2023-07-06, v3 2023-11-20)

### 5.2 Budgeting & count
Anthropic: more chunks help until they distract — top-20 beat top-10/top-5 in their tests; "more information can be distracting." Budget = f(model, task); measure, don't assume.
- Source: https://www.anthropic.com/news/contextual-retrieval

### 5.3 Deduplication & diversity
Deduplication of fused results is part of the standard hybrid recipe (Anthropic step 5: "Combine and deduplicate results … using rank fusion"). MMR-style diversity is available engine-side (Vespa result diversity grouping). MMR originates in Carbonell & Goldstein (SIGIR 1998); Vespa docs cover diversity grouping at https://docs.vespa.ai/en/querying/result-diversity.html.

### 5.4 Citation enforcement
Anthropic's Citations API returns exact supporting passages per claim: documents are sentence-chunked (or user-blocked for custom content), responses carry char/page/block-index citations, `cited_text` doesn't count toward output tokens, and citations are guaranteed valid pointers (vs prompt-based quoting). Key constraint: **citations are incompatible with structured outputs** (400 error if combined).
- Source: https://docs.claude.com/en/docs/build-with-claude/citations

### 5.5 Prompt injection defenses for retrieved content
Spotlighting (Microsoft Research) marks untrusted retrieved content so the model can distinguish data from instructions — three variants: **delimiting, data marking, encoding**. Result: attack success rate dropped from >50% to <2% on GPT-family models with minimal task-quality impact. Salted/rotating markers defeat marker-injection evasion.
- Source: https://arxiv.org/abs/2403.14720 (2024-03-20)

---

## 6. Generation

### 6.1 Structured outputs / JSON schema enforcement
OpenAI Structured Outputs (GA Aug 2024): `strict:true` function/tool params or `response_format=json_schema`; implemented via **constrained decoding over a context-free grammar** compiled from the schema (CFG chosen over FSM/regex to support recursive schemas). Guarantees schema conformance except on refusal or truncation (`refusal` field signals refusal). Caveats: subset of JSON Schema allowed, first-request latency penalty for grammar compilation, incompatible with parallel tool calls. This matches the pattern of provider-gated capabilities (Ollama uses `format=`, OpenAI-compatible APIs use `response_format`).
- Source: https://openai.com/index/introducing-structured-outputs-in-the-api/ (2024-08-06)

### 6.2 Hallucination detection / groundedness
Groundedness verification is operationalized in two places above: CRAG's retrieval evaluator grades retrieved-doc relevance before generation (below), and RAGAS exposes Faithfulness / Noise Sensitivity metrics for offline measurement (below). NLI-based claim decomposition is the underlying mechanism in both families.

### 6.3 Self-consistency
Sample diverse reasoning paths, marginalize to the most consistent answer: +17.9% GSM8K, +11.0% SVAMP, +6.4% StrategyQA over CoT greedy decoding (ICLR 2023). In RAG, applicable to answer-selection over multiple generated candidates — cost-proportional to sample count.
- Source: https://arxiv.org/abs/2203.11171 (2022-03-21, ICLR 2023)

---

## 7. Corrective loops

### 7.1 CRAG (Corrective RAG)
A lightweight **retrieval evaluator** assesses overall quality of retrieved docs and returns a confidence degree that triggers different actions: correct → use; incorrect → discard and fall back to **large-scale web search**; ambiguous → both. A decompose-then-recompose step strips irrelevant sentences from kept docs. Plug-and-play with existing RAG stacks; improvements shown on four datasets across short- and long-form generation.
- Source: https://arxiv.org/abs/2401.15884 (2024-01-29, v3 2024-10-07)

### 7.2 Self-RAG
Trains the LM to emit **reflection tokens** controlling: whether to retrieve at all (on-demand, adaptive retrieval), whether passages are relevant, whether the generation is supported by them, and whether it's useful. Inference-time controllable via reflection-token thresholds. Outperformed ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification, with gains in citation precision for long-form output. Requires fine-tuning — heavier lift than CRAG's prompting-based gating.
- Source: https://arxiv.org/abs/2310.11511 (2023-10-17)

### 7.3 Practical posture
Both papers converge on: gate hard refusals on *evidence* (no relevant docs / low confidence), fail open on auxiliary checks, and treat web-search fallback as a corrective action rather than a default path. (CRAG §retrieval evaluator; Self-RAG adaptive retrieval.)

---

## 8. Evaluation

### 8.1 The RAG triad & component metrics
RAGAS metric catalog (current stable docs): **Context Precision, Context Recall, Context Entities Recall, Noise Sensitivity, Faithfulness, Response Relevancy** for RAG; plus Factual Correctness, semantic/string similarity, aspect critics, rubrics-based scoring for general purpose; agent metrics (tool-call accuracy/F1, goal accuracy) for agentic flows.
- Source: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ (page dated 2025-12-09)

### 8.2 Retrieval-level IR metrics
precision@k, MRR, nDCG computed against curated qrels using libraries like ranx — Qdrant's prescribed workflow before changing any search mechanism ("None of the experiments makes sense if you don't measure the quality"). nDCG@10 over BeIR qrels is also how Jina validated late chunking.
- Sources: https://qdrant.tech/articles/hybrid-search/ ; https://jina.ai/news/late-chunking-in-long-context-embedding-models

### 8.3 Golden datasets & regression gates
Every primary source above gates changes on frozen evaluation sets: Anthropic ran fixed question sets per domain with recall@k; Qdrant requires qrels-based measurement before adopting Query-API changes; Jina used frozen BeIR splits. Pattern: freeze inputs, compare candidate config vs baseline, promote only on measured win.

### 8.4 LLM-as-judge bias mitigation
MT-Bench/Chatbot Arena (NeurIPS 2023) catalogs judge biases: **position, verbosity, self-enhancement**, limited reasoning ability — with mitigations (swap positions, tie handling, fine-grained rubrics) and finds strong judges reach >80% agreement with humans (parity with human-human agreement). Implication: use LLM judges with position-swapping and verbosity controls, and validate judge-vs-human agreement on your own data.
- Source: https://arxiv.org/abs/2306.05685 (2023-06-09, v4 2023-12-24)

---

## 9. Caching

### 9.1 Exact caching
Provider-side prompt caching: Anthropic reports >2× latency reduction and up to 90% cost reduction for cached prompt prefixes; foundational to making contextual retrieval affordable ($1.02/M doc tokens) and to caching citation source documents (`cache_control` on document blocks).
- Sources: https://www.anthropic.com/news/contextual-retrieval ; https://docs.claude.com/en/docs/build-with-claude/citations

### 9.2 Semantic caching
GPTCache is the reference design: embed incoming queries → vector-store similarity search → similarity-evaluator threshold decides hit. Explicitly documented pitfalls: **false positives on hits and false negatives on misses**; monitored via hit ratio, latency, and recall; temperature parameter can probabilistically bypass cache. Eviction (LRU/LFU/FIFO/RR) and distributed backends (Redis) are separate concerns from matching.
- Source: https://github.com/zilliztech/GPTCache

### 9.3 Scoping/invalidation pitfalls
Semantic caches keyed only on query text break under: index updates (stale answers), per-user/per-tenant scoping, and changed retrieval results. GPTCache's own false-positive framing plus the need for explicit eviction policies imply cache keys must include corpus version/index generation and tenant scope — consistent with immutable-generation index patterns (build new gen → validate → atomic alias switch).

---

## 10. GraphRAG / multi-hop

GraphRAG (Microsoft) targets **global sensemaking questions** ("what are the main themes in the dataset?") that vector RAG structurally fails because they are query-focused summarization, not retrieval. Pipeline: LLM-derived entity knowledge graph → pregenerated community summaries (Leiden-style communities) → map-reduce partial answers per community → final summary. Substantial comprehensiveness/diversity wins over vector-RAG baseline on ~1M-token corpora.
- **When worth it:** global/thematic questions, multi-hop entity traversal, corpus-level analytics. Not a replacement for vector retrieval on point lookups — cost of graph construction/maintenance is significant.
- Source: https://arxiv.org/abs/2404.16130 (2024-04-24, v2 2025-02-19)

For local multi-hop without a graph, Self-Ask style decomposition (§2.3) plus iterative retrieval covers most cases at far lower build cost.

---

## 11. Observability & production

### 11.1 Tracing standards
GenAI semantic conventions have **moved out of the main OTel semconv repo** into a dedicated repo: `open-telemetry/semantic-conventions-genai` — spans, metrics, and events for GenAI clients, MCP (Model Context Protocol), and provider-specific conventions (OpenAI etc.), generated from YAML models via Weaver. Instrumentation should track this repo, not the old `/docs/specs/semconv/gen-ai/` pages (now stubs).
- Sources: https://opentelemetry.io/docs/specs/semconv/gen-ai/ (moved notice) ; https://github.com/open-telemetry/semantic-conventions-genai

### 11.2 Drift & A/B gates
Primary sources converge on: frozen-input eval harnesses run per change (Anthropic appendix methodology; Qdrant "measure before change"; Jina frozen BeIR), health-scored provider routing with cached best-provider selection (implied by production stacks; explicit in this repo's ProviderSelector), and staged rollouts with rollback (Vespa deployment variants; this repo's gen-manifest/gen-build/gen-validate/gen-activate lifecycle).

---

## Consolidated reference checklist — SOTA 2026 RAG system, by stage

Legend: **[E]** established · **[Em]** emerging · **[X]** experimental

### Ingestion & chunking
- Structure-aware splitting with tuned boundaries/overlap — **[E]**
- Hybrid dense+sparse indexing (embeddings + BM25/SPLADE) — **[E]**
- Contextual retrieval (LLM-generated chunk context, prepended pre-embed & pre-BM25) — **[E]** (strong published numbers; now widely replicated)
- Late chunking with long-context embedders — **[Em]**
- Hierarchical/parent-child storage (retrieve small, return large) — **[E]**
- Immutable index generations with validate-then-activate promotion — **[E]** (ops pattern)

### Query understanding
- Intent classification & conditional routing — **[E]**
- Query rewriting/expansion — **[E]**
- HyDE — **[Em]** (eval-gated; helps vocabulary-mismatch cases, costs an LLM call)
- Step-back abstraction prompting — **[Em]**
- Decomposition/self-ask for multi-hop — **[E]** for agentic stacks, **[Em]** otherwise

### Retrieval
- Hybrid search with **RRF fusion** (k≈60), never raw-score linear combination — **[E]**
- Metadata/payload filtering composed into retrieval — **[E]**
- Matryoshka multistage retrieval (coarse→fine dimensions) — **[Em]**
- Server-side query pipelines (fusion+rerank in the DB) — **[Em]**

### Reranking
- Cross-encoder reranker over a 100–200 candidate pool, cut to top-K (~20) — **[E]**
- True late-interaction (ColBERTv2/PLAID) for quality-critical or high-QPS rerank — **[Em]** (PLAID makes it viable; integration burden remains)
- LLM rerankers — **[X]** (cost/latency not yet justified by primary-source evidence)

### Context assembly
- Top-K sizing measured per workload (evidence point: 20 > 10 > 5) — **[E]**
- Lost-in-the-middle mitigation (best evidence at edges; avoid mid-stuffing) — **[E]**
- Deduplication after fusion; diversity/MMR where sources overlap — **[E]**
- Citation enforcement via API-level citations where available — **[E]** (provider-dependent)
- Spotlighting/data-marking of retrieved content against indirect prompt injection — **[E]** (cheap, large ASR reduction)

### Generation
- Schema-enforced structured outputs (constrained decoding) for machine-consumed answers — **[E]**
- Groundedness/faithfulness verification with fail-open posture on auxiliary checks, fail-closed only on empty/irrelevant evidence — **[E]** (pattern), **[Em]** (specific verifiers)
- Self-consistency sampling for high-stakes answers — **[Em]**

### Corrective loops
- Retrieval confidence grading (CRAG-style evaluator) with web-search fallback — **[Em]**
- Self-RAG reflection tokens (requires fine-tuning) — **[X]** for most teams
- Hard-refusal only on evidence-based grounds — **[E]** (contractual posture)

### Evaluation
- Component-level IR metrics (recall@k, MRR, nDCG@10) on golden qrels — **[E]**
- RAG triad (faithfulness / response relevancy / context precision-recall) — **[E]**
- Frozen-input regression harnesses wired into CI with promotion gates — **[E]**
- LLM-as-judge with position-swap + verbosity-bias controls, validated against human labels — **[E]** (method), continuous validation **[Em]**

### Caching
- Prompt/prefix caching (provider-side) — **[E]**
- Exact query/response caching scoped by index generation + tenant — **[E]**
- Semantic caching with explicit false-positive tolerance and hit-ratio/recall monitoring — **[Em]**

### Graph / multi-hop
- Vector RAG + decomposition as default — **[E]**
- GraphRAG for global/thematic questions — **[Em]** (worth it only for corpus-level sensemaking workloads)

### Observability & ops
- OTel GenAI semantic conventions (new dedicated repo; spans/metrics/events incl. MCP) — **[Em]** (spec actively stabilizing)
- Tracing of full request path (retrieve→rerank→generate) — **[E]**
- Health-scored provider fallback chains — **[E]** (production pattern)
- Staged rollout with eval gates + rollback — **[E]**

---

## Source index (all fetched 2026-08-21)

| # | Source | Date |
|---|--------|------|
| 1 | https://arxiv.org/abs/2005.11401 — RAG (Lewis et al.) | 2020-05-22 |
| 2 | https://www.anthropic.com/news/contextual-retrieval | 2024-09-19 |
| 3 | https://arxiv.org/abs/2407.21059 — Modular RAG | 2024-07-26 |
| 4 | https://arxiv.org/abs/2501.09136 — Agentic RAG survey | 2025-01-15 (v4 2026-04-01) |
| 5 | https://docs.vespa.ai/en/rag/rag.html | current |
| 6 | https://arxiv.org/abs/2212.10496 — HyDE | 2022-12-20 |
| 7 | https://arxiv.org/abs/2310.06117 — Step-back prompting | 2023-10-09 (ICLR 2024) |
| 8 | https://arxiv.org/abs/2210.03350 — Self-Ask | 2022-10-07 (EMNLP-F 2023) |
| 9 | https://qdrant.tech/articles/hybrid-search/ | 2024-07-25 |
| 10 | https://weaviate.io/blog/hybrid-search-explained | 2025-01-27 |
| 11 | https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html | current |
| 12 | https://jina.ai/news/late-chunking-in-long-context-embedding-models | 2024-08-22 |
| 13 | https://docs.cohere.com/docs/rerank-overview | current |
| 14 | https://arxiv.org/abs/2205.09707 — PLAID | 2022-05-19 |
| 15 | https://arxiv.org/abs/2307.03172 — Lost in the Middle | 2023-07-06 (TACL 2023) |
| 16 | https://docs.claude.com/en/docs/build-with-claude/citations | current |
| 17 | https://arxiv.org/abs/2403.14720 — Spotlighting | 2024-03-20 |
| 18 | https://openai.com/index/introducing-structured-outputs-in-the-api/ | 2024-08-06 |
| 19 | https://arxiv.org/abs/2203.11171 — Self-consistency | 2022-03-21 (ICLR 2023) |
| 20 | https://arxiv.org/abs/2401.15884 — CRAG | 2024-01-29 (v3 2024-10-07) |
| 21 | https://arxiv.org/abs/2310.11511 — Self-RAG | 2023-10-17 |
| 22 | https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ | page dated 2025-12-09 |
| 23 | https://arxiv.org/abs/2306.05685 — MT-Bench / LLM-as-judge | 2023-06-09 (NeurIPS 2023) |
| 24 | https://github.com/zilliztech/GPTCache | current |
| 25 | https://arxiv.org/abs/2404.16130 — GraphRAG | 2024-04-24 (v2 2025-02-19) |
| 26 | https://opentelemetry.io/docs/specs/semconv/gen-ai/ (moved) + https://github.com/open-telemetry/semantic-conventions-genai | current |
