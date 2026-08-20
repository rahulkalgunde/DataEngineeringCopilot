# Context Assembly & Deduplication Research

Research date: 2026-08-20. Best practices for optimizing context assembly and
deduplication in production RAG pipelines. Primary sources cited throughout.

---

## 1. Content-Hash Deduplication

### Two-Level Strategy

Production systems implement deduplication at two distinct points:

**Index-time dedup (ingestion boundary):**
- Compute a normalized content hash before embedding. If the hash already exists in the metadata store, skip insertion entirely.
- Prevents duplicates from accumulating in the index in the first place.
- Source: MuninnDB (2026-04) uses SHA-256 content hashes at O(1) write time, returning the existing engram ID with a `duplicate_content` hint. Hard-delete cleans up the hash mapping so the same content can be re-stored.

**Query-time dedup (retrieval post-processing):**
- After retrieval, remove exact duplicates and near-duplicates before context assembly.
- Source: how2.sh (2026-02) describes a two-level pipeline: Level 1 uses `hashlib` for exact dedup; Level 2 uses `datasketch.MinHashLSH` for near-duplicate detection.

### Hash Algorithm Choices

| Algorithm | Use case | Notes |
|-----------|----------|-------|
| SHA-256 | Exact content-hash dedup | Industry standard; MuninnDB, production pipelines |
| MD5 | Fast exact dedup | Acceptable for dedup (not security); sub-microsecond |
| MinHash (LSH) | Near-duplicate detection | Jaccard similarity via shingles; threshold ~0.80–0.90 |

### Near-Duplicate Handling

- **MinHash LSH**: Estimate Jaccard similarity between documents using character n-grams (shingles). Chunks with similarity above a threshold (0.80–0.90) are near-duplicates. Source: how2.sh recommends k=5 shingle size for paragraph-length chunks, k=3 for short chunks, k=7 for long documents.
- **Cosine similarity on embeddings**: Compute pairwise cosine similarity between candidate embeddings. Thresholds of 0.88–0.94 are typical. Source: TokenGate (2026-05) uses `semantic_dedup_threshold=0.88`.
- **TF-IDF cosine**: More accurate for near-duplicate detection than MinHash but requires storing TF-IDF vectors. Use when accuracy matters more than performance.

### Key Failure Mode

The silent dedup regression: if a normalization step (e.g., lowercasing) breaks during ingestion refactoring, the same content gets indexed twice with different hashes. Retrieval scores look fine because both copies are genuinely relevant. The failure only surfaces downstream when the LLM sees the same paragraph three times and treats it as triple confirmation. Source: tianpan.co (2026-06).

### Quantitative Findings

- Clean academic retrieval (BeIR): 0.16% corpus-level byte reduction from exact dedup.
- Constructed enterprise patterns (versioned docs, multi-source): 24% byte reduction.
- Multi-turn conversational: 80% byte reduction.
- In production RAG with multi-source knowledge bases: expect 10–30% of retrieved chunks to be exact or near-duplicates. Higher than 30% suggests the indexing pipeline is re-ingesting without dedup. Source: arXiv 2605.09611, how2.sh.

---

## 2. Same-Parent Sibling Collapse

### Core Pattern: Hierarchical Parent-Child

The dominant production pattern is hierarchical chunking with parent-child relationships:

1. **Child chunks** (small, 150–400 words): Indexed for precise retrieval.
2. **Parent chunks** (large, 600–2000 chars): Returned as context for generation.
3. When a child matches, the full parent is returned — providing surrounding context.

Source: parent-child retrieval pattern documented across H-RAG (SemEval-2026), Auto-Merge RAG, LlamaIndex ParentDocumentSplitter, and multiple production implementations.

### Sibling Collapse Strategies

**Auto-merge (threshold-based):**
- If multiple children from the same parent are retrieved and their coverage exceeds a threshold (e.g., ≥50% of parent's children), return the entire parent instead of individual children.
- Source: Auto-Merge RAG implementation uses `coverage >= merge_threshold and avg_score > 0.3`.

**Adjacent merge:**
- When two or more adjacent leaf chunks from the same parent are both retrieved, merge them into a single contiguous passage rather than including the overlap twice.
- Source: TypeGraph context assembly pipeline (2026-03).

**Max-score aggregation:**
- Parent-level relevance score = max score among its children. This prevents multiple children from the same parent from each claiming a top-k slot.
- Source: H-RAG (SemEval-2026 Task 8) uses maximum-score aggregation.

### Implementation Pattern

```python
def collapse_siblings(chunks: list[dict], seen_parents: set) -> list[dict]:
    """Collapse same-parent siblings into parent chunks."""
    deduplicated = []
    for chunk in chunks:
        parent_id = chunk["metadata"].get("parent_id")
        # Skip if we already have a chunk from this parent
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)
        deduplicated.append(chunk)
    return deduplicated
```

Source: Production pattern from dev.to (2026-06), adapted from multiple implementations.

### Security Consideration

Parent expansion pulls in content that was not matched by the query. Attackers can place injections adjacent to benign content that will be retrieved. Consider selective expansion and scanning expanded content for injections.

Source: ZIVIS parent-child pattern analysis.

---

## 3. MMR (Maximal Marginal Relevance)

### The Formula

```
MMR(dᵢ) = argmax [ λ · Sim(dᵢ, query) - (1-λ) · max_{dⱼ ∈ S} Sim(dᵢ, dⱼ) ]
```

- S = already-selected documents
- λ = trade-off parameter [0, 1]
- λ = 1.0: pure relevance (identical to top-k)
- λ = 0.5: balanced (recommended starting point)
- λ = 0.0: pure diversity

### Lambda Tuning Guide

| Lambda | Behavior | Use case |
|--------|----------|----------|
| 0.7–0.8 | Mostly relevance, slight diversity | Technical docs, precise factual lookups |
| 0.5 | Balanced | General RAG, most use cases |
| 0.3 | Strongly diverse | Exploration, topic coverage |

### Practical Recipe

1. Fetch 4× the desired final count (e.g., fetch_k=20 for top_k=5).
2. Run MMR selection with λ=0.5.
3. MMR is greedy: O(k × fetch_k) pairwise comparisons.

Source: Learnixo (2026), Grafeo MMR Search, multiple LangChain/LlamaIndex implementations.

### Vector Similarity vs. Lexical Overlap

- **Embedding cosine** is the standard Sim function for MMR. Fast, works well for semantic diversity.
- **Lexical overlap (Jaccard, TF-IDF)** can complement embedding similarity for detecting surface-level duplicates that embeddings treat as distinct.
- **AdaGReS** (2025) generalizes MMR with a set-level objective: `F(q, C) = Σ sim(q, cᵢ) - β Σᵢ<j sim(cᵢ, cⱼ)`. Provides near-optimality guarantees under submodularity conditions.

### Advanced: Dynamic Lambda

```python
def get_dynamic_lambda(query: str) -> float:
    if "specific" in query.lower() or "what is" in query.lower():
        return 0.8  # Favor precision
    elif "overview" in query.lower() or "tell me about" in query.lower():
        return 0.4  # Favor diversity
    else:
        return 0.6  # Balanced default
```

Source: nickberens.me MMR guide.

### When to Skip MMR

- Knowledge base is well-curated with minimal redundancy.
- Query requires the most precise single answer.
- Latency is critical (MMR adds O(k × fetch_k) overhead).

---

## 4. Lost-in-the-Middle / Prompt Packing

### The Phenomenon

LLMs exhibit U-shaped attention: high attention at beginning and end of context, low in the middle. Same model, same question, same supporting material — only the position changed, and accuracy drops in the middle.

Source: Liu et al. (TACL 2024) "Lost in the Middle: How Language Models Use Long Contexts."

### 2026 Reproduction Findings

- The effect has softened with newer models but NOT disappeared.
- On single-hop NQ, performance is comparatively insensitive to order.
- On multi-hop HotpotQA, position matters significantly — larger contexts help mainly when high-value evidence is in high-attention positions.
- Frontiers models (2025–2026) show flatter curves than 2023 generation, but the bias persists.
- Source: Gabín et al. (SIGIR 2026) "Lost in the Evidence?", InContext.info.

### Attention Basin (ACL 2026)

A consistent phenomenon called the "attention basin" was documented: models systematically assign higher attention to items at the beginning and end of structured sequences. Key insight: allocating higher attention to critical information enhances performance.

Source: Yi et al. (ACL 2026) "Attention Basin: Why Contextual Position Matters in Large Language Models."

### Practical Positioning Strategy

**The Sandwich Pattern:**
1. **Top**: System instruction + most relevant chunk
2. **Middle**: Supporting chunks (ordered by relevance)
3. **Bottom**: Second most relevant chunk (recency bias)

```
[System prompt]
[Most relevant chunk — primacy bias]
[Chunk 3]
[Chunk 4]
[Chunk 5]
[Second most relevant — recency bias]
```

Source: InContext.info, multiple production guides.

### ScoreSpread Reordering

Interleave chunks by priority score — highest at start and end, lowest in the middle. This exploits the U-shaped attention curve.

```typescript
// From rag-chunk-reorder library
const reorderer = new Reorderer({ strategy: 'scoreSpread', topK: 8 });
const reordered = reorderer.reorderSync(chunks);
// High-relevance chunks land at positions 0 and N-1
```

Source: Mayureshju/rag-chunk-reorder library.

### OP-RAG (Document Order Preservation)

Maintains original document order within each source. Groups chunks by `sourceId`, sorts by `sectionIndex` within each group, and orders groups by highest relevance score. Preserves narrative coherence.

Source: rag-chunk-reorder library, TypeGraph (2026-03).

### Key Rule

"Always put your highest-scored chunks at the start and the end of the context, and always measure recall on your own eval set — not on MTEB, not on BEIR, not on marketing slides."

Source: Community consensus from r/MachineLearning, r/LocalLLaMA (2026).

---

## 5. Metadata Breadcrumbs

### Minimal Viable Metadata Format

Every chunk must carry:

```json
{
    "id": "doc-447::parent::3::child::1",
    "text": "The refund policy applies to purchases made within 30 days...",
    "source_document_id": "policy-v4.2",
    "source_url": "https://docs.internal/policy/refunds",
    "document_version": "4.2",
    "section": "Refund Policy",
    "last_updated": "2026-01-15",
    "content_hash": "sha256:abc123...",
    "embedding_model": "text-embedding-3-large",
    "embedding_model_version": "2024-02"
}
```

Source: Production checklist from dev.to (2026-06).

### Header Breadcrumb Pattern

Each chunk carries an in-document-order breadcrumb with markdown-level prefix. The H1 document title is injected into every chunk so downstream embeddings always have document-level context.

```
# DataEngineeringCopilot > ## RAG Pipeline > ### Context Assembly
```

Source: structchunk library, PrimeCut (POMA AI).

### Chunkset Pattern (Root-to-Leaf)

A chunkset is a root-to-leaf path through the document hierarchy — the leaf sentences plus every ancestor breadcrumb. Embed and retrieve at the chunkset level for self-explanatory context.

Source: POMA AI chunksets documentation.

### What Metadata Enables

| Metadata field | Purpose |
|---------------|---------|
| `parent_id` | Same-parent sibling collapse |
| `content_hash` | Exact dedup + staleness detection |
| `section` / `header_path` | Structural breadcrumbs for LLM |
| `document_version` | Version-aware retrieval |
| `source_document_id` | Source attribution |
| `embedding_model_version` | Re-embedding triggers |

---

## 6. Evaluation Metrics

### Context Assembly Quality Metrics

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Duplicate rate** | % of context window spent on duplicates | Count duplicate chunks / total chunks |
| **Source coverage** | Fraction of relevant sources represented | Required sources found / total required |
| **Compression ratio** | Tokens in candidate pool vs. final context | final_tokens / candidate_tokens |
| **Groundedness** | % of answer claims supported by context | LLM-judge or NLI entailment |
| **Needle-loss** | Key fact present in context but not in answer | Manual or LLM-judge inspection |

### Context Precision & Recall (CRUX Framework)

- **Coverage (Cov)**: Measures content of retrieval context based on answerability of sub-questions. More reflective than rank-based metrics for long-form RAG.
- **Ranked Coverage (α-nDCG)**: Coverage-aware novelty ranking metric.
- **Density**: Coverage normalized by redundancy — penalizes context bloat.

Source: CRUX framework (EMNLP 2025 Findings).

### RagChecker Fine-Grained Metrics

- **Retriever**: claim recall (how many ground-truth claims are in retrieved chunks), context precision (fraction of retrieved chunks that are relevant).
- **Generator**: faithfulness (% of answer claims entailed by context), noise sensitivity (incorrect claims entailed by irrelevant chunks), hallucination (incorrect claims not in any chunk), context utilization (% of available evidence actually used).

Source: RagChecker (2024).

### RA-nWG@K (Rarity-Aware Set Utility)

Order-free, per-query normalized metric that aggregates per-passage utilities, scales mid-grades by inverse prevalence, and normalizes by oracle score. Paired with N-Recall4+@K (coverage of high-utility evidence).

Source: arXiv 2511.09545 (2025).

### Production Monitoring Signals

| Stage | Signal | Alert pattern |
|-------|--------|---------------|
| Context | Duplicate ratio, truncation rate, token distribution | Growing duplicate ratio, required facet loss |
| Generation | Abstention rate, unsupported claims | Refusal spike, hallucination spike |

Source: RAG QA Testing Guide (2026-06).

---

## 7. Existing Libraries

### Context Assembly & Optimization

| Library | Key features | Notes |
|---------|-------------|-------|
| **TokenGate** | Exact dedup → embed → hybrid rank → rerank → adaptive cutoff → semantic dedup → MMR → token budget → prompt build. Full audit trail. | Python 3.12, BGE-M3 defaults, no lossy compression by default. 71% token reduction in benchmarks. |
| **context-engine** | Retrieval → re-ranking → memory decay → compression → token-budget enforcement. Slot-based budget (system → history → docs). | Pure Python, ~92ms on CPU. Extractive compression. |
| **rag-chunk-reorder** | ScoreSpread, PreserveOrder, Chronological, Auto strategies. Deduplication + token budget. | TypeScript. Score normalization, fuzzy dedup. |

### MMR Implementations

| Library | Approach | Notes |
|---------|----------|-------|
| **LangChain** | `search_type="mmr"` on any vector store. `fetch_k`, `lambda_mult` params. | Most widely used. |
| **LlamaIndex** | `vector_store_query_mode="mmr"` with `mmr_threshold`. | Built into SimpleVectorStore. |
| **Grafeo** | `mmr_search()` with `lambda_mult`, `fetch_k`, property filters. | PostgreSQL-based. |

### Deduplication

| Library | Approach | Notes |
|---------|----------|-------|
| **datasketch** | MinHash + MinHashLSH for near-duplicate detection. | Jaccard similarity estimation. Production-proven. |
| **hashlib** (stdlib) | SHA-256 / MD5 for exact dedup. | Sub-microsecond. Always use as first pass. |
| **sentence-transformers** | Cosine similarity on embeddings for semantic dedup. | Threshold 0.88–0.94 typical. |

### Chunking with Structure

| Library | Approach | Notes |
|---------|----------|-------|
| **structchunk** | Hierarchical + linear algorithms. Header breadcrumbs, Snowflake IDs, sibling merge. | Pure Python, zero deps. |
| **chunkana** | Markdown-aware. header_path, content_type, hierarchy support. | Never breaks code/tables/lists. |
| **LightRAG** | Paragraph semantic (P) chunking. Heading-aware with hierarchy-aware merging. | Production-ready, EMNLP 2025. |

### Full-Stack RAG Frameworks (with context assembly)

| Framework | Context assembly features |
|-----------|------------------------|
| **LightRAG** | Graph-based retrieval + vector hybrid. Reranker support. BM25 + RRF fusion. |
| **VORTEXRAG** | 7-layer pipeline: tri-vector encoding → vortex retrieval → drift correction → poison guard → rank fusion → causal context builder → faithfulness verifier. |
| **MacRAG** | Multi-scale adaptive: hierarchical indexing + bottom-up query-time expansion. Scaled top-chunk selection with α factor. |

---

## Recommended Pipeline Order

Based on the research, a production context assembly pipeline should execute these stages in order:

1. **Neighbor joining**: Expand retrieved chunks with adjacent chunks from the same document. Merge overlapping expansions.
2. **Deduplication**: Exact (hash) → Near-duplicate (embedding cosine or MinHash). Remove subsumed chunks.
3. **Re-ranking**: Cross-encoder or LLM-based relevance scoring.
4. **Adaptive cutoff**: Keep chunks above a per-query relevance threshold.
5. **MMR diversity**: Diversity-aware greedy selection (λ=0.5 default).
6. **Token budget fitting**: Greedy fill within token budget. Truncate at sentence boundaries.
7. **Position ordering**: ScoreSpread or sandwich ordering — highest at start/end, lowest in middle.
8. **Formatting**: Wrap each chunk in XML tags or markdown headers with source metadata.
9. **Audit**: Log per-chunk decisions, token math, and stage-level drop-off.

Source: Synthesized from TypeGraph (2026-03), TokenGate (2026-05), and multiple production implementations.

---

## Sources

- Liu et al. (TACL 2024) "Lost in the Middle: How Language Models Use Long Contexts"
- Yi et al. (ACL 2026) "Attention Basin: Why Contextual Position Matters in Large Language Models"
- Gabín et al. (SIGIR 2026) "Lost in the Evidence? Reproducing Document Position and Context Size Effects in RAG"
- Hsieh et al. (2024) "Found in the Middle: Calibrating Positional Attention Bias"
- arXiv 2605.09611 (2026-05) "Byte-Exact Deduplication in Retrieval-Augmented Generation"
- arXiv 2512.25052 (2025) "AdaGReS: Adaptive Greedy Context Selection via Redundancy-Aware Scoring"
- CRUX (EMNLP 2025 Findings) "Controlled Retrieval-augmented Context Evaluation"
- RagChecker (2024) "A Fine-grained Framework for Diagnosing RAG"
- arXiv 2511.09545 (2025) "Practical RAG Evaluation: A Rarity-Aware Set-Based Metric"
- H-RAG (SemEval-2026 Task 8) "Hierarchical Parent–Child Retrieval"
- MacRAG (arXiv 2505.06569v2) "Multi-scale Adaptive Context RAG"
- MuninnDB (2026-04) "content-hash deduplication at write time"
- tianpan.co (2026-06) "The RAG Dedup Step That Broke Silently"
- how2.sh (2026-02) "How to Add Retrieval Deduplication to RAG Pipelines"
- dev.to (2026-06) "Building Reliable RAG Pipelines: From Prototype to Production"
- TypeGraph (2026-03) "Context Window Assembly Strategies"
- TokenGate (2026-05) github.com/Mario-Vishal/tokengate
- context-engine (2026-04) github.com/Emmimal/context-engine
- rag-chunk-reorder (2026) github.com/Mayureshju/rag-chunk-reorder
- structchunk github.com/yzp0111/structchunk
- chunkana github.com/asukhodko/chunkana
- POMA AI "PrimeCut — POMA's RAG Ingestion Engine"
- LightRAG (EMNLP 2025) github.com/HKUDS/LightRAG
- VORTEXRAG (2025) github.com/vignesh2027/VORTEXRAG
- Auto-Merge RAG (2023) michaeljohnpena.com
- Parent-Child Retrieval (2023) michaeljohnpena.com
- ZIVIS "Parent-Child Chunking — RAG & Retrieval Pattern"
- RAG QA Testing Guide (2026-06) qaskills.sh
- Ranjan Kumar (2026-05) "Why Your RAG Pipeline Assembles Context Wrong"
- nickberens.me "Maximum Marginal Relevance in RAG"
- Grafeo "MMR Search"
- Learnixo "MMR: Maximum Marginal Relevance for Diversity"
