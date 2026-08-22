# Deferred Experiments Research — Late Chunking, Matryoshka, True ColBERT, Streaming JSON Repair

**Compiled:** 2026-08-21 · **Method:** primary sources fetched directly (Jina engineering blog + arXiv, Qdrant official docs/course, PyLate repo/docs/issues, json-repair repo README).
**Feeds:** `plans/2026-08-21_23-40_deferred_experiments_plan.md`

---

## 1. Late chunking

**Sources:** https://jina.ai/news/late-chunking-in-long-context-embedding-models (2024-08-22) · https://arxiv.org/abs/2409.04701 · jina-embeddings-v3 API exposes `late_chunking=True` (Qdrant Jina docs page)

**Algorithm.** Run the transformer stack of a long-context embedding model over the *entire* document → per-token vectors conditioned on the whole text → mean-pool token vectors per precomputed chunk span. Boundary cues are applied *after* encoding (hence "late"). Result: each chunk embedding is contextualized by its surroundings; naive chunking produces i.i.d. embeddings that lose anaphora/entity links ("the city" ↔ "Berlin" example: sim 0.708→0.825).

**Evidence (BeIR nDCG@10, jina-embeddings-v2-small-en, ~256-token chunks):**

| Dataset | Naive | Late |
|---|---|---|
| SciFact | 64.20 | **66.10** |
| TRECCOVID | 63.36 | **64.70** |
| FiQA2018 | 33.25 | **33.84** |
| NFCorpus | 23.46 | **29.98** |

Gains correlate with average document length — the longer the doc, the bigger the win.

**Implementation with sentence-transformers.**
- `model.encode(doc, output_value="token_embeddings")` yields per-token vectors.
- Map char spans → token indices via `model.tokenizer(text, return_offsets_mapping=True)`; pool only tokens whose char span overlaps the chunk span, excluding special/padding tokens; L2-normalize after pooling.
- Hard constraint: the whole document must fit the model context window (8192 for jina v2/v3 and Ollama's nomic-embed-text). Longer documents must fall back to naive per-chunk embedding — truncating mid-document silently degrades tail chunks.
- Our local `nvidia/Nemotron-3-Embed-1B-BF16` is dual-mode (query/passage prefixes); its context limit and query-prefix behavior for full-doc encoding must be probed before use (`dec probe`-style one-off script, not live paid calls).

### Adoption recipe (our stack)
- New `infrastructure/late_chunking.py::LateChunkEmbedder` wrapping `LocalSentenceTransformerEmbeddings`; protocol-compatible `embed_chunks(document_text, spans)`.
- Wire into `pinned_index_builder` + `claude_docs_ingestion` flush paths behind `late_chunking_enabled=False` (dark) — offline generations mean zero risk to the live alias.
- Gate: `make eval-fast` embedding sanity pairs still pass (relevant > irrelevant) AND `dec eval-retrieval --compare-baseline` Recall@10 ≥ baseline − 0.01, with expected gains concentrated on long documents.

---

## 2. Matryoshka (MRL) multistage retrieval

**Sources:** MRL paper https://arxiv.org/abs/2205.13147 · Qdrant Matryoshka docs https://qdrant.tech/documentation/inference/matryoshka-models/ · Qdrant multi-stage course module https://qdrant.tech/course/multi-vector-search/module-3/multi-stage-retrieval/ · nomic-embed-text-v1.5 model card (MRL dims 64–768) · Qdrant Gemini-embedding blog (MRL dims table pattern)

**Mechanism.** MRL-trained models concentrate semantics in the first k dimensions: slice the prefix and renormalize → a smaller vector usable as a first-stage index. Two-stage pattern: prefetch with the small named vector at high oversampling (e.g. 1000 candidates), then rescore survivors with the full-size vector. Qdrant Cloud's inference service can produce both sizes server-side (`mrl` option); self-hosted stacks store two named vectors ("small"/"large") computed client-side from ONE inference call (slice, don't re-run).

**Evidence.** jina-v3 retrieval nDCG@10 by dim: 256d = 62.72 vs 1024d = 63.35 (−0.6% at 4× compression); ≥128d degrades more. Qdrant course: oversampling factor is the quality knob; filters propagate automatically to prefetch stages.

**Model fit for us.** Ollama `nomic-embed-text` (= v1.5 lineage, 768d) is MRL-trained (dims 64–768 safe). `Nemotron-3-Embed-1B` MRL status unverified → treat as non-MRL until probed. Cloud embedders: no control over truncation contract → out of scope.

### Adoption recipe (our stack)
- During generation build: compute full dense vector once, store `dense_small` = renormalized first-K prefix (K=256 default) as a second named vector on every point.
- Query path in `AsyncQdrantVectorStore.query`: when `mrl_multistage_enabled`, wrap the dense prefetch in an outer prefetch — small-dim ANN with oversampled limit (e.g. 4× fused_limit) feeding the existing full-dim + RRF pipeline. Composes with hybrid/BM25 since prefetch chains nest.
- Gate: eval-retrieval Recall@10 within −0.01 of baseline while p95 retrieval latency improves ≥20% on the benchmark harness; else keep dark.

---

## 3. True ColBERT / PLAID rerank-only usage

**Sources:** ColBERTv2 https://arxiv.org/abs/2112.01488 · PLAID https://arxiv.org/abs/2205.09707 (7× GPU / 45× CPU latency cuts at 140M-passage scale) · PyLate https://github.com/lightonai/pylate + docs https://lightonai.github.io/pylate/api/models/ColBERT/ · Qdrant multistage course (multivector config) · CPU caveat: pylate issue #125 (2025-05-15, GTE-ModernColBERT)

**Rerank-only viability (no corpus indexing).** MaxSim needs only query token embeddings × doc token embeddings — no index required. PyLate gives exactly this:
```python
model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0", device="cpu")
q_emb   = model.encode([query], is_query=True)
d_embs  = model.encode(docs, is_query=False)
ranked  = rank.rerank(documents_ids=ids, queries_embeddings=q_emb, documents_embeddings=d_embs)
```
Cost driver is encoding the ~200-doc pool per query (token embeddings, 128-dim typical; colbertv2 ≈110M params). PLAID-class optimizations matter for corpus-scale indexing, NOT for pool reranking.

**Storage path (deferred).** If we ever index multivectors in Qdrant: `VectorParams(size=128, distance=DOT, multivector_config=MultiVectorConfig(comparator=MAX_SIM), hnsw_config=HnswConfigDiff(m=0))` — m=0 is mandatory for rerank-only spaces or indexing cost explodes.

**CPU caveats.** Models trained with flash-attention query expansion return garbage on CPU (pylate #125, GTE-specific) — colbertv2.0 is the safe choice; truncate docs (~256 tokens) like our other rerankers; expect slower-than-cross-encoder per-pool latency, which the gate must budget.

### Adoption recipe (our stack)
- Optional dependency extra `[colbert]` → pylate; new `services/pylate_colbert_reranker.py::PyLateColBERTReranker` implementing the existing reranker protocol (lazy load off-loop like `CrossEncoderReranker`, sigmoid-free min-max normalization already shared).
- `reranker_type="pylate_colbert"` setting value; factory branch alongside existing types.
- Gate: `dec eval-rerank` on frozen pools — nDCG@10 gain ≥ +0.02 vs cross-encoder AND p95 pool latency ≤ 2× cross-encoder; else keep off.

---

## 4. Streaming structured-output repair

**Sources:** json-repair repo https://github.com/mangiucugna/json_repair (MIT, 5.1k★, pinned `json-repair==0.*`) · OpenAI structured outputs GA post (constrained decoding ⇒ conformant output except refusal/truncation) — previously fetched in rag_best_practices_reference_2026-08-21.md §6.1

**State of the art.**
1. Schema-constrained streams are *guaranteed conformant* on strict providers — parse-at-end suffices there; failure cases come from degraded-fallback providers (Ollama raw mode, free-tier gateways) that ignore `response_format`.
2. For those, repair-at-end beats re-generation: `json_repair.loads(text)` fixes missing quotes/brackets/truncation and tries stdlib JSON first (no cost when valid). It also supports **schema-guided repair** (`schema=` param, beta): fills missing fields, coerces scalars, drops disallowed properties, raises if unsalvageable.
3. Incremental rendering (`stream_stable=True`) exists but is irrelevant to us: SSE consumers already receive raw tokens; the fix belongs in the terminal `done` event.

**Our actual gap.** `answer_stream()` streams raw tokens and its done event carries the raw text — for doc intents served by degraded-fallback providers this can be the raw JSON envelope. `chat_stream()` already unwraps envelopes (`_clean_chat_text`); the ask/stream path does not, and nothing applies the JSON-retry parity that `answer()` has.

### Adoption recipe (our stack)
- Add `json-repair==0.*` to dependencies.
- In `answer_stream()`: after generation completes, run the SAME envelope pipeline as `answer()` — `parse_structured_rag_response`; on parse failure, `json_repair.loads(full_text)` then re-parse; cleaned answer + `repaired: true/false` go into the `done` event (tokens already streamed stay raw — documented behavior).
- No eval gate needed (behavioral parity fix, hermetically testable); add regression tests for envelope, fenced-JSON, prose-wrapped, and truncated outputs.

---

## Cross-cutting sequencing note

Phases are independent. A (late chunking) and B (MRL) both change what gets stored per point → both belong in the next generation build; C (ColBERT rerank) and D (streaming repair) are query/generation-side only and can land anytime. Recommended order: D (pure win, no gate) → A → B → C (C has the highest latency risk on CPU).
