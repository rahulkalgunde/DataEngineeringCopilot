# RAG Pipeline Optimization Research

Research date: 2026-08-18. This evaluates the proposed Dynamic Routing and
Advanced Parsing Architecture against the current repository. Primary sources
were used for parser, HyDE, and Qdrant behavior. No application code was
changed.

## Executive Decision

The proposal is directionally sound, but several parts describe capabilities
that already exist or recommend replacing working components too broadly.

Recommended order:

1. Add namespace-aware BM25 terms and evaluate them on identifier-heavy queries.
2. Make HyDE selective by intent and query signals while retaining the original
   query, which the current pipeline already does correctly.
3. Add a lightweight chunker router at the ingestion boundary, but route by
   existing document metadata and source family before adding new classifiers.
4. Benchmark Markdown AST parsing and add parser golden fixtures before
   replacing the regex-based parser.
5. Extend code-aware splitting beyond Python only if the corpus and evaluation
   data justify Tree-sitter's dependency and maintenance cost.
6. Evaluate weighted RRF after BM25 and chunking changes; do not introduce raw
   dense/BM25 alpha mixing without score normalization and held-out evaluation.
7. Do not replace current context deduplication with embedding cosine alone.

## Current Implementation

| Area | Existing behavior | Gap relative to proposal |
|---|---|---|
| Generic routing | `factory.build_chunker()` selects one configured strategy: `semantic`, `header_aware`, `fixed_size`, or `sentence_preserving`. Spark and pinned index builders have separate source-aware paths. | No reusable `ChunkerRouter` dispatching per file or MIME type in the generic ingestion service. |
| Markdown parsing | `NativeDocumentParser` preserves Markdown/RST/code text and sections with regexes. `HeaderAwareChunker` preserves heading paths, fenced code metadata, heading-less paragraphs, and navigation-stub filtering. | No Markdown AST. Only selected heading levels are section boundaries in the native parser. |
| HTML conversion | `html_to_markdown.py` already uses BeautifulSoup, removes script/style/nav/footer/header/aside, prefers `main`/`article`/`body`, uses ATX headings, and cleans whitespace. | No explicit table/fenced-code options or parser golden corpus. Conversion fidelity is not measured. |
| Code splitting | `CodeBlockParser` uses Python `ast` for top-level `def`/`class`, regex fallback, scope prefixes, and code-size splitting. `SparkChunker` has Python top-level splitting and metadata-aware API/code paths. | Scala/Java/SQL/R do not have equivalent AST/tree-sitter boundaries. |
| Semantic splitting | `SemanticChunker` sentence-tokenizes with NLTK and clusters sentence embeddings. It does not mask code or inline code. | Code-safe sentence tokenization is a real gap, but semantic chunking is not the active pinned hierarchical path. |
| Hierarchical splitting | `hierarchical_chunk()` creates bounded parents and children, preserves reconstruction, and now merges whitespace-only pieces. | The proposed generic semantic/hierarchical fallback should be routed by source type rather than globally replacing existing chunkers. |
| BM25 | Regex is `[A-Za-z0-9_\\-]{2,}`, followed by lowercase, stopword removal, and Porter stemming. Vocabulary is frozen and persisted per index. | Dots and slashes are dropped; full identifiers and subterms are not both emitted. Unseen query terms disappear after freeze. |
| Intent | Rule-based intents include `api_lookup`, `code_example`, `comparative`, `debugging`, `how_to`, and `factual`, with optional LLM fallback. | The proposal's buckets mostly exist, but intent is not currently used to gate HyDE. |
| HyDE | HyDE is generated when enabled and appended as an additional retrieval query. The original query remains first, so HyDE does not replace the real query. | No intent/query-signal policy; every enabled query can incur HyDE latency and provider cost. |
| Hybrid retrieval | Qdrant dense+sparse prefetch with native RRF (`hybrid_rrf_k`, default 60), wider fused pools, optional metadata filters, and query-time BM25. | No weighted RRF or intent-dependent sparse bias. |
| Reranking/dedup | Local cross-encoder and optional cloud fallback exist. `ContextAssembler` collapses hierarchical siblings and uses >70% word overlap; `ContextCompressor` uses Jaccard; `reranker.py` already contains lexical MMR. | The proposed embedding-cosine dedup is not implemented in context assembly, but replacing lexical dedup wholesale would add cost and can remove complementary evidence. |
| Embedding cache | `CachedEmbedder` is wired when enabled. | Cached retrieval embeddings could support a measured cosine-diversity experiment, but the current retrieval result objects do not carry candidate embeddings. |

## Proposal Evaluation

### Phase 0: Dynamic Ingestion Router

**Verdict: worthwhile, but implement narrowly.**

The current system already has source-aware dispatch in `AsyncIngestionService`:
Spark documents use `SparkChunker` when `doc_type` is set; otherwise the
configured chunker is used. Separate pinned and rendered builders also choose
their chunking paths explicitly. A new generic router should therefore be an
adapter around existing strategies, not a second independent ingestion system.

Recommended design:

- Route by canonical `doc_type`, language, and source family first.
- Use extension/MIME only as fallback metadata.
- Register existing chunkers through a protocol that accepts
  `ParsedDocument`, not the proposal's untyped `Dict[str, Any]` payload.
- Make an unknown type fail closed to the existing deterministic chunker.
- Preserve chunk IDs, content hashes, generation metadata, and parent-child
  reconstruction contracts.

Do not add an LLM classifier. File metadata is cheaper, deterministic, and
adequate for the proposed Python/Markdown/structured/general categories.

### Phase 1: HTML and Markdown

#### HTML conversion

**Verdict: modify and test; do not blindly replace.**

The repository already implements most DOM scrubbing from the proposal. The
missing work is explicit conversion configuration and fidelity tests. The
`markdownify` upstream documentation supports heading styles, stripping,
custom converters, code-language handling, and table-related options, but HTML
conversion is not lossless. BeautifulSoup's own documentation notes that
different parsers produce different trees for malformed HTML.

Priority tests should cover:

- HTML tables, nested tables, and table headers.
- Fenced code with language classes.
- Links, images, inline code, and escaped Markdown.
- `main`/`article` selection and pages without either element.
- Malformed HTML under the chosen parser.
- Navigation/sidebar content that must not enter retrieval text.

The proposal's `extras=["tables", "fenced-code-blocks"]` is not a drop-in
guarantee for the installed `markdownify` API. Verify the package version and
use its supported options rather than copying configuration from another
Markdown library.

#### Markdown AST chunking

**Verdict: benchmark selectively.**

`HeaderAwareChunker` already handles headings, heading-less paragraphs,
fences, navigation stubs, heading paths, and minimum-size preservation. An AST
would improve explicit node boundaries for lists, tables, block quotes, and
HTML blocks, but it adds parser behavior and dependency surface.

`markdown-it-py` documents block tokens, nesting, source line maps, and a
`SyntaxTreeNode`; that is a better fit for this proposal than introducing
Mistune solely because it is called an AST parser. The correct experiment is
an AST-backed section extractor behind the existing chunker interface, with
byte-exact reconstruction tests and retrieval comparisons against the current
regex implementation.

Do not replace the existing parser before measuring:

- section count and empty-section rate,
- code/table preservation,
- reconstruction fidelity,
- source recall and MRR,
- indexing time and memory.

#### Semantic code masking

**Verdict: worthwhile if semantic chunking remains enabled.**

`SemanticChunker` currently sends raw document text to NLTK sentence
tokenization. It has no code-fence or inline-code masking. A masking layer is
appropriate, but UUID placeholders are more complexity than needed: use a
collision-resistant sentinel map scoped to one document, preserve exact spans,
and add tests for nested backticks, fenced languages, signatures, and malformed
fences.

This should not be the first optimization for the active pinned hierarchy,
which uses deterministic parent/child chunking. It is a targeted fix for the
semantic strategy.

### Phase 2: Retrieval

#### Namespace-aware BM25

**Verdict: highest-value proposal; implement behind an index-generation change.**

The current tokenizer drops dots and slashes, stems terms, and silently drops
query terms absent from the frozen vocabulary. This is a direct weakness for
`pyspark.sql.functions`, import paths, URLs, package coordinates, versions,
and method names.

Use dual representation rather than replacing the current token:

- full identifier: `pyspark.sql.functions.filter`,
- safe components: `pyspark`, `sql`, `functions`, `filter`,
- preserve hyphenated/version-like forms where useful,
- avoid unrestricted punctuation splitting that creates noisy tokens.

Required evaluation cases: dotted Python/Java/Scala names, slash paths, hyphen
versions, `FooBar`/case behavior, SQL identifiers, and queries containing a
term absent from the original frozen vocabulary.

Changing token IDs or token normalization requires rebuilding the BM25
vocabulary and sparse vectors for the generation. Do not mix old sparse vectors
with a new tokenizer.

#### Conditional HyDE

**Verdict: implement selectively.**

The current pipeline already preserves the original query and adds HyDE as a
separate variant. That is the correct safety property. The gap is that HyDE is
generated whenever enabled, including identifier-heavy API/debugging queries
where invented parameters can move dense retrieval away from exact evidence.

Recommended initial policy:

- Disable HyDE for `api_lookup` and queries containing exact dotted
  identifiers, version constraints, stack traces, or code fences.
- Disable or sample-test HyDE for `debugging` and `code_example` rather than
  assuming all such queries should bypass it.
- Keep HyDE for broad factual/how-to queries where vocabulary mismatch is the
  dominant failure mode.
- Always retain original-query retrieval and compare per-intent recall/MRR,
  answer groundedness, latency, and LLM cost.

The original HyDE paper supports hypothetical-document retrieval but does not
establish that intent-gated routing is universally better. Treat routing as an
experiment, not a theorem.

### Phase 3: Reranking and Context

#### Semantic deduplication

**Verdict: do not replace current deduplication wholesale.**

The repository already has three relevant mechanisms: content hashes in the
index payload, hierarchical sibling collapse, and lexical MMR/Jaccard
deduplication. The proposal correctly identifies that Jaccard is a weak proxy,
but embedding cosine over cached candidate vectors requires plumbing candidate
embeddings through retrieval and increases memory/latency.

Prefer this sequence:

1. Keep deterministic content-hash and same-parent deduplication.
2. Evaluate Qdrant MMR or local MMR on the reranker candidate pool.
3. Add embedding cosine only as a measured tie-breaker or second-stage filter.
4. Never deduplicate solely on cosine without source, parent, and chunk-type
   safeguards; complementary chunks can be semantically similar but necessary.

#### Weighted hybrid search

**Verdict: worthwhile after BM25 changes, evaluation required.**

The current Qdrant path already uses native dense+sparse RRF and a wider fused
pool. Qdrant documents weighted RRF and DBSF, and recommends tuning weights on
validation data. This directly supports a small sparse bias for identifier-heavy
queries.

Recommended implementation:

- Add a query feature detector for dotted identifiers, paths, versions, SQL
  keywords, and code syntax.
- Select between equal-weight RRF and a modest sparse-weighted RRF policy.
- Keep `hybrid_rrf_k` and weights configurable per generation/configuration.
- Evaluate source recall, MRR, exact API retrieval, and answer groundedness.
- Avoid raw dense/BM25 alpha mixing until scores are normalized; RRF/DBSF is
  safer with heterogeneous score scales.

## Prioritized Backlog

| Priority | Work | Why |
|---|---|---|
| P0 | Namespace-aware BM25 dual tokens + generation rebuild/eval | Largest direct gap for technical identifiers; low architectural risk. |
| P0 | Intent-gated HyDE experiment retaining original retrieval | Reduces unnecessary LLM calls and hallucinated dense queries without removing current safeguards. |
| P1 | Metadata-based chunker router adapter | Makes heterogeneous ingestion explicit while reusing existing chunkers. |
| P1 | Markdown/HTML golden corpus and AST parser benchmark | Measures whether AST parsing solves real failures before a broad replacement. |
| P1 | Semantic chunker code masking | Important only for semantic strategy; isolated and testable. |
| P2 | Weighted RRF experiment | Valuable after sparse-token changes; requires held-out tuning. |
| P2 | Candidate-vector cosine dedup experiment | More expensive and less urgent because MMR and sibling dedup already exist. |
| Defer | Immediate Mistune replacement, unrestricted Tree-sitter rollout, raw alpha mixing | High churn or tuning risk without evidence of a current failure. |

## Primary Sources

- [markdown-it-py usage and syntax trees](https://markdown-it-py.readthedocs.io/en/latest/using.html)
- [Mistune upstream repository](https://github.com/lepture/mistune)
- [BeautifulSoup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [markdownify upstream README](https://github.com/matthewwithanm/python-markdownify)
- [Python `ast` documentation](https://docs.python.org/3/library/ast.html)
- [Tree-sitter documentation](https://tree-sitter.github.io/tree-sitter/)
- [Original HyDE paper](https://arxiv.org/abs/2212.10496)
- [Qdrant hybrid queries and weighted RRF](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Qdrant BM25/sparse vectors](https://qdrant.tech/documentation/inference/inference-bm25/)
- [Qdrant points and upsert identity](https://qdrant.tech/documentation/manage-data/points/)
- [Qdrant search relevance and MMR](https://qdrant.tech/documentation/search/search-relevance/)

## Inspected Repository Touchpoints

- `data_engineering_copilot/factory.py`
- `data_engineering_copilot/services/async_ingestion.py`
- `data_engineering_copilot/services/header_aware_chunker.py`
- `data_engineering_copilot/services/semantic_chunker.py`
- `data_engineering_copilot/services/code_block_parser.py`
- `data_engineering_copilot/services/spark_chunker.py`
- `data_engineering_copilot/services/hierarchical_chunker.py`
- `data_engineering_copilot/infrastructure/native_document_parser.py`
- `data_engineering_copilot/infrastructure/html_to_markdown.py`
- `data_engineering_copilot/infrastructure/bm25_tokenizer.py`
- `data_engineering_copilot/services/query_rewriting.py`
- `data_engineering_copilot/services/async_rag.py`
- `data_engineering_copilot/infrastructure/async_qdrant_store.py`
- `data_engineering_copilot/services/context_assembler.py`
- `data_engineering_copilot/services/context_compression.py`
- `data_engineering_copilot/services/reranker.py`

## Measured Outcomes (2026-08-18)

Baseline: `pinned-88b00d6c1494` (nvidia nemotron-3-embed-1b, legacy BM25, technical_queries.jsonl, 20 rows).

| Feature | Gate | Measured | Decision |
|---|---|---|---|
| HyDE policy | ≥20% identifier-intent call reduction | 36→26 provider calls (27.8%) | **Enabled** (`hyde_policy_enabled=True`) |
| Namespace BM25 | identifier_recall ≥ +0.05 | 0.3333→0.3333 (+0.0) | **Rejected** |
| Weighted RRF | identifier_recall ≥ +0.05 | 0.3333→0.3333 (+0.0) | **Rejected** |
| Diversity benchmark | dup-rate reduction ≥10% | 0%→16.7% (MMR worse) | **Rejected** |

HyDE policy is the only feature that passed its fixed gate. Namespace BM25 was tested on a candidate generation built with the same nvidia embedder (pinned-ns-bm25, 71,916 chunks, namespace-v1 tokenizer) and validated; the identifier recall gate failed so the candidate was rolled back and deleted.

Rollback settings: all new settings default to their non-enabled state. `hyde_policy_enabled=True` is the only change from the pre-plan defaults. The old generation is the active rollback target (`pinned-88b00d6c1494`).
