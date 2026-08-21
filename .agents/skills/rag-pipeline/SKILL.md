---
name: rag-pipeline
description: Use for ANY task on the DataEngineeringCopilot RAG query/answer path — adding or modifying a stage, tracing where a stage runs, groundedness, scope checking, streaming, or the query/answer flow through AsyncRagService. Triggers: rag, answer, query, RAG pipeline, groundedness, scope check, rerank, context assembly, prompt injection, retrieval, query rewrite, cache hit, stage_times, AsyncRagService, HyDE, multi-hop, GraphRAG, CRAG, relevance grading, search mode, feedback telemetry.
---

# DataEngineeringCopilot RAG Pipeline

The single orchestrator is `services/async_rag.py` → **`AsyncRagService`**.
Everything below is project-specific and current as of the Spark/generation era.

## Query/answer flow (`AsyncRagService.answer()`, single-turn)

Stages execute in this order; each is timed and emitted to the pipeline
visualizer (`on_step`/`_emit_detail`) and Langfuse spans (`_record_stage`):

1. **Cache lookup** — `services/query_cache.py` `QueryCache` (exact + semantic
   tiers, Redis, scope-aware via `CacheScope`). Hit → record `cache_hit` trace, skip all LLM work.
2. **Rewrite** — `services/query_rewriting.py` `QueryRewriter`: intent
   classification, decomposition into steps, HyDE, multi-query expansion →
   `RewrittenQuery`. Uses the per-purpose `rewrite` LLM chain. HyDE is gated by
   the deterministic `HydePolicy` (`hyde_policy_enabled=True`): it runs only for
   factual/how_to intents and is suppressed for identifier-/version-qualified,
   code-fence, or stack-trace queries (`should_use_hyde`).
3. **Retrieval** — per query variant: `embedder` → `vector_store.query()` in
   `infrastructure/async_qdrant_store.py` `AsyncQdrantVectorStore` (dense +
   BM25 sparse prefetch + RRF `hybrid_rrf_k`), merged by
   `merge_retrieval_results()`. Then `_rejoin_sibling_chunks()` restores split
   parent context. `RetrievalError` is re-raised as-is.
   `services/query_signals.py` picks a `SearchMode` deterministically from the
   intent + signals (api_lookup/code_example → bm25_only, debugging →
   sparse_bias, factual/how_to/synthesis → dense_only). The weighted
   `identifier_sparse_rrf` profile (sparse 1.25 / dense 1.0) exists but is
   gated OFF (`identifier_sparse_rrf_enabled=False`) pending its benchmark gate.
4. **Input guardrails** — `services/input_guardrails.py` (indirect prompt
   injection scan of retrieved chunks, `services/prompt_injection.py`).
5. **Rerank** — selected by `reranker_type`: `services/reranker.py`
   `CrossEncoderReranker` (local `BAAI/bge-reranker-v2-m3`, default), or
   `services/colbert_reranker.py` `ColBERTReranker` when `reranker_type="colbert"`
   (a char-3gram MaxSim proxy — NOT neural late-interaction). With
   `llm_rerank_enabled=True` (default) `services/llm_reranker.py` `LLMReranker`
   tries the cloud rerank chain first (`rerank_fallback_order`:
   openrouter/nvidia/huggingface) with the local cross-encoder as degraded
   fallback. Pool = `max(top_k*4, reranker_top_k*8)` (`_rerank_pool_size`).
   Then `_reject_low_confidence()` drops weak chunks using
   `reranker_confidence_threshold` (when reranked) or `confidence_threshold`.
6. **Corrective retrieval (CRAG)** — `services/relevance_grader.py` grades the
   retrieved set; a score < 0.5 triggers exactly one expanded re-retrieval
   (`top_k * 2`, `_relevance_guarded_chunks`). Fail-open: grader errors keep
   the original chunks.
7. **Context augmentation** — `_augment_context()`: GraphRAG topological
   context (`services/graph_traversal.py` over `infrastructure/graph_store.py`,
   extracted by `services/graph_extractor.py`) plus multi-hop decomposition
   (`services/multi_hop_decomposer.py` plans dependent sub-queries and executes
   them stepwise). Both fail open — absent component or error returns no
   augmentation.
8. **Context assembly/compression** — `services/context_assembler.py`
   `ContextAssembler`: content-hash dedup, adjacent sibling merge, then MMR
   diversity or Jaccard dedup (`assembly_mmr_enabled`, default False →
   Jaccard), two-pass source-coverage budget selection (coverage pass = one
   chunk per source URL, depth pass capped at `max_chunks_per_source`),
   lost-in-the-middle reorder (alternating end placement), breadcrumb headers
   (`assembly_breadcrumb_format`: hierarchical/flat/none), XML content
   escaping (`&`/`<`), char budget `max_context_chars` + optional
   `services/context_compression.py`.
9. **Generation** — `services/prompt_builder.py` `PromptBuilder` (per-intent
   prompts; salted `<context_data_{salt}>` XML tags, XML content escaping,
   instruction sandwiching via trailing instructions, citation enforcement
   tri-state `prompt_citation_enforcement` strict/soft/off — strict and soft
   currently render identically; code intents → dedicated `code` LLM chain),
   schema-enforced structured output (`services/structured_output.py`; Ollama
   gets `format=`, OpenAI-style providers get `response_format=json_schema`,
   gated by `infrastructure/provider_capabilities.py`), per-purpose temperature
   (`generation_temperature`=0.15, `code_generation_temperature`=0.20) plus
   seed/penalties emitted only for capable providers, JSON retry,
   `_validate_and_fix_code_syntax` for code, `services/output_guardrails.py`,
   PII redaction. Uses the per-purpose `answer` LLM chain.
10. **Groundedness** — `services/groundedness.py` `GroundednessVerifier` (NLI,
    annotate-only / fail-open), `groundedness` chain.
11. **Scope check** — `services/scope_verifier.py` `ScopeVerifier` (topic-scope
    gate; refuses with `INSUFFICIENT_CONTEXT` markers) + deterministic
    domain-coherence guard (`_domain_mismatch`), `_apply_scope_gate`.
12. **Score & cache write-back** — Langfuse scores
    (confidence/groundedness/relevance/intent/cache_hit), store into Redis,
    feedback telemetry log (`services/feedback_telemetry.py`,
    `_log_feedback`).

Streaming variants: `answer_stream()` and `chat_stream()` (SSE) mirror the
stages; `_verify_stream_groundedness` runs post-stream.

## Hard conventions (non-negotiable)

- **Factory DI only** — never instantiate services manually. Build via
  `factory.py`: `build_rag_service()`, `build_llm_fallback_chain(purpose=...)`,
  `build_embedding_fallback_chain()`, etc.
- **Per-purpose LLM chains** — `answer`/`rewrite`/`groundedness`/`intent`/
  `enrichment`/`evaluation`/`code` are separate chains from
  `build_llm_fallback_chain()`. Do not override globally.
- **Three-valued returns** — methods like `extract_sentences` return `None`
  (unsupported), `[]` (empty), or a `list` (content). Always check `is None`
  explicitly — never truthiness alone.
- **All LLM/embedding/rerank calls** route through the unified
  `ProviderFallbackChain` (`infrastructure/provider_fallback.py`). Never call
  a provider directly.
- **Streaming semantics** — a provider that fails *before* emitting a token is
  skipped; failure *after* emission is re-raised (can't retry sent tokens).

## Key files

| Concern | File |
|---|---|
| Orchestrator | `services/async_rag.py` |
| Rewrite + HyDE policy | `services/query_rewriting.py` |
| Search-mode routing | `services/query_signals.py` |
| Retrieve + BM25 | `infrastructure/async_qdrant_store.py`, `infrastructure/bm25_tokenizer.py` |
| Rerank | `services/reranker.py`, `services/colbert_reranker.py`, `services/llm_reranker.py`, `infrastructure/rerank_clients.py` |
| CRAG / GraphRAG / multi-hop | `services/relevance_grader.py`, `services/graph_extractor.py`, `services/graph_traversal.py`, `infrastructure/graph_store.py`, `services/multi_hop_decomposer.py` |
| Prompt/answer | `services/prompt_builder.py` |
| Structured output / capabilities | `services/structured_output.py`, `infrastructure/provider_capabilities.py` |
| Assembly | `services/context_assembler.py` |
| Groundedness / scope | `services/groundedness.py`, `services/scope_verifier.py` |
| Cache | `services/query_cache.py`, `services/redis_query_cache.py` |
| Chat/multi-turn | `services/conversation_rag.py` |
| Feedback telemetry | `services/feedback_telemetry.py` |
| Telemetry | `observability/telemetry.py`, `observability/otel_telemetry.py` |

## Extension points

- **New stage**: add to `answer()` between rerank and generation, emit via
  `_emit_detail`/`on_step`, add a `_record_stage("...")` line, mirror it in
  `answer_stream()`/`chat_stream()`.
- **New intent/code path**: `PromptBuilder` prompt selection + route the intent
  to the right LLM chain in the factory.
- **New guardrail**: `services/input_guardrails.py` / `output_guardrails.py`
  are the injection points; register in `validate_all()` feature flags if gated.

## Verification

Always run the AGENTS.md loop after touching this path:
`ruff check --fix` → `ruff format` → `pyright` → targeted `pytest` on the
affected service. Integration/RAG tests auto-skip without Qdrant+Ollama
(`make test-integration`); do not fake infra availability.
