from __future__ import annotations

import ast
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, CachedAnswer, CacheScope, RagConfig, RetrievedChunk
from data_engineering_copilot.domain.protocols import (
    EmbedderProtocol,
    LLMClientProtocol,
    RerankerProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.infrastructure.pii_redactor import PiiRedactor
from data_engineering_copilot.observability.langfuse_prompts import get_langfuse_prompt, register_fallback
from data_engineering_copilot.observability.token_tracker import RetrievalTracker, TokenTracker
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.context_compression import ContextCompressor
from data_engineering_copilot.services.groundedness import GroundednessVerifier
from data_engineering_copilot.services.input_guardrails import InputGuardrails
from data_engineering_copilot.services.prompt_builder import CODE_INTENTS, PromptBuilder
from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
from data_engineering_copilot.services.query_rewriting import QueryRewriter
from data_engineering_copilot.services.structured_output import parse_rag_response, verify_citations

# Offline fallback for the Langfuse-managed ``rag-json-retry-suffix`` prompt.
_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Return ONLY raw JSON with no markdown, no code fences, no preamble."
)

register_fallback("rag-json-retry-suffix", _JSON_RETRY_SUFFIX)


def merge_retrieval_results(
    results: list[list[RetrievedChunk]],
    original_query: str,
) -> list[RetrievedChunk]:
    """Merge per-query retrieval results using rank fusion plus an original-query bonus.

    Each result list is ranked in retrieval order. Chunks are scored by the sum
    of ``1 / (rank + 3)`` across all queries that retrieved them. Chunks that
    match the original query (retrieved by the original query variant) receive
    an extra bonus. Chunks whose text defines a Spark function named in the
    query (e.g. ``def filter(``) receive an additional lexical bonus so the
    real docstring implementation surfaces above unrelated function-family
    chunks (e.g. ``inline``, ``array_*``). Returns chunks sorted by fused score
    descending.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for query_index, result_list in enumerate(results):
        for rank, result in enumerate(result_list):
            cid = result.chunk.chunk_id
            chunk_by_id[cid] = result
            score = 1.0 / (rank + 3.0)
            # Original-query bonus for the first result set.
            if query_index == 0:
                score += 0.5
            scores[cid] = scores.get(cid, 0.0) + score

    merged = sorted(chunk_by_id.values(), key=lambda c: -scores.get(c.chunk.chunk_id, 0.0))
    return merged


logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = "1"


def _chunk_provenance_ref(result: RetrievedChunk, rank: int) -> dict[str, object]:
    """Compact, JSON-safe per-candidate reference for retrieval provenance."""
    return {
        "rank": rank,
        "chunk_id": result.chunk.chunk_id,
        "url": result.chunk.url,
        "source_name": result.chunk.source_name,
        "distance": result.distance,
        "confidence": result.confidence,
    }


_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


def _scrub_pii(text: str) -> str:
    """Redact common PII patterns (email, phone, SSN) from trace input."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _llm_model_name(llm_client: Any) -> str:
    """Best-effort model name from an LLM client (LLMClient, ProviderFallbackChain)."""
    for attr in ("model", "model_name"):
        value = getattr(llm_client, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _rerank_pool_size(retrieval_top_k: int, reranker_top_k: int) -> int:
    """Candidate pool handed to the cross-encoder reranker.

    The pool is intentionally wider than ``retrieval_top_k`` so that URLs that
    just miss the dense/sparse cutoff can still be rescued by reranking. The
    ``* 8`` multiplier keeps the pool large enough for the fused rank of a
    relevant-but-poorly-scored document (observed at rank ~175) to remain
    inside the pool without running the cross-encoder over the whole corpus.
    """
    return max(retrieval_top_k * 8, reranker_top_k * 5)


class AsyncRagService:
    def __init__(
        self,
        config: RagConfig,
        vector_store: VectorStoreProtocol,
        llm_client: LLMClientProtocol,
        embedder: EmbedderProtocol,
        reranker: RerankerProtocol | None = None,
        telemetry: TelemetryTracerProtocol | None = None,
        cache: TwoTierCache | None = None,
        query_rewriter: QueryRewriter | None = None,
        groundedness_verifier: GroundednessVerifier | None = None,
        context_compressor: ContextCompressor | None = None,
        token_tracker: TokenTracker | None = None,
        retrieval_tracker: RetrievalTracker | None = None,
        code_llm_client: LLMClientProtocol | None = None,
        evaluation_llm_client: LLMClientProtocol | None = None,
        pii_redactor: PiiRedactor | None = None,
        input_guardrails: InputGuardrails | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.code_llm_client = code_llm_client
        self.evaluation_llm_client = evaluation_llm_client
        self.embedder = embedder
        self.reranker = reranker
        self.telemetry = telemetry
        self.cache = cache
        self.query_rewriter = query_rewriter
        self.groundedness_verifier = groundedness_verifier
        self.context_compressor = context_compressor
        self.token_tracker = token_tracker
        self.retrieval_tracker = retrieval_tracker
        self._pii_redactor = pii_redactor
        self.input_guardrails = input_guardrails
        self._prompt_builder = PromptBuilder()

    async def answer(
        self,
        question: str,
        on_step: Callable[[str], None] | None = None,
        source_filter: list[str] | None = None,
        cache_scope: CacheScope | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        provenance: list[dict] | None = None,
        bypass_cache: bool = False,
        expected_urls: list[str] | None = None,
    ) -> Answer:
        _t0 = time.monotonic()
        _stage_times: dict[str, float] = {}

        def _record_stage(name: str) -> None:
            _stage_times[name] = round((time.monotonic() - _t0) * 1000, 1)

        # Opt-in retrieval provenance capture (used by evaluation). Production
        # callers pass nothing and are unaffected.
        effective_query = question
        _prov_cache_hit = False
        _prov_variants: list[dict[str, object]] = []
        _prov_fused: list[dict[str, object]] = []
        _prov_rerank: dict[str, object] | None = None
        _prov_final: list[dict[str, object]] = []
        _prov_dropped: list[dict[str, object]] = []
        _prov_expected_urls = expected_urls or []

        def _emit_provenance() -> None:
            if provenance is None:
                return
            provenance.append(
                {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "question": question,
                    "effective_query": effective_query,
                    "cache_hit": _prov_cache_hit,
                    "query_variants": _prov_variants,
                    "fused": _prov_fused,
                    "rerank": _prov_rerank,
                    "final_context": _prov_final,
                    "dropped": _prov_dropped,
                    "expected_urls": _prov_expected_urls,
                    "candidate_pool_size": _prov_pool,
                    "stage_times": dict(_stage_times),
                }
            )

        query_emb_for_cache: list[float] | None = None
        cache_active = self.cache is not None and self.config.cache_enabled and not bypass_cache
        if cache_active and self.cache is not None:
            # Exact tier first — no embedding round-trip on an exact hit.
            cached = await self.cache.aget(question, scope=cache_scope)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                _record_stage("cache_lookup")
                _prov_cache_hit = True
                _emit_provenance()
                return Answer(
                    text=cached.text,
                    sources=cached.sources,
                    confidence=cached.confidence,
                    groundedness_score=cached.groundedness_score,
                    stage_times=_stage_times,
                )
            # Exact miss: only now pay the embedding cost for semantic lookup.
            with contextlib.suppress(Exception):
                query_emb_for_cache = await self.embedder.embed_query(question)
            cached = await self.cache.aget(question, query_embedding=query_emb_for_cache, scope=cache_scope)
            if cached is not None:
                logger.info("cache_hit_semantic question=%r", question[:80])
                _record_stage("cache_lookup")
                _prov_cache_hit = True
                _emit_provenance()
                return Answer(
                    text=cached.text,
                    sources=cached.sources,
                    confidence=cached.confidence,
                    groundedness_score=cached.groundedness_score,
                    stage_times=_stage_times,
                )

        trace = None
        if self.telemetry:
            trace_kwargs: dict[str, Any] = {
                "name": "rag-query-pipeline",
                "input": _scrub_pii(question),
                "as_type": "trace",
                "tags": ["app:data-engineering-copilot"],
                "metadata": {
                    "app_env": settings.langfuse_environment,
                    "git_sha": settings.image_git_sha,
                },
            }
            if user_id:
                trace_kwargs["user_id"] = user_id
            if session_id:
                trace_kwargs["session_id"] = session_id
            if settings.langfuse_environment:
                trace_kwargs["environment"] = settings.langfuse_environment
            trace = self.telemetry.start_observation(**trace_kwargs)

        # Phase 2A: Query rewriting — collect all queries for multi-step retrieval
        rewritten = None
        all_queries: list[str] = [question]  # Always include original

        rewrite_span = None
        if trace:
            rewrite_span = trace.start_observation(name="query-rewriting", as_type="span")

        if self.query_rewriter is not None:
            rewritten = await self.query_rewriter.async_rewrite(question)
            if rewritten.decomposed_steps:
                all_queries.extend(rewritten.decomposed_steps)
                effective_query = rewritten.decomposed_steps[0]
            # Multi-query expansion: generate additional variations
            expanded = await self.query_rewriter.expand_queries(
                question, max_variations=self.config.max_expansion_queries
            )
            for q in expanded:
                if q not in all_queries:
                    all_queries.append(q)
            _record_stage("rewrite")
            logger.info(
                "query_rewritten intent=%s steps=%d expanded=%d hyde=%s original=%r",
                rewritten.intent,
                len(rewritten.decomposed_steps),
                len(expanded),
                bool(rewritten.hyde_query),
                question[:80],
            )

            if rewrite_span:
                rewrite_span.update(
                    input=question,
                    output={
                        "intent": rewritten.intent,
                        "decomposed_steps": len(rewritten.decomposed_steps),
                        "expanded_queries": len(expanded),
                        "hyde_query": bool(rewritten.hyde_query),
                    },
                )
                rewrite_span.end()

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r queries=%d", question[:100], len(all_queries))

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        # Determine chunk type filter based on intent. When a hard ``modules``
        # filter is present the exact API page is already pinned, so the "api"
        # chunk-type restriction is dropped — rendered reference pages carry
        # ``chunk_type="text"`` and combining both would empty the result set.
        chunk_type_filter = None
        if rewritten is not None and rewritten.intent == "api_lookup":
            metadata_filters = rewritten.filters if rewritten is not None else None
            if metadata_filters is None or not metadata_filters.modules:
                chunk_type_filter = "api"

        # Structured metadata filters extracted during rewriting.
        metadata_filters = rewritten.filters if rewritten is not None else None

        # Retrieve with all query variations and merge results
        all_retrieved: list = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        try:
            # Collect per-query result sets for rank fusion; the original query
            # is always first so it receives the fusion bonus.
            per_query_results: list[list[RetrievedChunk]] = []
            queries_to_run = list(all_queries)
            if rewritten is not None and rewritten.hyde_query:
                queries_to_run.append(rewritten.hyde_query)

            # Human-readable labels for provenance diagnostics (original query,
            # decomposed steps, HyDE variant, or rewrite expansions).
            decomposed_count = len(rewritten.decomposed_steps) if rewritten is not None else 0
            variant_labels: list[str] = []
            for i in range(len(queries_to_run)):
                if i == 0:
                    variant_labels.append("original")
                elif i <= decomposed_count:
                    variant_labels.append("decomposed")
                elif rewritten is not None and rewritten.hyde_query and i == len(queries_to_run) - 1:
                    variant_labels.append("hyde")
                else:
                    variant_labels.append("expanded")

            for i, q in enumerate(queries_to_run):
                try:
                    # Embed every query separately; HyDE is an additional query,
                    # not a replacement for the real query embeddings.
                    q_emb = await self.embedder.embed_query(q)
                    results = await self.vector_store.query(
                        q_emb,
                        top_k=self.config.retrieval_top_k,
                        query_text=q,
                        source_filter=source_filter,
                        chunk_type_filter=chunk_type_filter,
                        metadata_filters=metadata_filters,
                        fused_limit=_rerank_pool_size(self.config.retrieval_top_k, self.config.reranker_top_k),
                    )
                    # Unfiltered fallback: if metadata filters removed everything,
                    # retry once without inferred filters.
                    if not results and metadata_filters is not None and not metadata_filters.is_empty:
                        results = await self.vector_store.query(
                            q_emb,
                            top_k=self.config.retrieval_top_k,
                            query_text=q,
                            source_filter=source_filter,
                            chunk_type_filter=chunk_type_filter,
                            fused_limit=_rerank_pool_size(self.config.retrieval_top_k, self.config.reranker_top_k),
                        )
                    any_success = True
                    per_query_results.append(results)
                    _prov_variants.append(
                        {
                            "variant": variant_labels[i],
                            "query": q,
                            "retrieved": [_chunk_provenance_ref(r, rank) for rank, r in enumerate(results)],
                        }
                    )
                    for r in results:
                        cid = r.chunk.chunk_id
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            all_retrieved.append(r)
                except Exception as sub_exc:
                    last_error = sub_exc
                    logger.warning("Failed to retrieve for sub-query %r: %s", q[:50], sub_exc)

            if not any_success and last_error is not None:
                raise RetrievalError(f"Vector store query failed: {last_error}") from last_error

            # Rank-fusion merge with original-query bonus.
            if per_query_results:
                retrieved_chunks = merge_retrieval_results(per_query_results, question)
            else:
                retrieved_chunks = sorted(all_retrieved, key=lambda c: c.confidence, reverse=True)
            _prov_pool = len(retrieved_chunks)
            _prov_fused = [_chunk_provenance_ref(c, rank) for rank, c in enumerate(retrieved_chunks)]

            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=effective_query,
                )
                retrieval_span.end()
            _record_stage("retrieval")
            logger.info("Multi-step retrieval: %d queries → %d unique chunks", len(all_queries), len(retrieved_chunks))

            # Track retrieval scores for observability
            if self.retrieval_tracker is not None and retrieved_chunks:
                scores = [c.confidence for c in retrieved_chunks]
                self.retrieval_tracker.record_retrieval(scores, query=question)
        except RetrievalError:
            if trace:
                trace.update(output="RetrievalError")
                trace.end()
            raise
        except Exception as exc:
            logger.exception("Failed during retrieval: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
                trace.end()
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
                trace.end()
            _emit_provenance()
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        if retrieved_chunks[0].confidence < self.config.confidence_threshold:
            if trace:
                trace.update(
                    output=f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {self.config.confidence_threshold:.4f}"
                )
                trace.end()
            _emit_provenance()
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        # Indirect prompt injection guard: drop retrieved chunks that look like
        # embedded instructions before they reach the prompt.
        if self.input_guardrails is not None:
            scan_result = self.input_guardrails.scan_chunks(retrieved_chunks)
            retrieved_chunks = scan_result.kept
            if not retrieved_chunks:
                logger.warning("All retrieved chunks rejected by input guardrails")
                if trace:
                    trace.update(output="All chunks rejected by input guardrails")
                    trace.end()
                _emit_provenance()
                return Answer(
                    text="I cannot answer this question because it is outside my knowledge repository.",
                    sources=tuple(),
                    confidence=0.0,
                )

        generation_span: Any = None
        try:
            if on_step:
                on_step("Reranking results")

            rerank_span = None
            if trace:
                rerank_span = trace.start_observation(name="reranking", as_type="span")

            pre_rerank_count = len(retrieved_chunks)
            reranker = self.reranker
            rerank_used = False
            if (
                self.config.reranker_enabled
                and reranker is not None
                and reranker.is_available()
                and pre_rerank_count > 1
            ):
                rerank_used = True
                # Rerank a broad candidate pool against the original question.
                # Multi-query retrieval and dense+sparse fusion generate recall;
                # the cross-encoder is the single generic relevance decision.
                rerank_pool = min(
                    pre_rerank_count, _rerank_pool_size(self.config.retrieval_top_k, self.config.reranker_top_k)
                )
                # Use the concise retrieval rewrite for pair scoring. The
                # original question remains the generation prompt, while the
                # rewrite removes conversational detail that can obscure the
                # technical terms in code and API documentation.
                rerank_query = effective_query.strip() or question
                retrieved_chunks = await reranker.rerank(rerank_query, retrieved_chunks, top_k=rerank_pool)

            _prov_rerank = {
                "enabled": rerank_used,
                "query": effective_query.strip() or question,
                "pool_size": pre_rerank_count,
                "top_k": (
                    min(pre_rerank_count, _rerank_pool_size(self.config.retrieval_top_k, self.config.reranker_top_k))
                    if rerank_used
                    else None
                ),
                "final_top_k": self.config.reranker_top_k,
            }

            # The reranker already returns the final relevance ordering. Do not
            # apply a second diversity objective here: lexical MMR can discard
            # complementary evidence needed to answer multi-part questions.
            if len(retrieved_chunks) > self.config.reranker_top_k:
                retrieved_chunks = retrieved_chunks[: self.config.reranker_top_k]

            # Phase 2D: Context compression (dedup + relevance re-ranking) — runs after reranking
            # to avoid wasted work (compressed results were previously discarded by re-fetch)
            if self.context_compressor is not None:
                retrieved_chunks = self.context_compressor.compress(retrieved_chunks, effective_query)
                logger.info("context_compressed chunks=%d", len(retrieved_chunks))
            _record_stage("rerank")

            if rerank_span:
                rerank_span.update(
                    input=f"{len(retrieved_chunks)} chunks before reranking",
                    output=f"{len(retrieved_chunks)} chunks after reranking",
                )
                rerank_span.end()

            assembler = ContextAssembler(max_context_chars=self.config.max_context_chars)
            context_str, source_names, dropped_records = assembler.assemble(
                retrieved_chunks,
                deduplicate=self.context_compressor is None,
            )
            _prov_dropped = dropped_records
            # The final context reflects only the segments actually placed in
            # the prompt; budget-dropped segments must not be claimed as
            # retrieved into the final context.
            _prov_dropped_ids = {record["chunk_id"] for record in dropped_records}
            _final_chunks = [c for c in retrieved_chunks if c.chunk.chunk_id not in _prov_dropped_ids]
            _prov_final = [_chunk_provenance_ref(c, rank) for rank, c in enumerate(_final_chunks)]

            if on_step:
                on_step("Generating answer")
            intent = rewritten.intent if rewritten else "factual"
            safe_question = PromptBuilder.sanitize_query(question)
            if self._pii_redactor is not None:
                safe_question, _pii_types = self._pii_redactor.redact(safe_question)
                if _pii_types:
                    logger.info("pii_redacted types=%s", _pii_types)
            prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=safe_question, intent=intent)

            llm_client = self._select_llm_client(intent)
            generation_span = None
            if trace:
                generation_span = trace.start_observation(
                    name="llm-generation",
                    as_type="generation",
                    model=_llm_model_name(llm_client),
                )
                generation_span.update(input=prompt)
            answer_text = await llm_client.generate(prompt)

            # JSON retry: if the intent expects JSON output but parsing fails,
            # retry once with a stricter instruction.
            if intent not in CODE_INTENTS and isinstance(answer_text, str):
                parsed_attempt = parse_rag_response(answer_text)
                if not parsed_attempt.answer and len(answer_text.strip()) > 20:
                    logger.info("json_parse_retry intent=%s response_len=%d", intent, len(answer_text))
                    retry_prompt = prompt + get_langfuse_prompt("rag-json-retry-suffix").compile()
                    retry_span = None
                    if trace:
                        retry_span = trace.start_observation(
                            name="llm-json-retry",
                            as_type="generation",
                            model=_llm_model_name(llm_client),
                            input=retry_prompt,
                        )
                    answer_text = await llm_client.generate(retry_prompt)
                    if retry_span:
                        retry_span.update(output=answer_text)
                        retry_span.end()

            _record_stage("generation")

            # Post-generation syntax check for code intents
            answer_text = await self._validate_and_fix_code_syntax(answer_text, intent, llm_client, trace)

            # Track token usage from LLM provider
            if self.token_tracker is not None and hasattr(llm_client, "last_usage"):
                usage = getattr(llm_client, "last_usage", None)
                if usage is not None:
                    self.token_tracker.record(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        model=usage.model,
                    )

                    # Also send token usage and cost to Langfuse
                    if self.telemetry and trace:
                        try:
                            trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
                            if trace_id:
                                # Calculate cost based on provider pricing
                                cost = self._estimate_cost(
                                    usage.prompt_tokens,
                                    usage.completion_tokens,
                                    usage.model,
                                )

                                # Record token usage + cost on the generation observation
                                if generation_span:
                                    generation_span.update(
                                        usage_details={
                                            "input": usage.prompt_tokens,
                                            "output": usage.completion_tokens,
                                            "total": usage.prompt_tokens + usage.completion_tokens,
                                            "unit": "TOKENS",
                                        },
                                        cost_details={
                                            "input": cost,
                                            "output": cost,
                                            "total": cost,
                                            "currency": "USD",
                                        },
                                    )

                                # Update trace with token usage and cost metadata
                                trace.update(
                                    metadata={
                                        "token_usage": {
                                            "prompt_tokens": usage.prompt_tokens,
                                            "completion_tokens": usage.completion_tokens,
                                            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                                            "model": usage.model,
                                        },
                                        "cost_usd": cost,
                                    }
                                )

                                # Score cost for monitoring
                                self.telemetry.score(
                                    trace_id=trace_id,
                                    name="cost_usd",
                                    value=cost,
                                    data_type="NUMERIC",
                                )

                                logger.debug(
                                    "Langfuse token usage and cost recorded: tokens=%d cost=$%.6f",
                                    usage.prompt_tokens + usage.completion_tokens,
                                    cost,
                                )
                        except Exception as exc:
                            logger.warning("Failed to record Langfuse token usage: %s", exc)

            # Output guardrails: verify structure and quality
            from data_engineering_copilot.services.output_guardrails import OutputGuardrails

            validated = OutputGuardrails.verify(answer_text, len(retrieved_chunks))
            if validated is not None:
                if validated.status == "INSUFFICIENT_CONTEXT" and validated.missing_info:
                    answer_text = f"{validated.answer}\n\nMissing information: {validated.missing_info}"
                else:
                    answer_text = validated.answer
                logger.info(
                    "output_guardrails passed status=%s confidence=%.2f citations=%d",
                    validated.status,
                    validated.confidence,
                    len(validated.citations),
                )
            else:
                logger.info("output_guardrails rejected answer, using raw output")

            # Post-LLM PII redaction: strip PII from answer before returning
            if self._pii_redactor is not None:
                answer_text, pii_types_answer = self._pii_redactor.redact(answer_text)
                if pii_types_answer:
                    logger.info("pii_redacted_in_answer types=%s", pii_types_answer)

            # Citation verification: keep only citations matching retrieved sources
            if not isinstance(answer_text, str):
                answer_text = str(answer_text) if answer_text else ""
            parsed = parse_rag_response(answer_text)
            source_names = [c.chunk.source_name for c in retrieved_chunks]
            verified_citations = verify_citations(parsed.citations, source_names)
            if verified_citations:
                answer_text = parsed.answer
                logger.info("citations_verified=%d total=%d", len(verified_citations), len(parsed.citations))

            if generation_span:
                generation_span.update(output=answer_text)
                generation_span.end()
            if trace:
                trace.update(
                    output=answer_text,
                    metadata={
                        "stage_times_ms": _stage_times,
                        "num_sources": len(retrieved_chunks),
                        "intent": intent,
                    },
                )
                trace.end()

            if self.telemetry:
                try:
                    await self.telemetry.flush_async()
                except Exception:
                    logger.warning("Telemetry flush failed, ignoring", exc_info=True)

            _record_stage("total")
            _emit_provenance()
            trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None) if trace else None
            result = Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in _final_chunks),
                confidence=retrieved_chunks[0].confidence,
                stage_times=_stage_times,
                trace_id=trace_id,
            )

            # Phase 2B: Groundedness verification (annotate-only, fail-open)
            groundedness_score = 1.0

            groundedness_span = None
            if trace:
                groundedness_span = trace.start_observation(name="groundedness-verification", as_type="span")

            if self.groundedness_verifier is not None:
                (
                    supported,
                    unsupported_claims,
                    groundedness_score,
                ) = await self.groundedness_verifier.async_verify_with_score(result, retrieved_chunks)
                logger.info(
                    "groundedness_supported=%s unsupported=%d score=%.2f",
                    supported,
                    len(unsupported_claims),
                    groundedness_score,
                )
                if not supported and unsupported_claims:
                    result = Answer(
                        text=result.text + "\n\n[Note: Some claims may not be fully supported by the documentation.]",
                        sources=result.sources,
                        confidence=result.confidence,
                        groundedness_score=groundedness_score,
                    )
                else:
                    result = Answer(
                        text=result.text,
                        sources=result.sources,
                        confidence=result.confidence,
                        groundedness_score=groundedness_score,
                    )

                if groundedness_span:
                    groundedness_span.update(
                        input="Answer and retrieved chunks",
                        output={
                            "supported": supported,
                            "unsupported_claims_count": len(unsupported_claims),
                            "groundedness_score": groundedness_score,
                        },
                    )
                    groundedness_span.end()

            # Phase 2E: Langfuse scoring (confidence, groundedness, quality)
            if self.telemetry and trace:
                try:
                    # Get trace ID from the trace object
                    trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
                    if trace_id:
                        # Score confidence
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="confidence",
                            value=result.confidence,
                            data_type="NUMERIC",
                        )
                        # Score groundedness
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="groundedness",
                            value=groundedness_score,
                            data_type="NUMERIC",
                        )

                        # LLM-as-Judge: compute relevance score
                        # Relevance = confidence * groundedness (composite metric)
                        relevance_score = result.confidence * groundedness_score
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="relevance",
                            value=relevance_score,
                            data_type="NUMERIC",
                        )

                        # Completeness heuristic: based on answer length and source count
                        # Longer answers with more sources tend to be more complete
                        source_count = len(result.sources)
                        answer_length = len(result.text.split())
                        completeness = min(1.0, (answer_length / 100) * (1 + source_count * 0.1))
                        self.telemetry.score(
                            trace_id=trace_id,
                            name="completeness",
                            value=completeness,
                            data_type="NUMERIC",
                        )

                        # Log quality metrics for monitoring
                        logger.info(
                            "quality_scores confidence=%.3f groundedness=%.3f relevance=%.3f completeness=%.3f",
                            result.confidence,
                            groundedness_score,
                            relevance_score,
                            completeness,
                        )
                        logger.debug("Langfuse scores recorded for trace %s", trace_id)
                except Exception as exc:
                    logger.warning("Failed to record Langfuse scores: %s", exc)

            # Phase 2C: Cache the result (exact + semantic tiers). Only cache
            # answers that pass the quality gate; the QueryCache also enforces
            # this at the set level (defense-in-depth).
            if cache_active and self.cache is not None:
                envelope = CachedAnswer(
                    text=result.text,
                    sources=result.sources,
                    confidence=result.confidence,
                    groundedness_score=result.groundedness_score,
                    cached_at=time.time(),
                )
                if self.cache.is_cacheable(envelope):
                    await self.cache.aset_exact(question, envelope, scope=cache_scope)
                    if query_emb_for_cache is not None:
                        await self.cache.aset_semantic(question, query_emb_for_cache, envelope, scope=cache_scope)

            # Log cache hit rate for observability
            if self.cache is not None:
                logger.info(
                    "query_cache_stats hits=%d misses=%d hit_rate=%.2f",
                    self.cache._hits,
                    self.cache._misses,
                    self.cache.hit_rate,
                )

            return result
        except LLMGenerationError:
            if trace:
                trace.update(output="LLMGenerationError")
                trace.end()
            if self.telemetry:
                with contextlib.suppress(Exception):
                    await self.telemetry.flush_async()
            raise
        except Exception as exc:
            logger.exception("Failed during answer generation: %s", exc)
            if generation_span:
                generation_span.update(output=str(exc), level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output=str(exc))
                trace.end()
            if self.telemetry:
                with contextlib.suppress(Exception):
                    await self.telemetry.flush_async()
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc

    async def answer_stream(
        self,
        question: str,
        source_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream answer tokens via SSE while performing retrieval.

        Yields JSON-encoded SSE events:
        - ``{"type": "status", "message": "..."}`` for pipeline progress
        - ``{"type": "token", "content": "..."}`` for answer tokens
        - ``{"type": "done", "text": "...", "confidence": 0.0}`` at completion
        """
        _t0 = time.monotonic()
        yield _sse({"type": "status", "message": "Sanitizing query"})

        # Create trace for streaming path
        trace = None
        if self.telemetry:
            trace_kwargs: dict[str, Any] = {
                "name": "rag-query-pipeline-stream",
                "input": _scrub_pii(question),
                "as_type": "trace",
                "tags": ["app:data-engineering-copilot"],
                "metadata": {
                    "app_env": settings.langfuse_environment,
                    "git_sha": settings.image_git_sha,
                },
            }
            if user_id:
                trace_kwargs["user_id"] = user_id
            if session_id:
                trace_kwargs["session_id"] = session_id
            if settings.langfuse_environment:
                trace_kwargs["environment"] = settings.langfuse_environment
            trace = self.telemetry.start_observation(**trace_kwargs)

        # PII redaction
        safe_question = PromptBuilder.sanitize_query(question)
        if self._pii_redactor is not None:
            safe_question, _pii_types = self._pii_redactor.redact(safe_question)
            if _pii_types:
                yield _sse({"type": "status", "message": f"PII redacted: {', '.join(_pii_types)}"})

        # Query rewriting
        rewritten = None
        rewrite_span = None
        if trace:
            rewrite_span = trace.start_observation(name="query-rewriting", as_type="span")
        if self.query_rewriter is not None:
            yield _sse({"type": "status", "message": "Rewriting query"})
            try:
                rewritten = await self.query_rewriter.async_rewrite(safe_question)
            except Exception:
                logger.warning("Query rewrite failed during streaming, using raw query", exc_info=True)
                rewritten = None
            if rewritten is not None:
                yield _sse({"type": "status", "message": f"Intent: {rewritten.intent}"})
        if rewrite_span:
            rewrite_span.update(
                input=question,
                output={"intent": rewritten.intent if rewritten else "factual"},
            )
            rewrite_span.end()

        if rewritten is not None and rewritten.decomposed_steps:
            effective_query = rewritten.decomposed_steps[0]
        else:
            effective_query = safe_question
        intent = rewritten.intent if rewritten else "factual"

        # Retrieval
        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")
        yield _sse({"type": "status", "message": "Retrieving documents"})
        try:
            query_emb = await self.embedder.embed_query(effective_query)
        except Exception as exc:
            if trace:
                trace.update(output=f"EmbeddingError: {exc}")
                trace.end()
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        top_k = self.config.retrieval_top_k
        retrieved_chunks = await self.vector_store.query(
            query_embedding=query_emb,
            top_k=top_k,
            query_text=effective_query,
            source_filter=source_filter,
            fused_limit=_rerank_pool_size(self.config.retrieval_top_k, self.config.reranker_top_k),
        )
        if retrieval_span:
            retrieval_span.update(
                input=effective_query,
                output=f"{len(retrieved_chunks)} chunks retrieved",
            )
            retrieval_span.end()
        yield _sse({"type": "status", "message": f"Retrieved {len(retrieved_chunks)} chunks"})

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
                trace.end()
            yield _sse({"type": "done", "text": "No relevant documents found.", "confidence": 0.0})
            return

        # Reranking
        rerank_span = None
        if trace:
            rerank_span = trace.start_observation(name="reranking", as_type="span")
        if self.reranker is not None and self.reranker.is_available() and len(retrieved_chunks) > 1:
            yield _sse({"type": "status", "message": "Reranking"})
            retrieved_chunks = await self.reranker.rerank(
                query=effective_query,
                chunks=retrieved_chunks,
                top_k=self.config.reranker_top_k,
            )
        if rerank_span:
            rerank_span.update(
                input=f"{len(retrieved_chunks)} chunks before reranking",
                output=f"{len(retrieved_chunks)} chunks after reranking",
            )
            rerank_span.end()

        # Context assembly
        sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
        assembler = ContextAssembler(max_context_chars=self.config.max_context_chars)
        context_str, _source_names, _dropped_records = assembler.assemble(sorted_chunks)

        # Build prompt
        prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=safe_question, intent=intent)
        yield _sse({"type": "status", "message": "Generating answer"})

        # Stream LLM tokens
        llm_client = self._select_llm_client(intent)
        generation_span = None
        if trace:
            generation_span = trace.start_observation(
                name="generation",
                as_type="generation",
                model=_llm_model_name(llm_client),
            )
        full_text = ""
        try:
            async for token in llm_client.generate_stream(prompt):
                full_text += token
                yield _sse({"type": "token", "content": token})
        except Exception:
            logger.exception("Streaming generation failed")
            if generation_span:
                generation_span.update(output="Generation failed", level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output="Streaming generation failed")
                trace.end()
            yield _sse({"type": "error", "message": "Generation failed"})
            return
        if generation_span:
            generation_span.update(input=prompt, output=full_text)
            generation_span.end()

        # Post-generation validation: PII redaction, output guardrails, caching
        if self._pii_redactor is not None:
            full_text, _pii_types = self._pii_redactor.redact(full_text)
            if _pii_types:
                logger.info("pii_redacted_in_stream types=%s", _pii_types)

        from data_engineering_copilot.services.output_guardrails import OutputGuardrails

        validated = OutputGuardrails.verify(full_text, len(retrieved_chunks))
        if validated is not None:
            full_text = validated.answer

        confidence = retrieved_chunks[0].confidence if retrieved_chunks else 0.0

        # Score and end trace
        if trace:
            trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
            if trace_id and self.telemetry:
                with contextlib.suppress(Exception):
                    self.telemetry.score(trace_id=trace_id, name="confidence", value=confidence)
            trace.update(
                output=full_text,
                metadata={
                    "intent": intent,
                    "num_sources": len(retrieved_chunks),
                    "streaming": True,
                    "total_ms": round((time.monotonic() - _t0) * 1000, 1),
                },
            )
            trace.end()

        if self.cache is not None:
            try:
                envelope = CachedAnswer(
                    text=full_text,
                    sources=tuple(c.chunk for c in retrieved_chunks),
                    confidence=confidence,
                    cached_at=time.time(),
                )
                await self.cache.aset_exact(safe_question, envelope)
                query_emb_for_cache = await self.embedder.embed_query(effective_query)
                await self.cache.aset_semantic(safe_question, query_emb_for_cache, envelope)
            except Exception:
                logger.warning("Cache write failed in stream", exc_info=True)

        if self.telemetry:
            with contextlib.suppress(Exception):
                await self.telemetry.flush_async()

        # Done event with metadata
        yield _sse({"type": "done", "text": full_text, "confidence": confidence})

    def _select_llm_client(self, intent: str) -> LLMClientProtocol:
        """Route code intents to the code-specific LLM if configured."""
        if self.code_llm_client and intent in CODE_INTENTS:
            return self.code_llm_client
        return self.llm_client

    async def _validate_and_fix_code_syntax(
        self, answer_text: str, intent: str, llm_client: LLMClientProtocol, trace: Any | None = None
    ) -> str:
        """Validate Python code blocks in answer. Retry once if syntax fails.

        Only validates Python code blocks — other languages (Scala, SQL, etc.)
        are returned as-is since we can't syntax-check them with ast.parse.
        """
        if intent not in CODE_INTENTS:
            return answer_text

        code_blocks = re.findall(r"```python\n(.*?)```", answer_text, re.DOTALL)
        if not code_blocks:
            return answer_text

        invalid_blocks = []
        for block in code_blocks:
            try:
                ast.parse(block.strip())
            except SyntaxError:
                invalid_blocks.append(block)

        if not invalid_blocks:
            return answer_text

        fix_prompt = (
            "The following Python code has syntax errors. Fix ONLY the code, "
            "keeping the same structure and imports. Return valid Python only.\n\n"
            f"Broken code:\n```python\n{invalid_blocks[0]}\n```"
        )
        fix_span = None
        if trace:
            fix_span = trace.start_observation(
                name="llm-code-syntax-fix",
                as_type="generation",
                model=_llm_model_name(llm_client),
                input=fix_prompt,
            )
        try:
            fixed = await llm_client.generate(fix_prompt)
            fixed_code = fixed.strip().strip("`").removeprefix("python").strip()
            if fix_span:
                fix_span.update(output=fixed_code)
                fix_span.end()
            return answer_text.replace(invalid_blocks[0], fixed_code)
        except Exception:
            if fix_span:
                fix_span.update(output="syntax fix failed", level="ERROR")
                fix_span.end()
            logger.warning("Code syntax fix retry failed, returning original")
            return answer_text

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD based on token usage and provider pricing.

        Pricing is approximate and based on common provider rates.
        For Ollama (local), cost is 0.
        """
        # Ollama is free (local inference)
        if not model or model.startswith(("llama", "qwen", "mistral", "gemma", "phi")):
            return 0.0

        # Approximate pricing per 1M tokens (as of 2024-2025)
        # Source: provider documentation
        pricing = {
            # OpenRouter models (approximate)
            "openrouter/": {"input": 0.0000003, "output": 0.0000006},
            # Cloudflare Workers AI
            "cf-": {"input": 0.000000011, "output": 0.000000036},
            # Groq
            "groq/": {"input": 0.00000005, "output": 0.00000008},
            # Cerebras
            "cerebras/": {"input": 0.0000001, "output": 0.0000001},
            # Gemini
            "gemini/": {"input": 0.000000125, "output": 0.000000375},
            # NVIDIA
            "nvidia/": {"input": 0.0000001, "output": 0.0000001},
        }

        # Find matching provider
        for prefix, rates in pricing.items():
            if model.lower().startswith(prefix.lower()) or prefix.lower() in model.lower():
                input_cost = prompt_tokens * rates["input"]
                output_cost = completion_tokens * rates["output"]
                return input_cost + output_cost

        # Default: assume free (Ollama or unknown)
        return 0.0

    async def close(self) -> None:
        """Close all underlying clients and connection pools. Idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        components = [
            self.vector_store,
            self.llm_client,
            self.embedder,
            self.code_llm_client,
            self.evaluation_llm_client,
            self.cache,
            self.reranker,
            self.groundedness_verifier,
            self.query_rewriter,
            self.context_compressor,
        ]
        for component in components:
            if component is None:
                continue
            with contextlib.suppress(TypeError, AttributeError, Exception):
                closer = None
                if hasattr(component, "aclose"):
                    closer = component.aclose
                elif hasattr(component, "close"):
                    closer = component.close
                if closer is not None:
                    await closer()
