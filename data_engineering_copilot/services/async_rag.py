from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from data_engineering_copilot.domain.exceptions import LLMGenerationError, RetrievalError
from data_engineering_copilot.domain.models import Answer, RagConfig
from data_engineering_copilot.domain.protocols import (
    EmbedderProtocol,
    LLMClientProtocol,
    RerankerProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.context_compression import ContextCompressor
from data_engineering_copilot.services.groundedness import GroundednessVerifier
from data_engineering_copilot.services.prompt_builder import PromptBuilder
from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
from data_engineering_copilot.services.query_rewriting import QueryRewriter
from data_engineering_copilot.services.rag_evaluation import FaithfulnessEvaluator

logger = logging.getLogger(__name__)


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
        token_tracker: object | None = None,
        retrieval_tracker: object | None = None,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedder = embedder
        self.reranker = reranker
        self.telemetry = telemetry
        self.cache = cache
        self.query_rewriter = query_rewriter
        self.groundedness_verifier = groundedness_verifier
        self.context_compressor = context_compressor
        self.token_tracker = token_tracker
        self.retrieval_tracker = retrieval_tracker
        self.faithfulness_evaluator = faithfulness_evaluator
        self._prompt_builder = PromptBuilder()

    async def answer(
        self,
        question: str,
        on_step: Callable[[str], None] | None = None,
        source_filter: list[str] | None = None,
    ) -> Answer:
        if self.cache is not None:
            query_emb_for_cache = None
            with contextlib.suppress(Exception):
                query_emb_for_cache = await self.embedder.embed_query(question)
            cached = self.cache.get(question, query_embedding=query_emb_for_cache)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                return Answer(
                    text=cached,
                    sources=tuple(),
                    confidence=1.0,
                )

        trace = None
        if self.telemetry:
            trace = self.telemetry.start_observation(
                name="rag-query-pipeline",
                input=question,
                as_type="trace",
            )

        # Phase 2A: Query rewriting — collect all queries for multi-step retrieval
        rewritten = None
        all_queries: list[str] = [question]  # Always include original
        effective_query = question
        if self.query_rewriter is not None:
            rewritten = await self.query_rewriter.async_rewrite(question)
            if rewritten.decomposed_steps:
                all_queries.extend(rewritten.decomposed_steps)
                effective_query = rewritten.decomposed_steps[0]
            # Multi-query expansion: generate additional variations
            expanded = await self.query_rewriter.expand_queries(question, max_variations=2)
            for q in expanded:
                if q not in all_queries:
                    all_queries.append(q)
            logger.info(
                "query_rewritten intent=%s steps=%d expanded=%d hyde=%s original=%r",
                rewritten.intent,
                len(rewritten.decomposed_steps),
                len(expanded),
                bool(rewritten.hyde_query),
                question[:80],
            )

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r queries=%d", question[:100], len(all_queries))

        # Compute a representative embedding for reranking/MMR
        try:
            query_emb = await self.embedder.embed_query(effective_query)
        except Exception as exc:
            logger.exception("Failed to compute representative embedding: %s", exc)
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        # Retrieve with all query variations and merge results
        all_retrieved: list = []
        seen_ids: set[str] = set()
        any_success = False
        last_error: Exception | None = None
        try:
            for q in all_queries:
                try:
                    if self.query_rewriter is not None and rewritten is not None and rewritten.hyde_query:
                        q_emb = await self.query_rewriter.hyde(q, self.embedder)
                    else:
                        q_emb = await self.embedder.embed_query(q)
                    results = await self.vector_store.query(
                        q_emb, top_k=self.config.retrieval_top_k, query_text=q,
                        source_filter=source_filter,
                    )
                    any_success = True
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

            # Sort merged results by confidence
            retrieved_chunks = sorted(all_retrieved, key=lambda c: c.confidence, reverse=True)

            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=effective_query,
                )
                retrieval_span.end()
            logger.info("Multi-step retrieval: %d queries → %d unique chunks", len(all_queries), len(retrieved_chunks))

            # Track retrieval scores for observability
            if self.retrieval_tracker is not None and retrieved_chunks:
                scores = [c.confidence for c in retrieved_chunks]
                self.retrieval_tracker.record_retrieval(scores, query=question)
        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("Failed during retrieval: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
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
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        generation_span = None
        if trace:
            generation_span = trace.start_observation(
                name="ollama-generation",
                as_type="generation",
            )

        try:
            if on_step:
                on_step("Reranking results")
            if self.config.reranker_enabled and self.reranker is not None and self.reranker.is_available():
                expanded_chunks = await self.vector_store.query(query_emb, top_k=self.config.retrieval_top_k * 2)
                retrieved_chunks = self.reranker.rerank(
                    effective_query, expanded_chunks, top_k=self.config.reranker_top_k
                )

            # MMR diversity reranking — ensures diverse context
            if len(retrieved_chunks) > 3 and self.reranker is not None:
                retrieved_chunks = self.reranker.max_marginal_relevance(
                    query_emb, retrieved_chunks, top_k=self.config.reranker_top_k
                )

            # Phase 2D: Context compression (dedup + relevance re-ranking) — runs after reranking
            # to avoid wasted work (compressed results were previously discarded by re-fetch)
            if self.context_compressor is not None:
                retrieved_chunks = self.context_compressor.compress(retrieved_chunks, effective_query)
                logger.info("context_compressed chunks=%d", len(retrieved_chunks))

            sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
            assembler = ContextAssembler(max_context_chars=self.config.max_context_chars)
            context_str, source_names = assembler.assemble(sorted_chunks)

            if on_step:
                on_step("Generating answer")
            prompt = self._prompt_builder.build_rag_prompt(context=context_str, question=question)

            if generation_span:
                generation_span.update(input=prompt)

            answer_text = await self.llm_client.generate(prompt)

            # Track token usage from OllamaClient
            if self.token_tracker is not None and hasattr(self.llm_client, "last_usage"):
                usage = getattr(self.llm_client, "last_usage", None)
                if usage is not None:
                    self.token_tracker.record(
                        prompt_tokens=usage.prompt_eval_count,
                        completion_tokens=usage.eval_count,
                        model=usage.model,
                    )

            if generation_span:
                generation_span.update(output=answer_text)
                generation_span.end()
            if trace:
                trace.update(output=answer_text)
                trace.end()

            if self.telemetry:
                self.telemetry.flush()

            result = Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
            )

            # Phase 2B: Groundedness verification (annotate-only, fail-open)
            groundedness_score = 1.0
            if self.groundedness_verifier is not None:
                supported, unsupported_claims, groundedness_score = (
                    await self.groundedness_verifier.async_verify_with_score(
                        result, retrieved_chunks
                    )
                )
                logger.info(
                    "groundedness_supported=%s unsupported=%d score=%.2f",
                    supported, len(unsupported_claims), groundedness_score,
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

            # Phase 2B2: Faithfulness evaluation (RAGAS-compatible)
            if self.faithfulness_evaluator is not None:
                context_text = " ".join(c.chunk.text for c in retrieved_chunks[:5])
                faith_result = await self.faithfulness_evaluator.evaluate(result.text, context_text)
                logger.info(
                    "faithfulness_score=%.2f supported=%d unsupported=%d",
                    faith_result.faithfulness_score,
                    faith_result.supported_claims,
                    faith_result.unsupported_claims,
                )
                # Update groundedness_score with faithfulness if lower
                if faith_result.faithfulness_score < groundedness_score:
                    groundedness_score = faith_result.faithfulness_score
                    result = Answer(
                        text=result.text,
                        sources=result.sources,
                        confidence=result.confidence,
                        groundedness_score=groundedness_score,
                    )

            # Phase 2C: Cache the result (exact + semantic tiers)
            if self.cache is not None:
                self.cache.set_exact(question, result.text)
                if query_emb_for_cache is not None:
                    self.cache.set_semantic(question, query_emb_for_cache, result.text)

            return result
        except LLMGenerationError:
            raise
        except Exception as exc:
            logger.exception("Failed during answer generation: %s", exc)
            if generation_span:
                generation_span.update(output=str(exc), level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output=str(exc))
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
