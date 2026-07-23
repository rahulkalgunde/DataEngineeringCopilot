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
from data_engineering_copilot.services.mmr_reranker import mmr_rerank
from data_engineering_copilot.services.prompt_builder import PromptBuilder
from data_engineering_copilot.services.query_cache import QueryCache as TwoTierCache
from data_engineering_copilot.services.query_rewriting import QueryRewriter

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
        self._prompt_builder = PromptBuilder()

    async def answer(self, question: str, on_step: Callable[[str], None] | None = None) -> Answer:
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

        # Phase 2A: Query rewriting
        effective_query = question
        if self.query_rewriter is not None:
            rewritten = self.query_rewriter.rewrite(question)
            if rewritten.decomposed_steps:
                effective_query = rewritten.decomposed_steps[0]
            logger.info(
                "query_rewritten intent=%s steps=%d original=%r",
                rewritten.intent,
                len(rewritten.decomposed_steps),
                question[:80],
            )

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r", question[:100])

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        try:
            query_emb = await self.embedder.embed_query(effective_query)
            logger.info("Query embedding successful: dimension=%d", len(query_emb))
        except Exception as exc:
            logger.exception("Failed to embed query: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        if on_step:
            on_step("Retrieving chunks")
        try:
            retrieved_chunks = await self.vector_store.query(query_emb, top_k=self.config.retrieval_top_k)
            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=effective_query,
                )
                retrieval_span.end()
        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("Failed to query vector store: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

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

        # Phase 2D: Context compression (dedup + relevance re-ranking)
        if self.context_compressor is not None:
            retrieved_chunks = self.context_compressor.compress(retrieved_chunks, effective_query)
            logger.info("context_compressed chunks=%d", len(retrieved_chunks))

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
            if len(retrieved_chunks) > 3:
                retrieved_chunks = mmr_rerank(retrieved_chunks, top_k=self.config.reranker_top_k)

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
                usage = getattr(self.llm_client, "last_usage", {})
                if usage:
                    self.token_tracker.record(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        model=getattr(self.llm_client, "model", "unknown"),
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

            # Phase 2B: Groundedness verification (annotate-only)
            if self.groundedness_verifier is not None:
                groundedness = self.groundedness_verifier.verify(answer_text, [c.chunk for c in retrieved_chunks])
                logger.info(
                    "groundedness_score=%.3f claims=%d",
                    groundedness.overall_score,
                    len(groundedness.annotations),
                )

            # Phase 2C: Cache the result
            if self.cache is not None:
                self.cache.set_exact(question, answer_text)

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
