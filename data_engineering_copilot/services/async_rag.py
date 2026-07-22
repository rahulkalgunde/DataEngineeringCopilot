from __future__ import annotations

import logging
from collections.abc import Callable

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import Answer
from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol, VectorStoreProtocol
from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class AsyncRagService:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        ollama_client: LLMClientProtocol,
        embedder: EmbedderProtocol,
        cache: QueryCache | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.embedder = embedder
        self.cache = cache
        self.langfuse = get_langfuse_instance()

    async def answer(self, question: str, on_step: Callable[[str], None] | None = None) -> Answer:
        if self.cache is not None:
            cached = self.cache.get(question)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                return cached

        trace = None
        if self.langfuse:
            trace = self.langfuse.start_observation(
                name="rag-query-pipeline",
                input=question,
                as_type="trace",
            )

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r", question[:100])

        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        try:
            query_emb = await self.embedder.embed_query(question)
            logger.info("Query embedding successful: dimension=%d", len(query_emb))
        except Exception as exc:
            logger.exception("Failed to embed query: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
            return Answer(
                text="I cannot answer this question because embedding failed.", sources=tuple(), confidence=0.0
            )

        if on_step:
            on_step("Retrieving chunks")
        try:
            retrieved_chunks = await self.vector_store.query(query_emb, top_k=settings.retrieval_top_k)
            if retrieval_span:
                retrieval_span.update(
                    output=[c.chunk.text for c in retrieved_chunks],
                    input=question,
                )
                retrieval_span.end()
        except Exception as exc:
            logger.exception("Failed to query vector store: %s", exc)
            if retrieval_span:
                retrieval_span.update(output=str(exc), level="ERROR")
                retrieval_span.end()
            if trace:
                trace.update(output=str(exc))
            return Answer(
                text="I cannot answer this question because vector store query failed.",
                sources=tuple(),
                confidence=0.0,
            )

        if not retrieved_chunks:
            if trace:
                trace.update(output="No chunks retrieved")
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        if retrieved_chunks[0].confidence < settings.confidence_threshold:
            if trace:
                trace.update(
                    output=f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {settings.confidence_threshold:.4f}"
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
                model=settings.ollama_model,
                as_type="generation",
            )

        try:
            if on_step:
                on_step("Reranking results")
            if settings.reranker_enabled:
                reranker = CrossEncoderReranker(model_name=settings.reranker_model)
                if reranker.is_available():
                    expanded_chunks = await self.vector_store.query(query_emb, top_k=settings.retrieval_top_k * 2)
                    retrieved_chunks = reranker.rerank(question, expanded_chunks, top_k=settings.reranker_top_k)

            sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
            assembler = ContextAssembler(max_context_chars=settings.max_context_chars)
            context_str, source_names = assembler.assemble(sorted_chunks)

            if on_step:
                on_step("Generating answer")
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

            if generation_span:
                generation_span.update(input=prompt)

            answer_text = await self.ollama_client.generate(prompt)

            if generation_span:
                generation_span.update(output=answer_text)
                generation_span.end()
            if trace:
                trace.update(output=answer_text)
                trace.end()

            if self.langfuse:
                self.langfuse.flush()

            result = Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
            )

            if self.cache is not None:
                self.cache.set(question, result)

            return result
        except Exception as exc:
            logger.exception("Failed during answer generation: %s", exc)
            if generation_span:
                generation_span.update(output=str(exc), level="ERROR")
                generation_span.end()
            if trace:
                trace.update(output=str(exc))
            return Answer(
                text=f"I encountered an error while generating the answer: {exc}",
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
            )
