"""Async RAG answer service with optional query caching."""

from __future__ import annotations

import logging
from collections.abc import Callable

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import Answer
from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol, VectorStoreProtocol
from data_engineering_copilot.infrastructure.async_rag_cache import QueryCache
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

    async def answer(self, question: str, on_step: Callable[[str], None] | None = None) -> Answer:
        if self.cache is not None:
            cached = self.cache.get(question)
            if cached is not None:
                logger.info("cache_hit question=%r", question[:80])
                return cached

        if on_step:
            on_step("Embedding query")
        logger.info("async_rag.answer question=%r", question[:100])

        try:
            query_emb = self.embedder.embed_query(question)
            logger.info("Query embedding successful: dimension=%d", len(query_emb))
        except Exception as exc:
            logger.exception("Failed to embed query: %s", exc)
            return Answer(
                text="I cannot answer this question because embedding failed.", sources=tuple(), confidence=0.0
            )

        if on_step:
            on_step("Retrieving chunks")
        try:
            retrieved_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k)
        except Exception as exc:
            logger.exception("Failed to query vector store: %s", exc)
            return Answer(
                text="I cannot answer this question because vector store query failed.",
                sources=tuple(),
                confidence=0.0,
            )

        if not retrieved_chunks:
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        if retrieved_chunks[0].confidence < settings.confidence_threshold:
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        try:
            if settings.reranker_enabled:
                reranker = CrossEncoderReranker(model_name=settings.reranker_model)
                if reranker.is_available():
                    expanded_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k * 2)
                    retrieved_chunks = reranker.rerank(question, expanded_chunks, top_k=settings.reranker_top_k)

            sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)
            assembler = ContextAssembler(max_context_chars=settings.max_context_chars)
            context_str, source_names = assembler.assemble(sorted_chunks)

            if on_step:
                on_step("Generating answer")
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
            answer_text = self.ollama_client.generate(prompt)

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
            return Answer(
                text=f"I encountered an error while generating the answer: {exc}",
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
            )
