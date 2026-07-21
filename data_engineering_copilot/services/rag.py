import logging
from collections.abc import Callable

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import Answer
from data_engineering_copilot.domain.protocols import EmbedderProtocol, LLMClientProtocol, VectorStoreProtocol
from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RagAnswerService:
    def __init__(self, vector_store: VectorStoreProtocol, ollama_client: LLMClientProtocol, embedder: EmbedderProtocol):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.embedder = embedder
        # Base class: no Langfuse tracing by default.
        # ProductionRagService overrides this after super().__init__().
        self.langfuse = None

    def answer(self, question: str, on_step: Callable[[str], None] | None = None) -> Answer:
        # Log the incoming question
        if on_step:
            on_step("Embedding query")
        logger.info("RAG answer() called with question: %r", question[:100])

        # Start a Langfuse trace for this query (v3 API)
        trace = None
        if self.langfuse:
            trace = self.langfuse.start_observation(
                name="rag-query-pipeline",
                input=question,
                as_type="trace",
            )

        # Embed the query and log embedding details
        retrieval_span = None
        if trace:
            retrieval_span = trace.start_observation(name="retrieval", as_type="span")

        try:
            query_emb = self.embedder.embed_query(question)
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

        # Query the vector store and log retrieval details
        if on_step:
            on_step("Retrieving chunks")
        try:
            retrieved_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k)
            logger.info(
                "Vector store query completed: retrieved_count=%d, threshold=%.2f",
                len(retrieved_chunks),
                settings.confidence_threshold,
            )

            # Log details of each retrieved chunk
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                logger.info(
                    "Retrieved chunk #%d: source=%r, confidence=%.4f, distance=%.4f, title=%r",
                    idx,
                    chunk.chunk.source_name,
                    chunk.confidence,
                    chunk.distance,
                    chunk.chunk.title[:50] if chunk.chunk.title else "N/A",
                )

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
                text="I cannot answer this question because vector store query failed.", sources=tuple(), confidence=0.0
            )

        # Check if we have results and if they meet confidence threshold
        if not retrieved_chunks:
            logger.warning("No chunks retrieved from vector store for question: %r", question[:100])
            if trace:
                trace.update(output="No chunks retrieved")
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        if retrieved_chunks[0].confidence < settings.confidence_threshold:
            logger.warning(
                "Top chunk confidence below threshold: top_confidence=%.4f, threshold=%.4f, question=%r",
                retrieved_chunks[0].confidence,
                settings.confidence_threshold,
                question[:100],
            )
            if trace:
                trace.update(
                    output=f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {settings.confidence_threshold:.4f}"
                )
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0,
            )

        logger.info("Confidence check passed. Proceeding to retrieval reranking and answer generation.")

        # Build context and generate answer
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
            # Apply reranking if enabled
            if settings.reranker_enabled:
                reranker = CrossEncoderReranker(model_name=settings.reranker_model)
                if reranker.is_available():
                    # Retrieve 2x top_k candidates, then rerank to final top_k
                    expanded_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k * 2)
                    retrieved_chunks = reranker.rerank(question, expanded_chunks, top_k=settings.reranker_top_k)
                    logger.info("Reranking applied: %d → %d chunks", len(expanded_chunks), len(retrieved_chunks))
                else:
                    logger.info("Reranker model not available; using embedding similarity only")

            # Sort chunks by confidence (highest first) for better context ordering
            sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)

            # Assemble intelligent context with deduplication and truncation
            assembler = ContextAssembler(max_context_chars=settings.max_context_chars)
            context_str, source_names = assembler.assemble(sorted_chunks)

            if on_step:
                on_step("Generating answer")
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
            logger.debug("Ollama prompt (first 200 chars): %r", prompt[:200])

            if generation_span:
                generation_span.update(input=prompt)

            answer_text = self.ollama_client.generate(prompt)
            logger.info("Ollama generation successful: answer_length=%d", len(answer_text))

            if generation_span:
                generation_span.update(output=answer_text)
                generation_span.end()
            if trace:
                trace.update(output=answer_text)
                trace.end()

            if self.langfuse:
                self.langfuse.flush()

            return Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence,
            )
        except Exception as exc:
            logger.exception("Failed during Ollama generation: %s", exc)
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


class ProductionRagService(RagAnswerService):
    """Production RAG Service Implementation

    This service integrates:
    - Langfuse tracing
    - Qdrant vector store for retrieval
    - Ollama LLM for response generation
    """

    def __init__(self):
        from data_engineering_copilot.infrastructure.embeddings import OllamaEmbeddings
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
        from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

        self.db = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
        self.embedder = OllamaEmbeddings(
            model_name=settings.embedding_model_name,
        )
        self.ollama_client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
        )
        # Must call super().__init__() first so self.langfuse starts as None,
        # then overwrite with the real Langfuse instance below.
        super().__init__(self.db, self.ollama_client, self.embedder)

        # Initialize Langfuse client using centralized settings with graceful fallback
        self.langfuse = get_langfuse_instance()
        if self.langfuse is None:
            logger.warning("Langfuse tracing is disabled because the client could not be initialized")
