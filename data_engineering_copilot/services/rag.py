import logging
import requests
from typing import List, Dict

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.domain.models import RetrievedChunk, Answer
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.services.context_assembler import ContextAssembler
from data_engineering_copilot.services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except ImportError:
    # No-op fallback when langfuse is not installed (v3 API)
    class _NoopObservation:
        def update(self, *args, **kwargs):
            pass
        def end(self, *args, **kwargs):
            pass
        def start_observation(self, *args, **kwargs):
            return _NoopObservation()

    class Langfuse:
        def __init__(self, *args, **kwargs):
            pass
        def start_observation(self, *args, **kwargs):
            return _NoopObservation()
        def flush(self, *args, **kwargs):
            pass


class RagAnswerService:
    def __init__(self, vector_store, ollama_client, embedder):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.embedder = embedder
        # Base class: no Langfuse tracing by default.
        # ProductionRagService overrides this after super().__init__().
        self.langfuse = None

    def answer(self, question: str) -> Answer:
        # Log the incoming question
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
                text="I cannot answer this question because embedding failed.",
                sources=tuple(),
                confidence=0.0
            )

        # Query the vector store and log retrieval details
        try:
            retrieved_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k)
            logger.info(
                "Vector store query completed: retrieved_count=%d, threshold=%.2f",
                len(retrieved_chunks),
                settings.confidence_threshold
            )

            # Log details of each retrieved chunk
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                logger.info(
                    "Retrieved chunk #%d: source=%r, confidence=%.4f, distance=%.4f, title=%r",
                    idx,
                    chunk.chunk.source_name,
                    chunk.confidence,
                    chunk.distance,
                    chunk.chunk.title[:50] if chunk.chunk.title else "N/A"
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
                text="I cannot answer this question because vector store query failed.",
                sources=tuple(),
                confidence=0.0
            )

        # Check if we have results and if they meet confidence threshold
        if not retrieved_chunks:
            logger.warning("No chunks retrieved from vector store for question: %r", question[:100])
            if trace:
                trace.update(output="No chunks retrieved")
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0
            )

        if retrieved_chunks[0].confidence < settings.confidence_threshold:
            logger.warning(
                "Top chunk confidence below threshold: top_confidence=%.4f, threshold=%.4f, question=%r",
                retrieved_chunks[0].confidence,
                settings.confidence_threshold,
                question[:100]
            )
            if trace:
                trace.update(
                    output=f"Low confidence: {retrieved_chunks[0].confidence:.4f} < threshold {settings.confidence_threshold:.4f}"
                )
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0
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

            if self.langfuse:
                self.langfuse.flush()

            return Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence
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
                confidence=retrieved_chunks[0].confidence
            )


class ProductionRagService(RagAnswerService):
    """Production RAG Service Implementation

    This service integrates:
    - Langfuse tracing
    - Qdrant vector store for retrieval
    - Ollama LLM for response generation
    """

    def __init__(self):
        self.db = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
        self.embedder = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model_name,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
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

        # Initialize Langfuse client using centralized settings
        self.langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            debug=True
        )

        try:
            auth_result = self.langfuse.auth_check()
            print(auth_result)
        except Exception as e:
            logger.warning("Langfuse auth check failed: %s", e)
            # Attempt fallback to localhost when running outside Docker
            fallback_host = "http://localhost:3000"
            if getattr(settings, "langfuse_host", "") != fallback_host:
                try:
                    self.langfuse = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=fallback_host,
                        debug=True,
                    )
                    auth_result = self.langfuse.auth_check()
                    print("Langfuse fallback auth succeeded:", auth_result)
                except Exception as e2:
                    logger.error("Langfuse fallback auth also failed: %s", e2)

    def answer_question(
        self,
        user_id: str,
        session_id: str,
        question: str,
        top_k: int = 4,
    ) -> Dict:
        """Execute the full RAG pipeline: retrieve -> generate -> stream.

        The function:
        1. Embeds the question using the SentenceTransformerEmbeddings model.
        2. Queries Qdrant for the top-K most relevant chunks.
        3. Uses the context to generate an answer via local Ollama HTTP API.
        4. Returns the answer with metadata and sources.
        """
        logger.info(
            "answer_question() called: user_id=%r, session_id=%r, question_preview=%r",
            user_id,
            session_id,
            question[:100]
        )

        if not self.langfuse:
            # Fallback: answer without tracing
            logger.warning("Langfuse not available; answering without tracing")
            answer_result = self.answer(question)
            return {
                "answer": answer_result.text,
                "sources": [c.source_name for c in answer_result.sources],
                "confidence": answer_result.confidence,
            }

        trace = self.langfuse.start_observation(
            name="rag-query-pipeline",
            input=question,
            as_type="trace",
        )

        # Retrieval Phase
        retrieval_span = trace.start_observation(name="retrieval", as_type="span")
        try:
            logger.info("Starting retrieval phase: embedding query...")
            query_emb = self.embedder.embed_query(question)
            logger.info("Query embedding successful: dimension=%d", len(query_emb))

            logger.info("Querying Qdrant vector store with top_k=%d", top_k)
            retrieved_chunks: List[RetrievedChunk] = self.db.query(query_emb, top_k=top_k)
            logger.info("Qdrant query returned %d chunks", len(retrieved_chunks))

            # Log retrieved chunks details
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                logger.info(
                    "Retrieved chunk #%d: source=%r, confidence=%.4f, title=%r (first 80 chars of text: %r)",
                    idx,
                    chunk.chunk.source_name,
                    chunk.confidence,
                    chunk.chunk.title[:50] if chunk.chunk.title else "N/A",
                    chunk.chunk.text[:80]
                )

            retrieval_span.update(output=[c.chunk.text for c in retrieved_chunks])
            retrieval_span.end()
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            retrieval_span.update(output=str(exc), level="ERROR")
            retrieval_span.end()
            raise

        # Context Construction for Ollama Generation
        # Sort chunks by confidence (highest first) for better context ordering
        sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.confidence, reverse=True)

        # Assemble intelligent context with deduplication and truncation
        assembler = ContextAssembler(max_context_chars=settings.max_context_chars)
        context_str, source_names = assembler.assemble(sorted_chunks)

        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        logger.debug("Constructed prompt with context_length=%d, prompt_preview=%r", len(context_str), prompt[:150])

        # Generation Phase (Ollama HTTP API)
        generation_span = trace.start_observation(
            name="ollama-generation",
            model=settings.ollama_model,
            input=prompt,
            as_type="generation",
        )

        try:
            logger.info("Starting Ollama generation: model=%s, timeout=%d", settings.ollama_model, settings.ollama_timeout_seconds)
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            answer_text = response.json().get("response", "")
            logger.info("Ollama generation successful: answer_length=%d, answer_preview=%r", len(answer_text), answer_text[:150])
            generation_span.update(output=answer_text)
            generation_span.end()
        except Exception as exc:
            logger.exception("Ollama generation failed: %s", exc)
            generation_span.update(output=str(exc), level="ERROR")
            generation_span.end()
            raise

        # Finalization
        trace.update(output=answer_text)
        if self.langfuse:
            self.langfuse.flush()

        logger.info("answer_question() completed successfully")
        return {
            "answer": answer_text,
            "sources": [c.chunk.source_name for c in retrieved_chunks],
            "confidence": min(c.confidence for c in retrieved_chunks) if retrieved_chunks else 0.0,
        }
