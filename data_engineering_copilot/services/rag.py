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
    class Langfuse:
        def __init__(self, *args, **kwargs):
            pass
        def trace(self, *args, **kwargs):
            return self.DummyTrace()
        class DummyTrace:
            def span(self, *args, **kwargs):
                return self.DummySpan()
            def end(self, *args):
                pass
        class DummySpan:
            def end(self, *args, **kwargs):
                pass

class RagAnswerService:
    def __init__(self, vector_store, ollama_client, embedder):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.embedder = embedder

    def answer(self, question: str) -> Answer:
        # Log the incoming question
        logger.info("RAG answer() called with question: %r", question[:100])
        
        # Embed the query and log embedding details
        try:
            query_emb = self.embedder.embed_query(question)
            logger.info("Query embedding successful: dimension=%d", len(query_emb))
        except Exception as exc:
            logger.exception("Failed to embed query: %s", exc)
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
        except Exception as exc:
            logger.exception("Failed to query vector store: %s", exc)
            return Answer(
                text="I cannot answer this question because vector store query failed.",
                sources=tuple(),
                confidence=0.0
            )
        
        # Check if we have results and if they meet confidence threshold
        if not retrieved_chunks:
            logger.warning("No chunks retrieved from vector store for question: %r", question[:100])
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
            return Answer(
                text="I cannot answer this question because it is outside my knowledge repository.",
                sources=tuple(),
                confidence=0.0
            )
        
        logger.info("Confidence check passed. Proceeding to retrieval reranking and answer generation.")
        
        # Build context and generate answer
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
            
            answer_text = self.ollama_client.generate(prompt)
            logger.info("Ollama generation successful: answer_length=%d", len(answer_text))
            
            return Answer(
                text=answer_text,
                sources=tuple(c.chunk for c in retrieved_chunks),
                confidence=retrieved_chunks[0].confidence
            )
        except Exception as exc:
            logger.exception("Failed during Ollama generation: %s", exc)
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
        # Only initialize Langfuse if proper env vars are configured
        import os
        public_key="pk-lf-c2eebd11-9735-47b0-b4e8-c20670c703b5"
        secret_key="sk-lf-c105cd37-b4b1-48cc-85cc-d4c9f3652da1"
        if public_key and secret_key:
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            )
        else:
            # Use a no-op Langfuse instance
            self.langfuse = Langfuse()
        
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
        super().__init__(self.db, self.ollama_client, self.embedder)

    def answer_question(
        self,
        user_id: str,
        session_id: str,
        question: str,
        top_k: int = 4,
    ) -> Dict:
        """Execute the full RAG pipeline: retrieve → generate → stream.
        
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
        
        trace = self.langfuse.trace(
            name="rag-query-pipeline",
            user_id=user_id,
            session_id=session_id,
            input=question,
        )

        # Retrieval Phase
        retrieval_span = trace.span(name="retrieval")
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
            
            retrieval_span.end(output=[c.chunk.text for c in retrieved_chunks])
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            retrieval_span.end(output=str(exc), status="error")
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
        generation_span = trace.span(name="ollama-generation", input=prompt)

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
            generation_span.end(output=answer_text)
        except Exception as exc:
            logger.exception("Ollama generation failed: %s", exc)
            generation_span.end(output=str(exc), status="error")
            raise

        # Finalization
        trace.end(output=answer_text)
        logger.info("answer_question() completed successfully")
        return {
            "answer": answer_text,
            "sources": [c.chunk.source_name for c in retrieved_chunks],
            "confidence": min(c.confidence for c in retrieved_chunks) if retrieved_chunks else 0.0,
        }
