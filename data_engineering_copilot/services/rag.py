import logging
import requests
from typing import List, Dict

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.domain.models import RetrievedChunk, Answer
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

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
        query_emb = self.embedder.embed_query(question)
        retrieved_chunks = self.vector_store.query(query_emb, top_k=settings.retrieval_top_k)
        if not retrieved_chunks or retrieved_chunks[0].confidence < settings.confidence_threshold:
            return Answer(text="I cannot answer this question because it is outside my knowledge repository.", sources=tuple(), confidence=0.0)

        context_str = "\n".join(c.chunk.text for c in retrieved_chunks)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        answer_text = self.ollama_client.generate(prompt)
        
        return Answer(
            text=answer_text,
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
            model_name=settings.ollama_model
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
        trace = self.langfuse.trace(
            name="rag-query-pipeline",
            user_id=user_id,
            session_id=session_id,
            input=question,
        )

        # Retrieval Phase
        retrieval_span = trace.span(name="retrieval")
        try:
            query_emb = self.embedder.embed_query(question)
            retrieved_chunks: List[RetrievedChunk] = self.db.query(query_emb, top_k=top_k)
            retrieval_span.end(output=[c.chunk.text for c in retrieved_chunks])
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            retrieval_span.end(output=str(exc), status="error")
            raise

        # Context Construction for Ollama Generation
        context_str = "\n".join(chunk.chunk.text for chunk in retrieved_chunks)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        
        # Generation Phase (Ollama HTTP API)
        generation_span = trace.span(name="ollama-generation", input=prompt)

        try:
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
            generation_span.end(output=answer_text)
        except Exception as exc:
            logger.exception("Ollama generation failed: %s", exc)
            generation_span.end(output=str(exc), status="error")
            raise

        # Finalization
        trace.end(output=answer_text)
        return {
            "answer": answer_text,
            "sources": [c.chunk.source_name for c in retrieved_chunks],
            "confidence": min(c.confidence for c in retrieved_chunks) if retrieved_chunks else 0.0,
        }