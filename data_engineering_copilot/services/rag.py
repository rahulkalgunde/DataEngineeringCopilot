"""Production RAG service with Langfuse tracing and direct Ollama HTTP calls.

The service now uses the persistent Chroma vector store (ChromaVectorStore) which matches the rest of the codebase. It embeds the user question using SentenceTransformerEmbeddings, retrieves relevant chunks via the store's query method, and sends a prompt to Ollama.
"""

from __future__ import annotations

import logging
from typing import List, Dict

import requests
try:
    from langfuse import Langfuse
except ImportError:
    class Langfuse:  # Dummy placeholder
        def __init__(self, *args, **kwargs):
            pass
        def trace(self, *args, **kwargs):
            return self.DummyTrace()
        class DummyTrace:
            def span(self, *args, **kwargs):
                return self.DummySpan()
            def end(self, *args, **kwargs):
                pass
            class DummySpan:
                def end(self, *args, **kwargs):
                    pass

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)


class ProductionRagService:
    def __init__(self):
        # Initialise Langfuse tracing – configuration is read from env vars
        self.langfuse = Langfuse()
        # Initialise vector store (persistent Chroma DB)
# Initialise vector store (Qdrant)
        self.db = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
# Initialise vector store (Qdrant)
        self.db = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
        # Initialise embedder for query embedding
        self.db = ChromaVectorStore(
            persist_directory=str(settings.chroma_dir),
            collection_name=settings.collection_name,
        )
        # Initialise embedder for query embedding
        self.embedder = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model_name,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )

    def answer_question(
        self,
        user_id: str,
        session_id: str,
        question: str,
        top_k: int = 4,
    ) -> Dict:
        """Run the full RAG pipeline and return answer + sources.

        Returns a dictionary with keys ``answer``, ``sources`` and ``confidence``.
        """
        trace = self.langfuse.trace(
            name="rag-query-pipeline",
            user_id=user_id,
            session_id=session_id,
            input=question,
        )

        # ---------- Retrieval ----------
        retrieval_span = trace.span(name="retrieval")
        try:
            query_emb = self.embedder.embed_query(question)
            retrieved_chunks: List[RetrievedChunk] = self.db.query(query_emb, top_k=top_k)
            retrieval_span.end(output=[c.chunk.text for c in retrieved_chunks])
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            retrieval_span.end(output=str(exc), status="error")
            raise

        # ---------- Generation ----------
        context_str = "\n".join(chunk.chunk.text for chunk in retrieved_chunks)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
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

        # ---------- Finalise ----------
        trace.end(output=answer_text)

        return {
            "answer": answer_text,
            "sources": [c.chunk.source_name for c in retrieved_chunks],
            "confidence": min(c.confidence for c in retrieved_chunks) if retrieved_chunks else 0.0,
        }