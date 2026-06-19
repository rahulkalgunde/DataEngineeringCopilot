from __future__ import annotations

import logging

from data_engineering_copilot.domain.models import Answer, RetrievedChunk
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError
from data_engineering_copilot.infrastructure.vector_store import ChromaVectorStore, VectorStoreReadError


OUTSIDE_REPOSITORY_MESSAGE = "I cannot answer this question because it is outside my knowledge repository."
logger = logging.getLogger(__name__)


class RagAnswerService:
    def __init__(
        self,
        embeddings: SentenceTransformerEmbeddings,
        vector_store: ChromaVectorStore,
        ollama_client: OllamaClient,
        top_k: int,
        max_context_chars: int,
        confidence_threshold: float,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.confidence_threshold = confidence_threshold

    def answer(self, question: str) -> Answer:
        logger.info("RAG answer started question=%r", question[:200])
        query_embedding = self.embeddings.embed_query(question)
        try:
            matches = self.vector_store.query(query_embedding, top_k=self.top_k)
        except VectorStoreReadError:
            logger.exception("Vector store unreadable during RAG answer")
            return Answer(text=OUTSIDE_REPOSITORY_MESSAGE, sources=(), confidence=0.0)
        confidence = matches[0].confidence if matches else 0.0
        logger.info(
            "RAG retrieval completed matches=%s top_confidence=%.4f threshold=%.4f",
            len(matches),
            confidence,
            self.confidence_threshold,
        )

        if confidence < self.confidence_threshold:
            logger.info("RAG answer rejected by confidence gate confidence=%.4f", confidence)
            return Answer(text=OUTSIDE_REPOSITORY_MESSAGE, sources=(), confidence=confidence)

        prompt = self._build_prompt(question, matches)
        try:
            generated = self.ollama_client.generate(prompt)
        except OllamaError as exc:
            logger.exception("Ollama failed during RAG answer")
            return Answer(text=str(exc), sources=self._unique_sources(matches), confidence=confidence)
        unique_sources = self._unique_sources(matches)
        logger.info(
            "RAG answer completed answer_chars=%s sources=%s confidence=%.4f",
            len(generated),
            len(unique_sources),
            confidence,
        )
        return Answer(text=generated, sources=unique_sources, confidence=confidence)

    def _build_prompt(self, question: str, matches: list[RetrievedChunk]) -> str:
        context_blocks = []
        for index, match in enumerate(matches, start=1):
            chunk = match.chunk
            remaining_chars = max(0, self.max_context_chars - sum(len(block) for block in context_blocks))
            if remaining_chars == 0:
                break
            chunk_text = chunk.text[:remaining_chars]
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Source: {chunk.source_name}",
                        f"Title: {chunk.title}",
                        f"URL: {chunk.url}",
                        f"Chunk ID: {chunk.chunk_id}",
                        f"Text: {chunk_text}",
                    ]
                )
            )

        context = "\n\n".join(context_blocks)
        logger.info(
            "RAG prompt built context_chars=%s blocks=%s max_context_chars=%s",
            len(context),
            len(context_blocks),
            self.max_context_chars,
        )
        return f"""You are DataEngineeringCopilot, an offline assistant for data engineering documentation.
Answer only from the provided repository context.
If the context does not contain the answer, reply exactly:
{OUTSIDE_REPOSITORY_MESSAGE}

Write a concise, practical answer in no more than 5 bullet points or 2 short paragraphs.
Prefer direct facts, commands, and caveats from the context. Do not show hidden reasoning. Do not invent sources.

Repository context:
{context}

Question:
{question}

Answer:"""

    def _unique_sources(self, matches: list[RetrievedChunk]) -> tuple:
        seen: set[tuple[str, str]] = set()
        sources = []
        for match in matches:
            chunk = match.chunk
            key = (chunk.title, chunk.url)
            if key in seen:
                continue
            seen.add(key)
            sources.append(chunk)
        return tuple(sources)
