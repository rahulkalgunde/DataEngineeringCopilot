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
        retry_context_ratio: float,
        retry_extra_num_predict: int,
        retry_max_num_predict: int,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.confidence_threshold = confidence_threshold
        self.retry_context_ratio = retry_context_ratio
        self.retry_extra_num_predict = retry_extra_num_predict
        self.retry_max_num_predict = retry_max_num_predict

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
        logger.info("RAG sending prompt prompt_chars=%s", len(prompt))
        try:
            generated = self.ollama_client.generate(prompt)
        except OllamaError as exc:
            logger.warning("Ollama generation failed first attempt: %s", exc)
            if "length" in str(exc).lower():
                reduced_chars = max(200, int(self.max_context_chars * self.retry_context_ratio))
                logger.info("Retrying Ollama with reduced context max_context_chars=%s", reduced_chars)
                prompt_retry = self._build_prompt(question, matches, max_context_chars=reduced_chars)
                logger.info("RAG retry sending prompt prompt_chars=%s", len(prompt_retry))
                try:
                    generated = self.ollama_client.generate(prompt_retry)
                except OllamaError as exc2:
                    logger.warning("Ollama retry with reduced context failed: %s", exc2)
                    try:
                        original_np = getattr(self.ollama_client, "num_predict", None) or 0
                        increased_np = min(original_np + self.retry_extra_num_predict, self.retry_max_num_predict)
                        logger.info("Final attempt: increasing num_predict to %s and retrying", increased_np)
                        generated = self.ollama_client.generate(prompt_retry, num_predict=increased_np)
                    except OllamaError:
                        logger.exception("Ollama failed during RAG answer on final retry")
                        return Answer(text=str(exc), sources=self._unique_sources(matches), confidence=confidence)
            else:
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

    def _build_prompt(self, question: str, matches: list[RetrievedChunk], max_context_chars: int = None) -> str:
        if max_context_chars is None:
            max_context_chars = self.max_context_chars
        context_blocks: list[str] = []
        used_chars = 0
        for index, match in enumerate(matches, start=1):
            chunk = match.chunk
            remaining_chars = max(0, max_context_chars - used_chars)
            if remaining_chars <= 0:
                break
            headroom = 200
            chunk_text = chunk.text[: max(0, remaining_chars - headroom)]
            used_chars += len(chunk_text)
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
            max_context_chars,
        )
        return f"""You are DataEngineeringCopilot, an offline assistant for data engineering documentation.
    Answer only from the provided repository context.
    If the context does not contain the answer, reply exactly:
    {OUTSIDE_REPOSITORY_MESSAGE}

    Provide a concise, practical answer using at most 3 bullet points or a single short paragraph.
    Limit the answer to approximately 150 words (or ~800 characters). If the answer exceeds this, summarize the key points.
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
