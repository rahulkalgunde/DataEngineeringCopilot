from __future__ import annotations

import logging

import chromadb
from chromadb.errors import InternalError

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk


logger = logging.getLogger(__name__)


class VectorStoreReadError(RuntimeError):
    """Raised when the persisted Chroma index cannot be read."""


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        logger.info("Opening Chroma vector store path=%s collection=%s", persist_directory, collection_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            logger.info("Skipping vector upsert because chunk list is empty")
            return
        if len(chunks) != len(embeddings):
            logger.error("Vector upsert length mismatch chunks=%s embeddings=%s", len(chunks), len(embeddings))
            raise ValueError("chunks and embeddings must have the same length")

        logger.info("Upserting chunks count=%s first_chunk_id=%s", len(chunks), chunks[0].chunk_id)
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_name": chunk.source_name,
                    "title": chunk.title,
                    "url": chunk.url,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        logger.info("Vector query started top_k=%s embedding_dimensions=%s", top_k, len(query_embedding))
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except InternalError as exc:
            if "Nothing found on disk" in str(exc):
                logger.exception("Vector query failed because Chroma index is incomplete")
                raise VectorStoreReadError(
                    "The ChromaDB index is incomplete or corrupted. Run `python main.py reset-index`, then ingest again."
                ) from exc
            logger.exception("Vector query failed with Chroma internal error")
            raise

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            confidence = max(0.0, min(1.0, 1.0 - float(distance)))
            chunk = DocumentChunk(
                chunk_id=str(metadata["chunk_id"]),
                source_name=str(metadata["source_name"]),
                title=str(metadata["title"]),
                url=str(metadata["url"]),
                text=text,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, distance=float(distance), confidence=confidence))
        logger.info(
            "Vector query completed results=%s top_confidence=%.4f",
            len(retrieved),
            retrieved[0].confidence if retrieved else 0.0,
        )
        return retrieved

    def count(self) -> int:
        try:
            count = self.collection.count()
            logger.info("Vector store count completed count=%s", count)
            return count
        except InternalError as exc:
            if "Nothing found on disk" in str(exc):
                logger.exception("Vector count failed because Chroma index is incomplete")
                raise VectorStoreReadError(
                    "The ChromaDB index is incomplete or corrupted. Run `python main.py reset-index`, then ingest again."
                ) from exc
            logger.exception("Vector count failed with Chroma internal error")
            raise
