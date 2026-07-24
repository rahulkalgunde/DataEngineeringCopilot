"""
Semantic chunker: Groups sentences by embedding similarity and topic coherence.

Strategy:
1. Split document into sentences
2. Embed each sentence using the configured embedding model
3. Cluster sentences by semantic similarity using cosine distance
4. Merge clusters into chunks respecting size constraints
5. Validate chunk quality before inclusion
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from nltk.tokenize import sent_tokenize

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunker: Groups sentences by embedding similarity (topic coherence).

    Unlike simple boundary-aware chunking, semantic chunking clusters semantically
    related sentences together, preserving topical coherence within chunks.

    Attributes:
        chunk_size_words: Target chunk size in words
        overlap_words: Overlap between chunks in words (semantic overlap)
        embedding_model: OllamaEmbeddings instance for encoding
        min_semantic_similarity: Minimum cosine similarity to group sentences (0.0-1.0)
        min_chunk_words: Minimum chunk size to avoid empty/tiny chunks
        max_chunk_words: Maximum chunk size (hard limit)
    """

    def __init__(
        self,
        chunk_size_words: int,
        overlap_words: int,
        embedding_model=None,
        min_semantic_similarity: float = 0.5,
        min_chunk_words: int = 20,
        max_chunk_words: int | None = None,
    ) -> None:
        """
        Initialize semantic chunker.

        Args:
            chunk_size_words: Target chunk size in words
            overlap_words: Overlap between chunks in words
            embedding_model: OllamaEmbeddings instance
            min_semantic_similarity: Min cosine similarity threshold for grouping (0-1)
            min_chunk_words: Minimum chunk size
            max_chunk_words: Maximum chunk size (optional, defaults to chunk_size_words * 1.5)
        """
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be >= 0 and less than chunk_size_words")
        if not 0.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be between 0.0 and 1.0")
        if min_chunk_words < 0:
            raise ValueError("min_chunk_words must be non-negative")
        if embedding_model is None:
            raise ValueError("embedding_model must not be None")

        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.embedding_model = embedding_model
        self.min_semantic_similarity = min_semantic_similarity
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words or int(chunk_size_words * 1.5)

    @staticmethod
    def extract_sentences(text: str) -> list[str] | None:
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning("Sentence tokenization failed: %s", str(e))
            return None
        return sentences

    async def chunk(
        self,
        document: ParsedDocument,
        precomputed_embeddings: list[list[float]] | None = None,
    ) -> list[DocumentChunk]:
        sentences = self.extract_sentences(document.text)
        if not sentences:
            logger.warning("No sentences found in document url=%s", document.url)
            return []

        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        elif self.embedding_model is not None:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.embedding_model.embed_texts):
                    embeddings = await self.embedding_model.embed_texts(sentences)
                else:
                    embeddings = self.embedding_model.embed_texts(sentences)
            except Exception as e:
                logger.warning(
                    "Embedding failed for url=%s, cannot perform semantic chunking: %s",
                    document.url,
                    str(e),
                )
                return []
        else:
            logger.error(
                "Semantic chunking requires either precomputed_embeddings or an embedding_model",
            )
            return []

        if len(embeddings) != len(sentences):
            logger.error(
                "Embedding count mismatch: got %d embeddings for %d sentences",
                len(embeddings),
                len(sentences),
            )
            return []

        # Cluster sentences by semantic similarity
        sentence_groups = self._cluster_sentences(sentences, embeddings)

        # Merge clusters into chunks respecting size constraints
        chunks = self._merge_clusters_into_chunks(document, sentence_groups)

        logger.info(
            "Chunked document (semantic) source=%s url=%s title=%r sentences=%s clusters=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            len(sentences),
            len(sentence_groups),
            len(chunks),
        )
        return chunks

    def _cluster_sentences(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
    ) -> list[list[int]]:
        """
        Cluster sentences by semantic similarity using greedy clustering.

        Algorithm:
        1. Start with first sentence in its own cluster
        2. For each subsequent sentence, compute similarity to all existing clusters
        3. If max similarity >= min_semantic_similarity, add to most similar cluster
        4. Otherwise, create new cluster

        Args:
            sentences: List of sentence texts
            embeddings: List of embedding vectors (same length as sentences)

        Returns:
            List of clusters, where each cluster is a list of sentence indices
        """
        if not embeddings:
            return []

        # Convert to numpy for efficient similarity computation
        embedding_array = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
        embedding_array = embedding_array / (norms + 1e-10)

        # Initialize clustering: each sentence starts in its own cluster
        clusters: list[list[int]] = [[0]]

        # Cluster remaining sentences
        for i in range(1, len(sentences)):
            # Compute similarity to each existing cluster (use cluster center)
            cluster_similarities = []
            for cluster_idx in range(len(clusters)):
                # Cluster center: mean of all embeddings in cluster
                cluster_embedding = np.mean(embedding_array[clusters[cluster_idx]], axis=0)
                # Normalize cluster center
                cluster_embedding = cluster_embedding / (np.linalg.norm(cluster_embedding) + 1e-10)
                # Cosine similarity
                similarity = np.dot(embedding_array[i], cluster_embedding)
                cluster_similarities.append(similarity)

            # Find most similar cluster
            max_similarity = max(cluster_similarities)
            most_similar_cluster_idx = cluster_similarities.index(max_similarity)

            # Add to most similar cluster if above threshold, otherwise create new cluster
            if max_similarity >= self.min_semantic_similarity:
                clusters[most_similar_cluster_idx].append(i)
            else:
                clusters.append([i])

        return clusters

    def _merge_clusters_into_chunks(
        self,
        document: ParsedDocument,
        sentence_groups: list[list[int]],
    ) -> list[DocumentChunk]:
        """
        Merge semantic clusters into chunks respecting size constraints.

        Strategy:
        1. Group clusters into chunks respecting target/max word count
        2. Validate chunk quality before inclusion
        3. For overlap: keep semantically similar sentences from previous chunk

        Args:
            document: Source document
            sentence_groups: List of sentence index clusters

        Returns:
            List of DocumentChunk objects
        """
        if not sentence_groups:
            return []

        # Group sentences by their cluster
        sentences = sent_tokenize(document.text)
        sentences_by_cluster = [[sentences[i] for i in cluster] for cluster in sentence_groups]

        chunks: list[DocumentChunk] = []
        current_chunk_clusters: list[list[str]] = []
        current_chunk_words = 0

        for cluster_sentences in sentences_by_cluster:
            cluster_text = " ".join(cluster_sentences)
            cluster_words = len(cluster_text.split())

            # If adding this cluster exceeds target size and we have content, finalize chunk
            if current_chunk_words + cluster_words > self.chunk_size_words and current_chunk_clusters:
                chunk_text = " ".join(" ".join(cluster) for cluster in current_chunk_clusters).strip()
                if self._is_valid_chunk(chunk_text):
                    chunk_id = self._chunk_id(document, len(chunks))
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_name=document.source_name,
                            title=document.title,
                            url=document.url,
                            text=f"[Source: {document.title}]\n{chunk_text}",
                        )
                    )

                # Start new chunk with overlap
                current_chunk_clusters = []
                current_chunk_words = 0

                # Semantic overlap: keep sentences from end of previous chunk if available
                if chunks and self.overlap_words > 0:
                    overlap_words = chunk_text.split()[-self.overlap_words :]
                    if overlap_words:
                        overlap_text = " ".join(overlap_words)
                        current_chunk_clusters.append([overlap_text])
                        current_chunk_words = len(overlap_words)

            # Add cluster to current chunk
            current_chunk_clusters.append(cluster_sentences)
            current_chunk_words += cluster_words

            # Hard limit: if we exceed max_chunk_words, finalize even mid-cluster
            if current_chunk_words > self.max_chunk_words:
                chunk_text = " ".join(" ".join(cluster) for cluster in current_chunk_clusters).strip()
                if self._is_valid_chunk(chunk_text):
                    chunk_id = self._chunk_id(document, len(chunks))
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_name=document.source_name,
                            title=document.title,
                            url=document.url,
                            text=f"[Source: {document.title}]\n{chunk_text}",
                        )
                    )
                current_chunk_clusters = []
                current_chunk_words = 0

        # Handle final chunk
        if current_chunk_clusters:
            chunk_text = " ".join(" ".join(cluster) for cluster in current_chunk_clusters).strip()
            if self._is_valid_chunk(chunk_text):
                chunk_id = self._chunk_id(document, len(chunks))
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_name=document.source_name,
                        title=document.title,
                        url=document.url,
                        text=f"[Source: {document.title}]\n{chunk_text}",
                    )
                )

        return chunks

    def _is_valid_chunk(self, text: str) -> bool:
        """
        Validate chunk quality before inclusion.

        Rules:
        - Must have at least min_chunk_words words
        - Must not be empty after stripping
        - Must contain at least some alphanumeric content

        Args:
            text: Chunk text to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        text = text.strip()
        if not text:
            return False

        words = text.split()
        if len(words) < self.min_chunk_words:
            return False

        # Ensure chunk has meaningful content
        has_alphanumeric = any(c.isalnum() for c in text)
        return has_alphanumeric

    def _chunk_id(self, document: ParsedDocument, index: int) -> str:
        """
        Generate deterministic chunk ID.

        Format: {source_slug}:{url_digest}:{index:04d}

        Args:
            document: Source document
            index: Chunk index within document

        Returns:
            Unique chunk identifier
        """
        digest = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
        source = slugify(document.source_name)
        return f"{source}:{digest}:semantic:{index:04d}"
