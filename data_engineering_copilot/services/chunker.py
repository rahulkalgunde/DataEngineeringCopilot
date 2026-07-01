from __future__ import annotations

import hashlib
import logging
from enum import Enum

import nltk
from nltk.tokenize import sent_tokenize

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.utils.text import slugify


logger = logging.getLogger(__name__)

# Download required NLTK data on first import
def _setup_nltk_data():
    """Ensure required NLTK data is available."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            # Fallback to older punkt tokenizer if punkt_tab fails
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

_setup_nltk_data()


class ChunkingStrategy(Enum):
    """Enum for supported chunking strategies."""
    FIXED_SIZE = "fixed_size"  # Legacy word-based fixed-size chunking
    SENTENCE_PRESERVING = "sentence_preserving"  # New: sentence-boundary aware chunking


class DocumentChunker:
    """
    Improved document chunker with support for multiple strategies.
    
    Attributes:
        chunk_size_words: Target chunk size in words
        overlap_words: Overlap between chunks in words
        strategy: Chunking strategy to use
        min_chunk_words: Minimum chunk size to avoid empty/tiny chunks
    """
    
    def __init__(
        self,
        chunk_size_words: int,
        overlap_words: int,
        strategy: ChunkingStrategy | str = ChunkingStrategy.SENTENCE_PRESERVING,
        min_chunk_words: int = 10,
    ) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be >= 0 and less than chunk_size_words")
        if min_chunk_words < 0:
            raise ValueError("min_chunk_words must be non-negative")
        if min_chunk_words > chunk_size_words:
            raise ValueError("min_chunk_words must not exceed chunk_size_words")
            
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words
        
        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                self.strategy = ChunkingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
        else:
            self.strategy = strategy

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk a parsed document using the configured strategy.
        
        Args:
            document: ParsedDocument to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document)
        elif self.strategy == ChunkingStrategy.SENTENCE_PRESERVING:
            return self._chunk_sentence_preserving(document)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")

    def _chunk_fixed_size(self, document: ParsedDocument) -> list[DocumentChunk]:
        """
        Legacy fixed-size word-based chunking.
        Splits text into fixed word chunks regardless of sentence boundaries.
        """
        words = document.text.split()
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        step = self.chunk_size_words - self.overlap_words

        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            text = " ".join(words[start:end])
            
            # Apply quality validation
            if self._is_valid_chunk(text):
                chunk_id = self._chunk_id(document, len(chunks))
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_name=document.source_name,
                        title=document.title,
                        url=document.url,
                        text=text,
                    )
                )
            
            if end == len(words):
                break
            start += step
            index += 1

        logger.info(
            "Chunked document (fixed-size) source=%s url=%s title=%r words=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            len(words),
            len(chunks),
        )
        return chunks

    def _chunk_sentence_preserving(self, document: ParsedDocument) -> list[DocumentChunk]:
        """
        Sentence-boundary aware chunking.
        
        Strategy:
        1. Split text into sentences using NLTK
        2. Group sentences into chunks respecting target word size
        3. Preserve paragraph boundaries when possible
        4. Apply min/max chunk size constraints
        5. Validate chunk quality before inclusion
        
        Returns:
            List of DocumentChunk objects with sentence boundaries preserved
        """
        try:
            sentences = sent_tokenize(document.text)
        except Exception as e:
            logger.warning(
                "Sentence tokenization failed for url=%s, falling back to fixed-size: %s",
                document.url,
                str(e),
            )
            return self._chunk_fixed_size(document)
        
        if not sentences:
            logger.warning("No sentences found in document url=%s", document.url)
            return []
        
        chunks: list[DocumentChunk] = []
        current_chunk_sentences: list[str] = []
        current_chunk_words = 0
        step = max(1, self.chunk_size_words - self.overlap_words)
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds target, finalize current chunk
            if current_chunk_words + sentence_words > self.chunk_size_words and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                if self._is_valid_chunk(chunk_text):
                    chunk_id = self._chunk_id(document, len(chunks))
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_name=document.source_name,
                            title=document.title,
                            url=document.url,
                            text=chunk_text,
                        )
                    )
                
                # Start new chunk with overlap: keep the last N words from previous chunk
                current_chunk_sentences = []
                current_chunk_words = 0
                
                # Add overlap: reuse sentences from end of previous chunk if available
                if chunks and self.overlap_words > 0:
                    overlap_text = chunk_text.split()[-self.overlap_words:]
                    if overlap_text:
                        current_chunk_sentences.append(" ".join(overlap_text))
                        current_chunk_words = len(overlap_text)
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_words += sentence_words
        
        # Handle final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if self._is_valid_chunk(chunk_text):
                chunk_id = self._chunk_id(document, len(chunks))
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_name=document.source_name,
                        title=document.title,
                        url=document.url,
                        text=chunk_text,
                    )
                )
        
        logger.info(
            "Chunked document (sentence-preserving) source=%s url=%s title=%r sentences=%s chunks=%s",
            document.source_name,
            document.url,
            document.title,
            len(sentences),
            len(chunks),
        )
        return chunks

    def _is_valid_chunk(self, text: str) -> bool:
        """
        Validate chunk quality before inclusion.
        
        Rules:
        - Must have at least min_chunk_words words
        - Must not be empty after stripping
        - Must contain at least some alphanumeric content (not just punctuation)
        
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
        
        # Ensure chunk has meaningful content (not just punctuation)
        has_alphanumeric = any(c.isalnum() for c in text)
        if not has_alphanumeric:
            return False
        
        return True

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
        return f"{source}:{digest}:{index:04d}"


