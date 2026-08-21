"""True sentence-preserving chunker.

Splits documents on NLTK sentence boundaries and greedily packs complete
sentences into budget-bounded chunks, so sentences stay together whenever the
limits allow. A single sentence that exceeds the budget is split losslessly
with the shared token-budget utility — content is never silently discarded.

Sentence spans are computed over the original text and extended to cover
inter-sentence whitespace, so ``"".join(chunk_texts)`` reconstructs the
normalized source exactly.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import pathlib
import uuid

import nltk
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    count_tokens,
    split_text_losslessly,
)
from data_engineering_copilot.services.code_span_masker import mask_code_spans, unmask_code_spans

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _ensure_punkt_tab() -> None:
    """Ensure NLTK's ``punkt_tab`` tokenizer data is available (idempotent)."""
    try:
        sent_tokenize("A short test sentence.")
    except LookupError:
        for candidate in nltk.data.path:
            candidate = pathlib.Path(candidate)
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                nltk.download("punkt_tab", quiet=True, download_dir=str(candidate))
            except Exception:
                continue
            if (candidate / "tokenizers" / "punkt_tab").is_dir():
                break
        sent_tokenize("A short test sentence.")


class SentencePreservingChunker:
    """Chunker that preserves complete sentences within budget-bounded chunks.

    Parameters
    ----------
    max_tokens:
        Hard token budget per chunk (matches the embedding input budget).
    max_chars:
        Hard character budget per chunk. Sentences that fit whole are never
        cut; a sentence alone over budget is split losslessly.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_tokens = max_tokens
        self.max_chars = max_chars

    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]:
        """Chunk *document* by packing complete sentences into bounded chunks."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_chunk, document)

    def extract_sentences(self, text: str) -> list[str] | None:
        """Sentence pre-extraction is not supported: no embeddings are needed."""
        return None

    def _sync_chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = document.text.strip()
        if not text:
            return []

        sentences = self._split_sentences_losslessly(text)
        if not sentences:
            return []

        pieces = self._pack_sentences(sentences)

        chunks: list[DocumentChunk] = []
        total_chunks = len(pieces)
        cursor = 0
        for index, piece in enumerate(pieces):
            start_offset = cursor
            end_offset = cursor + len(piece)
            cursor = end_offset
            chunk = DocumentChunk(
                chunk_id=self._chunk_id(document, index),
                source_name=document.source_name,
                title=document.title,
                url=document.url,
                text=piece,
                start_offset=start_offset,
                end_offset=end_offset,
                content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                chunk_type="text",
                word_count=len(piece.split()),
                chunk_index=index,
                total_chunks=total_chunks,
                doc_type=document.doc_type,
                language=document.language,
                file_path=document.file_path,
                token_count=count_tokens(piece),
                character_count=len(piece),
            )
            assert chunk.text == document.text[chunk.start_offset : chunk.end_offset]
            chunks.append(chunk)

        logger.info(
            "Sentence-preserving chunking: source=%s url=%s sentences=%d chunks=%d",
            document.source_name,
            document.url,
            len(sentences),
            len(chunks),
        )
        return chunks

    def _split_sentences_losslessly(self, text: str) -> list[str]:
        """Split *text* on sentence boundaries, preserving every character.

        Code spans are masked before tokenization so a sentence boundary never
        falls inside code; each sentence is unmasked afterwards. Inter-sentence
        whitespace is attached to the following sentence so
        ``"".join(sentences) == text`` exactly. Falls back to ``[text]`` when
        sentence tokenization is unavailable so content is never dropped.
        """
        masked = mask_code_spans(text)
        try:
            _ensure_punkt_tab()
            spans = list(PunktSentenceTokenizer().span_tokenize(masked.text))
        except Exception as exc:
            logger.warning("Sentence tokenization failed: %s", exc)
            return [text]
        if not spans:
            return [text]

        sentences: list[str] = []
        for index, (_start, end) in enumerate(spans):
            seg_start = spans[index - 1][1] if index > 0 else 0
            sentences.append(unmask_code_spans(masked.text[seg_start:end], masked))
        return sentences

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        """Greedily pack complete sentences into budget-bounded pieces."""
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if current and not self._fits(current + sentence):
                pieces.append(current)
                current = ""
            if self._fits(sentence):
                current += sentence
                continue
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(self._split_sentence_losslessly(sentence))
        if current:
            pieces.append(current)
        return pieces

    def _split_sentence_losslessly(self, sentence: str) -> list[str]:
        """Split an over-budget sentence losslessly, preserving boundary whitespace.

        ``split_text_losslessly`` normalizes (strips) its input, which would
        otherwise drop the inter-sentence whitespace attached to this sentence;
        re-attach the stripped leading/trailing whitespace so
        ``"".join(pieces)`` still reconstructs the source.
        """
        lead = sentence[: len(sentence) - len(sentence.lstrip())]
        trail = sentence[len(sentence.rstrip()) :]
        segments = split_text_losslessly(sentence.strip(), max_tokens=self.max_tokens, max_chars=self.max_chars)
        if lead:
            segments[0] = lead + segments[0]
        if trail:
            segments[-1] = segments[-1] + trail
        return segments

    def _fits(self, text: str) -> bool:
        return len(text) <= self.max_chars and count_tokens(text) <= self.max_tokens

    @staticmethod
    def _chunk_id(document: ParsedDocument, index: int) -> str:
        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, document.url)
        return str(uuid.uuid5(namespace, f"{document.source_name}:sent:{index:04d}"))
