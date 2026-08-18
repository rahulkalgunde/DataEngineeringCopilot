"""Metadata-based chunker router.

Selects the chunking strategy for a parsed document by its metadata, in fixed
precedence order: Spark ``doc_type`` → JSON/structured → explicit code language
→ Markdown/RST guide → the configured generic strategy. Unknown metadata always
falls through to the configured generic strategy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, cast

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.services.spark_chunker import SparkChunker

logger = logging.getLogger(__name__)


class ChunkerStrategy(Protocol):
    async def chunk(
        self, document: ParsedDocument, precomputed_embeddings: list[list[float]] | None = None
    ) -> list[DocumentChunk]: ...
    def extract_sentences(self, text: str) -> list[str] | None: ...


@dataclass(frozen=True)
class ChunkerRoute:
    """A routing decision: which strategy to use and why."""

    key: str
    strategy: ChunkerStrategy
    reason: str


# Structured (JSON) signals. The text sniff only runs on small inputs so
# routing stays cheap on ordinary markdown pages.
_STRUCTURED_DOC_TYPES = frozenset({"json", "structured", "ndjson"})
_STRUCTURED_URL_SUFFIXES = (".json", ".jsonl", ".ndjson")
_MAX_JSON_SNIFF_CHARS = 8192

# Explicit code-language signals, mirroring DocumentChunker._detect_language.
_CODE_LANGUAGES = frozenset(
    {"python", "scala", "java", "r", "sql", "rust", "go", "typescript", "javascript", "c", "cpp", "csharp"}
)
_CODE_URL_MARKERS = ("/api/python/", "/pyspark", "/api/scala/", "/api/java/", "/api/r/")
_CODE_FILE_SUFFIXES = (
    ".py",
    ".scala",
    ".java",
    ".r",
    ".sql",
    ".rs",
    ".go",
    ".ts",
    ".js",
    ".c",
    ".cpp",
    ".cs",
)

# Markdown/RST guide signals.
_GUIDE_DOC_TYPES = frozenset({"guide"})
_GUIDE_URL_SUFFIXES = (".md", ".mdx", ".markdown", ".rst", ".rst.txt")


class ChunkerRouter:
    """Routes documents to chunking strategies by metadata precedence.

    Parameters
    ----------
    generic_strategy:
        The configured fallback strategy used for any document with unknown
        metadata.
    structured_strategy:
        Strategy for JSON/structured documents (``StructuredDataChunker``).
    code_strategy:
        Language-aware strategy for explicit code documents (``DocumentChunker``).
    guide_strategy:
        Strategy for Markdown/RST guide documents (``HeaderAwareChunker``).
    spark_chunker:
        Spark chunker. When wired, any document carrying a ``doc_type`` routes
        to Spark first (mirroring the ingestion service's Spark path). Must be
        the same instance wired into ``AsyncIngestionService``.
    """

    def __init__(
        self,
        generic_strategy: ChunkerStrategy,
        structured_strategy: ChunkerStrategy | None = None,
        code_strategy: ChunkerStrategy | None = None,
        guide_strategy: ChunkerStrategy | None = None,
        spark_chunker: SparkChunker | None = None,
    ) -> None:
        if generic_strategy is None:
            raise ValueError("generic_strategy is required")
        self._generic = generic_strategy
        self._structured = structured_strategy
        self._code = code_strategy
        self._guide = guide_strategy
        self._spark_chunker = spark_chunker

    def route(self, document: ParsedDocument) -> ChunkerRoute:
        if self._spark_chunker is not None and document.doc_type:
            return ChunkerRoute(
                key="spark",
                strategy=cast(ChunkerStrategy, self._spark_chunker),
                reason=f"Spark doc_type present: {document.doc_type}",
            )
        if self._structured is not None and _looks_structured(document):
            return ChunkerRoute(
                key="structured",
                strategy=self._structured,
                reason="JSON/structured content detected",
            )
        if self._code is not None and _is_code_document(document):
            return ChunkerRoute(
                key="code",
                strategy=self._code,
                reason=f"explicit code language: {document.language or document.url}",
            )
        if self._guide is not None and _is_guide_document(document):
            return ChunkerRoute(
                key="guide",
                strategy=self._guide,
                reason="Markdown/RST guide detected",
            )
        return ChunkerRoute(key="generic", strategy=self._generic, reason="configured generic strategy")


def _looks_structured(document: ParsedDocument) -> bool:
    if document.doc_type and document.doc_type.lower() in _STRUCTURED_DOC_TYPES:
        return True
    url = document.url.lower().split("?")[0]
    if url.endswith(_STRUCTURED_URL_SUFFIXES):
        return True
    text = document.text.strip()
    if not text or len(text) > _MAX_JSON_SNIFF_CHARS:
        return False
    if text[0] not in ("{", "["):
        return False
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _is_code_document(document: ParsedDocument) -> bool:
    if document.language and document.language.lower() in _CODE_LANGUAGES:
        return True
    url = document.url.lower()
    if any(marker in url for marker in _CODE_URL_MARKERS):
        return True
    path = (document.file_path or "").lower()
    return path.endswith(_CODE_FILE_SUFFIXES)


def _is_guide_document(document: ParsedDocument) -> bool:
    if document.doc_type and document.doc_type.lower() in _GUIDE_DOC_TYPES:
        return True
    url = document.url.lower().split("?")[0]
    return url.endswith(_GUIDE_URL_SUFFIXES)
