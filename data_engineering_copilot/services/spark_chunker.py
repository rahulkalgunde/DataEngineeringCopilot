"""Stream-specific chunking for Spark guide, API, example, and SQL function docs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.domain.protocols import ChunkerProtocol
from data_engineering_copilot.services.spark_metadata import SparkMetadata

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# FunctionRegistry.scala registration regexes. The ``expression[...]`` form maps
# a case class to its SQL name(s); builder registrations map a *builder object*
# to SQL name(s) (generators and expressions built from ExpressionBuilder).
_EXPR_REG_RE = re.compile(r'expression\[([A-Za-z0-9_]+)\]\(\s*"([^"]+)"')
_BUILDER_REG_RE = re.compile(
    r'(?:expressionBuilder|expressionGeneratorBuilderOuter|generatorBuilder)\(\s*"([^"]+)",\s*([A-Za-z0-9_]+)',
    re.DOTALL,
)
_INTERNAL_REG_RE = re.compile(r'registerInternalExpression\[([A-Za-z0-9_]+)\]\(\s*"([^"]+)"', re.DOTALL)

_FUNC_TOKEN = "_FUNC_"
_ANNOTATION_RE = re.compile(r"@ExpressionDescription\(")
_CASE_CLASS_RE = re.compile(r"\s*(?:case\s+)?class\s+([A-Za-z0-9_]+)\s*(?:private\s*)?\(")
_OBJECT_RE = re.compile(r"\s*object\s+([A-Za-z0-9_]+)\s+extends\s+([A-Za-z0-9_]+)")
_BUILDER_SUPERTYPES = ("ExpressionBuilder", "GeneratorBuilder", "ExpressionBuilderBase")


@dataclass(frozen=True)
class SparkChunker:
    """Route each Spark document to the correct chunking strategy."""

    header_chunker: ChunkerProtocol
    function_registry_text: str | None = None

    async def chunk(
        self,
        document: ParsedDocument,
        metadata: SparkMetadata,
    ) -> list[DocumentChunk]:
        """Chunk a parsed Spark document according to its document type.

        Returns an empty list for empty document text. Raises ``ValueError``
        for unsupported ``doc_type`` values.
        """
        if not document.text.strip():
            return []

        if metadata.doc_type == "guide":
            chunks = await self._chunk_guide(document, metadata)
        elif metadata.doc_type == "api_reference":
            chunks = self._chunk_api(document, metadata)
        elif metadata.doc_type == "code_example":
            chunks = self._chunk_code(document, metadata, chunk_type="code")
        elif metadata.doc_type == "sql_function_ref":
            chunks = _chunk_sql_functions(
                document,
                metadata,
                self.function_registry_text,
                chunk_type="api",
            )
        else:
            raise ValueError(f"Unsupported doc_type for Spark chunking: {metadata.doc_type!r}")

        return self._number_chunks(chunks)

    # ------------------------------------------------------------------
    # Stream-specific chunking
    # ------------------------------------------------------------------

    async def _chunk_guide(self, document: ParsedDocument, metadata: SparkMetadata) -> list[DocumentChunk]:
        """Header-aware chunking for conceptual guides."""
        chunks = await self.header_chunker.chunk(document)
        if not chunks:
            return []
        return [self._with_metadata(chunk, metadata) for chunk in chunks]

    def _chunk_api(self, document: ParsedDocument, metadata: SparkMetadata) -> list[DocumentChunk]:
        """Group API signatures, parameters, description, and examples."""
        return self._split_into_chunks(document, metadata, chunk_type="api")

    def _chunk_code(self, document: ParsedDocument, metadata: SparkMetadata, chunk_type: str) -> list[DocumentChunk]:
        """Split code at top-level boundaries with a fixed-size fallback."""
        return self._split_into_chunks(document, metadata, chunk_type=chunk_type)

    def _split_into_chunks(
        self,
        document: ParsedDocument,
        metadata: SparkMetadata,
        chunk_type: str,
    ) -> list[DocumentChunk]:
        """Split non-guide content into deterministic chunks."""
        text = document.text
        pieces = _split_python_top_level(text) if metadata.language == "python" else _split_on_blank_lines(text)
        if not pieces:
            pieces = [text]

        chunks: list[DocumentChunk] = []
        for index, piece in enumerate(pieces):
            if not piece.strip():
                continue
            content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunk_id = _deterministic_chunk_id(metadata, index, piece)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=piece,
                    content_hash=content_hash,
                    section_header=document.title,
                    chunk_type=chunk_type,
                    word_count=len(piece.split()),
                    heading_path=(),
                    chunk_index=index,
                    total_chunks=len(pieces),
                    doc_type=metadata.doc_type,
                    language=metadata.language,
                    spark_version=metadata.spark_version,
                    module=metadata.module,
                    source_commit=metadata.source_commit,
                    file_path=metadata.file_path,
                    license=metadata.license,
                    deployment_mode=metadata.deployment_mode,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _with_metadata(chunk: DocumentChunk, metadata: SparkMetadata) -> DocumentChunk:
        from dataclasses import replace

        return replace(
            chunk,
            doc_type=metadata.doc_type,
            language=metadata.language,
            spark_version=metadata.spark_version,
            module=metadata.module,
            source_commit=metadata.source_commit,
            file_path=metadata.file_path,
            license=metadata.license,
            deployment_mode=metadata.deployment_mode,
        )

    def _number_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        from dataclasses import replace

        total = len(chunks)
        return [replace(chunk, chunk_index=index, total_chunks=total) for index, chunk in enumerate(chunks)]


def chunk_spark_document(
    document: ParsedDocument,
    metadata: SparkMetadata,
    function_registry_text: str | None = None,
) -> list[DocumentChunk]:
    """Chunk a Spark document using a lightweight synchronous header splitter.

    Uses the same deterministic rules as ``SparkChunker`` but without the
    async header chunker, so it can run in CLI/offline contexts.
    """
    if not document.text.strip():
        return []

    chunks: list[DocumentChunk] = []
    if metadata.doc_type == "guide":
        sections = _split_markdown_sections(document.text)
        for index, (header, body) in enumerate(sections):
            if not body.strip():
                continue
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            chunk_id = _deterministic_chunk_id(metadata, index, body)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=header or document.title,
                    url=document.url,
                    text=body,
                    content_hash=content_hash,
                    section_header=header,
                    chunk_type="text",
                    word_count=len(body.split()),
                    heading_path=(header,) if header else (),
                    chunk_index=index,
                    total_chunks=len(sections),
                    doc_type=metadata.doc_type,
                    language=metadata.language,
                    spark_version=metadata.spark_version,
                    module=metadata.module,
                    source_commit=metadata.source_commit,
                    file_path=metadata.file_path,
                    license=metadata.license,
                    deployment_mode=metadata.deployment_mode,
                )
            )
    elif metadata.doc_type == "api_reference":
        for index, piece in enumerate(_split_python_top_level(document.text) or [document.text]):
            if not piece.strip():
                continue
            content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunk_id = _deterministic_chunk_id(metadata, index, piece)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=piece,
                    content_hash=content_hash,
                    section_header=document.title,
                    chunk_type="api",
                    word_count=len(piece.split()),
                    heading_path=(),
                    chunk_index=index,
                    total_chunks=max(1, len(_split_python_top_level(document.text) or [document.text])),
                    doc_type=metadata.doc_type,
                    language=metadata.language,
                    spark_version=metadata.spark_version,
                    module=metadata.module,
                    source_commit=metadata.source_commit,
                    file_path=metadata.file_path,
                    license=metadata.license,
                    deployment_mode=metadata.deployment_mode,
                )
            )
    elif metadata.doc_type == "code_example":
        for index, piece in enumerate(_split_python_top_level(document.text) or [document.text]):
            if not piece.strip():
                continue
            content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunk_id = _deterministic_chunk_id(metadata, index, piece)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    title=document.title,
                    url=document.url,
                    text=piece,
                    content_hash=content_hash,
                    section_header=document.title,
                    chunk_type="code",
                    word_count=len(piece.split()),
                    heading_path=(),
                    chunk_index=index,
                    total_chunks=max(1, len(_split_python_top_level(document.text) or [document.text])),
                    doc_type=metadata.doc_type,
                    language=metadata.language,
                    spark_version=metadata.spark_version,
                    module=metadata.module,
                    source_commit=metadata.source_commit,
                    file_path=metadata.file_path,
                    license=metadata.license,
                    deployment_mode=metadata.deployment_mode,
                )
            )
    elif metadata.doc_type == "sql_function_ref":
        chunks = _chunk_sql_functions(
            document,
            metadata,
            function_registry_text,
            chunk_type="api",
        )
    else:
        raise ValueError(f"Unsupported doc_type for Spark chunking: {metadata.doc_type!r}")
    return chunks


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------


def _split_python_top_level(text: str) -> list[str]:
    """Split Python source at top-level ``def``/``class`` boundaries."""
    lines = text.splitlines()
    pieces: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("def ") or stripped.startswith("class ")) and (line[:1] not in (" ", "\t")) and current:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _split_on_blank_lines(text: str, max_lines: int = 200) -> list[str]:
    """Split non-Python code on blank-line boundaries capped by max_lines."""
    pieces: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip() and current and len(current) >= max_lines:
            pieces.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (header, body) sections at heading boundaries."""
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_parts: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m and m.group(1) in ("#", "##", "###"):
            if current_parts:
                sections.append((current_header, "\n".join(current_parts)))
            current_header = m.group(2).strip()
            current_parts = [line]
        else:
            current_parts.append(line)
    if current_parts:
        sections.append((current_header, "\n".join(current_parts)))
    return sections


def _deterministic_chunk_id(metadata: SparkMetadata, index: int, text: str) -> str:
    """Build a deterministic chunk ID from commit, path, index, and content."""
    digest = hashlib.sha256(f"{metadata.source_commit}|{metadata.file_path}|{index}|{text}".encode()).hexdigest()
    return f"spark-{metadata.source_commit[:8]}-{index}-{digest[:12]}"


# ---------------------------------------------------------------------------
# SQL function reference chunking (@ExpressionDescription sources)
# ---------------------------------------------------------------------------


def parse_function_registry(registry_text: str) -> dict[str, tuple[str, ...]]:
    """Parse ``FunctionRegistry.scala`` into a class/builder → SQL name map.

    Returns a mapping from case-class or builder-object name to the tuple of
    SQL function names it registers (canonical name first, then aliases). An
    empty dict is returned when no registrations can be parsed.
    """
    names_by_class: dict[str, list[str]] = {}
    names_by_builder: dict[str, list[str]] = {}
    names_by_internal: dict[str, list[str]] = {}

    for m in _EXPR_REG_RE.finditer(registry_text):
        names_by_class.setdefault(m.group(1), []).append(m.group(2))
    for m in _BUILDER_REG_RE.finditer(registry_text):
        names_by_builder.setdefault(m.group(2), []).append(m.group(1))
    for m in _INTERNAL_REG_RE.finditer(registry_text):
        names_by_internal.setdefault(m.group(1), []).append(m.group(2))

    merged: dict[str, list[str]] = {}
    for source in (names_by_class, names_by_internal):
        for key, names in source.items():
            merged.setdefault(key, []).extend(names)
    for key, names in names_by_builder.items():
        merged.setdefault(key, []).extend(names)

    result: dict[str, tuple[str, ...]] = {}
    for key, names in merged.items():
        result[key] = tuple(dict.fromkeys(names))
    return result


def _find_annotation_spans(text: str) -> list[tuple[int, int]]:
    """Return (start_line, end_line) spans of every ``@ExpressionDescription(`` block."""
    lines = text.splitlines()
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _ANNOTATION_RE.search(lines[i]):
            depth = 0
            started = False
            j = i
            while j < len(lines):
                for ch in lines[j]:
                    if ch == "(":
                        depth += 1
                        started = True
                    elif ch == ")":
                        depth -= 1
                        if started and depth == 0:
                            break
                if started and depth == 0:
                    break
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def _resolve_annotation_owner(lines: list[str], start: int, end: int) -> str | None:
    """Return the case-class or builder-object name an annotation belongs to."""
    # Prefer a builder object declared immediately after the annotation.
    for k in range(end + 1, min(end + 21, len(lines))):
        m = _OBJECT_RE.match(lines[k])
        if m and (m.group(1).endswith("Builder") or m.group(2).endswith("Builder")):
            return m.group(1)
    # Then the nearest case class after the annotation (generators declare the
    # class before the builder but some files annotate forward).
    for k in range(end + 1, min(end + 71, len(lines))):
        m = _CASE_CLASS_RE.match(lines[k])
        if m:
            return m.group(1)
    # Fall back to the nearest case class before the annotation.
    for k in range(start - 1, max(start - 71, -1), -1):
        m = _CASE_CLASS_RE.match(lines[k])
        if m:
            return m.group(1)
    return None


def _chunk_sql_functions(
    document: ParsedDocument,
    metadata: SparkMetadata,
    registry_text: str | None,
    chunk_type: str,
) -> list[DocumentChunk]:
    """Split an ``@ExpressionDescription`` source into one chunk per annotation.

    ``_FUNC_`` is replaced with the registered SQL name(s) resolved from
    ``FunctionRegistry.scala`` (aliases emit an additional chunk per name). A
    chunk whose owner cannot be resolved keeps ``_FUNC_`` literal — content is
    never dropped. Files without annotations fall back to blank-line splitting.
    """
    spans = _find_annotation_spans(document.text)
    if not spans:
        pieces = _split_on_blank_lines(document.text) or [document.text]
        return _build_plain_chunks(document, metadata, pieces, chunk_type)

    registry = parse_function_registry(registry_text) if registry_text else {}
    lines = document.text.splitlines()
    chunks: list[DocumentChunk] = []
    for index, (start, end) in enumerate(spans):
        owner = _resolve_annotation_owner(lines, start, end)
        names = registry.get(owner, ()) if owner else ()
        block = "\n".join(lines[start : end + 1])
        if not names or _FUNC_TOKEN not in block:
            chunk_text = block
            header = owner or document.title
            chunks.append(_build_chunk(document, metadata, index, chunk_text, chunk_type, header))
            continue
        for name in names:
            chunk_text = block.replace(_FUNC_TOKEN, name)
            header = f"{name} ({owner})" if owner else name
            chunks.append(_build_chunk(document, metadata, index, chunk_text, chunk_type, header))
    return chunks


def _build_plain_chunks(
    document: ParsedDocument,
    metadata: SparkMetadata,
    pieces: list[str],
    chunk_type: str,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for index, piece in enumerate(pieces):
        if not piece.strip():
            continue
        chunks.append(_build_chunk(document, metadata, index, piece, chunk_type, document.title))
    return chunks


def _build_chunk(
    document: ParsedDocument,
    metadata: SparkMetadata,
    index: int,
    text: str,
    chunk_type: str,
    header: str,
) -> DocumentChunk:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk_id = _deterministic_chunk_id(metadata, index, text)
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name=document.source_name,
        title=header,
        url=document.url,
        text=text,
        content_hash=content_hash,
        section_header=header,
        chunk_type=chunk_type,
        word_count=len(text.split()),
        heading_path=(),
        chunk_index=index,
        total_chunks=0,
        doc_type=metadata.doc_type,
        language=metadata.language,
        spark_version=metadata.spark_version,
        module=metadata.module,
        source_commit=metadata.source_commit,
        file_path=metadata.file_path,
        license=metadata.license,
        deployment_mode=metadata.deployment_mode,
    )
