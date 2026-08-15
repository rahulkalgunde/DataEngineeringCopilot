"""Generic GitHub source preparer: manifest → parse → chunk (no embedding).

The Spark source routes through ``SparkChunker`` (full fidelity: guides, API
references, code examples, and SQL function references with the FunctionRegistry
resolution); all other GitHub sources use ``HeaderAwareChunker`` with metadata
attached post-chunking (its ``chunk()`` takes no metadata parameter).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
from data_engineering_copilot.infrastructure.spark_source_resolver import (
    SparkFileRecord,
    SparkManifest,
    SparkSourceResolver,
)
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.spark_chunker import SparkChunker
from data_engineering_copilot.services.spark_index_builder import (
    _FUNCTION_REGISTRY_RELATIVE_PATH,
    CoverageRecord,
)
from data_engineering_copilot.services.spark_metadata import derive_spark_metadata

_SPARK_SLUG = "spark"


class GithubSourcePreparer:
    """Materialize a pinned GitHub source and chunk it into a ``PreparedSource``.

    Parameters
    ----------
    config:
        Pinned GitHub source (``PinnedSourceConfig`` of type ``github``).
    cache_dir:
        Directory under which the materialized source tree is cached.
    generation:
        Generation ID stamped on every produced chunk.
    parser:
        Injected native parser (defaults to ``NativeDocumentParser``).
    header_chunker:
        Injected header chunker used for Spark guides and all non-Spark sources.
    """

    def __init__(
        self,
        config: PinnedSourceConfig,
        cache_dir: Path,
        generation: str,
        parser: NativeDocumentParser | None = None,
        header_chunker: HeaderAwareChunker | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = Path(cache_dir)
        self._generation = generation
        self._parser = parser or NativeDocumentParser()
        self._header_chunker = header_chunker or HeaderAwareChunker()

    async def prepare(self) -> PreparedSource:
        """Materialize the release and chunk every selected file."""
        resolver = SparkSourceResolver(self._config, self._cache_dir)
        manifest = resolver.resolve()
        chunks, coverage = await self._chunk_manifest(manifest)
        return PreparedSource(
            slug=self._config.slug,
            source_name=self._config.name,
            generation=self._generation,
            commit=self._config.commit,
            chunks=tuple(chunks),
            coverage=tuple(coverage),
        )

    async def _chunk_manifest(self, manifest: SparkManifest) -> tuple[list[DocumentChunk], list[CoverageRecord]]:
        if self._config.slug == _SPARK_SLUG:
            return await self._chunk_spark(manifest)
        return await self._chunk_generic(manifest)

    async def _chunk_spark(self, manifest: SparkManifest) -> tuple[list[DocumentChunk], list[CoverageRecord]]:
        chunker = SparkChunker(
            header_chunker=self._header_chunker,
            function_registry_text=self._function_registry_text(manifest),
        )
        chunks: list[DocumentChunk] = []
        chunk_count_by_path: dict[str, int] = {}
        for record in manifest.files:
            parsed = self._parse_record(record)
            if _is_rst(record.relative_path) and record.doc_type == "guide":
                parsed = replace(parsed, text=_rst_to_markdown_headings(parsed.text))
            metadata = derive_spark_metadata(record, self._config, title=parsed.title, text=parsed.text)
            doc_chunks = await chunker.chunk(parsed, metadata)
            doc_chunks = [self._attach_spark_metadata(chunk) for chunk in doc_chunks]
            chunk_count_by_path[record.relative_path] = len(doc_chunks)
            chunks.extend(doc_chunks)
        return chunks, self._build_coverage(manifest, chunk_count_by_path)

    async def _chunk_generic(self, manifest: SparkManifest) -> tuple[list[DocumentChunk], list[CoverageRecord]]:
        chunks: list[DocumentChunk] = []
        chunk_count_by_path: dict[str, int] = {}
        for record in manifest.files:
            parsed = self._parse_record(record)
            if _is_rst(record.relative_path):
                parsed = replace(parsed, text=_rst_to_markdown_headings(parsed.text))
            doc_chunks = await self._header_chunker.chunk(parsed)
            doc_chunks = [self._attach_generic_metadata(chunk) for chunk in doc_chunks]
            chunk_count_by_path[record.relative_path] = len(doc_chunks)
            chunks.extend(doc_chunks)
        return chunks, self._build_coverage(manifest, chunk_count_by_path)

    def _parse_record(self, record: SparkFileRecord) -> ParsedDocument:
        native = self._parser.parse(record.absolute_path, record.doc_type)
        return ParsedDocument(
            source_name=self._config.name,
            title=native.title,
            url=record.source_url,
            text=native.text,
            sections=(),
            doc_type=record.doc_type,
            language=record.language,
            spark_version="",
            module="",
            source_commit=self._config.commit,
            file_path=record.relative_path,
            license=self._config.license,
        )

    def _attach_generic_metadata(self, chunk: DocumentChunk) -> DocumentChunk:
        return replace(
            chunk,
            doc_type=chunk.doc_type or "guide",
            language=chunk.language or "conceptual",
            source_commit=self._config.commit,
            file_path=chunk.file_path,
            license=self._config.license,
            index_generation=self._generation,
            chunker_version="header-aware-v1",
        )

    def _attach_spark_metadata(self, chunk: DocumentChunk) -> DocumentChunk:
        return replace(
            chunk,
            index_generation=self._generation,
            chunker_version="spark-chunker-v1",
        )

    @staticmethod
    def _function_registry_text(manifest: SparkManifest) -> str | None:
        registry = manifest.root / _FUNCTION_REGISTRY_RELATIVE_PATH
        if not registry.is_file():
            return None
        return registry.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _build_coverage(
        manifest: SparkManifest,
        chunk_count_by_path: dict[str, int],
    ) -> list[CoverageRecord]:
        records: list[CoverageRecord] = []
        for record in manifest.files:
            chunk_count = chunk_count_by_path.get(record.relative_path, 0)
            records.append(
                CoverageRecord(
                    relative_path=record.relative_path,
                    representation="native",
                    doc_type=record.doc_type,
                    canonical_url=record.source_url,
                    status="indexed" if chunk_count > 0 else "no_content",
                    chunk_count=chunk_count,
                    content_hash="",
                    failure_reason="" if chunk_count > 0 else "parsed to empty text",
                )
            )
        return records


# RST heading underline characters mapped to Markdown heading levels. The
# HeaderAwareChunker only splits on ``#`` headings, so RST guides (Airflow) are
# converted to Markdown headings before chunking.
_RST_UNDERLINE_LEVELS = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "'": 6}


def _is_rst(relative_path: str) -> bool:
    return Path(relative_path).suffix in {".rst", ".rst.txt"}


def _rst_to_markdown_headings(text: str) -> str:
    """Convert RST underlined headings into Markdown ``#`` headings."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and i + 1 < len(lines):
            underline = lines[i + 1].strip()
            chars = set(underline)
            if len(chars) == 1 and chars & set(_RST_UNDERLINE_LEVELS) and len(underline) >= len(line):
                level = _RST_UNDERLINE_LEVELS[chars.pop()]
                out.append("#" * level + " " + line)
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)
