"""Offline Spark index builder: manifest → parse → chunk → embed → Qdrant."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.domain.protocols import EmbedderProtocol
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import MAX_SAFE_TOKENS
from data_engineering_copilot.infrastructure.native_document_parser import NativeDocumentParser
from data_engineering_copilot.infrastructure.spark_html_parser import RenderedParseResult, SparkHtmlParser
from data_engineering_copilot.infrastructure.spark_rendered_builder import (
    RenderedFileRecord,
    RenderedManifest,
)
from data_engineering_copilot.infrastructure.spark_source_resolver import (
    SparkFileRecord,
    SparkManifest,
    SparkSourceResolver,
)
from data_engineering_copilot.infrastructure.token_budget import DEFAULT_MAX_CHARS, count_tokens, split_text_losslessly
from data_engineering_copilot.services.spark_chunker import SparkChunker
from data_engineering_copilot.services.spark_metadata import derive_spark_metadata
from data_engineering_copilot.services.spark_rendered_chunker import SparkRenderedChunker

# Relative path (inside the pinned Spark tree) of the function registry that
# maps expression case classes/builder objects to their registered SQL names.
_FUNCTION_REGISTRY_RELATIVE_PATH = (
    "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/analysis/FunctionRegistry.scala"
)

if TYPE_CHECKING:
    from data_engineering_copilot.config.settings import (
        SparkRenderedBuildConfig,
        SparkRenderedSourceConfig,
        SparkSourceConfig,
    )
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore


@dataclass(frozen=True)
class _MergedDocument:
    """A single corpus document selected for chunking after the hybrid merge."""

    key: str  # canonical docs URL used to match native and rendered records
    parsed: ParsedDocument
    metadata_relative_path: str
    metadata_doc_type: str
    metadata_language: str
    representation: str  # "native" | "rendered"
    chunker: SparkChunker | SparkRenderedChunker


@dataclass(frozen=True)
class CoverageRecord:
    """Per-file coverage status for a selected native or rendered record.

    ``status`` is one of:
    - ``indexed``: produced at least one chunk;
    - ``replaced``: native file superseded by its rendered counterpart;
    - ``no_content``: parsed to empty text (skipped, not a failure);
    - ``zero_chunks``: parsed to non-empty text but produced no chunks
      (validation failure);
    - ``missing_output``: rendered build output file does not exist
      (validation failure).
    """

    relative_path: str
    representation: str  # "native" | "rendered"
    doc_type: str
    canonical_url: str
    status: str
    chunk_count: int
    content_hash: str
    failure_reason: str = ""


@dataclass(frozen=True)
class IndexBuildReport:
    """Result of an offline Spark index build."""

    generation: str
    manifest_hash: str
    chunk_count: int
    source_file_count: int
    bm25_vocabulary_size: int
    qdrant_collection: str
    validation_passed: bool
    rendered_file_count: int = 0
    coverage_count: int = 0


class SparkIndexBuilder:
    """Build a complete dense + BM25 Qdrant generation from a Spark manifest."""

    def __init__(
        self,
        config: SparkSourceConfig,
        resolver: SparkSourceResolver,
        parser: NativeDocumentParser,
        chunker: SparkChunker,
        store: AsyncQdrantVectorStore,
        generation: str,
        embedder: EmbedderProtocol,
        embedding_batch_size: int = 128,
        max_embed_tokens: int = MAX_SAFE_TOKENS,
        max_embed_chars: int = DEFAULT_MAX_CHARS,
        rendered_config: SparkRenderedSourceConfig | None = None,
        rendered_manifest: RenderedManifest | None = None,
        chunks_path: Path | None = None,
    ) -> None:
        self._config = config
        self._resolver = resolver
        self._parser = parser
        self._chunker = chunker
        self._store = store
        self._generation = generation
        self._embedder = embedder
        self._embedding_batch_size = embedding_batch_size
        self._max_embed_tokens = max_embed_tokens
        self._max_embed_chars = max_embed_chars
        self._rendered_config = rendered_config
        self._rendered_manifest = rendered_manifest
        self._chunks_path = chunks_path

    async def build(self) -> IndexBuildReport:
        """Build a generation collection from the pinned Spark source.

        The build never activates the collection (activation is Phase 10) and
        never touches the live crawl frontier.
        """
        await self._store.initialize()
        manifest = self._resolver.resolve()
        return await self._build_from_manifest(manifest)

    async def build_from_manifest(self, manifest: SparkManifest) -> IndexBuildReport:
        """Build a generation collection from an existing manifest.

        Intended for tests and offline rebuilds that already have a manifest.
        """
        await self._store.initialize()
        return await self._build_from_manifest(manifest)

    async def _build_from_manifest(self, manifest: SparkManifest) -> IndexBuildReport:
        chunks, coverage = await self._chunk_all(manifest)
        chunks = self._dedup_by_content_hash(chunks)
        self._reject_duplicate_chunk_ids(chunks)

        # Split every chunk losslessly BEFORE fitting BM25/embedding so the
        # indexed segment texts are exactly what is embedded and persisted.
        # No content is truncated.
        normalized: list[DocumentChunk] = []
        for chunk in chunks:
            normalized.extend(self._normalize_chunk(chunk))
        self._reject_duplicate_chunk_ids(normalized)

        # Persist the exact segment list BEFORE embedding so validation can
        # prove persisted text == embedded text.
        self._write_chunks_jsonl(normalized)
        self._write_coverage(coverage)

        corpus_texts = [c.text for c in normalized]
        self._store.fit_bm25_corpus(corpus_texts)

        vectors = await self._embed_all(normalized)

        await self._store.upsert_frozen_chunks(normalized, vectors)
        validation = await self._store.validate_index_generation(len(normalized))
        bm25_vocab = 0
        if self._store._bm25 is not None:
            bm25_vocab = self._store._bm25.vocab_size

        rendered_file_count = len(self._rendered_manifest.files) if self._rendered_manifest is not None else 0
        self._write_build_report(manifest, normalized, validation, len(coverage), rendered_file_count, bm25_vocab)

        return IndexBuildReport(
            generation=self._generation,
            manifest_hash=manifest.manifest_hash,
            chunk_count=len(normalized),
            source_file_count=len(manifest.files),
            bm25_vocabulary_size=bm25_vocab,
            qdrant_collection=self._store._collection_name,
            validation_passed=bool(validation.get("passed")),
            rendered_file_count=rendered_file_count,
            coverage_count=len(coverage),
        )

    async def _chunk_all(self, manifest: SparkManifest) -> tuple[list[DocumentChunk], list[CoverageRecord]]:
        """Chunk the merged corpus and build per-file coverage records.

        Returns the flat chunk list and a ``CoverageRecord`` per selected
        native/rendered file so validation can prove every selected non-empty
        file produced chunks.
        """
        from dataclasses import replace

        native_chunker = self._with_function_registry(manifest)
        merged = self._merge_documents(manifest, native_chunker)
        chunks: list[DocumentChunk] = []
        chunk_count_by_key: dict[str, int] = {}
        for doc in merged:
            metadata = derive_spark_metadata(
                SparkFileRecord(
                    stream="",
                    relative_path=doc.metadata_relative_path,
                    absolute_path=Path("/nonexistent"),
                    doc_type=doc.metadata_doc_type,
                    language=doc.metadata_language,
                    source_url=doc.parsed.url,
                ),
                self._config,
                title=doc.parsed.title,
                text=doc.parsed.text,
            )
            doc_chunks = await doc.chunker.chunk(doc.parsed, metadata)
            chunk_count_by_key[doc.key] = len(doc_chunks)
            chunks.extend(replace(c, representation=doc.representation) for c in doc_chunks)
        coverage = self._build_coverage(manifest, merged, chunk_count_by_key)
        return chunks, coverage

    # ------------------------------------------------------------------
    # Hybrid native + rendered merge
    # ------------------------------------------------------------------

    def _merge_documents(
        self,
        manifest: SparkManifest,
        native_chunker: SparkChunker,
    ) -> list[_MergedDocument]:
        """Merge native and rendered records into one chunking corpus.

        Records are matched on their canonical docs URL. When a rendered page
        exists for a canonical URL and parses to valid main content, the
        rendered representation replaces the native one while retaining native
        provenance (source path + commit). Native-only records without a
        rendered counterpart and rendered-only pages are both retained.
        """
        docs = [self._native_document(record, native_chunker) for record in manifest.files]
        if self._rendered_manifest is None:
            return docs

        native_by_key = {self._native_canonical_url(record): record for record in manifest.files}
        docs_by_key: dict[str, _MergedDocument] = {doc.key: doc for doc in docs}

        for rendered in self._rendered_manifest.files:
            result = self._parse_rendered(rendered)
            if result is None:
                continue
            native = native_by_key.get(rendered.canonical_url)
            docs_by_key[rendered.canonical_url] = self._rendered_document(result, rendered, native)

        # Preserve deterministic order: native order first, then rendered-only.
        # Use the (possibly replaced) docs from ``docs_by_key``.
        ordered: list[_MergedDocument] = []
        seen_keys: set[str] = set()
        for doc in docs:
            merged_doc = docs_by_key.get(doc.key, doc)
            if merged_doc.key in seen_keys:
                continue
            ordered.append(merged_doc)
            seen_keys.add(merged_doc.key)
        for key in sorted(docs_by_key):
            if key not in seen_keys:
                ordered.append(docs_by_key[key])
        return ordered

    def _native_canonical_url(self, record: SparkFileRecord) -> str:
        """Map a native file record to its canonical docs URL.

        Guide pages render to ``<version>/<path>.html``; PySpark API modules
        render to ``<version>/api/python/reference/<module>.html``. Records
        with no rendered counterpart (examples, Scala internals, RST) fall back
        to their raw source URL so they never collide with rendered pages.
        """
        version = self._config.ref.lstrip("v")
        rel = record.relative_path
        if rel.startswith("docs/") and rel.endswith(".md"):
            page = rel[len("docs/") :]
            page = page[:-3] + ".html"
            return f"https://spark.apache.org/docs/{version}/{page}"
        if record.language == "python" and rel.startswith("python/pyspark/"):
            module = rel[len("python/") :]
            module = module[:-3] if module.endswith(".py") else module
            parts = module.split("/")
            parent = ".".join(parts[:-1])
            leaf = parts[-1]
            return f"https://spark.apache.org/docs/{version}/api/python/reference/{parent}/{leaf}.html"
        return record.source_url

    def _native_document(self, record: SparkFileRecord, chunker: SparkChunker) -> _MergedDocument:
        native = self._parser.parse(record.absolute_path, record.doc_type)
        parsed = ParsedDocument(
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
        return _MergedDocument(
            key=self._native_canonical_url(record),
            parsed=parsed,
            metadata_relative_path=record.relative_path,
            metadata_doc_type=record.doc_type,
            metadata_language=record.language,
            representation="native",
            chunker=chunker,
        )

    def _rendered_document(
        self,
        result: RenderedParseResult,
        rendered: RenderedFileRecord,
        native: SparkFileRecord | None,
    ) -> _MergedDocument:
        """Build a merged document for a rendered page.

        ``native`` provenance (source path) is retained whenever a rendered
        page replaces a native file so chunks still resolve to the repo path.
        """
        provenance_path = native.relative_path if native is not None else result.source_path
        parsed = ParsedDocument(
            source_name=self._config.name,
            title=result.title,
            url=result.canonical_url,
            text=result.text,
            sections=(),
            doc_type=rendered.doc_type,
            language=rendered.language,
            spark_version="",
            module="",
            source_commit=self._config.commit,
            file_path=provenance_path,
            license=self._config.license,
        )
        return _MergedDocument(
            key=result.canonical_url,
            parsed=parsed,
            metadata_relative_path=provenance_path,
            metadata_doc_type=rendered.doc_type,
            metadata_language=rendered.language,
            representation="rendered",
            chunker=SparkRenderedChunker(),
        )

    def _parse_rendered(self, rendered: RenderedFileRecord) -> RenderedParseResult | None:
        """Parse a rendered HTML file using its build's content-root settings."""
        build = _rendered_build(self._rendered_config, rendered.build)
        if build is None:
            return None
        parser = SparkHtmlParser(
            content_root_selector=build.content_root_selector,
            excluded_selectors=build.excluded_selectors,
        )
        html = rendered.absolute_path.read_text(encoding="utf-8", errors="replace")
        return parser.parse(html, rendered.canonical_url, rendered.relative_path)

    def _write_chunks_jsonl(self, chunks: list[DocumentChunk]) -> None:
        """Persist the exact segment list as JSONL before embedding."""
        if self._chunks_path is None:
            return
        self._chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self._chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_coverage(self, coverage: list[CoverageRecord]) -> None:
        """Persist per-file coverage records to ``coverage.json``."""
        if self._chunks_path is None:
            return
        from dataclasses import asdict

        path = self._chunks_path.parent / "coverage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(record) for record in coverage], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_build_report(
        self,
        manifest: SparkManifest,
        chunks: list[DocumentChunk],
        validation: dict[str, object],
        coverage_count: int,
        rendered_file_count: int,
        bm25_vocab: int,
    ) -> None:
        """Persist the build report artifact (``build_report.json``)."""
        if self._chunks_path is None:
            return
        report = {
            "generation": self._generation,
            "source_commit": self._config.commit,
            "native_manifest_hash": manifest.manifest_hash,
            "rendered_manifest_hash": (
                self._rendered_manifest.manifest_hash if self._rendered_manifest is not None else None
            ),
            "selected_file_count": len(manifest.files),
            "rendered_file_count": rendered_file_count,
            "final_chunk_count": len(chunks),
            "qdrant_point_count": validation.get("point_count"),
            "bm25_vocabulary_size": bm25_vocab,
            "embedding_dimension": self._store._embedding_dim(),
            "validation_result": bool(validation.get("passed")),
            "coverage_count": coverage_count,
        }
        path = self._chunks_path.parent / "build_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _build_coverage(
        self,
        manifest: SparkManifest,
        merged: list[_MergedDocument],
        chunk_count_by_key: dict[str, int],
    ) -> list[CoverageRecord]:
        """Build per-file coverage records for native and rendered records.

        Every selected native file gets a record: ``indexed`` when it produced
        chunks, ``replaced`` when a rendered page superseded it, ``no_content``
        when its parsed text was empty, or ``zero_chunks`` otherwise. Every
        rendered file gets ``indexed``, ``no_content``, ``zero_chunks``, or
        ``missing_output`` when its output file is absent.
        """
        merged_by_key = {doc.key: doc for doc in merged}
        records: list[CoverageRecord] = []

        for record in manifest.files:
            key = self._native_canonical_url(record)
            doc = merged_by_key.get(key)
            if doc is not None and doc.representation == "rendered":
                records.append(
                    CoverageRecord(
                        relative_path=record.relative_path,
                        representation="native",
                        doc_type=record.doc_type,
                        canonical_url=key,
                        status="replaced",
                        chunk_count=chunk_count_by_key.get(key, 0),
                        content_hash="",
                        failure_reason="superseded by rendered page",
                    )
                )
                continue
            chunk_count = chunk_count_by_key.get(key, 0)
            if chunk_count > 0:
                records.append(
                    CoverageRecord(
                        relative_path=record.relative_path,
                        representation="native",
                        doc_type=record.doc_type,
                        canonical_url=key,
                        status="indexed",
                        chunk_count=chunk_count,
                        content_hash="",
                    )
                )
            elif doc is not None and not doc.parsed.text.strip():
                records.append(
                    CoverageRecord(
                        relative_path=record.relative_path,
                        representation="native",
                        doc_type=record.doc_type,
                        canonical_url=key,
                        status="no_content",
                        chunk_count=0,
                        content_hash="",
                        failure_reason="parsed to empty text",
                    )
                )
            elif doc is not None and self._is_redirect_stub(doc.parsed.text):
                records.append(
                    CoverageRecord(
                        relative_path=record.relative_path,
                        representation="native",
                        doc_type=record.doc_type,
                        canonical_url=key,
                        status="no_content",
                        chunk_count=0,
                        content_hash="",
                        failure_reason="redirect or under-construction stub page",
                    )
                )
            else:
                records.append(
                    CoverageRecord(
                        relative_path=record.relative_path,
                        representation="native",
                        doc_type=record.doc_type,
                        canonical_url=key,
                        status="zero_chunks",
                        chunk_count=0,
                        content_hash="",
                        failure_reason="selected non-empty file produced no chunks",
                    )
                )

        if self._rendered_manifest is not None:
            for rendered in self._rendered_manifest.files:
                if not rendered.absolute_path.is_file():
                    records.append(
                        CoverageRecord(
                            relative_path=rendered.relative_path,
                            representation="rendered",
                            doc_type=rendered.doc_type,
                            canonical_url=rendered.canonical_url,
                            status="missing_output",
                            chunk_count=0,
                            content_hash="",
                            failure_reason="rendered build output file missing",
                        )
                    )
                    continue
                chunk_count = chunk_count_by_key.get(rendered.canonical_url, 0)
                doc = merged_by_key.get(rendered.canonical_url)
                if chunk_count > 0:
                    records.append(
                        CoverageRecord(
                            relative_path=rendered.relative_path,
                            representation="rendered",
                            doc_type=rendered.doc_type,
                            canonical_url=rendered.canonical_url,
                            status="indexed",
                            chunk_count=chunk_count,
                            content_hash="",
                        )
                    )
                elif doc is not None and not doc.parsed.text.strip():
                    records.append(
                        CoverageRecord(
                            relative_path=rendered.relative_path,
                            representation="rendered",
                            doc_type=rendered.doc_type,
                            canonical_url=rendered.canonical_url,
                            status="no_content",
                            chunk_count=0,
                            content_hash="",
                            failure_reason="parsed to empty text",
                        )
                    )
                elif self._parse_rendered(rendered) is None:
                    # Navigation-only / redirect stub: the HTML parser rejects
                    # pages with too little content. Not a failure — the page
                    # has no retrieval content by design.
                    records.append(
                        CoverageRecord(
                            relative_path=rendered.relative_path,
                            representation="rendered",
                            doc_type=rendered.doc_type,
                            canonical_url=rendered.canonical_url,
                            status="no_content",
                            chunk_count=0,
                            content_hash="",
                            failure_reason="navigation-only or redirect page (no main content)",
                        )
                    )
                else:
                    records.append(
                        CoverageRecord(
                            relative_path=rendered.relative_path,
                            representation="rendered",
                            doc_type=rendered.doc_type,
                            canonical_url=rendered.canonical_url,
                            status="zero_chunks",
                            chunk_count=0,
                            content_hash="",
                            failure_reason="selected non-empty file produced no chunks",
                        )
                    )

        return records

    @staticmethod
    def _is_redirect_stub(text: str) -> bool:
        """Return True when *text* is a redirect or under-construction stub.

        These pages carry no retrieval content (e.g. ``redirect:`` front-matter
        with "This document has moved", or a body consisting only of
        "under construction") and legitimately produce zero chunks. The
        Jekyll front-matter license block is excluded before judging size.
        """
        lower = text.lower()
        if "redirect:" in lower and "has moved" in lower:
            return True
        body = lower
        if body.startswith("---"):
            end = body.find("\n---", 4)
            if end != -1:
                body = body[end + 4 :]
        return "under construction" in body and len(body.strip()) < 400

    @staticmethod
    def _dedup_by_content_hash(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Remove exact duplicate normalized content, keeping the first copy.

        Dedup compares the normalized (stripped) text so identical content that
        differs only in leading/trailing whitespace (e.g. the ASF license
        header appearing in many files) collapses to a single copy.
        """
        seen: set[str] = set()
        deduped: list[DocumentChunk] = []
        for chunk in chunks:
            normalized_key = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            deduped.append(chunk)
        return deduped

    def _with_function_registry(self, manifest: SparkManifest) -> SparkChunker:
        """Return a chunker preloaded with FunctionRegistry.scala, when present."""
        from dataclasses import replace

        if self._chunker.function_registry_text is not None:
            return self._chunker
        registry = manifest.root / _FUNCTION_REGISTRY_RELATIVE_PATH
        if not registry.is_file():
            return self._chunker
        text = registry.read_text(encoding="utf-8", errors="replace")
        return replace(self._chunker, function_registry_text=text)

    async def _embed_all(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), self._embedding_batch_size):
            batch = [c.text for c in chunks[i : i + self._embedding_batch_size]]
            batch_vectors = await self._embedder.embed_texts(batch)
            vectors.extend(batch_vectors)
        return vectors

    @staticmethod
    def _reject_duplicate_chunk_ids(chunks: list[DocumentChunk]) -> None:
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise ValueError(f"Duplicate chunk_id in Spark corpus: {chunk.chunk_id!r}")
            seen.add(chunk.chunk_id)

    def _normalize_chunk(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        """Split *chunk* into lossless, budget-safe segments with metadata.

        Never truncates text. Every returned segment satisfies the token and
        character budgets, carries segment identity (``parent_content_hash``,
        ``segment_index``, ``segment_total``, ``token_count``,
        ``character_count``), and ``"".join(segment_texts)`` reconstructs the
        parent chunk text. A chunk already within budget yields a single
        segment with ``segment_index=0``.
        """
        from dataclasses import replace

        # The reconstruction target is the normalized (stripped) source that
        # ``split_text_losslessly`` actually reproduces. The chunker's raw
        # content_hash is computed on the unstripped text (which can carry
        # leading/trailing whitespace), so hash the stripped text so the
        # parent hash stays consistent with lossless reconstruction.
        parent_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        segment_texts = split_text_losslessly(
            chunk.text,
            max_tokens=self._max_embed_tokens,
            max_chars=self._max_embed_chars,
        )

        normalized: list[DocumentChunk] = []
        for index, text in enumerate(segment_texts):
            segment_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            normalized.append(
                replace(
                    chunk,
                    chunk_id=f"{chunk.chunk_id}:seg:{index}",
                    text=text,
                    content_hash=segment_hash,
                    word_count=len(text.split()),
                    index_generation=self._generation,
                    chunker_version="spark-chunker-v1",
                    parent_content_hash=parent_hash,
                    segment_index=index,
                    segment_total=len(segment_texts),
                    token_count=count_tokens(text),
                    character_count=len(text),
                )
            )
        return normalized


async def build_spark_index(
    manifest: SparkManifest,
    generation: str,
    config: SparkSourceConfig,
    resolver: SparkSourceResolver,
    parser: NativeDocumentParser,
    chunker: SparkChunker,
    store: AsyncQdrantVectorStore,
    embedder: EmbedderProtocol,
) -> IndexBuildReport:
    """Convenience wrapper around ``SparkIndexBuilder.build``."""
    builder = SparkIndexBuilder(
        config=config,
        resolver=resolver,
        parser=parser,
        chunker=chunker,
        store=store,
        generation=generation,
        embedder=embedder,
    )
    return await builder.build()


def _rendered_build(
    config: SparkRenderedSourceConfig | None,
    build_name: str,
) -> SparkRenderedBuildConfig | None:
    """Return the rendered build config matching *build_name*, or ``None``."""
    if config is None:
        return None
    for build in config.builds:
        if build.name == build_name:
            return build
    return None


def _chunk_to_dict(chunk: DocumentChunk) -> dict[str, object]:
    """Serialize a chunk for ``chunks.jsonl`` (JSON-compatible values only)."""
    from dataclasses import asdict

    data = asdict(chunk)
    data["heading_path"] = list(chunk.heading_path)
    return data


def validate_generation_artifacts(
    *,
    generation: str,
    expected_commit: str,
    chunks: list[DocumentChunk],
    coverage: list[CoverageRecord],
    native_manifest_paths: list[str],
    rendered_manifest_paths: list[str] | None = None,
    qdrant_point_count: int | None = None,
    bm25_ready: bool | None = None,
    sparse_configured: bool | None = None,
) -> list[str]:
    """Return a list of validation failures for a built Spark generation.

    An empty list means the generation is valid. Checks:

    - every selected non-empty native/rendered file produced chunks
      (``zero_chunks`` coverage records fail);
    - no rendered build output file is missing (``missing_output`` fails);
    - native and rendered manifest paths are unique;
    - chunk IDs are unique;
    - Qdrant point count matches the persisted chunk count;
    - every chunk carries the expected generation and source commit metadata;
    - BM25 and sparse vectors are available when required;
    - every persisted segment satisfies the token/character budgets and the
      per-parent segment metadata is complete and lossless (Task 10).
    """
    failures: list[str] = []

    failures.extend(_validate_coverage(coverage))
    failures.extend(_validate_manifest_paths(native_manifest_paths, rendered_manifest_paths))
    failures.extend(_validate_chunk_ids(chunks))
    failures.extend(_validate_chunk_metadata(chunks, generation, expected_commit))
    failures.extend(_validate_segment_budgets(chunks))

    if qdrant_point_count is not None and qdrant_point_count != len(chunks):
        failures.append(f"Qdrant point count {qdrant_point_count} differs from chunks.jsonl count {len(chunks)}")
    if bm25_ready is False:
        failures.append("BM25 tokenizer is not ready")
    if sparse_configured is False:
        failures.append("Sparse vectors are not configured")

    return failures


def _validate_coverage(coverage: list[CoverageRecord]) -> list[str]:
    failures: list[str] = []
    for record in coverage:
        if record.status == "zero_chunks":
            failures.append(
                f"{record.representation} file {record.relative_path!r} produced no chunks: {record.failure_reason}"
            )
        elif record.status == "missing_output":
            failures.append(f"rendered output missing for {record.relative_path!r}: {record.failure_reason}")
    return failures


def _validate_manifest_paths(
    native_paths: list[str],
    rendered_paths: list[str] | None,
) -> list[str]:
    failures: list[str] = []
    seen: dict[str, str] = {}
    for path in native_paths:
        if path in seen:
            failures.append(f"native manifest path duplicated: {path!r} (sources {seen[path]})")
        seen[path] = "native"
    for path in rendered_paths or []:
        if path in seen:
            failures.append(f"rendered manifest path duplicated: {path!r} (sources {seen[path]})")
        seen[path] = "rendered"
    return failures


def _validate_chunk_ids(chunks: list[DocumentChunk]) -> list[str]:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            return [f"duplicate chunk_id in chunks.jsonl: {chunk.chunk_id!r}"]
        seen.add(chunk.chunk_id)
    return []


def _validate_chunk_metadata(chunks: list[DocumentChunk], generation: str, expected_commit: str) -> list[str]:
    failures: list[str] = []
    for chunk in chunks:
        if chunk.index_generation != generation:
            failures.append(
                f"chunk {chunk.chunk_id!r} has generation {chunk.index_generation!r}, expected {generation!r}"
            )
        if chunk.source_commit != expected_commit:
            failures.append(
                f"chunk {chunk.chunk_id!r} has commit {chunk.source_commit!r}, expected {expected_commit!r}"
            )
        if not chunk.doc_type:
            failures.append(f"chunk {chunk.chunk_id!r} lacks doc_type metadata")
    return failures


def _validate_segment_budgets(chunks: list[DocumentChunk]) -> list[str]:
    """Validate per-segment budgets and per-parent segment metadata (Task 10).

    Verifies, for every persisted segment:
    - ``token_count <= 3800`` and ``character_count <= 6000``;
    - segment metadata is complete (``parent_content_hash``, ``segment_index``,
      ``segment_total`` present);
    - segment indices are contiguous from zero;
    - ``segment_total`` matches the parent's actual segment count;
    - reconstruction ``"".join(segment texts)`` (normalized) reproduces the
      parent content hash — proving no truncation occurred.
    """
    failures: list[str] = []
    max_tokens = MAX_SAFE_TOKENS
    max_chars = DEFAULT_MAX_CHARS

    for chunk in chunks:
        if chunk.token_count > max_tokens:
            failures.append(f"segment {chunk.chunk_id!r} token_count {chunk.token_count} exceeds budget {max_tokens}")
        if chunk.character_count > max_chars:
            failures.append(
                f"segment {chunk.chunk_id!r} character_count {chunk.character_count} exceeds budget {max_chars}"
            )
        if chunk.segment_index < 0:
            failures.append(f"segment {chunk.chunk_id!r} has negative segment_index {chunk.segment_index}")

    # Per-parent completeness and lossless reconstruction.
    by_parent: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        if chunk.parent_content_hash:
            by_parent.setdefault(chunk.parent_content_hash, []).append(chunk)

    for parent_hash, segments in by_parent.items():
        segments_by_index = {segment.segment_index: segment for segment in segments}
        expected_total = max(segments_by_index) + 1 if segments_by_index else 0
        for index in range(expected_total):
            if index not in segments_by_index:
                failures.append(
                    f"parent {parent_hash[:12]} is missing segment index {index} "
                    "(segment indices must be contiguous from zero)"
                )
        for segment in segments:
            if segment.segment_total != len(segments):
                failures.append(
                    f"segment {segment.chunk_id!r} segment_total {segment.segment_total} "
                    f"does not match parent segment count {len(segments)}"
                )
        # Lossless reconstruction: the normalized joined texts reproduce the
        # parent content hash, proving no characters were truncated.
        joined = "".join(segments_by_index[index].text for index in sorted(segments_by_index))
        reconstructed_hash = hashlib.sha256(joined.strip().encode("utf-8")).hexdigest()
        if reconstructed_hash != parent_hash:
            failures.append(
                f"parent {parent_hash[:12]} reconstruction hash {reconstructed_hash[:12]} "
                "does not match the recorded parent content hash (truncation detected)"
            )

    return failures
