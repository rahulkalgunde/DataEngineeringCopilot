"""Generic pinned generation builder: prepare → embed → validate → upsert.

Combines ``PreparedSource`` packages (one per pinned source) into a single
frozen Qdrant generation collection: one combined BM25 corpus, batch-embedded
vectors, per-source commit validation, and chunk/coverage/report artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from data_engineering_copilot.config.naming import resolve_naming, validate_naming
from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.domain.protocols import EmbedderProtocol
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import MAX_SAFE_TOKENS
from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_CHARS,
    coalesce_blank_segments,
    count_tokens,
    split_text_losslessly,
)
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.spark_index_builder import (
    CoverageRecord,
    IndexBuildReport,
    _chunk_to_dict,
    _embed_batch_with_retry,
    _validate_chunk_metadata,
    _validate_coverage,
    _validate_segment_budgets,
)

logger = logging.getLogger(__name__)

from data_engineering_copilot.infrastructure.late_chunking import LateChunkEmbedder  # noqa: E402

if TYPE_CHECKING:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore


class PinnedIndexBuilder:
    """Build a single generation collection from prepared pinned sources.

    Parameters
    ----------
    store:
        Vector store configured with the target generation collection name.
    embedder:
        Injected embedding client (via the fallback chain).
    generation:
        Generation ID stamped on every upserted chunk.
    embedding_batch_size:
        Texts per ``embed_texts`` call.
    output_dir:
        Directory for ``chunks.jsonl`` / ``coverage.json`` / ``build_report.json``.
    """

    def __init__(
        self,
        store: AsyncQdrantVectorStore,
        embedder: EmbedderProtocol,
        generation: str,
        embedding_batch_size: int = 128,
        max_embed_tokens: int = MAX_SAFE_TOKENS,
        max_embed_chars: int = DEFAULT_MAX_CHARS,
        output_dir: Path | None = None,
        late_chunking_enabled: bool = False,
        late_chunking_max_tokens: int = 8192,
        late_chunking_model_name: str = "",
    ) -> None:
        naming = resolve_naming(generation)
        validate_naming(naming)
        self._store = store
        self._late_chunking_enabled = late_chunking_enabled
        self._late_chunking_max_tokens = late_chunking_max_tokens
        self._late_chunking_model_name = late_chunking_model_name
        self._embedder = embedder
        self._generation = generation
        self._embedding_batch_size = embedding_batch_size
        self._max_embed_tokens = max_embed_tokens
        self._max_embed_chars = max_embed_chars
        self._output_dir = Path(output_dir) if output_dir is not None else None

    async def build(self, packages: Sequence[PreparedSource]) -> IndexBuildReport:
        """Combine, normalize, embed, and persist the prepared sources."""
        await self._store.initialize()

        # Per-source metadata validation BEFORE normalization so failures point
        # at the offending package. url_index sources pin no commit ("").
        for package in packages:
            failures = _validate_chunk_metadata(list(package.chunks), self._generation, expected_commit=package.commit)
            failures.extend(_validate_coverage(list(package.coverage)))
            if failures:
                raise ValueError(f"pinned build validation failed for {package.slug}: {failures[:3]}")

        combined = [chunk for package in packages for chunk in package.chunks]
        combined = self._dedup_by_content_hash(combined)
        _reject_duplicate_ids(combined)

        normalized: list[DocumentChunk] = []
        for chunk in combined:
            normalized.extend(self._normalize_chunk(chunk))
        _reject_duplicate_ids(normalized)

        # Hierarchical pass: split each normalized segment into a parent chunk
        # plus smaller children for precise retrieval. The output carries
        # ``parent_chunk_id`` links and satisfies segment-budget validation.
        from data_engineering_copilot.services.hierarchical_chunker import hierarchical_chunk

        hierarchical: list[DocumentChunk] = []
        for chunk in normalized:
            hierarchical.extend(
                hierarchical_chunk(
                    chunk,
                    parent_offset_start=chunk.start_offset,
                    parent_offset_end=chunk.end_offset,
                )
            )
        normalized = hierarchical
        _reject_duplicate_ids(normalized)

        segment_failures = _validate_segment_budgets(normalized)
        if segment_failures:
            raise ValueError(f"pinned build segment validation failed: {segment_failures[:3]}")

        self._write_chunks_jsonl(normalized)
        self._write_coverage([record for package in packages for record in package.coverage])
        self._write_provenance(packages)

        per_source_chunk_counts = {
            package.slug: sum(1 for c in normalized if c.source_name == package.source_name) for package in packages
        }

        corpus_texts = [c.text for c in normalized]
        self._store.fit_bm25_corpus(corpus_texts)

        vectors = await self._embed_all(normalized)

        await self._store.upsert_frozen_chunks(normalized, vectors)
        validation = await self._store.validate_index_generation(len(normalized))
        bm25_vocab = 0
        if getattr(self._store, "_bm25", None) is not None:
            bm25_vocab = self._store._bm25.vocab_size  # type: ignore[attr-defined]

        self._write_build_report(normalized, packages, validation, bm25_vocab, per_source_chunk_counts)

        return IndexBuildReport(
            generation=self._generation,
            manifest_hash=self._combined_hash(normalized),
            chunk_count=len(normalized),
            source_file_count=sum(len(package.coverage) for package in packages),
            bm25_vocabulary_size=bm25_vocab,
            qdrant_collection=self._store._collection_name,
            validation_passed=bool(validation.get("passed")),
            coverage_count=sum(len(package.coverage) for package in packages),
        )

    async def _embed_all(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        if self._late_chunking_enabled:
            from data_engineering_copilot.infrastructure.late_chunking import embed_document_grouped

            if not self._late_chunking_model_name:
                logger.warning("late_chunking enabled without model name; using naive embedding")
                self._late_chunking_enabled = False

            def _late() -> LateChunkEmbedder:
                return LateChunkEmbedder(self._late_chunking_model_name, max_tokens=self._late_chunking_max_tokens)

            async def naive(batch_texts: list[str]) -> list[list[float]]:
                out: list[list[float]] = []
                for i in range(0, len(batch_texts), self._embedding_batch_size):
                    out.extend(
                        await _embed_batch_with_retry(self._embedder, batch_texts[i : i + self._embedding_batch_size])
                    )
                return out

            return await embed_document_grouped(
                chunks,
                naive_embed=naive,
                late_embedder=_late,
                max_group_tokens=self._late_chunking_max_tokens,
            )
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), self._embedding_batch_size):
            batch = [c.text for c in chunks[i : i + self._embedding_batch_size]]
            vectors.extend(await _embed_batch_with_retry(self._embedder, batch))
        return vectors

    @staticmethod
    def _dedup_by_content_hash(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        seen: set[str] = set()
        deduped: list[DocumentChunk] = []
        for chunk in chunks:
            key = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    def _normalize_chunk(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        """Split *chunk* into lossless, budget-safe segments with metadata."""
        from dataclasses import replace

        parent_hash = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        segment_texts = coalesce_blank_segments(
            split_text_losslessly(
                chunk.text,
                max_tokens=self._max_embed_tokens,
                max_chars=self._max_embed_chars,
            )
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
                    chunker_version=chunk.chunker_version or "pinned-builder-v1",
                    parent_content_hash=parent_hash,
                    segment_index=index,
                    segment_total=len(segment_texts),
                    token_count=count_tokens(text),
                    character_count=len(text),
                    crawled_at=datetime.now(UTC).isoformat(),
                )
            )
        return normalized

    @staticmethod
    def _combined_hash(chunks: list[DocumentChunk]) -> str:
        canonical = json.dumps(
            [(c.chunk_id, c.content_hash) for c in chunks],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _write_chunks_jsonl(self, chunks: list[DocumentChunk]) -> None:
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with (self._output_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_coverage(self, coverage: Sequence[CoverageRecord]) -> None:
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "coverage.json").write_text(
            json.dumps([asdict(record) for record in coverage], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_build_report(
        self,
        chunks: list[DocumentChunk],
        packages: Sequence[PreparedSource],
        validation: dict[str, object],
        bm25_vocab: int,
        per_source_chunk_counts: dict[str, int],
    ) -> None:
        if self._output_dir is None:
            return
        report = {
            "generation": self._generation,
            "sources": [
                {
                    "slug": package.slug,
                    "source_name": package.source_name,
                    "commit": package.commit,
                    "chunk_count": per_source_chunk_counts.get(package.slug, 0),
                    "coverage_count": len(package.coverage),
                }
                for package in packages
            ],
            "final_chunk_count": len(chunks),
            "qdrant_point_count": validation.get("point_count"),
            "bm25_vocabulary_size": bm25_vocab,
            "validation_result": bool(validation.get("passed")),
        }
        (self._output_dir / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_provenance(self, packages: Sequence[PreparedSource]) -> None:
        """Write a ``.provenance.json`` sidecar per source for staleness gating.

        Each file is named ``provenance-<slug>.json`` and contains:
        - ``generated_at``: ISO timestamp
        - ``generator``: fixed string "pinned-index-builder"
        - ``generation``: the generation ID
        - ``source``: metadata about the source (slug, name, type, commit, manifest_hash)
        - ``sources``: mapping of source file path (relative to cwd) → sha12 hash

        This schema is backward-compatible with ``check_derived_staleness.py``
        which reads the top-level ``sources`` dict and re-hashes each file.
        """
        if self._output_dir is None:
            return
        from data_engineering_copilot.config.settings import load_pinned_sources

        config_map = {src.slug: src for src in load_pinned_sources()}
        for package in packages:
            config = config_map.get(package.slug)
            source_type = config.type if config else "unknown"
            prov = {
                "generated_at": datetime.now(UTC).isoformat(),
                "generator": "pinned-index-builder",
                "generation": self._generation,
                "source": {
                    "slug": package.slug,
                    "name": package.source_name,
                    "type": source_type,
                    "commit": package.commit,
                    "manifest_hash": package.manifest_hash,
                },
                "sources": package.provenance_sources(),
            }
            (self._output_dir / f"provenance-{package.slug}.json").write_text(
                json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def _reject_duplicate_ids(chunks: list[DocumentChunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id in pinned corpus: {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)


def validate_pinned_generation_artifacts(
    generation: str,
    expected_commits: set[str],
    chunks: list[DocumentChunk],
    coverage: Sequence[CoverageRecord],
    qdrant_point_count: int | None,
    bm25_ready: bool,
    sparse_configured: bool,
) -> list[str]:
    """Return the list of validation failures for a built pinned generation.

    Mirrors ``validate_generation_artifacts`` for the combined generation:
    every chunk must carry the generation and a source commit from
    ``expected_commits`` (which includes ``""`` for unpinned url_index
    sources), chunk IDs must be unique, coverage paths must cover every chunk
    ``file_path``, and the Qdrant side must be frozen and sparse-ready.
    """
    failures: list[str] = []
    if not chunks:
        failures.append("no chunks in generation")
    chunk_ids: list[str] = []
    covered_paths = {record.relative_path for record in coverage}
    for chunk in chunks:
        chunk_ids.append(chunk.chunk_id)
        if chunk.index_generation != generation:
            failures.append(f"chunk {chunk.chunk_id}: generation mismatch")
        if chunk.source_commit not in expected_commits:
            failures.append(f"chunk {chunk.chunk_id}: commit {chunk.source_commit!r} not pinned")
        if chunk.file_path not in covered_paths:
            failures.append(f"chunk {chunk.chunk_id}: file_path {chunk.file_path!r} missing from coverage")
    if len(set(chunk_ids)) != len(chunk_ids):
        failures.append("duplicate chunk ids")
    # Parent-child referential integrity: every child's parent_chunk_id must
    # resolve to a persisted parent chunk in the generation.
    parent_ids = {c.chunk_id for c in chunks if not c.parent_chunk_id}
    for chunk in chunks:
        if chunk.parent_chunk_id and chunk.parent_chunk_id not in parent_ids:
            failures.append(f"chunk {chunk.chunk_id}: parent {chunk.parent_chunk_id!r} not found in generation")
    if qdrant_point_count is not None and qdrant_point_count != len(chunks):
        failures.append(f"qdrant point count {qdrant_point_count} != chunks {len(chunks)}")
    if not bm25_ready:
        failures.append("bm25 not ready")
    if not sparse_configured:
        failures.append("sparse vectors not configured")
    return failures
