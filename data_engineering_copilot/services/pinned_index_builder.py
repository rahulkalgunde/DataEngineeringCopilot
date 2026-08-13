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
from pathlib import Path
from typing import TYPE_CHECKING

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.domain.protocols import EmbedderProtocol
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import MAX_SAFE_TOKENS
from data_engineering_copilot.infrastructure.token_budget import DEFAULT_MAX_CHARS, count_tokens, split_text_losslessly
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
    ) -> None:
        self._store = store
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

        segment_failures = _validate_segment_budgets(normalized)
        if segment_failures:
            raise ValueError(f"pinned build segment validation failed: {segment_failures[:3]}")

        self._write_chunks_jsonl(normalized)
        self._write_coverage([record for package in packages for record in package.coverage])

        corpus_texts = [c.text for c in normalized]
        self._store.fit_bm25_corpus(corpus_texts)

        vectors = await self._embed_all(normalized)

        await self._store.upsert_frozen_chunks(normalized, vectors)
        validation = await self._store.validate_index_generation(len(normalized))
        bm25_vocab = 0
        if getattr(self._store, "_bm25", None) is not None:
            bm25_vocab = self._store._bm25.vocab_size  # type: ignore[attr-defined]

        self._write_build_report(normalized, packages, validation, bm25_vocab)

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
                    chunker_version=chunk.chunker_version or "pinned-builder-v1",
                    parent_content_hash=parent_hash,
                    segment_index=index,
                    segment_total=len(segment_texts),
                    token_count=count_tokens(text),
                    character_count=len(text),
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
                    "chunk_count": len(package.chunks),
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


def _reject_duplicate_ids(chunks: list[DocumentChunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id in pinned corpus: {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
