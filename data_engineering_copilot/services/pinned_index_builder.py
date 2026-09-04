"""Generic pinned generation builder: prepare → embed → validate → upsert.

Combines ``PreparedSource`` packages (one per pinned source) into a single
frozen Qdrant generation collection: one combined BM25 corpus, batch-embedded
vectors, per-source commit validation, and chunk/coverage/report artifacts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from data_engineering_copilot.config.naming import resolve_naming, validate_naming
from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.domain.protocols import EmbedderProtocol
from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import MAX_SAFE_TOKENS
from data_engineering_copilot.infrastructure.token_budget import (
    DEFAULT_MAX_CHARS,
    coalesce_blank_segments,
    count_tokens,
    split_text_losslessly,
)
from data_engineering_copilot.services.chunker import deduplicate_chunks, embedding_text_for_chunk
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
_structlog = structlog.get_logger(__name__)


if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore


CHECKPOINT_BATCH_SIZE = 32
EMBEDDING_MAX_RETRIES = 2
EMBEDDING_COOLDOWN_BASE_S = 5


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
        embedding_batch_size: int = 256,
        max_embed_tokens: int = MAX_SAFE_TOKENS,
        max_embed_chars: int = DEFAULT_MAX_CHARS,
        output_dir: Path | None = None,
        late_chunking_enabled: bool = False,
        late_chunking_max_tokens: int = 8192,
        late_chunking_model_name: str = "",
        settings: AppSettings | None = None,
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
        self._settings = settings or AppSettings()
        self._transformer_pool: ProcessPoolExecutor | None = None

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
        combined = deduplicate_chunks(combined)
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

        corpus_texts = [embedding_text_for_chunk(c) for c in normalized]
        self._store.fit_bm25_corpus(corpus_texts)

        await self._embed_all_with_checkpoint(normalized)

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

    async def _embed_all_with_checkpoint(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Embed with crash-resilient checkpointing.

        Embeds in batches sized dynamically based on model context window and
        corpus token distribution. Checkpointing progress after every
        ``CHECKPOINT_BATCH_SIZE`` batches. On restart, already-embedded
        chunks are skipped — saving hours of re-work on crashes.
        """
        from data_engineering_copilot.infrastructure.dynamic_batch_sizer import DynamicBatchSizer

        # Compute optimal batch size at runtime based on actual corpus
        # When resuming, keep the batch size that was checkpointed so slicing
        # stays deterministic across restarts.
        checkpoint = self._load_checkpoint()
        last_batch = checkpoint.get("last_batch", 0)
        ckpt_batch_size = checkpoint.get("batch_size")
        if ckpt_batch_size is not None and isinstance(ckpt_batch_size, int) and ckpt_batch_size > 0:
            self._embedding_batch_size = ckpt_batch_size
            _structlog.info("embedding_batch_size_from_checkpoint", batch_size=ckpt_batch_size)
        else:
            sample_texts = [embedding_text_for_chunk(c) for c in chunks[:100]]  # sample first 100 chunks
            sizer = DynamicBatchSizer(self._settings)
            # Detect provider from embedder (fallback chain wraps multiple)
            provider = getattr(self._embedder, "name", "nvidia")
            model_name = getattr(self._embedder, "model_name", None)
            dynamic_batch = sizer.compute_batch_size(provider, sample_texts, model_name)
            self._embedding_batch_size = dynamic_batch

            # Propagate to inner embedder if it supports runtime batch updates
            inner = getattr(self._embedder, "inner", self._embedder)
            set_batch = getattr(inner, "set_batch_size", None)
            if callable(set_batch):
                set_batch(dynamic_batch)
        total_batches = (len(chunks) + self._embedding_batch_size - 1) // self._embedding_batch_size

        if last_batch > 0:
            _structlog.info("embedding_resuming", resume_batch=last_batch, total_batches=total_batches)

        vectors: list[list[float]] = []
        # Track what's already been upserted — on resume, chunks before last_batch
        # are already in Qdrant, so we must NOT re-send them with a fresh vectors list
        # (which would cause a length mismatch: chunks[:processed_count] vs only new vectors).
        already_upserted_chunks = last_batch * self._embedding_batch_size
        upserted_since_resume = 0  # how many vectors in `vectors` have been upserted
        batch_idx = 0
        for batch_idx in range(last_batch, total_batches):
            batch_start = batch_idx * self._embedding_batch_size
            batch_end = min(batch_start + self._embedding_batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]
            batch_texts = [embedding_text_for_chunk(c) for c in batch]

            try:
                batch_vectors = await self._embed_batch_with_crash_recovery(batch_texts, batch_idx, total_batches)
            except Exception as exc:
                # Offline wait budget exhausted → checkpoint & pause gracefully
                from data_engineering_copilot.infrastructure.offline_embedding_wait import OfflineEmbeddingPaused

                if isinstance(exc, OfflineEmbeddingPaused) or isinstance(exc.__cause__, OfflineEmbeddingPaused):
                    paused = exc if isinstance(exc, OfflineEmbeddingPaused) else exc.__cause__  # type: ignore[assignment]
                    # Flush any embedded-but-not-yet-upserted vectors
                    if upserted_since_resume < len(vectors):
                        new_chunks = chunks[already_upserted_chunks:batch_start]
                        new_vectors = vectors[upserted_since_resume:]
                        if new_chunks and new_vectors:
                            _structlog.info(
                                "embedding_paused_upsert",
                                new_chunks=len(new_chunks),
                                new_vectors=len(new_vectors),
                            )
                            await self._store.upsert_frozen_chunks(new_chunks, new_vectors)
                    self._save_checkpoint({"last_batch": batch_idx, "batch_size": self._embedding_batch_size})
                    _structlog.warning(
                        "embedding_paused_budget_exhausted",
                        batch=batch_idx,
                        total_batches=total_batches,
                        waited=getattr(paused, "waited_s", 0),
                        max_wait=getattr(paused, "max_wait_s", 0),
                    )
                raise
            vectors.extend(batch_vectors)

            pct = (batch_idx + 1) / total_batches * 100
            _structlog.info("embedding_progress", batch=batch_idx, total_batches=total_batches, pct=f"{pct:.1f}%")

            if (batch_idx + 1) % CHECKPOINT_BATCH_SIZE == 0:
                # Only upsert NEW chunks+vectors since last checkpoint/resume
                upsert_end = (batch_idx + 1) * self._embedding_batch_size
                new_chunks = chunks[already_upserted_chunks:upsert_end]
                new_vectors = vectors[upserted_since_resume:]
                _structlog.info(
                    "embedding_checkpoint_upsert",
                    new_chunks=len(new_chunks),
                    new_vectors=len(new_vectors),
                )
                await self._store.upsert_frozen_chunks(new_chunks, new_vectors)
                upserted_since_resume = len(vectors)
                already_upserted_chunks = upsert_end
                self._save_checkpoint({"last_batch": batch_idx + 1, "batch_size": self._embedding_batch_size})

        # Final upsert for remaining chunks not yet upserted
        if upserted_since_resume < len(vectors):
            new_end = len(chunks)
            new_chunks = chunks[already_upserted_chunks:new_end]
            new_vectors = vectors[upserted_since_resume:]
            _structlog.info(
                "embedding_final_upsert",
                new_chunks=len(new_chunks),
                new_vectors=len(new_vectors),
            )
            await self._store.upsert_frozen_chunks(new_chunks, new_vectors)
        self._clear_checkpoint()
        return vectors

    async def _embed_batch_with_crash_recovery(
        self, texts: list[str], batch_idx: int, total_batches: int
    ) -> list[list[float]]:
        """Embed a batch with crash isolation and escalating cooldown."""
        from data_engineering_copilot.domain.exceptions import EmbeddingCrashError

        last_exc: Exception | None = None
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                return await _embed_batch_with_retry(self._embedder, texts)
            except EmbeddingCrashError as exc:
                last_exc = exc
                cooldown = EMBEDDING_COOLDOWN_BASE_S * (attempt + 1)
                _structlog.warning(
                    "embedding_batch_crashed",
                    batch=batch_idx,
                    total=total_batches,
                    attempt=attempt + 1,
                    cooldown=cooldown,
                    error=str(exc)[:160],
                )
                await asyncio.sleep(cooldown)
            except Exception as exc:
                last_exc = exc
                # When offline wait is active, rate-limit / 503 errors are
                # handled by the OfflineEmbeddingWaitController's collective
                # 10→60s backoff (wait-time only). Don't waste 5s/10s here and
                # don't escalate to pure transformers — let the controller drive.
                if self._settings.offline_embedding_wait_enabled:
                    from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
                    from data_engineering_copilot.infrastructure.offline_embedding_wait import OfflineEmbeddingPaused

                    if isinstance(exc, OfflineEmbeddingPaused) or isinstance(
                        getattr(exc, "__cause__", None), OfflineEmbeddingPaused
                    ):
                        raise
                    cat2 = None
                    if isinstance(exc, ProviderError):
                        cat2 = exc.category
                    else:
                        cause2 = getattr(exc, "__cause__", None)
                        if isinstance(cause2, ProviderError):
                            cat2 = cause2.category
                    msg2 = str(exc).lower() + str(getattr(exc, "__cause__", "")).lower()
                    is_rate = cat2 in (
                        ProviderErrorCategory.RATE_LIMITED,
                        ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
                        ProviderErrorCategory.QUOTA_EXCEEDED,
                        ProviderErrorCategory.RETRYABLE,
                    ) or any(k in msg2 for k in ("503", "429", "rate_limited", "temporary_unavailable"))
                    if is_rate:
                        raise
                cooldown = EMBEDDING_COOLDOWN_BASE_S * (attempt + 1)
                _structlog.warning(
                    "embedding_batch_failed",
                    batch=batch_idx,
                    total=total_batches,
                    attempt=attempt + 1,
                    cooldown=cooldown,
                    error=str(exc)[:160],
                )
                await asyncio.sleep(cooldown)

        # Offline bulk path: when the pool is rate-limited (nvidia/openrouter 503
        # or 429), the OfflineEmbeddingWaitController handles the collective
        # 10→60s backoff with 1h wait-time budget. Escalating to the local
        # pure-transformers ProcessPool (1.14GB model) would OOM on this host
        # (8GB ollama + 2GB qdrant) and masks the rate-limit as a crash.
        # The user explicitly disabled local-hf, so never fall back for those.
        from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
        from data_engineering_copilot.infrastructure.offline_embedding_wait import OfflineEmbeddingPaused

        if isinstance(last_exc, OfflineEmbeddingPaused) or isinstance(
            getattr(last_exc, "__cause__", None), OfflineEmbeddingPaused
        ):
            raise last_exc  # type: ignore[misc]
        # Unwrap ProviderError from fallback chain (LLMClientError cause)
        cat = None
        if isinstance(last_exc, ProviderError):
            cat = last_exc.category
        else:
            cause = getattr(last_exc, "__cause__", None)
            if isinstance(cause, ProviderError):
                cat = cause.category
            # Fallback chain wraps as "All providers in fallback chain failed" string
            msg = str(last_exc).lower() if last_exc else ""
            if "temporary_unavailable" in msg or "rate_limited" in msg or "503" in msg or "429" in msg:
                raise last_exc  # type: ignore[misc]
        if cat in (
            ProviderErrorCategory.RATE_LIMITED,
            ProviderErrorCategory.TEMPORARY_UNAVAILABLE,
            ProviderErrorCategory.QUOTA_EXCEEDED,
            ProviderErrorCategory.RETRYABLE,
        ):
            raise last_exc  # type: ignore[misc]
        # Also never use local fallback when offline wait is enabled (user asked no local-hf)
        if self._settings.offline_embedding_wait_enabled:
            raise last_exc  # type: ignore[misc]

        _structlog.warning(
            "embedding_escalating_to_pure_transformers",
            batch=batch_idx,
            total=total_batches,
            error=str(last_exc)[:160] if last_exc else "unknown",
        )
        return await self._embed_batch_pure_transformers(texts)

    async def _embed_batch_pure_transformers(self, texts: list[str]) -> list[list[float]]:
        """Fallback: embed using pure transformers (no sentence-transformers).

        Verified to produce cos=0.999996-identical vectors to sentence-transformers
        for the Nemotron model, with zero crash risk.
        Uses a persistent ProcessPoolExecutor to avoid reloading the model on every batch.
        """
        from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
            _encode_batch_pure_transformers_cached,
        )

        loop = asyncio.get_event_loop()
        if self._transformer_pool is None:
            import concurrent.futures

            self._transformer_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._transformer_pool, _encode_batch_pure_transformers_cached, texts),
                timeout=300,  # 5 min max per batch — escalate if exceeded
            )
        except TimeoutError as exc:
            _structlog.error(
                "local_hf_batch_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                texts_count=len(texts),
            )
            # Reset pool so next call gets a fresh worker
            if self._transformer_pool:
                self._transformer_pool.shutdown(wait=False, cancel_futures=True)
            self._transformer_pool = None
            raise
        except Exception as exc:
            import concurrent.futures

            if isinstance(exc, concurrent.futures.process.BrokenProcessPool):
                _structlog.error(
                    "local_hf_pool_broken",
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                    texts_count=len(texts),
                )
                # Reset pool so next call gets a fresh worker
                if self._transformer_pool:
                    self._transformer_pool.shutdown(wait=False, cancel_futures=True)
                self._transformer_pool = None
            raise

    def _load_checkpoint(self) -> dict:
        if self._output_dir is None:
            return {}
        path = self._output_dir / "embedding_checkpoint.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_checkpoint(self, data: dict) -> None:
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "embedding_checkpoint.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # ensure data is on disk before atomic rename
        tmp.replace(path)

    def _clear_checkpoint(self) -> None:
        if self._output_dir is None:
            return
        path = self._output_dir / "embedding_checkpoint.json"
        if path.exists():
            path.unlink()

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
        - ``source``: metadata about the source (slug, name, type, commit, manifest_hash, chunk_count)
        - ``sources``: mapping of source file path (relative to cwd) → sha12 hash

        Delegates to :mod:`data_engineering_copilot.services.source_manifest`
        for the actual writing. The schema is backward-compatible with
        ``check_derived_staleness.py`` which reads the top-level ``sources``
        dict and re-hashes each file.
        """
        if self._output_dir is None:
            return
        from data_engineering_copilot.config.settings import (
            load_pinned_sources,
            settings,
        )
        from data_engineering_copilot.services.source_manifest import (
            write_all_source_provenances,
        )

        config_map = {src.slug: src.type for src in load_pinned_sources(settings.pinned_sources_path)}
        write_all_source_provenances(
            packages=list(packages),
            generation=self._generation,
            output_dir=self._output_dir,
            source_type_map=config_map,
            generator="pinned-index-builder",
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
