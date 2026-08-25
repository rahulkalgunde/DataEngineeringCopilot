"""Shared types for the generic generation pipeline.

A ``PreparedSource`` is the output of a per-source preparer (parse + chunk,
no embedding) and the input to ``PinnedIndexBuilder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.spark_index_builder import CoverageRecord


@dataclass(frozen=True)
class PreparedSource:
    """Chunked, not-yet-embedded corpus for one pinned source in a generation."""

    slug: str
    source_name: str
    generation: str
    commit: str
    chunks: tuple[DocumentChunk, ...]
    coverage: tuple[CoverageRecord, ...]
    cache_root: Path = Path()
    manifest_hash: str = ""

    def provenance_sources(self) -> dict[str, str]:
        """Map each covered source file (path→sha12) for staleness checking.

        Only files with a non-empty ``content_hash`` (i.e. present on disk)
        are included; missing-output rows are omitted so ``check_derived_staleness``
        won't flag an already-failed download as "stale".
        """
        root = self.cache_root
        out: dict[str, str] = {}
        for record in self.coverage:
            if not record.content_hash:
                continue
            try:
                rel = (root / record.relative_path).relative_to(Path.cwd())
                key = str(rel)
            except ValueError:
                key = str(root / record.relative_path)
            out[key] = record.content_hash[:12]
        return out

