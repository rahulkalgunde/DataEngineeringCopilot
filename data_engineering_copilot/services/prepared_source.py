"""Shared types for the generic generation pipeline.

A ``PreparedSource`` is the output of a per-source preparer (parse + chunk,
no embedding) and the input to ``PinnedIndexBuilder``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
