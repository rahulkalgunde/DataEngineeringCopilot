"""Single source of truth for generation naming contracts.

All consumers (CLI commands, index builders, validation) must derive every
name from a generation ID through :func:`resolve_naming` instead of string
concatenation. This prevents the cross-module mismatch class of bugs where a
collection name, artifact directory, or Qdrant alias drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

_PREFIX = "data_engineering_docs__"


@dataclass(frozen=True)
class GenerationNaming:
    """Derived naming for one frozen generation."""

    generation_id: str  # e.g. "pinned-abc123def4"
    collection_name: str  # e.g. "data_engineering_docs__pinned-abc123def4"
    artifact_dir_name: str  # MUST equal collection_name
    active_alias: str  # "data_engineering_docs"
    chunks_file: str = "chunks.jsonl"
    coverage_file: str = "coverage.json"
    build_report_file: str = "build_report.json"


def resolve_naming(generation_id: str) -> GenerationNaming:
    """Derive all naming from the generation ID."""
    if not generation_id:
        raise ValueError("generation_id must not be empty")
    collection = f"{_PREFIX}{generation_id}"
    return GenerationNaming(
        generation_id=generation_id,
        collection_name=collection,
        artifact_dir_name=collection,
        active_alias=_PREFIX.rstrip("_"),
    )


def validate_naming(naming: GenerationNaming) -> None:
    """Fail-fast contract validation before any artifact I/O or upsert."""
    if naming.artifact_dir_name != naming.collection_name:
        raise RuntimeError(
            f"Contract violated: artifact_dir_name ({naming.artifact_dir_name}) "
            f"!= collection_name ({naming.collection_name})"
        )
    if not naming.collection_name.startswith(_PREFIX):
        raise RuntimeError(f"collection_name must start with {_PREFIX!r}: {naming.collection_name}")
