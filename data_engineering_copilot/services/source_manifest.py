"""Generic source manifest and provenance utilities.

Provides helpers for writing per-source provenance sidecars alongside
generated manifests. Used by the pinned/Spark generation pipelines.

The provenance schema is designed for backward compatibility with
``scripts/check_derived_staleness.py`` which reads the top-level
``sources`` mapping (path -> sha12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.services.prepared_source import PreparedSource


@dataclass(frozen=True)
class SourceProvenance:
    """Provenance metadata for a single source in a generation.

    Attributes
    ----------
    source_name:
        Human-readable source name (e.g., "Apache Spark").
    slug:
        URL-safe slug for the source.
    generation:
        Generation ID (e.g., "pinned-abc123").
    commit_hash:
        Pinned commit SHA (or "" for non-versioned sources).
    manifest_hash:
        Hash of the source manifest (12+ hex chars).
    chunk_count:
        Number of chunks produced for this source.
    source_type:
        Type of source: "github", "url_index", "local_mirror", or "unknown".
    timestamp:
        ISO-8601 UTC timestamp when the provenance was written.
    source_files:
        Mapping of source file path -> sha12 content hash (for staleness).
    generator:
        Identifier for the tool that wrote the provenance.
    """

    source_name: str
    slug: str
    generation: str
    commit_hash: str
    manifest_hash: str
    chunk_count: int
    source_type: str
    timestamp: str
    source_files: dict[str, str]
    generator: str = "source-manifest-writer"

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict matching the existing schema.

        Top-level ``sources`` key maps file path -> sha12 for
        backward compatibility with ``check_derived_staleness.py``.
        """
        return {
            "generated_at": self.timestamp,
            "generator": self.generator,
            "generation": self.generation,
            "source": {
                "slug": self.slug,
                "name": self.source_name,
                "type": self.source_type,
                "commit": self.commit_hash,
                "manifest_hash": self.manifest_hash,
                "chunk_count": self.chunk_count,
            },
            "sources": self.source_files,
        }


def build_source_provenance(
    package: PreparedSource,
    generation: str,
    source_type: str,
    generator: str = "pinned-index-builder",
) -> SourceProvenance:
    """Build a SourceProvenance from a PreparedSource package.

    Parameters
    ----------
    package:
        The prepared source (parse + chunk, no embedding).
    generation:
        The generation ID.
    source_type:
        The source type: "github", "url_index", "local_mirror", or "unknown".
    generator:
        Identifier of the writer (e.g., "pinned-index-builder").
    """
    return SourceProvenance(
        source_name=package.source_name,
        slug=package.slug,
        generation=generation,
        commit_hash=package.commit,
        manifest_hash=package.manifest_hash,
        chunk_count=len(package.chunks),
        source_type=source_type,
        timestamp=datetime.now(UTC).isoformat(),
        source_files=package.provenance_sources(),
        generator=generator,
    )


def write_source_provenance(
    provenance: SourceProvenance,
    output_dir: Path,
) -> Path:
    """Write a provenance sidecar for a single source.

    Writes to ``output_dir / f"provenance-{provenance.slug}.json"``.
    Returns the path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prov_path = output_dir / f"provenance-{provenance.slug}.json"
    prov_path.write_text(
        json.dumps(provenance.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prov_path


def write_all_source_provenances(
    packages: list[PreparedSource],
    generation: str,
    output_dir: Path,
    source_type_map: dict[str, str] | None = None,
    generator: str = "pinned-index-builder",
) -> list[Path]:
    """Write provenance sidecars for all prepared sources.

    Parameters
    ----------
    packages:
        Prepared sources from the generation pipeline.
    generation:
        Generation ID.
    output_dir:
        Directory to write provenance files.
    source_type_map:
        Optional mapping of slug -> source type. Defaults to "unknown"
        for any slug not in the map.
    generator:
        Identifier of the writer.

    Returns
    -------
    List of paths written, one per source.
    """
    written: list[Path] = []
    for package in packages:
        source_type = (source_type_map or {}).get(package.slug, "unknown")
        prov = build_source_provenance(package, generation, source_type, generator=generator)
        written.append(write_source_provenance(prov, output_dir))
    return written


def read_source_provenance(prov_path: Path) -> SourceProvenance:
    """Read a provenance sidecar back into a SourceProvenance object."""
    data = json.loads(prov_path.read_text(encoding="utf-8"))
    source = data.get("source", {})
    return SourceProvenance(
        source_name=source.get("name", ""),
        slug=source.get("slug", ""),
        generation=data.get("generation", ""),
        commit_hash=source.get("commit", ""),
        manifest_hash=source.get("manifest_hash", ""),
        chunk_count=source.get("chunk_count", 0),
        source_type=source.get("type", "unknown"),
        timestamp=data.get("generated_at", ""),
        source_files=data.get("sources", {}),
        generator=data.get("generator", "unknown"),
    )
