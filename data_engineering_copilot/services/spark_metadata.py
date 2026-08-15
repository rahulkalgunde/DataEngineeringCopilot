"""Deterministic Spark metadata derivation from manifest records."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.domain.protocols import GitRepoSource
from data_engineering_copilot.infrastructure.spark_source_resolver import SparkFileRecord

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class SparkMetadata:
    """Provenance metadata attached to every Spark-derived chunk."""

    doc_type: str
    language: str
    spark_version: str
    module: str
    source_commit: str
    file_path: str
    license: str
    deployment_mode: str = ""


def derive_spark_metadata(
    record: SparkFileRecord,
    source: GitRepoSource,
    title: str,
    text: str,
) -> SparkMetadata:
    """Derive deterministic provenance metadata for a Spark file record.

    ``title`` must come from parser output, never from arbitrary text tokens.
    Empty strings are used for unknown values; the Spark version is taken from
    ``source.ref`` only when it is not a mutable branch marker.
    """
    if not _SHA_RE.match(source.commit):
        raise ValueError(f"source_commit must be a 40-character hexadecimal SHA, got {source.commit!r}")

    language = record.language.lower()
    spark_version = _version_from_ref(source.ref)
    module = _derive_module(record.relative_path, language)

    return SparkMetadata(
        doc_type=record.doc_type,
        language=language,
        spark_version=spark_version,
        module=module,
        source_commit=source.commit,
        file_path=record.relative_path,
        license=source.license,
        deployment_mode=_derive_deployment_mode(record.relative_path),
    )


# Deployment-mode documentation files in the Spark repo; everything else is
# mode-agnostic and gets an empty deployment_mode.
_DEPLOYMENT_MODE_FILES: dict[str, str] = {
    "docs/running-on-yarn.md": "yarn",
    "docs/running-on-kubernetes.md": "kubernetes",
    "docs/spark-standalone.md": "standalone",
}


def _derive_deployment_mode(relative_path: str) -> str:
    """Derive the deployment mode from a docs file path, or '' when not mode-specific."""
    normalized = relative_path.replace("\\", "/").lower()
    return _DEPLOYMENT_MODE_FILES.get(normalized, "")


def _version_from_ref(ref: str) -> str:
    """Extract a concrete release version from a tag, or '' for mutable refs."""
    stripped = ref.strip()
    if stripped.startswith("v"):
        stripped = stripped[1:]
    if not stripped or any(marker in stripped.lower() for marker in ("master", "latest", "main", "snapshot")):
        return ""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*", stripped):
        return stripped
    return ""


def _derive_module(relative_path: str, language: str) -> str:
    """Derive a canonical dotted module for Python files; '' otherwise."""
    if language != "python":
        return ""
    path = Path(relative_path)
    parts = list(path.parts)
    # Strip the leading python/pyspark prefix segments and the file stem.
    try:
        idx = parts.index("pyspark")
    except ValueError:
        return ""
    module_parts = parts[idx:]
    if module_parts and module_parts[-1].endswith(".py"):
        module_parts[-1] = Path(module_parts[-1]).stem
    return ".".join(module_parts)


def metadata_to_parsed_document(
    parsed_doc: ParsedDocument,
    metadata: SparkMetadata,
) -> ParsedDocument:
    """Attach Spark metadata to a parsed document via dataclasses.replace."""

    return replace(
        parsed_doc,
        doc_type=metadata.doc_type,
        language=metadata.language,
        spark_version=metadata.spark_version,
        module=metadata.module,
        source_commit=metadata.source_commit,
        file_path=metadata.file_path,
        license=metadata.license,
    )


def metadata_field_names() -> tuple[str, ...]:
    """Return the metadata field names for Qdrant payload serialization."""
    return tuple(f.name for f in fields(SparkMetadata))
