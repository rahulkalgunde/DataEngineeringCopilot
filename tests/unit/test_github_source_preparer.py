"""Phase 2 tests: generic GitHub source preparer."""

from __future__ import annotations

import asyncio
from pathlib import Path

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.infrastructure.spark_source_resolver import SparkFileRecord, SparkManifest
from data_engineering_copilot.services.github_source_preparer import GithubSourcePreparer

_COMMIT = "a" * 40
_GENERATION = "gen-test-abc123"


def _config(slug: str, name: str) -> PinnedSourceConfig:
    return PinnedSourceConfig(
        type="github",
        name=name,
        slug=slug,
        version="1.0",
        license="Apache-2.0",
        repository="https://github.com/apache/example.git",
        ref="v1.0",
        commit=_COMMIT,
        streams=(),
    )


def _manifest(root: Path, files: list[SparkFileRecord]) -> SparkManifest:
    return SparkManifest(
        source_name="x",
        ref="v1.0",
        commit=_COMMIT,
        root=root,
        files=tuple(files),
        manifest_hash="h" * 64,
    )


def _record(root: Path, rel: str, doc_type: str, language: str, content: str) -> SparkFileRecord:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return SparkFileRecord(
        stream="stream",
        relative_path=rel,
        absolute_path=path,
        doc_type=doc_type,
        language=language,
        source_url=f"https://raw.githubusercontent.com/apache/example/{_COMMIT}/{rel}",
    )


def test_chunk_spark_uses_spark_chunker_with_metadata() -> None:
    config = _config("spark", "Apache Spark 4.0.0")
    preparer = GithubSourcePreparer(config, Path("/tmp/pinned-src-cache"), _GENERATION)
    root = Path("/tmp/pinned-src-cache") / "repo"
    guide = _record(
        root,
        "docs/guide.md",
        "guide",
        "conceptual",
        "# Spark Guide\n\n"
        "Header-aware body with useful content covering the core concepts "
        "and best practices for the reader.\n\n"
        "## Section\n\nMore body here with plenty of words to pass the "
        "minimum chunk length threshold.\n",
    )
    api = _record(
        root,
        "python/pyspark/sql/functions.py",
        "api_reference",
        "python",
        "def col(colName):\n    ...\n\nclass Column:\n    ...\n",
    )
    manifest = _manifest(root, [guide, api])

    chunks, coverage = asyncio.run(preparer._chunk_manifest(manifest))

    assert len(chunks) >= 2
    assert len(coverage) == 2
    for chunk in chunks:
        assert chunk.source_commit == _COMMIT
        assert chunk.source_name == "Apache Spark 4.0.0"
        assert chunk.index_generation == _GENERATION
        assert chunk.chunker_version == "spark-chunker-v1"
    by_doc_type = {c.doc_type for c in chunks}
    assert {"guide", "api_reference"}.issubset(by_doc_type)
    api_chunks = [c for c in chunks if c.doc_type == "api_reference"]
    assert api_chunks and api_chunks[0].language == "python"
    assert all(record.status == "indexed" for record in coverage)


def test_chunk_spark_converts_rst_guide_headings_before_chunking(tmp_path) -> None:
    config = _config("spark", "Apache Spark 4.0.0")
    preparer = GithubSourcePreparer(config, tmp_path, _GENERATION)
    record = _record(
        tmp_path / "repo",
        "python/docs/source/tutorial/sql/arrow_pandas.rst",
        "guide",
        "conceptual",
        "Apache Arrow in PySpark\n=======================\n\n"
        "Real content about Arrow transfers between JVM and Python.\n\n"
        "Pandas UDFs\n-----------\n\nMore useful body about vectorized operations.\n",
    )
    manifest = _manifest(tmp_path / "repo", [record])

    chunks, coverage = asyncio.run(preparer._chunk_manifest(manifest))

    assert chunks
    for chunk in chunks:
        assert chunk.source_commit == _COMMIT
        assert chunk.index_generation == _GENERATION
        assert chunk.source_name == "Apache Spark 4.0.0"
        assert chunk.doc_type == "guide"
    assert coverage[0].status == "indexed"


def test_chunk_generic_uses_header_aware_with_attached_metadata(tmp_path) -> None:
    config = _config("airflow", "Apache Airflow Documentation")
    preparer = GithubSourcePreparer(config, tmp_path, _GENERATION)
    record = _record(
        tmp_path / "repo",
        "docs/apache-airflow/start.rst",
        "guide",
        "conceptual",
        "Starting Airflow\n===============\n\nReal content body.\n\nIntroduction\n------------\n\nMore useful body here.\n",
    )
    manifest = _manifest(tmp_path / "repo", [record])

    chunks, coverage = asyncio.run(preparer._chunk_manifest(manifest))

    assert chunks
    for chunk in chunks:
        assert chunk.source_commit == _COMMIT
        assert chunk.index_generation == _GENERATION
        assert chunk.source_name == "Apache Airflow Documentation"
        assert chunk.doc_type == "guide"
    assert coverage[0].status == "indexed"
