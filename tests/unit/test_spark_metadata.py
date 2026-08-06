"""Phase 5 tests: metadata and provenance derivation and propagation."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import SparkSourceConfig
from data_engineering_copilot.infrastructure.spark_source_resolver import SparkFileRecord
from data_engineering_copilot.services.spark_metadata import derive_spark_metadata


def _source() -> SparkSourceConfig:
    return SparkSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4",
        license="Apache-2.0",
        streams=(),
    )


def _record(
    relative_path: str, stream: str = "api", doc_type: str = "api_reference", language: str = "python"
) -> SparkFileRecord:
    return SparkFileRecord(
        stream=stream,
        relative_path=relative_path,
        absolute_path=Path(relative_path),
        doc_type=doc_type,
        language=language,
        source_url=f"https://raw.githubusercontent.com/apache/spark/{_source().commit}/{relative_path}",
    )


def test_derive_metadata_basic() -> None:
    meta = derive_spark_metadata(
        _record("python/pyspark/sql/functions.py"),
        _source(),
        title="Functions",
        text="",
    )
    assert meta.doc_type == "api_reference"
    assert meta.language == "python"
    assert meta.spark_version == "4.0.0"
    assert meta.module == "pyspark.sql.functions"
    assert meta.source_commit == "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"
    assert meta.file_path == "python/pyspark/sql/functions.py"
    assert meta.license == "Apache-2.0"


def test_derive_metadata_guide() -> None:
    meta = derive_spark_metadata(
        _record("docs/sql-guide.md", stream="guides", doc_type="guide", language="conceptual"),
        _source(),
        title="SQL Guide",
        text="",
    )
    assert meta.doc_type == "guide"
    assert meta.language == "conceptual"
    assert meta.module == ""


def test_derive_metadata_example_language() -> None:
    meta = derive_spark_metadata(
        _record("examples/src/main/scala/foo.scala", stream="examples", doc_type="code_example", language="scala"),
        _source(),
        title="Foo",
        text="",
    )
    assert meta.language == "scala"
    assert meta.spark_version == "4.0.0"


def test_derive_metadata_version_from_ref() -> None:
    source = SparkSourceConfig(
        name="Spark",
        repository="https://github.com/apache/spark.git",
        ref="v3.5.1",
        commit="a" * 40,
        license="Apache-2.0",
        streams=(),
    )
    meta = derive_spark_metadata(_record("docs/a.md", "guides", "guide", "conceptual"), source, "A", "")
    assert meta.spark_version == "3.5.1"


def test_derive_metadata_rejects_bad_commit() -> None:
    source = SparkSourceConfig(
        name="Spark",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit="not-a-sha",
        license="Apache-2.0",
        streams=(),
    )
    with pytest.raises(ValueError, match="commit"):
        derive_spark_metadata(_record("docs/a.md"), source, "A", "")


def test_derive_metadata_empty_version_for_mutable_ref() -> None:
    source = SparkSourceConfig(
        name="Spark",
        repository="https://github.com/apache/spark.git",
        ref="master",
        commit="b" * 40,
        license="Apache-2.0",
        streams=(),
    )
    meta = derive_spark_metadata(_record("docs/a.md", "guides", "guide", "conceptual"), source, "A", "")
    assert meta.spark_version == ""
